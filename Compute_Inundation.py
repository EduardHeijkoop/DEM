import numpy as np
import geopandas as gpd
from osgeo import gdal,osr,gdalconst
import os, sys
import datetime
import argparse
import subprocess
import warnings
import configparser

from dem_utils import get_lonlat_gdf,find_corner_points_gdf
from dem_utils import get_raster_extents,get_gsw
from inundation_utils import resample_vlm,resample_geoid,clip_coast
from inundation_utils import get_coastal_sealevel,get_sealevel_high
from inundation_utils import inundate_loc,sigma_to_quantiles

def main():
    warnings.simplefilter(action='ignore')
    config_file = 'dem_config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',help='Path to input DEM to run inundation on.')
    parser.add_argument('--loc_name',help='Name of location to run inundation on.')
    parser.add_argument('--machine',default='t',help='Machine to run on (t, b or local)')
    parser.add_argument('--N_cpus',help='Number of CPUs to use for parallel processing.',default=1,type=int)
    parser.add_argument('--downsample_res',help='Resolution at which to compute inundation.',default=None,type=float)
    parser.add_argument('--geoid',help='Path to geoid file to calculate orthometric heights with.',default=None)
    parser.add_argument('--vlm',help='Path to VLM file to propagate input file in time.',default=None)
    parser.add_argument('--clip_vlm',help='Clip DEM to VLM extents?',default=False,action='store_true')
    parser.add_argument('--icesat2',help='Path to ICESat-2 file to calculate coastal sea level with.')
    parser.add_argument('--sealevel_grid',help='Path to sea level grid to calculate coastal sea level with.')
    parser.add_argument('--grid_extents',help='Extents of grid to be used in calculation (x_min x_max y_min y_max)',nargs=4)
    parser.add_argument('--coastline',help='Path to coastline file to calculate coastal sea level on.')
    parser.add_argument('--clip_coast',help='Clip DEM to coastline?',default=False,action='store_true')
    parser.add_argument('--years',help='Years to compute inundation for.',nargs='*',default='2020')
    parser.add_argument('--rcp',help='RCP to use.',choices=['2.6','4.5','8.5'])
    parser.add_argument('--ssp',help='RCP to use.')
    parser.add_argument('--slr',help='Sea level rise to use.',nargs='*',default=None)
    parser.add_argument('--t0',help='Time to use as t0 to zero SLR.',default='2020')
    parser.add_argument('--return_period',help='Return period of CoDEC in years')
    parser.add_argument('--fes2014',help='Flag to use FES2014 max tidal heights.',default=False,action='store_true')
    parser.add_argument('--mhhw',help='Flag to use MHHW instead of max tidal heights.',default=False,action='store_true')
    parser.add_argument('--high_tide',help='Value to use for high tide.',default=None,type=float)
    parser.add_argument('--connectivity',help='Calculate inundation connectivity to sea?',default=False,action='store_true')
    parser.add_argument('--uncertainty',help='Calculate inundation uncertainty?',default=False,action='store_true')
    parser.add_argument('--sigma',help='Sigma value to use for uncertainty calculation.',default=None)
    parser.add_argument('--of',help='Output format to use.',choices=['shp','geojson'],default='shp')
    args = parser.parse_args()

    dem_file = args.input_file
    loc_name = args.loc_name
    machine_name = args.machine
    N_cpus = args.N_cpus
    downsample_res = args.downsample_res
    geoid_file = args.geoid
    vlm_file = args.vlm
    clip_vlm_flag = args.clip_vlm
    icesat2_file = args.icesat2
    sl_grid_file = args.sealevel_grid
    sl_grid_extents = args.grid_extents
    coastline_file = args.coastline
    clip_coast_flag = args.clip_coast
    years = args.years
    years = [int(yr) for yr in np.atleast_1d(years)]
    rcp = args.rcp
    ssp = args.ssp
    slr = args.slr
    if slr is not None:
        slr = [float(s) for s in np.atleast_1d(slr)]
    if args.t0 is not None:
        t0 = int(args.t0)
    if args.return_period is not None:
        return_period = int(args.return_period)
    else:
        return_period = None
    return_period_options = np.asarray([2,5,10,25,50,100,250,500,1000])
    fes2014_flag = args.fes2014
    mhhw_flag = args.mhhw
    high_tide = args.high_tide
    connectivity_flag = args.connectivity
    uncertainty_flag = args.uncertainty
    output_format = args.of

    if icesat2_file is not None and sl_grid_file is not None:
        print('ICESat-2 file and sea level grid given, cannot handle both!')
        sys.exit()
    if vlm_file is None:
        print('No VLM file supplied to propagate in time!')
        print('Still running inundation with sea level rise.')
    if sl_grid_file is not None and sl_grid_extents is None:
        print('Warning, selecting whole grid as input!')
        src_sl_grid = gdal.Open(sl_grid_file,gdalconst.GA_ReadOnly)
        sl_grid_extents = get_raster_extents(sl_grid_file,'global')
    if vlm_file is None and clip_vlm_flag == True:
        print('No VLM file supplied, but clipping desired!')
        sys.exit()
    if np.sum((ssp is not None, rcp is not None, slr is not None)) > 1:
        print('Please only select SSP, RCP or SLR!')
        sys.exit()
    if np.sum((ssp is not None, rcp is not None, slr is not None)) < 1:
        print('Please select one of SSP, RCP or SLR!')
        sys.exit()
    if np.sum((fes2014_flag == True, return_period is not None, high_tide is not None)) > 1:
        print('Cannot use FES2014, CoDEC and/or high tide together!')
        sys.exit()
    if (high_tide is None and fes2014_flag == False) and return_period not in return_period_options:
        print('Invalid return period selected!')
        print('Must be 2, 5, 10, 25, 50, 100, 250, 500 or 1000 years.')
        sys.exit()
    if uncertainty_flag == True:
        sigma = int(args.sigma)
        if sigma not in [1,2,3]:
            print('Invalid sigma value selected!')
            print('Must be 1, 2 or 3.')
            sys.exit()
    else:
        sigma = args.sigma
    if vlm_file is not None:
        try:
            vlm_rate = float(vlm_file)
            vlm_file = None
        except ValueError:
            vlm_rate = None
    else:
        vlm_rate = None
    
    if os.path.dirname(os.path.abspath(dem_file)).split('/')[-1] == 'Mosaic':
        inundation_dir = f'{"/".join(os.path.dirname(os.path.abspath(dem_file)).split("/")[:-1])}/Inundation/'
    else:
        inundation_dir = f'{os.path.dirname(os.path.abspath(dem_file))}/Inundation/'

    if not os.path.exists(inundation_dir):
        os.mkdir(inundation_dir)
    
    SROCC_dir = config.get('INUNDATION_PATHS','SROCC_dir')
    AR6_dir = config.get('INUNDATION_PATHS','AR6_dir')
    CODEC_file = config.get('INUNDATION_PATHS','CODEC_file')
    fes2014_file = config.get('INUNDATION_PATHS','fes2014_file')
    tmp_dir = config.get('GENERAL_PATHS','tmp_dir')
    gsw_dir = config.get('GENERAL_PATHS','gsw_dir')
    landmask_c_file = config.get('GENERAL_PATHS','landmask_c_file')
    osm_shp_file = config.get('GENERAL_PATHS','osm_shp_file')

    if machine_name == 'b':
        SROCC_dir = SROCC_dir.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
        AR6_dir = AR6_dir.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
        CODEC_file = CODEC_file.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
        fes2014_file = fes2014_file.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
        tmp_dir = tmp_dir.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
        gsw_dir = gsw_dir.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
        osm_shp_file = osm_shp_file.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
    elif machine_name == 'local':
        AR6_dir = AR6_dir.replace('/BhaltosMount/Bhaltos/EDUARD/NASA_SEALEVEL/DATABASE/','/media/heijkoop/DATA/')
        CODEC_file = CODEC_file.replace('/BhaltosMount/Bhaltos/EDUARD/NASA_SEALEVEL/DATABASE/','/media/heijkoop/DATA/')
        fes2014_file = fes2014_file.replace('/BhaltosMount/Bhaltos/EDUARD/DATA_REPOSITORY/','/media/heijkoop/DATA/')
        tmp_dir = tmp_dir.replace('/BhaltosMount/Bhaltos/EDUARD/','/home/heijkoop/Desktop/Projects/')
        gsw_dir = gsw_dir.replace('/BhaltosMount/Bhaltos/EDUARD/DATA_REPOSITORY/','/media/heijkoop/DATA/')
        landmask_c_file = landmask_c_file.replace('/home/eheijkoop/Scripts/','/media/heijkoop/DATA/Dropbox/TU/PhD/Github/')
        osm_shp_file = osm_shp_file.replace('/BhaltosMount/Bhaltos/EDUARD/DATA_REPOSITORY/','/media/heijkoop/DATA/')

    VLM_NODATA = config.getfloat('VLM_CONSTANTS','VLM_NODATA')
    N_PTS = config.getint('INUNDATION_CONSTANTS','N_PTS')
    INTERPOLATE_METHOD = config.get('INUNDATION_CONSTANTS','INTERPOLATE_METHOD')
    REGGRID_INTERPOLATE_METHOD = config.get('INUNDATION_CONSTANTS','REGGRID_INTERPOLATE_METHOD')
    ICESAT2_GRID_RESOLUTION = config.getfloat('INUNDATION_CONSTANTS','ICESAT2_GRID_RESOLUTION')
    GRID_ALGORITHM = config.get('INUNDATION_CONSTANTS','GRID_ALGORITHM')
    GRID_NODATA = config.getint('INUNDATION_CONSTANTS','GRID_NODATA')
    GRID_SMOOTHING = config.getfloat('INUNDATION_CONSTANTS','GRID_SMOOTHING')
    GRID_POWER = config.getfloat('INUNDATION_CONSTANTS','GRID_POWER')
    GRID_MAX_PTS = config.getint('INUNDATION_CONSTANTS','GRID_MAX_PTS')
    GRID_NUM_THREADS = config.getint('INUNDATION_CONSTANTS','GRID_NUM_THREADS')
    GRID_INTERMEDIATE_RES = config.getint('INUNDATION_CONSTANTS','GRID_INTERMEDIATE_RES')
    INUNDATION_NODATA = config.getfloat('INUNDATION_CONSTANTS','INUNDATION_NODATA')
    GSW_BUFFER = config.getfloat('INUNDATION_CONSTANTS','GSW_BUFFER')

    algorithm_dict = {'grid_algorithm':GRID_ALGORITHM,
        'grid_nodata':GRID_NODATA,
        'grid_smoothing':GRID_SMOOTHING,
        'grid_power':GRID_POWER,
        'grid_max_pts':GRID_MAX_PTS,
        'grid_num_threads':GRID_NUM_THREADS,
        'grid_res':GRID_INTERMEDIATE_RES
    }
    
    dir_dict = {'tmp_dir':tmp_dir,
        'inundation_dir':inundation_dir,
        'AR6_dir':AR6_dir
    }

    constants_dict = {'GRID_NODATA':GRID_NODATA,
        'REGGRID_INTERPOLATE_METHOD':REGGRID_INTERPOLATE_METHOD,
        'INTERPOLATE_METHOD':INTERPOLATE_METHOD,
        'ICESAT2_GRID_RESOLUTION':ICESAT2_GRID_RESOLUTION,
        'N_PTS':N_PTS,
        'GRID_INTERMEDIATE_RES':GRID_INTERMEDIATE_RES,
        'INUNDATION_NODATA':INUNDATION_NODATA,
        'CODEC_file':CODEC_file,
        'fes2014_file':fes2014_file,
        'output_format':output_format,
        'landmask_c_file':landmask_c_file,
        'osm_shp_file':osm_shp_file
    }

    flag_dict = {'return_period':return_period,
        'fes2014_flag':fes2014_flag,
        'mhhw_flag':mhhw_flag,
        'high_tide':high_tide,
        'connectivity_flag':connectivity_flag,
        'geoid_file':geoid_file
    }

    if loc_name is None:
        loc_name = '_'.join(dem_file.split('/')[-1].split('_')[0:2])
    src = gdal.Open(dem_file,gdalconst.GA_ReadOnly)
    dem_nodata = src.GetRasterBand(1).GetNoDataValue()
    epsg_code = osr.SpatialReference(wkt=src.GetProjection()).GetAttrValue('AUTHORITY',1)

    if ssp is not None:
        ssp = ssp.replace('ssp','').replace('SSP','').replace('.','').replace('-','')
        if ssp not in ['119','126','245','370','585']:
            print('Invalid SSP pathway selected!')
            sys.exit()

    quantiles = sigma_to_quantiles(sigma,uncertainty_flag)

    gdf_coast = gpd.read_file(coastline_file)
    lon_coast,lat_coast = get_lonlat_gdf(gdf_coast)
    idx_corners = find_corner_points_gdf(lon_coast,lat_coast,gdf_coast)
    lon_coast[idx_corners] = np.nan
    lat_coast[idx_corners] = np.nan
    gdf_coast = gdf_coast.to_crs(f'EPSG:{epsg_code}')
    x_coast,y_coast = get_lonlat_gdf(gdf_coast)
    x_coast[idx_corners] = np.nan
    y_coast[idx_corners] = np.nan
    
    print(f'Working on {loc_name}.')

    if vlm_file is not None:
        dem_file,vlm_resampled_file = resample_vlm(vlm_file,dem_file,clip_vlm_flag,VLM_NODATA)
    else:
        vlm_resampled_file = None
    if geoid_file is not None:
        dem_file = resample_geoid(geoid_file,dem_file,loc_name,dem_nodata)
    if clip_coast_flag == True:
        dem_file = clip_coast(dem_file,coastline_file,epsg_code,GRID_NODATA)

    vlm_dict = {'t0':t0,
        'vlm_rate':vlm_rate,
        'vlm_resampled_file':vlm_resampled_file
    }

    if downsample_res is not None:
        xres = src.GetGeoTransform()[1]
        if downsample_res < xres:
            print('Intermediate resolution must be larger than default resolution!')
        else:
            downsample_res_str = f'{downsample_res:.1f}'.replace('.','p')
            dem_downsampled_file = dem_file.replace('.tif',f'_resampled_{downsample_res_str}m.tif')
            resample_command_downsampled_dem = f'gdalwarp -q -overwrite -tr {downsample_res:.1f} {downsample_res:.1f} -r bilinear {dem_file} {dem_downsampled_file}'
            subprocess.run(resample_command_downsampled_dem,shell=True)
            dem_file = dem_downsampled_file
    
    lon_dem_min,lon_dem_max,lat_dem_min,lat_dem_max = get_raster_extents(dem_file,'global')

    t_start = datetime.datetime.now()
    print(f'Resampling DEM to {GRID_INTERMEDIATE_RES} meters.')
    dem_resampled_file = dem_file.replace('.tif',f'_resampled_{GRID_INTERMEDIATE_RES}m.tif')
    resample_dem_command = f'gdalwarp -q -overwrite -tr {GRID_INTERMEDIATE_RES} {GRID_INTERMEDIATE_RES} -r bilinear {dem_file} {dem_resampled_file}'
    subprocess.run(resample_dem_command,shell=True)
    src_resampled = gdal.Open(dem_resampled_file,gdalconst.GA_ReadOnly)
    t_end = datetime.datetime.now()
    delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
    delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
    print(f'Resampling DEM took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')

    dem_resampled_x_size = src_resampled.RasterXSize
    dem_resampled_y_size = src_resampled.RasterYSize
    xres_dem_resampled,yres_dem_resampled = src_resampled.GetGeoTransform()[1],-src_resampled.GetGeoTransform()[5]
    x_dem_resampled_min,x_dem_resampled_max,y_dem_resampled_min,y_dem_resampled_max = get_raster_extents(dem_resampled_file,'local')
    dx_dem_resampled = np.abs(x_dem_resampled_max - x_dem_resampled_min)
    dy_dem_resampled = np.abs(y_dem_resampled_max - y_dem_resampled_min)
    grid_max_dist = np.max((dx_dem_resampled,dy_dem_resampled))
    algorithm_dict['grid_max_dist'] = grid_max_dist

    dem_dict = {'xmin':x_dem_resampled_min,
                'xmax':x_dem_resampled_max,
                'xres':xres_dem_resampled,
                'ymin':y_dem_resampled_min,
                'ymax':y_dem_resampled_max,
                'yres':yres_dem_resampled
    }

    resampled_dict = {'src_resampled':src_resampled,
        'dem_resampled_x_size':dem_resampled_x_size,
        'dem_resampled_y_size':dem_resampled_y_size
    }

    x_coast,y_coast,lon_coast,lat_coast,h_coast,output_file_coastline = get_coastal_sealevel(loc_name,x_coast,y_coast,lon_coast,lat_coast,sl_grid_extents,sl_grid_file,icesat2_file,dir_dict,constants_dict,epsg_code,geoid_file)
    sealevel_high_grid_full_res = get_sealevel_high(dem_file,high_tide,return_period,fes2014_flag,mhhw_flag,loc_name,epsg_code,
                                                    x_coast,y_coast,lon_coast,lat_coast,
                                                    dir_dict,constants_dict,dem_dict,algorithm_dict,resampled_dict)

    if connectivity_flag == True:
        if 'NDWI' in coastline_file:
            surface_water_file = coastline_file.replace('Coastline','Surface_Water')
            gdf_surface_water = gpd.read_file(surface_water_file)
            gdf_surface_water = gdf_surface_water.to_crs(f'EPSG:{epsg_code}')
        else:
            gdf_surface_water,surface_water_file = get_gsw(inundation_dir,tmp_dir,gsw_dir,epsg_code,lon_dem_min,lon_dem_max,lat_dem_min,lat_dem_max,loc_name)
            if loc_name not in os.path.basename(surface_water_file):
                surface_water_file = f'{os.path.dirname(surface_water_file)}/{loc_name}_{os.path.basename(surface_water_file)}'
        surface_water_file_buffered = surface_water_file.replace('.shp',f"_buffered_{int(GSW_BUFFER)}m.shp")
        gdf_surface_water_buffered = gdf_surface_water.buffer(GSW_BUFFER)
        gdf_surface_water_buffered.to_file(surface_water_file_buffered)

    inundate_loc(dem_file,slr,years,quantiles,loc_name,high_tide,ssp,
                 x_coast,y_coast,h_coast,
                 dir_dict,flag_dict,constants_dict,dem_dict,algorithm_dict,vlm_dict,
                 output_file_coastline,epsg_code,gdf_surface_water,sealevel_high_grid_full_res,N_cpus)

    print(f'Finished with {loc_name} at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.')

if __name__ == '__main__':
    main()