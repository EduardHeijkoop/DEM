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
    # parser.add_argument('--machine',default='t',help='Machine to run on (t, b or local)')
    parser.add_argument('--config',default='dem_config.ini',help='Path to configuration file.')
    parser.add_argument('--N_cpus',help='Number of CPUs to use for parallel processing.',default=1,type=int)
    parser.add_argument('--downsample_res',help='Resolution at which to compute inundation.',default=None,type=float)
    parser.add_argument('--geoid',help='Path to geoid file to calculate orthometric heights with.',default=None)
    parser.add_argument('--vlm',help='Path to VLM file to propagate input file in time.',default=None)
    parser.add_argument('--clip_vlm',help='Clip DEM to VLM extents?',default=False,action='store_true')
    parser.add_argument('--t0',help='Time to use as t0 to zero VLM/DEM.',default='2020')
    # parser.add_argument('--icesat2',help='Path to ICESat-2 file to calculate coastal sea level with.')
    # parser.add_argument('--sealevel_grid',help='Path to sea level grid to calculate coastal sea level with.') #move to config file
    parser.add_argument('--grid_extents',help='Extents of grid to be used in calculation (x_min x_max y_min y_max)',nargs=4)
    parser.add_argument('--coastline',help='Path to coastline file to calculate coastal sea level on.')
    parser.add_argument('--clip_coast',help='Clip DEM to coastline?',default=False,action='store_true')
    # parser.add_argument('--rcp',help='RCP to use.',choices=['2.6','4.5','8.5'])
    parser.add_argument('--ssp',help='SSP to use.',choices=['119','126','245','370','585'])
    parser.add_argument('--years',help='Years to compute inundation for.',nargs='*',default='2020')
    parser.add_argument('--confidence',help='Confidence level for which to to use SSP.',choices=['low','medium'],default='medium')
    parser.add_argument('--slr',help='Sea level rise to use.',nargs='*',default=None)
    parser.add_argument('--return_period',help='Return period of CoDEC in years')
    parser.add_argument('--fes2014',help='Flag to use FES2014 max tidal heights.',default=False,action='store_true')
    parser.add_argument('--fes2022',help='Flag to use FES2022 max tidal heights.',default=False,action='store_true')
    parser.add_argument('--mhhw',help='Flag to use MHHW instead of max tidal heights.',default=False,action='store_true')
    parser.add_argument('--high_tide',help='Value to use for high tide.',default=None,type=float)
    parser.add_argument('--connectivity',help='Calculate inundation connectivity to sea?',default=False,action='store_true')
    parser.add_argument('--separate',help='Separate disconnected file?',default=False,action='store_true')
    parser.add_argument('--surface_water',help='Path to surface water file to calculate connectivity with.')
    parser.add_argument('--uncertainty',help='Calculate inundation uncertainty?',default=False,action='store_true')
    parser.add_argument('--sigma',help='Sigma value to use for uncertainty calculation.',default=None)
    parser.add_argument('--of',help='Output format to use.',choices=['shp','geojson'],default='shp')
    args = parser.parse_args()

    dem_file = args.input_file
    loc_name = args.loc_name
    # machine_name = args.machine
    N_cpus = args.N_cpus
    downsample_res = args.downsample_res
    geoid_file = args.geoid
    vlm_file = args.vlm
    clip_vlm_flag = args.clip_vlm
    # icesat2_file = args.icesat2
    # sl_grid_file = args.sealevel_grid
    sl_grid_extents = args.grid_extents
    coastline_file = args.coastline
    clip_coast_flag = args.clip_coast
    years = args.years
    years = [int(yr) for yr in np.atleast_1d(years)]
    # rcp = args.rcp
    ssp = args.ssp
    confidence_level = args.confidence
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
    fes2022_flag = args.fes2022
    mhhw_flag = args.mhhw
    high_tide = args.high_tide
    connectivity_flag = args.connectivity
    separate_flag = args.separate
    surface_water_input_file = args.surface_water 
    uncertainty_flag = args.uncertainty
    output_format = args.of

    # if icesat2_file is not None and sl_grid_file is not None:
    #     print('ICESat-2 file and sea level grid given, cannot handle both!')
    #     sys.exit()
    if vlm_file is None:
        print('No VLM file supplied to propagate in time!')
        print('Still running inundation with sea level rise.')
    if vlm_file is None and clip_vlm_flag == True:
        print('No VLM file supplied, but clipping desired!')
        sys.exit()
    if np.sum((ssp is not None, slr is not None)) > 1:
        print('Please only select SSP or SLR!')
        sys.exit()
    if np.sum((ssp is not None, slr is not None)) < 1:
        print('Please select either SSP or SLR!')
        sys.exit()
    if np.sum((fes2014_flag == True, fes2022_flag == True, return_period is not None, high_tide is not None)) > 1:
        print('Cannot use FES2014, CoDEC and/or high tide together!')
        sys.exit()
    if (high_tide is None and fes2014_flag == False and fes2022_flag == False) and return_period not in return_period_options:
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

    if loc_name is None:
        loc_name = '_'.join(dem_file.split('/')[-1].split('_')[0:2])

    if os.path.dirname(os.path.abspath(dem_file)).split('/')[-1] == 'Mosaic':
        inundation_dir = f'{"/".join(os.path.dirname(os.path.abspath(dem_file)).split("/")[:-1])}/Inundation/'
    else:
        inundation_dir = f'{os.path.dirname(os.path.abspath(dem_file))}/Inundation/'

    if not os.path.exists(inundation_dir):
        os.mkdir(inundation_dir)

    write_file = f'{inundation_dir}{loc_name}_Input_{datetime.datetime.strftime(datetime.datetime.now(),"%Y%m%dT%H%M%S")}.txt'
    f_write = open(write_file,'w')
    args_dict = vars(args)
    for k in args_dict.keys():
        f_write.write(f'{k}: {args_dict[k]}\n')
    f_write.close()
    
    tmp_dir = config.get('GENERAL_PATHS','tmp_dir')
    gsw_dir = config.get('GENERAL_PATHS','gsw_dir')
    landmask_c_file = config.get('GENERAL_PATHS','landmask_c_file')
    osm_shp_file = config.get('GENERAL_PATHS','osm_shp_file')
    # SROCC_dir = config.get('INUNDATION_PATHS','SROCC_dir')
    sl_grid_file = config.get('INUNDATION_PATHS','sealevel_grid')
    AR6_dir = config.get('INUNDATION_PATHS','AR6_dir')
    CODEC_file = config.get('INUNDATION_PATHS','CODEC_file')
    if fes2014_flag is not None:
        fes_file = config.get('INUNDATION_PATHS','fes2014_file')
    elif fes2022_flag is not None:
        fes_file = config.get('INUNDATION_PATHS','fes2022_file')
    VLM_NODATA = config.getfloat('VLM_CONSTANTS','VLM_NODATA')
    # ICESAT2_GRID_RESOLUTION = config.getfloat('INUNDATION_CONSTANTS','ICESAT2_GRID_RESOLUTION')
    # N_PTS = config.getint('INUNDATION_CONSTANTS','N_PTS')
    INTERPOLATE_METHOD = config.get('INUNDATION_CONSTANTS','INTERPOLATE_METHOD')
    GRID_NUM_THREADS = config.getint('INUNDATION_CONSTANTS','GRID_NUM_THREADS')
    GRID_INTERMEDIATE_RES = config.getint('INUNDATION_CONSTANTS','GRID_INTERMEDIATE_RES')
    GRID_ALGORITHM = config.get('INUNDATION_CONSTANTS','GRID_ALGORITHM')
    GRID_SMOOTHING = config.getfloat('INUNDATION_CONSTANTS','GRID_SMOOTHING')
    GRID_POWER = config.getfloat('INUNDATION_CONSTANTS','GRID_POWER')
    GRID_NODATA = config.getint('INUNDATION_CONSTANTS','GRID_NODATA')
    GRID_MAX_PTS = config.getint('INUNDATION_CONSTANTS','GRID_MAX_PTS')
    INUNDATION_NODATA = config.getfloat('INUNDATION_CONSTANTS','INUNDATION_NODATA')
    GSW_BUFFER = config.getfloat('INUNDATION_CONSTANTS','GSW_BUFFER')
    REGGRID_INTERPOLATE_METHOD = config.get('INUNDATION_CONSTANTS','REGGRID_INTERPOLATE_METHOD')

    if sl_grid_extents is None:
        print('Warning, selecting whole grid as input!')
        # src_sl_grid = gdal.Open(sl_grid_file,gdalconst.GA_ReadOnly)
        sl_grid_extents = get_raster_extents(sl_grid_file,'global')

    algorithm_dict = {
        'grid_num_threads':GRID_NUM_THREADS,
        'grid_res':GRID_INTERMEDIATE_RES,
        'grid_algorithm':GRID_ALGORITHM,
        'grid_smoothing':GRID_SMOOTHING,
        'grid_power':GRID_POWER,
        'grid_nodata':GRID_NODATA,
        'grid_max_pts':GRID_MAX_PTS
    }
    
    dir_dict = {
        'inundation_dir':inundation_dir,
        'tmp_dir':tmp_dir,
        'AR6_dir':AR6_dir
    }

    constants_dict = {
        'landmask_c_file':landmask_c_file,
        'osm_shp_file':osm_shp_file,
        'CODEC_file':CODEC_file,
        'fes_file':fes_file,
        'interpolate_method':INTERPOLATE_METHOD,
        'grid_intermediate_res':GRID_INTERMEDIATE_RES,
        'grid_nodata':GRID_NODATA,
        'inundation_nodata':INUNDATION_NODATA,
        'reggrid_interpolate_method':REGGRID_INTERPOLATE_METHOD,
        'output_format':output_format
    }

    flag_dict = {
        'geoid_file':geoid_file,
        'return_period':return_period,
        'fes2014_flag':fes2014_flag,
        'fes2022_flag':fes2022_flag,
        'mhhw_flag':mhhw_flag,
        'high_tide':high_tide,
        'connectivity_flag':connectivity_flag,
        'separate_flag':separate_flag
    }

    src = gdal.Open(dem_file,gdalconst.GA_ReadOnly)
    dem_nodata = src.GetRasterBand(1).GetNoDataValue()
    epsg_code = osr.SpatialReference(wkt=src.GetProjection()).GetAttrValue('AUTHORITY',1)

    # if ssp is not None:
    #     ssp = ssp.replace('ssp','').replace('SSP','').replace('.','').replace('-','')
    #     if ssp not in ['119','126','245','370','585']:
    #         print('Invalid SSP pathway selected!')
    #         sys.exit()

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
            if vlm_resampled_file is not None:
                vlm_downsampled_file = vlm_resampled_file.replace('.tif',f'_{downsample_res_str}m.tif')
                resample_command_downsampled_vlm = f'gdalwarp -q -overwrite -tr {downsample_res:.1f} {downsample_res:.1f} -r bilinear {vlm_resampled_file} {vlm_downsampled_file}'
                subprocess.run(resample_command_downsampled_vlm,shell=True)
                vlm_resampled_file = vlm_downsampled_file
    
    vlm_dict = {'t0':t0,
        'vlm_rate':vlm_rate,
        'vlm_resampled_file':vlm_resampled_file
    }

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

    x_coast,y_coast,lon_coast,lat_coast,h_coast,output_file_coastline = get_coastal_sealevel(loc_name,x_coast,y_coast,lon_coast,lat_coast,sl_grid_extents,sl_grid_file,dir_dict,constants_dict,geoid_file)
    sealevel_high_grid_full_res = get_sealevel_high(dem_file,flag_dict,loc_name,epsg_code,
                                                    x_coast,y_coast,lon_coast,lat_coast,
                                                    dir_dict,constants_dict,dem_dict,algorithm_dict,resampled_dict)

    if connectivity_flag == True:
        if surface_water_input_file is not None:
            surface_water_file = surface_water_input_file
            gdf_surface_water = gpd.read_file(surface_water_input_file)
            gdf_surface_water = gdf_surface_water.to_crs(f'EPSG:{epsg_code}')
        elif 'NDWI' in coastline_file:
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

    inundate_loc(dem_file,slr,years,quantiles,loc_name,ssp,confidence_level,
                 x_coast,y_coast,h_coast,
                 dir_dict,flag_dict,constants_dict,dem_dict,algorithm_dict,vlm_dict,
                 output_file_coastline,epsg_code,gdf_surface_water_buffered,sealevel_high_grid_full_res,N_cpus)

    print(f'Finished with {loc_name} at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.')

if __name__ == '__main__':
    main()