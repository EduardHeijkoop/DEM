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
    parser.add_argument('--sigma',help='Sigma value to use for uncertainty calculation.')
    parser.add_argument('--of',help='Output format to use.',choices=['shp','geojson'],default='shp')
    args = parser.parse_args()

    dem_file = args.input_file
    loc_name = args.loc_name
    machine_name = args.machine
    N_cpus = args.N_cpus
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
        projection_select = 'AR6'
        ssp = ssp.replace('ssp','').replace('SSP','').replace('.','').replace('-','')
        if ssp not in ['119','126','245','370','585']:
            print('Invalid SSP pathway selected!')
            sys.exit()
    elif rcp is not None:
        projection_select = 'SROCC'
        if rcp not in ['2.6','4.5','8.5']:
            print('Invalid RCP pathway selected!')
            sys.exit()

    quantiles = sigma_to_quantiles(sigma,uncertainty_flag)

    gdf_coast = gpd.read_file(coastline_file)
    epsg_coastline = gdf_coast.crs.to_epsg()
    lon_coast,lat_coast = get_lonlat_gdf(gdf_coast)
    idx_corners = find_corner_points_gdf(lon_coast,lat_coast,gdf_coast)
    lon_coast[idx_corners] = np.nan
    lat_coast[idx_corners] = np.nan
    gdf_coast = gdf_coast.to_crs(f'EPSG:{epsg_code}')
    x_coast,y_coast = get_lonlat_gdf(gdf_coast)
    x_coast[idx_corners] = np.nan
    y_coast[idx_corners] = np.nan
    x_coast_orig = x_coast.copy()
    y_coast_orig = y_coast.copy()
    
    print(f'Working on {loc_name}.')

    if vlm_file is not None:
        dem_file,vlm_resampled_file = resample_vlm(vlm_file,dem_file,clip_vlm_flag,VLM_NODATA)
    if geoid_file is not None:
        dem_file = resample_geoid(geoid_file,dem_file,loc_name,dem_nodata)
    if clip_coast_flag == True:
        dem_file = clip_coast(dem_file,coastline_file,epsg_code,GRID_NODATA)

    vlm_dict = {'t0':t0,
        'vlm_rate':vlm_rate,
        'vlm_resampled_file':vlm_resampled_file
    }
    
    lon_dem_min,lon_dem_max,lat_dem_min,lat_dem_max = get_raster_extents(dem_file,'global')
    lon_center_dem = (lon_dem_min + lon_dem_max)/2
    lat_center_dem = (lat_dem_min + lat_dem_max)/2

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

    # dem_x_size = src.RasterXSize
    # dem_y_size = src.RasterYSize
    # src_proj = src.GetProjection()
    # src_geotransform = src.GetGeoTransform()
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


    # t_start = datetime.datetime.now()
    # print('Generating coastal sea level grid...')
    # if sl_grid_extents is not None:
    #     h_coast = interpolate_grid(lon_coast,lat_coast,sl_grid_file,sl_grid_extents,loc_name,tmp_dir,grid_nodata=GRID_NODATA,method=REGGRID_INTERPOLATE_METHOD)
    #     idx_fillvalue = h_coast==GRID_NODATA
    #     idx_keep = ~np.logical_or(idx_fillvalue,np.isnan(h_coast))
    #     x_coast = x_coast[idx_keep]
    #     y_coast = y_coast[idx_keep]
    #     h_coast = h_coast[idx_keep]
    #     lon_coast = lon_coast[idx_keep]
    #     lat_coast = lat_coast[idx_keep]
    #     if loc_name in sl_grid_file:
    #         output_file_coastline = f'{tmp_dir}{os.path.basename(sl_grid_file).replace(".tif","_subset_interpolated_coastline.csv")}'
    #     else:
    #         output_file_coastline = f'{tmp_dir}{loc_name}_{os.path.basename(sl_grid_file).replace(".tif","_subset_interpolated_coastline.csv")}'
    #     if geoid_file is not None:
    #         h_geoid = interpolate_grid(lon_coast,lat_coast,geoid_file,None,loc_name,tmp_dir,grid_nodata=GRID_NODATA,method=REGGRID_INTERPOLATE_METHOD)
    #         h_geoid = h_geoid[idx_keep]
    #         h_coast = h_coast - h_geoid
    #         output_file_coastline = output_file_coastline.replace('.csv','_orthometric.csv')
    # elif icesat2_file is not None:
    #     df_icesat2 = pd.read_csv(icesat2_file,header=None,names=['lon','lat','height','time'],dtype={'lon':'float','lat':'float','height':'float','time':'str'})
    #     x_icesat2_grid_array,y_icesat2_grid_array,h_icesat2_grid_array = create_icesat2_grid(df_icesat2,epsg_code,geoid_file,tmp_dir,loc_name,ICESAT2_GRID_RESOLUTION,N_PTS)
    #     h_coast = interpolate_points(x_icesat2_grid_array,y_icesat2_grid_array,h_icesat2_grid_array,x_coast,y_coast,INTERPOLATE_METHOD)
    #     idx_keep = ~np.isnan(x_coast)
    #     x_coast = x_coast[idx_keep]
    #     y_coast = y_coast[idx_keep]
    #     if loc_name in icesat2_file:
    #         output_file_coastline = f'{tmp_dir}{os.path.basename(icesat2_file).replace(".txt",f"_subset_{INTERPOLATE_METHOD}BivariateSpline_coastline.csv")}'
    #     else:
    #         output_file_coastline = f'{tmp_dir}{loc_name}_{os.path.basename(icesat2_file).replace(".txt",f"_subset_{INTERPOLATE_METHOD}BivariateSpline_coastline.csv")}'
    #     if geoid_file is not None:
    #         output_file_coastline = output_file_coastline.replace('.csv','_orthometric.csv')
    # t_end = datetime.datetime.now()
    # delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
    # delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
    # print(f'Generating coastal sea level took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')


    x_coast,y_coast,h_coast,output_file_coastline = get_coastal_sealevel(loc_name,sl_grid_extents,sl_grid_file,icesat2_file,dir_dict,constants_dict,epsg_code,geoid_file)
    sealevel_high_grid_full_res = get_sealevel_high(dem_file,high_tide,return_period,fes2014_flag,mhhw_flag,loc_name,epsg_code,
                                                    lon_coast,lat_coast,x_coast,y_coast,
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


    # if slr is not None:
    #     for slr_value in slr:
    #         t_start = datetime.datetime.now()
    #         print(f'\nCreating inundation for {slr_value:.2f} m...')
    #         slr_value_str = f'SLR_{slr_value:.2f}m'.replace('.','p').replace('-','neg')
    #         if high_tide is not None:
    #             high_tide_str = f'{high_tide:.2f}m'.replace('.','p')
    #             output_inundation_file = f'{inundation_dir}{loc_name}_Inundation_{slr_value_str}_HT_{high_tide_str}.tif'
    #         elif return_period is not None:
    #             output_inundation_file = f'{inundation_dir}{loc_name}_Inundation_{slr_value_str}_CoDEC_RP_{return_period}_yrs.tif'
    #         elif fes2014_flag == True:
    #             output_inundation_file = f'{inundation_dir}{loc_name}_Inundation_{slr_value_str}_FES2014.tif'
    #             if mhhw_flag == True:
    #                 output_inundation_file = output_inundation_file.replace('FES2014','FES2014_MHHW')
    #         if geoid_file is not None:
    #             output_inundation_file = output_inundation_file.replace('_Inundation_','_Orthometric_Inundation_')
    #         if vlm_file is None:
    #             output_inundation_file = output_inundation_file.replace('_Inundation_','_Inundation_No_VLM_')
    #         h_coast_slr = h_coast + slr_value
    #         output_file_coastline_slr = output_file_coastline.replace('.csv',f'_{slr_value_str}.csv')
    #         np.savetxt(output_file_coastline_slr,np.c_[x_coast,y_coast,h_coast_slr],fmt='%f',delimiter=',',comments='')
    #         sl_grid_file_intermediate_res = csv_to_grid(output_file_coastline_slr,algorithm_dict,x_dem_resampled_min,x_dem_resampled_max,xres_dem_resampled,y_dem_resampled_min,y_dem_resampled_max,yres_dem_resampled,epsg_code)
    #         sl_grid_file_full_res = sl_grid_file_intermediate_res.replace(f'_{GRID_INTERMEDIATE_RES}m','')
    #         resample_raster(sl_grid_file_intermediate_res,dem_file,sl_grid_file_full_res,quiet_flag=True)
    #         if vlm_file is not None:
    #             dt = int(yr - t0)
    #             inundation_command = f'gdal_calc.py --quiet -A {dem_file} -B {vlm_resampled_file} -C {sl_grid_file_full_res} -D {sealevel_high_grid_full_res} --outfile={tmp_dir}tmp_inundation.tif --calc="A+B*{dt} < C+D" --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --co "TILED=YES"'
    #             edit_nodata_command = f'gdal_calc.py --quiet -A {output_inundation_file} --outfile={output_inundation_file} --calc="A*(A<1E38) + {INUNDATION_NODATA}*(A>1E38)" --NoDataValue={INUNDATION_NODATA} --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --co "TILED=YES"'
    #         elif vlm_rate is not None:
    #             inundation_command = f'gdal_calc.py --quiet -A {dem_file} -C {sl_grid_file_full_res} -D {sealevel_high_grid_full_res} --outfile={output_inundation_file} --calc="A+{vlm_rate}*{dt} < C+D" --NoDataValue={INUNDATION_NODATA} --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --co "TILED=YES"'
    #             edit_nodata_command = None
    #         else:
    #             inundation_command = f'gdal_calc.py --quiet -A {dem_file} -C {sl_grid_file_full_res} -D {sealevel_high_grid_full_res} --outfile={output_inundation_file} --calc="A < C+D" --NoDataValue={INUNDATION_NODATA} --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --co "TILED=YES"'
    #             edit_nodata_command = None
    #         subprocess.run(inundation_command,shell=True)
    #         if edit_nodata_command is not None:
    #             subprocess.run(edit_nodata_command,shell=True)
    #             subprocess.run(f'rm {tmp_dir}tmp_inundation.tif',shell=True)
    #         output_inundation_vec_file = output_inundation_file.replace('.tif',f'.{output_format}')
    #         polygonize_command = f'gdal_polygonize.py -q {output_inundation_file} {output_inundation_vec_file}'
    #         subprocess.run(polygonize_command,shell=True)
    #         t_end = datetime.datetime.now()
    #         delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
    #         delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
    #         print(f'Inundation creation took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')
    #         if connectivity_flag == True:
    #             print('Computing connectivity to the ocean...')
    #             t_start = datetime.datetime.now()
    #             compute_connectivity(output_inundation_vec_file,gdf_surface_water_buffered)
    #             t_end = datetime.datetime.now()
    #             delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
    #             delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
    #             print(f'Connectivity took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')
    #         subprocess.run(f'rm {output_file_coastline_slr}',shell=True)
    #         subprocess.run(f'rm {output_file_coastline_slr.replace(".csv",".vrt")}',shell=True)
    #         subprocess.run(f'rm {sl_grid_file_intermediate_res}',shell=True)
    #         subprocess.run(f'rm {sl_grid_file_full_res}',shell=True)
    # else:
    #     for yr,quantile_select in itertools.product(years,quantiles):
    #         t_start = datetime.datetime.now()
    #         if high_tide is not None:
    #             output_inundation_file = f'{inundation_dir}{loc_name}_Inundation_{yr}_PROJECTION_METHOD_HT_{high_tide:.2f}.tif'.replace('.','p')
    #         elif return_period is not None:
    #             output_inundation_file = f'{inundation_dir}{loc_name}_Inundation_{yr}_PROJECTION_METHOD_CoDEC_RP_{return_period}_yrs.tif'
    #         elif fes2014_flag == True:
    #             output_inundation_file = f'{inundation_dir}{loc_name}_Inundation_{yr}_PROJECTION_METHOD_FES2014.tif'
    #             if mhhw_flag == True:
    #                 output_inundation_file = output_inundation_file.replace('FES2014','FES2014_MHHW')
    #         if projection_select == 'SROCC':
    #             print(f'\nCreating inundation in {yr} using RCP{rcp}...')
    #             output_inundation_file = output_inundation_file.replace('PROJECTION_METHOD',f'SROCC_RCP_{str(rcp).replace(".","p")}')
    #             lon_projection,lat_projection,slr_projection = upscale_SROCC_grid(SROCC_dir,dem_file,rcp,t0,yr,)
    #         elif projection_select == 'AR6':
    #             output_inundation_file = output_inundation_file.replace('PROJECTION_METHOD',f'AR6_SSP_{ssp}')
    #             if quantile_select < 0.5:
    #                 output_inundation_file = output_inundation_file.replace('_Inundation_',f'_Inundation_Minus_{sigma}sigma_')
    #                 print(f'\nCreating inundation in {yr} using SSP{ssp} (Median minus {sigma} sigma)...')
    #             elif quantile_select > 0.5:
    #                 output_inundation_file = output_inundation_file.replace('_Inundation_',f'_Inundation_Plus_{sigma}sigma_')
    #                 print(f'\nCreating inundation in {yr} using SSP{ssp} (Median plus {sigma} sigma)...')
    #             else:
    #                 print(f'\nCreating inundation in {yr} using SSP{ssp}...')
    #             lon_projection,lat_projection,slr_projection = upscale_ar6_data(AR6_dir,tmp_dir,landmask_c_file,dem_file,ssp,osm_shp_file,yr,quantile_select=quantile_select)
    #         if geoid_file is not None:
    #             output_inundation_file = output_inundation_file.replace('_Inundation_','_Orthometric_Inundation_')
    #         if vlm_file is None:
    #             output_inundation_file = output_inundation_file.replace('_Inundation_','_Inundation_No_VLM_')
    #         h_projection_coast = interpolate_points(lon_projection,lat_projection,slr_projection,x_coast,y_coast,INTERPOLATE_METHOD)
    #         h_coast_yr = h_coast + h_projection_coast
    #         output_file_coastline_yr = output_file_coastline.replace('.csv',f'_{yr}.csv')
    #         np.savetxt(output_file_coastline_yr,np.c_[x_coast,y_coast,h_coast_yr],fmt='%f',delimiter=',',comments='')
    #         sl_grid_file_intermediate_res = csv_to_grid(output_file_coastline_yr,algorithm_dict,x_dem_resampled_min,x_dem_resampled_max,xres_dem_resampled,y_dem_resampled_min,y_dem_resampled_max,yres_dem_resampled,epsg_code)
    #         sl_grid_file_full_res = sl_grid_file_intermediate_res.replace(f'_{GRID_INTERMEDIATE_RES}m','')
    #         resample_raster(sl_grid_file_intermediate_res,dem_file,sl_grid_file_full_res,quiet_flag=True)
    #         if vlm_file is not None:
    #             dt = int(yr - t0)
    #             inundation_command = f'gdal_calc.py --quiet -A {dem_file} -B {vlm_resampled_file} -C {sl_grid_file_full_res} -D {sealevel_high_grid_full_res} --outfile={output_inundation_file} --calc="A+B*{dt} < C+D" --NoDataValue={INUNDATION_NODATA} --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --co "TILED=YES"'
    #         elif vlm_rate is not None:
    #             inundation_command = f'gdal_calc.py --quiet -A {dem_file} -C {sl_grid_file_full_res} -D {sealevel_high_grid_full_res} --outfile={output_inundation_file} --calc="A+{vlm_rate}*{dt} < C+D" --NoDataValue={INUNDATION_NODATA} --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --co "TILED=YES"'
    #         else:
    #             inundation_command = f'gdal_calc.py --quiet -A {dem_file} -C {sl_grid_file_full_res} -D {sealevel_high_grid_full_res} --outfile={output_inundation_file} --calc="A < C+D" --NoDataValue={INUNDATION_NODATA} --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --co "TILED=YES"'
    #         subprocess.run(inundation_command,shell=True)
    #         output_inundation_vec_file = output_inundation_file.replace('.tif',f'.{output_format}')
    #         polygonize_command = f'gdal_polygonize.py -q {output_inundation_file} {output_inundation_vec_file}'
    #         subprocess.run(polygonize_command,shell=True)
    #         t_end = datetime.datetime.now()
    #         delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
    #         delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
    #         print(f'Inundation creation took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')
    #         if connectivity_flag == True:
    #             print('Computing connectivity to the ocean...')
    #             t_start = datetime.datetime.now()
    #             compute_connectivity(output_inundation_vec_file,gdf_surface_water_buffered)
    #             t_end = datetime.datetime.now()
    #             delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
    #             delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
    #             print(f'Connectivity took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')
    #         subprocess.run(f'rm {output_file_coastline_yr}',shell=True)
    #         subprocess.run(f'rm {output_file_coastline_yr.replace(".csv",".vrt")}',shell=True)
    #         subprocess.run(f'rm {sl_grid_file_intermediate_res}',shell=True)
    #         subprocess.run(f'rm {sl_grid_file_full_res}',shell=True)
    # if os.path.isfile(sealevel_csv_output):
    #     subprocess.run(f'rm {sealevel_csv_output}',shell=True)
    #     subprocess.run(f'rm {sealevel_csv_output.replace(".csv",".vrt")}',shell=True)
    # subprocess.run(f'rm {sealevel_high_grid_intermediate_res}',shell=True)
    # subprocess.run(f'rm {sealevel_high_grid_full_res}',shell=True)
    print(f'Finished with {loc_name} at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.')

if __name__ == '__main__':
    main()