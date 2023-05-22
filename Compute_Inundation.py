import numpy as np
import pandas as pd
import netCDF4 as nc
import geopandas as gpd
import glob
from osgeo import gdal,osr,gdalconst
import os, sys
import datetime
import argparse
import subprocess
import matplotlib.pyplot as plt
import warnings
import configparser
import itertools
from dem_utils import get_lonlat_gdf,find_corner_points_gdf
from dem_utils import get_raster_extents,resample_raster,get_gsw
from inundation_utils import create_icesat2_grid, interpolate_grid, interpolate_points,get_codec,get_fes,csv_to_grid
from inundation_utils import upscale_SROCC_grid,upscale_ar6_data,compute_connectivity


def main():
    warnings.simplefilter(action='ignore')
    config_file = 'dem_config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dem',help='Path to input DEM to run inundation on.')
    parser.add_argument('--loc_name',help='Name of location to run inundation on.')
    parser.add_argument('--geoid',help='Path to geoid file to calculate orthometric heights with.',default=None)
    parser.add_argument('--vlm',help='Path to VLM file to propagate input file in time.',default=None)
    parser.add_argument('--clip_vlm',help='Clip DEM to VLM extents?',default=False,action='store_true')
    parser.add_argument('--icesat2',help='Path to ICESat-2 file to calculate coastal sea level with.')
    parser.add_argument('--sealevel_grid',help='Path to sea level grid to calculate coastal sea level with.')
    parser.add_argument('--grid_extents',help='Extents of grid to be used in calculation (x_min x_max y_min y_max)',nargs=4)
    parser.add_argument('--coastline',help='Path to coastline file to calculate coastal sea level on.')
    parser.add_argument('--clip_coast',help='Clip DEM to coastline?',default=False,action='store_true')
    parser.add_argument('--years',help='Years to compute inundation for.',nargs='*',default='2020')
    parser.add_argument('--rcp',help='RCP to use.')
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
    parser.add_argument('--machine',default='t',help='Machine to run on (t, b or local)')
    args = parser.parse_args()

    dem_file = args.dem
    loc_name = args.loc_name
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
    machine_name = args.machine


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
        'grid_res':GRID_INTERMEDIATE_RES}

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

    if uncertainty_flag == True:
        if sigma == 1:
            quantiles = [0.16,0.5,0.84]
        elif sigma == 2:
            quantiles = [0.02,0.5,0.98]
        elif sigma == 3:
            quantiles = [0.001,0.5,0.999]
        else:
            quantiles = [0.5] #sigma is undefined, so just use median
    else:
        quantiles = [0.5]



    print(f'Working on {loc_name}.')

    if vlm_file is not None:
        t_start = datetime.datetime.now()
        print('Resampling VLM...')
        lon_vlm_min,lon_vlm_max,lat_vlm_min,lat_vlm_max = get_raster_extents(vlm_file,'global')
        src_vlm = gdal.Open(vlm_file,gdalconst.GA_ReadOnly)
        epsg_vlm_file = osr.SpatialReference(wkt=src_vlm.GetProjection()).GetAttrValue('AUTHORITY',1)
        if clip_vlm_flag == True:
            dem_clipped_to_vlm_file = dem_file.replace('.tif','_clipped_to_vlm.tif')
            clip_dem_to_vlm_command = f'gdal_translate -projwin {lon_vlm_min} {lat_vlm_max} {lon_vlm_max} {lat_vlm_min} -projwin_srs EPSG:{epsg_vlm_file} {dem_file} {dem_clipped_to_vlm_file} -co "COMPRESS=LZW"'
            subprocess.run(clip_dem_to_vlm_command,shell=True)
            dem_file = dem_clipped_to_vlm_file
            src = gdal.Open(dem_file,gdalconst.GA_ReadOnly)
        vlm_resampled_file = vlm_file.replace('.tif',f'_resampled.tif')
        resample_raster(vlm_file,dem_file,vlm_resampled_file,VLM_NODATA)
        t_end = datetime.datetime.now()
        delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
        delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
        print(f'Resampling VLM took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')


    if geoid_file is not None:
        t_start = datetime.datetime.now()
        print('Resampling geoid...')
        geoid_name = geoid_file.split('/')[-1].split('.')[0]
        geoid_resampled_file = f'{os.path.dirname(os.path.abspath(dem_file))}/{loc_name}_{geoid_name}_resampled.tif'
        resample_raster(geoid_file,dem_file,geoid_resampled_file,None)
        dem_file_orthometric = dem_file.replace('.tif','_orthometric.tif')
        orthometric_command = f'gdal_calc.py --quiet -A {dem_file} -B {geoid_resampled_file} --outfile={dem_file_orthometric} --calc="A-B" --NoDataValue={dem_nodata} --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --co "TILED=YES"'
        subprocess.run(orthometric_command,shell=True)
        dem_file = dem_file_orthometric
        src = gdal.Open(dem_file,gdalconst.GA_ReadOnly)
        t_end = datetime.datetime.now()
        delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
        delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
        print(f'Resampling geoid took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')

    if clip_coast_flag == True:
        t_start = datetime.datetime.now()
        print('Clipping DEM to coastline...')
        dem_file_clipped = dem_file.replace('.tif','_clipped.tif')
        clip_command = f'gdalwarp -s_srs EPSG:{epsg_code} -t_srs EPSG:{epsg_code} -of GTiff -cutline {coastline_file} -cl {os.path.splitext(os.path.basename(coastline_file))[0]} -dstnodata {GRID_NODATA} {dem_file} {dem_file_clipped} -overwrite -co "COMPRESS=LZW" -co "BIGTIFF=IF_SAFER" -co "TILED=YES"'
        subprocess.run(clip_command,shell=True)
        dem_file = dem_file_clipped
        src = gdal.Open(dem_file,gdalconst.GA_ReadOnly)
        t_end = datetime.datetime.now()
        delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
        delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
        print(f'Clipping took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')
    
    lon_dem_min,lon_dem_max,lat_dem_min,lat_dem_max = get_raster_extents(dem_file,'global')
    lon_center_dem = (lon_dem_min + lon_dem_max)/2
    lat_center_dem = (lat_dem_min + lat_dem_max)/2

    t_start = datetime.datetime.now()
    print(f'Resampling DEM to {GRID_INTERMEDIATE_RES} meters.')
    dem_resampled_file = dem_file.replace('.tif',f'_resampled_{GRID_INTERMEDIATE_RES}m.tif')
    resample_dem_command = f'gdalwarp -overwrite -tr {GRID_INTERMEDIATE_RES} {GRID_INTERMEDIATE_RES} -r bilinear {dem_file} {dem_resampled_file}'
    subprocess.run(resample_dem_command,shell=True)
    src_resampled = gdal.Open(dem_resampled_file,gdalconst.GA_ReadOnly)
    t_end = datetime.datetime.now()
    delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
    delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
    print(f'Resampling DEM took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')

    dem_x_size = src.RasterXSize
    dem_y_size = src.RasterYSize
    src_proj = src.GetProjection()
    src_geotransform = src.GetGeoTransform()
    dem_resampled_x_size = src_resampled.RasterXSize
    dem_resampled_y_size = src_resampled.RasterYSize
    src_resampled_proj = src_resampled.GetProjection()
    src_resampled_geotransform = src_resampled.GetGeoTransform()
    xres_dem_resampled,yres_dem_resampled = src_resampled.GetGeoTransform()[1],-src_resampled.GetGeoTransform()[5]
    x_dem_resampled_min,x_dem_resampled_max,y_dem_resampled_min,y_dem_resampled_max = get_raster_extents(dem_resampled_file,'local')
    dx_dem_resampled = np.abs(x_dem_resampled_max - x_dem_resampled_min)
    dy_dem_resampled = np.abs(y_dem_resampled_max - y_dem_resampled_min)
    grid_max_dist = np.max((dx_dem_resampled,dy_dem_resampled))
    algorithm_dict['grid_max_dist'] = grid_max_dist

    t_start = datetime.datetime.now()
    print('Generating coastal sea level grid...')
    if sl_grid_extents is not None:
        h_coast = interpolate_grid(lon_coast,lat_coast,sl_grid_file,sl_grid_extents,loc_name,tmp_dir,GRID_NODATA)
        idx_fillvalue = h_coast==GRID_NODATA
        idx_keep = ~np.logical_or(idx_fillvalue,np.isnan(h_coast))
        x_coast = x_coast[idx_keep]
        y_coast = y_coast[idx_keep]
        h_coast = h_coast[idx_keep]
        lon_coast = lon_coast[idx_keep]
        lat_coast = lat_coast[idx_keep]
        if loc_name in sl_grid_file:
            output_file_coastline = f'{tmp_dir}{os.path.basename(sl_grid_file).replace(".tif","_subset_interpolated_coastline.csv")}'
        else:
            output_file_coastline = f'{tmp_dir}{loc_name}_{os.path.basename(sl_grid_file).replace(".tif","_subset_interpolated_coastline.csv")}'
        if geoid_file is not None:
            h_geoid = interpolate_grid(lon_coast,lat_coast,geoid_file,None,loc_name,tmp_dir,GRID_NODATA)
            h_geoid = h_geoid[idx_keep]
            h_coast = h_coast - h_geoid
            output_file_coastline = output_file_coastline.replace('.csv','_orthometric.csv')
    elif icesat2_file is not None:
        df_icesat2 = pd.read_csv(icesat2_file,header=None,names=['lon','lat','height','time'],dtype={'lon':'float','lat':'float','height':'float','time':'str'})
        x_icesat2_grid_array,y_icesat2_grid_array,h_icesat2_grid_array = create_icesat2_grid(df_icesat2,epsg_code,geoid_file,tmp_dir,loc_name,ICESAT2_GRID_RESOLUTION,N_PTS)
        h_coast = interpolate_points(x_icesat2_grid_array,y_icesat2_grid_array,h_icesat2_grid_array,x_coast,y_coast,INTERPOLATE_METHOD)
        idx_keep = ~np.isnan(x_coast)
        x_coast = x_coast[idx_keep]
        y_coast = y_coast[idx_keep]
        if loc_name in icesat2_file:
            output_file_coastline = f'{tmp_dir}{os.path.basename(icesat2_file).replace(".txt",f"_subset_{INTERPOLATE_METHOD}BivariateSpline_coastline.csv")}'
        else:
            output_file_coastline = f'{tmp_dir}{loc_name}_{os.path.basename(icesat2_file).replace(".txt",f"_subset_{INTERPOLATE_METHOD}BivariateSpline_coastline.csv")}'
        if geoid_file is not None:
            output_file_coastline = output_file_coastline.replace('.csv','_orthometric.csv')
    t_end = datetime.datetime.now()
    delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
    delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
    print(f'Generating coastal sea level took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')

    if high_tide is not None:
        '''
        Implement manual high tide value here
        Create a grid of high tide values at DEM resampled resolution and resample to full res
        '''
        t_start = datetime.datetime.now()
        print(f'Generating high tide file with a value of {high_tide:.2f} m...')
        high_tide_array = np.ones((dem_resampled_y_size,dem_resampled_x_size))*high_tide
        high_tide_full_res = f'{tmp_dir}{loc_name}_high_tide.tif'
        high_tide_intermediate_res = f'{tmp_dir}{loc_name}_high_tide_resampled.tif'
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(sealevel_high_grid_intermediate_res,high_tide_array.shape[1],high_tide_array.shape[0],1,gdal.GDT_Float32)
        dataset.SetGeoTransform(src_resampled_geotransform)
        dataset.SetProjection(src_resampled_proj)
        dataset.GetRasterBand(1).WriteArray(high_tide_array)
        dataset.FlushCache()
        dataset = None
        resample_raster(high_tide_intermediate_res,dem_file,high_tide_full_res)
        t_end = datetime.datetime.now()
        delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
        delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
        print(f'Generating high tide file took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')
        sealevel_high_grid_intermediate_res = high_tide_intermediate_res
        sealevel_high_grid_full_res = high_tide_full_res
    elif return_period is not None:
        t_start = datetime.datetime.now()
        print(f'Finding CoDEC sea level extremes for return period of {return_period} years...')
        rps_coast = get_codec(lon_coast,lat_coast,CODEC_file,return_period)
        output_file_codec = f'{tmp_dir}{loc_name}_CoDEC_{return_period}_yrs_coastline.csv'
        np.savetxt(output_file_codec,np.c_[x_coast,y_coast,rps_coast],fmt='%f',delimiter=',',comments='')
        codec_grid_intermediate_res = csv_to_grid(output_file_codec,algorithm_dict,x_dem_resampled_min,x_dem_resampled_max,xres_dem_resampled,y_dem_resampled_min,y_dem_resampled_max,yres_dem_resampled,epsg_code)
        codec_grid_full_res = codec_grid_intermediate_res.replace(f'_{GRID_INTERMEDIATE_RES}m','')
        resample_raster(codec_grid_intermediate_res,dem_file,codec_grid_full_res)
        t_end = datetime.datetime.now()
        delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
        delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
        print(f'Generating CoDEC sea level extremes took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')
        sealevel_csv_output = output_file_codec
        sealevel_high_grid_intermediate_res = codec_grid_intermediate_res
        sealevel_high_grid_full_res = codec_grid_full_res
    elif fes2014_flag == True:
        t_start = datetime.datetime.now()
        if mhhw_flag == False:
            print(f'Finding FES2014 max tidal heights...')
        else:
            print(f'Finding FES2014 MHHW values...')
        fes_heights_coast = get_fes(lon_coast,lat_coast,fes2014_file,mhhw_flag=mhhw_flag)
        output_file_fes = f'{tmp_dir}{loc_name}_FES2014_coastline.csv'
        np.savetxt(output_file_fes,np.c_[x_coast,y_coast,fes_heights_coast],fmt='%f',delimiter=',',comments='')
        fes_grid_intermediate_res = csv_to_grid(output_file_fes,algorithm_dict,x_dem_resampled_min,x_dem_resampled_max,xres_dem_resampled,y_dem_resampled_min,y_dem_resampled_max,yres_dem_resampled,epsg_code)
        fes_grid_full_res = fes_grid_intermediate_res.replace(f'_{GRID_INTERMEDIATE_RES}m','')
        resample_raster(fes_grid_intermediate_res,dem_file,fes_grid_full_res)
        t_end = datetime.datetime.now()
        delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
        delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
        print(f'Generating FES2014 values took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')
        sealevel_csv_output = output_file_fes
        sealevel_high_grid_intermediate_res = fes_grid_intermediate_res
        sealevel_high_grid_full_res = fes_grid_full_res


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

    if slr is not None:
        for slr_value in slr:
            t_start = datetime.datetime.now()
            print(f'Creating inundation for {slr_value:.2f} m...')
            slr_value_str = f'SLR_{slr_value:.2f}m'.replace('.','p').replace('-','neg')
            if fes2014_flag == True:
                output_inundation_file = f'{inundation_dir}{loc_name}_Inundation_{slr_value_str}_FES2014.tif'
                if mhhw_flag == True:
                    output_inundation_file = output_inundation_file.replace('FES2014','FES2014_MHHW')
            elif return_period is not None:
                output_inundation_file = f'{inundation_dir}{loc_name}_Inundation_{slr_value_str}_CoDEC_RP_{return_period}_yrs.tif'
            if geoid_file is not None:
                output_inundation_file = output_inundation_file.replace('_Inundation_','_Orthometric_Inundation_')
            if vlm_file is None:
                output_inundation_file = output_inundation_file.replace('_Inundation_','_Inundation_No_VLM_')
            h_coast_slr = h_coast + slr_value
            output_file_coastline_slr = output_file_coastline.replace('.csv',f'_{slr_value_str}.csv')
            np.savetxt(output_file_coastline_slr,np.c_[x_coast,y_coast,h_coast_slr],fmt='%f',delimiter=',',comments='')
            sl_grid_file_intermediate_res = csv_to_grid(output_file_coastline_slr,algorithm_dict,x_dem_resampled_min,x_dem_resampled_max,xres_dem_resampled,y_dem_resampled_min,y_dem_resampled_max,yres_dem_resampled,epsg_code)
            sl_grid_file_full_res = sl_grid_file_intermediate_res.replace(f'_{GRID_INTERMEDIATE_RES}m','')
            resample_raster(sl_grid_file_intermediate_res,dem_file,sl_grid_file_full_res)
            if vlm_file is not None:
                dt = int(yr - t0)
                inundation_command = f'gdal_calc.py --quiet -A {dem_file} -B {vlm_resampled_file} -C {sl_grid_file_full_res} -D {sealevel_high_grid_full_res} --outfile={output_inundation_file} --calc="A+B*{dt} < C+D" --NoDataValue={INUNDATION_NODATA} --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --co "TILED=YES"'
            elif vlm_rate is not None:
                inundation_command = f'gdal_calc.py --quiet -A {dem_file} -C {sl_grid_file_full_res} -D {sealevel_high_grid_full_res} --outfile={output_inundation_file} --calc="A+{vlm_rate}*{dt} < C+D" --NoDataValue={INUNDATION_NODATA} --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --co "TILED=YES"'
            else:
                inundation_command = f'gdal_calc.py --quiet -A {dem_file} -C {sl_grid_file_full_res} -D {sealevel_high_grid_full_res} --outfile={output_inundation_file} --calc="A < C+D" --NoDataValue={INUNDATION_NODATA} --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --co "TILED=YES"'
            subprocess.run(inundation_command,shell=True)
            output_inundation_shp_file = output_inundation_file.replace('.tif','.shp')
            polygonize_command = f'gdal_polygonize.py -f "ESRI Shapefile" {output_inundation_file} {output_inundation_shp_file}'
            subprocess.run(polygonize_command,shell=True)
            t_end = datetime.datetime.now()
            delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
            delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
            print(f'Inundation creation took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')
            if connectivity_flag == True:
                print('Computing connectivity to the ocean...')
                t_start = datetime.datetime.now()
                compute_connectivity(output_inundation_shp_file,gdf_surface_water_buffered)
                t_end = datetime.datetime.now()
                delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
                delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
                print(f'Connectivity took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')
            subprocess.run(f'rm {output_file_coastline_slr}',shell=True)
            subprocess.run(f'rm {output_file_coastline_slr.replace(".csv",".vrt")}',shell=True)
            subprocess.run(f'rm {sl_grid_file_intermediate_res}',shell=True)
            subprocess.run(f'rm {sl_grid_file_full_res}',shell=True)
    else:
        for yr,quantile_select in itertools.product(years,quantiles):
            t_start = datetime.datetime.now()
            if fes2014_flag == True:
                output_inundation_file = f'{inundation_dir}{loc_name}_Inundation_{yr}_PROJECTION_METHOD_FES2014.tif'
                if mhhw_flag == True:
                    output_inundation_file = output_inundation_file.replace('FES2014','FES2014_MHHW')
            elif return_period is not None:
                output_inundation_file = f'{inundation_dir}{loc_name}_Inundation_{yr}_PROJECTION_METHOD_CoDEC_RP_{return_period}_yrs.tif'
            if projection_select == 'SROCC':
                print(f'Creating inundation in {yr} using RCP{rcp}...')
                output_inundation_file = output_inundation_file.replace('PROJECTION_METHOD',f'SROCC_RCP_{str(rcp).replace(".","p")}')
                lon_projection,lat_projection,slr_projection = upscale_SROCC_grid(SROCC_dir,dem_file,rcp,t0,yr,)
            elif projection_select == 'AR6':
                print(f'Creating inundation in {yr} using SSP{ssp}...')
                output_inundation_file = output_inundation_file.replace('PROJECTION_METHOD',f'AR6_SSP_{ssp}')
                if quantile_select < 0.5:
                    output_inundation_file = output_inundation_file.replace('_Inundation_',f'_Inundation_Minus_{sigma}sigma_')
                elif quantile_select > 0.5:
                    output_inundation_file = output_inundation_file.replace('_Inundation_',f'_Inundation_Plus_{sigma}sigma_')
                lon_projection,lat_projection,slr_projection = upscale_ar6_data(AR6_dir,tmp_dir,landmask_c_file,dem_file,ssp,osm_shp_file,yr,quantile_select=quantile_select)
            if geoid_file is not None:
                output_inundation_file = output_inundation_file.replace('_Inundation_','_Orthometric_Inundation_')
            if vlm_file is None:
                output_inundation_file = output_inundation_file.replace('_Inundation_','_Inundation_No_VLM_')
            h_projection_coast = interpolate_points(lon_projection,lat_projection,slr_projection,x_coast,y_coast,INTERPOLATE_METHOD)
            h_coast_yr = h_coast + h_projection_coast
            output_file_coastline_yr = output_file_coastline.replace('.csv',f'_{yr}.csv')
            np.savetxt(output_file_coastline_yr,np.c_[x_coast,y_coast,h_coast_yr],fmt='%f',delimiter=',',comments='')
            sl_grid_file_intermediate_res = csv_to_grid(output_file_coastline_yr,algorithm_dict,x_dem_resampled_min,x_dem_resampled_max,xres_dem_resampled,y_dem_resampled_min,y_dem_resampled_max,yres_dem_resampled,epsg_code)
            sl_grid_file_full_res = sl_grid_file_intermediate_res.replace(f'_{GRID_INTERMEDIATE_RES}m','')
            resample_raster(sl_grid_file_intermediate_res,dem_file,sl_grid_file_full_res)
            if vlm_file is not None:
                dt = int(yr - t0)
                inundation_command = f'gdal_calc.py --quiet -A {dem_file} -B {vlm_resampled_file} -C {sl_grid_file_full_res} -D {sealevel_high_grid_full_res} --outfile={output_inundation_file} --calc="A+B*{dt} < C+D" --NoDataValue={INUNDATION_NODATA} --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --co "TILED=YES"'
            elif vlm_rate is not None:
                inundation_command = f'gdal_calc.py --quiet -A {dem_file} -C {sl_grid_file_full_res} -D {sealevel_high_grid_full_res} --outfile={output_inundation_file} --calc="A+{vlm_rate}*{dt} < C+D" --NoDataValue={INUNDATION_NODATA} --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --co "TILED=YES"'
            else:
                inundation_command = f'gdal_calc.py --quiet -A {dem_file} -C {sl_grid_file_full_res} -D {sealevel_high_grid_full_res} --outfile={output_inundation_file} --calc="A < C+D" --NoDataValue={INUNDATION_NODATA} --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --co "TILED=YES"'
            subprocess.run(inundation_command,shell=True)
            output_inundation_shp_file = output_inundation_file.replace('.tif','.shp')
            polygonize_command = f'gdal_polygonize.py -f "ESRI Shapefile" {output_inundation_file} {output_inundation_shp_file}'
            subprocess.run(polygonize_command,shell=True)
            t_end = datetime.datetime.now()
            delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
            delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
            print(f'Inundation creation took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')
            if connectivity_flag == True:
                print('Computing connectivity to the ocean...')
                t_start = datetime.datetime.now()
                compute_connectivity(output_inundation_shp_file,gdf_surface_water_buffered)
                t_end = datetime.datetime.now()
                delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
                delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
                print(f'Connectivity took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')
            subprocess.run(f'rm {output_file_coastline_yr}',shell=True)
            subprocess.run(f'rm {output_file_coastline_yr.replace(".csv",".vrt")}',shell=True)
            subprocess.run(f'rm {sl_grid_file_intermediate_res}',shell=True)
            subprocess.run(f'rm {sl_grid_file_full_res}',shell=True)
    if os.path.isfile(sealevel_csv_output):
        subprocess.run(f'rm {sealevel_csv_output}',shell=True)
        subprocess.run(f'rm {sealevel_csv_output.replace(".csv",".vrt")}',shell=True)
    subprocess.run(f'rm {sealevel_high_grid_intermediate_res}',shell=True)
    subprocess.run(f'rm {sealevel_high_grid_full_res}',shell=True)
    print(f'Finished with {loc_name} at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.')

if __name__ == '__main__':
    main()