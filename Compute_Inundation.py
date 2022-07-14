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

import warnings
import configparser
from dem_utils import get_lonlat_gdf,icesat2_df2array,deg2utm
from dem_utils import filter_utm,get_raster_extents,resample_raster
from inundation_utils import create_csv_vrt, create_icesat2_grid, interpolate_grid, get_SROCC_data, kriging_inundation


def main():
    warnings.simplefilter(action='ignore')
    config_file = 'dem_config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dem',help='Path to input DEM to run inundation on.')
    parser.add_argument('--vlm',help='Path to VLM file to propagate input file in time.')
    parser.add_argument('--icesat2',help='Path to ICESat-2 file to calculate coastal sea level with.')
    parser.add_argument('--sealevel_grid',help='Path to sea level grid to calculate coastal sea level with.')
    parser.add_argument('--grid_extents',help='Extents of grid to be used in calculation (x_min x_max y_min y_max)',nargs=4)
    parser.add_argument('--coastline',help='Path to coastline file to calculate coastal sea level on.')
    parser.add_argument('--years',help='Years to compute inundation for.',nargs='*',default='2020')
    
    parser.add_argument('--rcp',help='RCP to use.',default='4.5')
    parser.add_argument('--t0',help='Time to use as t0 to zero SLR.',default='2020')
    #parser.add_argument('--date_filter',help='Filter by date of input file.',default=False,action='store_true')
    args = parser.parse_args()

    dem_file = args.dem
    vlm_file = args.vlm
    icesat2_file = args.icesat2
    sl_grid_file = args.sealevel_grid
    sl_grid_extents = args.grid_extents
    coastline_file = args.coastline
    rcp = args.rcp
    t0 = int(args.t0)
    years = args.years
    years = [int(yr) for yr in np.atleast_1d(years)]

    if icesat2_file is not None and sl_grid_file is not None:
        print('ICESat-2 file and sea level grid given, cannot handle both!')
        sys.exit()
    if dem_file is None:
        print('No DEM file supplied to run inundation on!')
        sys.exit()
    if vlm_file is None:
        print('No VLM file supplied to propagate in time!')
        print('Still running inundation with sea level rise.')
    if sl_grid_file is not None and sl_grid_extents is None:
        print('Warning, selecting whole grid as input!')
        src_sl_grid = gdal.Open(sl_grid_file,gdalconst.GA_ReadOnly)

    if os.path.dirname(os.path.abspath(dem_file)).split('/')[-1] == 'Mosaic':
        inundation_dir = f'{"/".join(os.path.dirname(os.path.abspath(dem_file)).split("/")[:-1])}/Inundation/'
    else:
        inundation_dir = f'{os.path.dirname(os.path.abspath(dem_file))}/Inundation/'

    if not os.path.exists(inundation_dir):
        os.mkdir(inundation_dir)

    '''    
    temporary paths
    dem_file = '/media/heijkoop/DATA/DEM/Locations/Nigeria_Lagos/May_2022/Mosaic/Nigeria_Lagos_Full_Mosaic_0_32631_Nigeria_Lagos_ATL03_high_conf_masked_SRTM_filtered_threshold_10_m_32631-DEM_nuth_x+0.91_y+0.76_z+1.21_align.tif'
    icesat2_file = '/media/heijkoop/DATA/DEM/Locations/Nigeria_Lagos/May_2022/Nigeria_Lagos_ATL03_FES2014_high_med_conf_masked_DTU21_filtered_threshold_10p0_m.txt'
    coastline_file = '/media/heijkoop/DATA/DEM/Locations/Nigeria_Lagos/Nigeria_Lagos_S2_NDWI_20211221_simplified.shp'
    sl_grid_file = '/media/heijkoop/DATA/DTU21/1min/DTU21MSS_WGS84_lon180.tif'
    sl_grid_extents = [2.987,3.799,6.218,6.766]

    dem_file = '/media/heijkoop/DATA/DEM/Locations/India_Mumbai/MOSAIC/July_2022/India_Mumbai_Full_Mosaic_0_32643.tif'
    icesat2_file = '/home/heijkoop/Desktop/tmp/Mumbai/ICESat-2/India_Mumbai_20220529_ATL03_FES2014_high_med_conf_masked_DTU21_filtered_threshold_10_m_buffer_5000m.txt'
    coastline_file = '/media/heijkoop/DATA/DEM/Locations/India_Mumbai/Mumbai_Large.shp'
    sl_grid_file = '/media/heijkoop/DATA/DTU21/1min/DTU21MSS_WGS84_lon180.tif'
    sl_grid_extents = [72.6,73.1,18.0,20.0]

    tmp_dir = '/home/heijkoop/Desktop/tmp/Inundation/'
    '''
    
    SROCC_dir = config.get('INUNDATION_PATHS','SROCC_dir')
    # tmp_dir = config.get('GENERAL_PATHS','tmp_dir')
    tmp_dir = '/home/heijkoop/Desktop/tmp/Inundation/'
    gsw_dir = config.get('GENERAL_PATHS','gsw_dir')
    GRID_RESOLUTION = config.getfloat('INUNDATION_CONSTANTS','GRID_RESOLUTION')
    N_PTS = config.getint('INUNDATION_CONSTANTS','N_PTS')
    KRIGING_METHOD = config.get('INUNDATION_CONSTANTS','KRIGING_METHOD')
    KRIGING_VARIOGRAM = config.get('INUNDATION_CONSTANTS','KRIGING_VARIOGRAM')
    NUM_THREADS = config.getint('INUNDATION_CONSTANTS','NUM_THREADS')
    DEM_INTERMEDIATE_RES = config.get('INUNDATION_CONSTANTS','DEM_INTERMEDIATE_RES')
    GRID_ALGORITHM = config.get('INUNDATION_CONSTANTS','GRID_ALGORITHM')
    GRID_SMOOTHING = config.getfloat('INUNDATION_CONSTANTS','GRID_SMOOTHING')
    GRID_POWER = config.getfloat('INUNDATION_CONSTANTS','GRID_POWER')
    GRID_NODATA = config.getint('INUNDATION_CONSTANTS','GRID_NODATA')
    GRID_MAX_PTS = config.getint('INUNDATION_CONSTANTS','GRID_MAX_PTS')

    loc_name = '_'.join(dem_file.split('/')[-1].split('_')[0:2])
    dem_resampled_file = dem_file.replace('.tif',f'_resampled_{DEM_INTERMEDIATE_RES}m.tif')
    resample_dem_command = f'gdalwarp -overwrite -tr {DEM_INTERMEDIATE_RES} {DEM_INTERMEDIATE_RES} -r bilinear {dem_file} {dem_resampled_file}'
    subprocess.run(resample_dem_command,shell=True)

    src = gdal.Open(dem_file,gdalconst.GA_ReadOnly)
    src_resampled = gdal.Open(dem_resampled_file,gdalconst.GA_ReadOnly)
    epsg_code = osr.SpatialReference(wkt=src.GetProjection()).GetAttrValue('AUTHORITY',1)

    dem_x_size = src.RasterXSize
    dem_y_size = src.RasterYSize
    dem_resampled_x_size = src_resampled.RasterXSize
    dem_resampled_y_size = src_resampled.RasterYSize
    res_dem_resampled_x,res_dem_resampled_y = src_resampled.GetGeoTransform()[1],-src_resampled.GetGeoTransform()[5]
    x_dem_resampled_min,x_dem_resampled_max,y_dem_resampled_min,y_dem_resampled_max = get_raster_extents(dem_resampled_file,'local')
    dx_dem_resampled = np.abs(x_dem_resampled_max - x_dem_resampled_min)
    dy_dem_resampled = np.abs(y_dem_resampled_max - y_dem_resampled_min)
    grid_max_dist = np.max((dx_dem_resampled,dy_dem_resampled))

    gdf_coast = gpd.read_file(coastline_file)
    lon_coast,lat_coast = get_lonlat_gdf(gdf_coast)
    gdf_coast = gdf_coast.to_crs(f'EPSG:{epsg_code}')
    x_coast,y_coast = get_lonlat_gdf(gdf_coast)
    
    if sl_grid_extents is not None:
        # x_sl_grid_array,y_sl_grid_array,sl_grid_array = create_sl_grid(sl_grid_file,sl_grid_extents,loc_name,epsg_code,tmp_dir)
        # h_coast,var_coast = kriging_inundation(x_sl_grid_array,y_sl_grid_array,sl_grid_array,x_coast,y_coast,KRIGING_METHOD,KRIGING_VARIOGRAM)
        h_coast = interpolate_grid(lon_coast,lat_coast,sl_grid_file,sl_grid_extents,loc_name,tmp_dir,GRID_NODATA)
        var_coast = None
        if loc_name in sl_grid_file:
            output_file_coastline = f'{tmp_dir}{os.path.basename(sl_grid_file).replace(".tif","_subset_interpolated_coastline.csv")}'
        else:
            output_file_coastline = f'{tmp_dir}{loc_name}_{os.path.basename(sl_grid_file).replace(".tif","_subset_interpolated_coastline.csv")}'
    elif icesat2_file is not None:
        df_icesat2 = pd.read_csv(icesat2_file,header=None,names=['lon','lat','height','time'],dtype={'lon':'float','lat':'float','height':'float','time':'str'})
        x_icesat2_grid_array,y_icesat2_grid_array,h_icesat2_grid_array = create_icesat2_grid(df_icesat2,epsg_code,GRID_RESOLUTION,N_PTS)
        h_coast,var_coast = kriging_inundation(x_icesat2_grid_array,y_icesat2_grid_array,h_icesat2_grid_array,x_coast,y_coast,KRIGING_METHOD,KRIGING_VARIOGRAM)
        if loc_name in icesat2_file:
            output_file_coastline = f'{tmp_dir}{os.path.basename(icesat2_file).replace(".txt",f"_subset_{KRIGING_METHOD}_{KRIGING_VARIOGRAM}_kriging_coastline.csv")}'
        else:
            output_file_coastline = f'{tmp_dir}{loc_name}_{os.path.basename(icesat2_file).replace(".txt",f"_{KRIGING_METHOD}_{KRIGING_VARIOGRAM}_kriging_coastline.csv")}'
    output_file_vrt = output_file_coastline.replace('.csv','.vrt')
    output_file_grid_resampled = output_file_vrt.replace('.vrt',f'_{GRID_ALGORITHM}_grid_{DEM_INTERMEDIATE_RES}m.tif')
    output_file_grid = output_file_vrt.replace('.vrt','_{GRID_ALGORITHM}_grid.tif')
    idx_fillvalue = h_coast==GRID_NODATA
    idx_keep = ~np.logical_or(idx_fillvalue,np.isnan(h_coast))
    x_coast = x_coast[idx_keep]
    y_coast = y_coast[idx_keep]
    h_coast = h_coast[idx_keep]
    if var_coast is not None:
        var_coast = var_coast[idx_keep]
        np.savetxt(output_file_coastline,np.c_[x_coast,y_coast,h_coast,var_coast],fmt='%f',delimiter=',',comments='')
    else:
        np.savetxt(output_file_coastline,np.c_[x_coast,y_coast,h_coast],fmt='%f',delimiter=',',comments='')
    layer_name = output_file_coastline.split('/')[-1].replace('.csv','')
    vrt_flag = create_csv_vrt(output_file_vrt,output_file_coastline,layer_name)
    
    #X_MIN X_MAX Y_MAX Y_MIN is correct to get pixel size to N, -N
    #This is because the origin is top-left (i.e. Northwest)
    if GRID_ALGORITHM == 'nearest':
        build_grid_command = f'gdal_grid -a {GRID_ALGORITHM}:nodata={GRID_NODATA} -txe {x_dem_resampled_min} {x_dem_resampled_max} -tye {y_dem_resampled_max} {y_dem_resampled_min} -tr {res_dem_resampled_x} {res_dem_resampled_y} -a_srs EPSG:{epsg_code} -of GTiff -ot Float32 -l {layer_name} {output_file_vrt} {output_file_grid_resampled} --config GDAL_NUM_THREADS {NUM_THREADS} -co "COMPRESS=LZW"'
    elif GRID_ALGORITHM == 'invdist':
        build_grid_command = f'gdal_grid -a {GRID_ALGORITHM}:nodata={GRID_NODATA}:smoothing={GRID_SMOOTHING}:power={GRID_POWER} -txe {x_dem_resampled_min} {x_dem_resampled_max} -tye {y_dem_resampled_max} {y_dem_resampled_min} -tr {res_dem_resampled_x} {res_dem_resampled_y} -a_srs EPSG:{epsg_code} -of GTiff -ot Float32 -l {layer_name} {output_file_vrt} {output_file_grid_resampled} --config GDAL_NUM_THREADS {NUM_THREADS} -co "COMPRESS=LZW"'
    elif GRID_ALGORITHM == 'invdistnn':
        build_grid_command = f'gdal_grid -a {GRID_ALGORITHM}:nodata={GRID_NODATA}:smoothing={GRID_SMOOTHING}:power={GRID_POWER}:max_points={GRID_MAX_PTS}:radius={grid_max_dist} -txe {x_dem_resampled_min} {x_dem_resampled_max} -tye {y_dem_resampled_max} {y_dem_resampled_min} -tr {res_dem_resampled_x} {res_dem_resampled_y} -a_srs EPSG:{epsg_code} -of GTiff -ot Float32 -l {layer_name} {output_file_vrt} {output_file_grid_resampled} --config GDAL_NUM_THREADS {NUM_THREADS} -co "COMPRESS=LZW"'
    subprocess.run(build_grid_command,shell=True)
    #RADIUS NEEDS TO BE A FUNCTION OF THE DEM


    # resample_sl_flag = resample_raster(output_file_grid_resampled,dem_file,output_file_grid,'nearest',True)

    ####
    # Read sea level extreme from CoDEC file, use return periods as input in argparse
    # Find closest x_loc,y_loc to center of DEM, in EPSG:4326 
    ####

    # vlm_file_resampled = vlm_file.replace('.tif',f'_resampled_{dem_x_size}.tif')
    # t_SROCC,slr_md_closest_minus_t0,slr_he_closest_minus_t0,slr_le_closest_minus_t0 = get_SROCC_data(SROCC_dir,dem_file,rcp,t0)
    # resample_vlm_flag = resample_raster(vlm_file,dem_file,vlm_file_resampled,'nearest',True)
    '''
    find slr for each given year
    update dem to those years, using t0
    update sl grid with SROCC to those years, using t0
    gdal_calc it all
    '''

if __name__ == '__main__':
    main()