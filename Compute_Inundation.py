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
from dem_utils import filter_utm,get_raster_extents
from inundation_utils import create_csv_vrt, create_icesat2_grid, get_SROCC_data, kriging_icesat2


def main():
    warnings.simplefilter(action='ignore')
    config_file = 'dem_config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dem',help='Path to input DEM to run inundation on.')
    parser.add_argument('--vlm',help='Path to VLM file to propagate input file in time.')
    parser.add_argument('--icesat2',help='Path to ICESat-2 file to calculate coastal sea level with.')
    parser.add_argument('--coastline',help='Path to coastline file to calculate coastal sea level on.')
    parser.add_argument('--years',help='Years to compute inundation for.',nargs='*')
    
    parser.add_argument('--rcp',help='RCP to use.',default='4.5')
    parser.add_argument('--t0',help='Time to use as t0 to zero SLR.',default='2020')
    args = parser.parse_args()

    dem_file = args.dem
    vlm_file = args.vlm
    icesat2_file = args.icesat2
    coastline_file = args.coastline
    rcp = args.rcp
    t0 = int(args.t0)
    years = args.years
    years = [int(yr) for yr in np.atleast_1d(years)]

    output_file_kriging = icesat2_file.replace('.txt','_kriging_coastline.csv')
    output_file_vrt = output_file_kriging.replace('.csv','.vrt')
    output_file_grid = output_file_vrt.replace('.vrt','_grid.tif')

    SROCC_dir = config.get('INUNDATION_PATHS','SROCC_dir')
    gsw_dir = config.get('GENERAL_PATHS','gsw_dir')
    GRID_RESOLUTION = config.getfloat('INUNDATION_CONSTANTS','GRID_RESOLUTION')
    N_PTS = config.getint('INUNDATION_CONSTANTS','N_PTS')
    KRIGING_METHOD = config.get('INUNDATION_CONSTANTS','KRIGING_METHOD')
    KRIGING_VARIOGRAM = config.get('INUNDATION_CONSTANTS','KRIGING_VARIOGRAM')
    LAYER_NAME = config.get('INUNDATION_CONSTANTS','LAYER_NAME')
    NUM_THREADS = config.getint('INUNDATION_CONSTANTS','NUM_THREADS')

    if os.path.dirname(os.path.abspath(dem_file)).split('/')[-1] == 'Mosaic':
        inundation_dir = f'{"/".join(os.path.dirname(os.path.abspath(dem_file)).split("/")[:-1])}/Inundation/'
    else:
        inundation_dir = f'{os.path.dirname(os.path.abspath(dem_file))}/Inundation/'
    if not os.path.exists(inundation_dir):
        os.mkdir(inundation_dir)

    src = gdal.Open(dem_file,gdalconst.GA_ReadOnly)
    epsg_code = osr.SpatialReference(wkt=src.GetProjection()).GetAttrValue('AUTHORITY',1)
    dem_x_size = src.RasterXSize
    dem_y_size = src.RasterYSize
    res_dem_x,res_dem_y = src.GetGeoTransform()[1],-src.GetGeoTransform()[5]
    x_dem_min,x_dem_max,y_dem_min,y_dem_max = get_raster_extents(dem_file,'local')


    df_icesat2 = pd.read_csv(icesat2_file,header=None,names=['lon','lat','height','time'],dtype={'lon':'float','lat':'float','height':'float','time':'str'})

    gdf_coast = gpd.read_file(coastline_file)
    gdf_coast = gdf_coast.to_crs(f'EPSG:{epsg_code}')
    x_coast,y_coast = get_lonlat_gdf(gdf_coast)
    
    
    
    # t_SROCC,slr_md_closest_minus_t0,slr_he_closest_minus_t0,slr_le_closest_minus_t0 = get_SROCC_data(SROCC_dir,dem_file,rcp,t0)
    x_grid,y_grid,h_grid,x_grid_array,y_grid_array,h_grid_array = create_icesat2_grid(df_icesat2,epsg_code,GRID_RESOLUTION,N_PTS)
    h_coast,var_coast = kriging_icesat2(x_grid_array,y_grid_array,h_grid_array,x_coast,y_coast,KRIGING_METHOD,KRIGING_VARIOGRAM)
    x_coast = x_coast[~np.isnan(x_coast)]
    y_coast = y_coast[~np.isnan(y_coast)]
    h_coast = h_coast[~np.isnan(h_coast)]
    var_coast = var_coast[~np.isnan(var_coast)]
    np.savetxt('tmp.csv',np.c_[x_coast,y_coast,h_coast],fmt='%f',delimiter=',')
    # LAYER_NAME = output_file_kriging.split('/')[-1].replace('.csv','')
    LAYER_NAME = 'tmp'
    output_file_kriging = 'tmp.csv'
    output_file_vrt = 'tmp.vrt'
    # np.savetxt(output_file_kriging,np.c_[x_coast,y_coast,h_coast],fmt='%f',delimiter=',',header='Easting,Northing,Elevation',comments='')
    vrt_flag = create_csv_vrt(output_file_vrt,output_file_kriging,LAYER_NAME)

    # build_grid_command = f'gdal_grid -a nearest:nodata=-9999 -txe {x_dem_min} {x_dem_max} -tye {y_dem_min} {y_dem_max} -tr {res_dem_x} {res_dem_y} -a_srs EPSG:{epsg_code} -of GTiff -ot Float32 -l {LAYER_NAME} {output_file_vrt} {output_file_grid}'
    build_grid_command = f'gdal_grid -a nearest:nodata=-9999 -txe {x_dem_min} {x_dem_max} -tye {y_dem_min} {y_dem_max} -tr {res_dem_x} {res_dem_y} -a_srs EPSG:{epsg_code} -of GTiff -ot Float32 -l tmp {output_file_vrt} {output_file_grid} --config GDAL_NUM_THREADS {NUM_THREADS} -co "COMPRESS=LZW"'
    print(build_grid_command)
    print(' ')
    subprocess.run(build_grid_command,shell=True)
if __name__ == '__main__':
    main()