import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import shapely
from osgeo import gdal,gdalconst,osr
import os,sys
import glob
import argparse
import subprocess
import datetime
import warnings
import configparser

from dem_utils import get_strip_list,get_strip_extents
from dem_utils import get_gsw,get_strip_shp,filter_strip_gsw,find_cloud_water
from dem_utils import get_contained_strips,get_valid_strip_overlaps,get_minimum_spanning_tree
from dem_utils import find_mosaic,build_mosaic,copy_single_strips

def main():
    warnings.simplefilter(action='ignore')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='dem_config.ini',help='Path to configuration file.')
    # parser.add_argument('--input_file',default=config.get('MOSAIC_PATHS','input_file'),help='Path to input file containing directories of strips.')
    parser.add_argument('--list',default=None,help='Path to list of strips to mosaic.')
    parser.add_argument('--output_dir',default=None,help='Path to output directory.')
    parser.add_argument('--loc_name',default=None,help='Name of location.')
    parser.add_argument('--gsw',default=None,help='Path to GSW shapefile')
    # parser.add_argument('--machine',default='t',help='Machine to run on.',choices=['t','b','local'])
    parser.add_argument('--dir_structure',default='sealevel',help='Directory structure of input strips (sealevel, simple or scenes)',choices=['sealevel','simple','scenes'])
    parser.add_argument('--N_cpus',help='Number of CPUs to use',default=1,type=int)
    parser.add_argument('--horizontal',default=False,help='Incorperate horizontal alignment in mosaic?',action='store_true')
    parser.add_argument('--cloud_water_filter',default='default',nargs='?',help='Use cloud and water filter?')
    parser.add_argument('--corrected',default=False,help='Find corrected strips instead?',action='store_true')
    parser.add_argument('--all_strips',default=False,help='Mosaic all strips in directory? (No geometry filtering.)',action='store_true')
    parser.add_argument('--no_gsw',default=False,help='Skip GSW filter?',action='store_true')
    parser.add_argument('--simplify',default=False,help='Apply simplify operation to shapefile of strips?',action='store_true')
    args = parser.parse_args()
    config_file = args.config
    # input_file = args.input_file
    list_file = args.list
    single_output_dir = args.output_dir
    single_loc_name = args.loc_name
    # machine_name = args.machine
    dir_structure = args.dir_structure
    N_cpus = args.N_cpus
    horizontal_flag = args.horizontal
    cloud_water_filter_flag = args.cloud_water_filter
    corrected_flag = args.corrected
    all_strips_flag = args.all_strips
    gsw_file = args.gsw
    no_gsw_flag = args.no_gsw
    simplify_flag = args.simplify

    config = configparser.ConfigParser()
    config.read(config_file)

    input_file = config.get('MOSAIC_PATHS','input_file')
    tmp_dir = config.get('GENERAL_PATHS','tmp_dir')
    gsw_dir = config.get('GENERAL_PATHS','gsw_dir')
    landmask_c_file = config.get('GENERAL_PATHS','landmask_c_file')

    df_input = pd.read_csv(input_file,header=0,names=['loc_dirs','output_dirs','input_types'],dtype={'loc_dirs':'str','output_dirs':'str','input_types':'object'})
    df_input.input_types = df_input.input_types.fillna('0').astype(int)

    if list_file is not None:
        df_list = pd.read_csv(list_file,header=None,names=['strip'],dtype={'strip':'str'})
        if single_output_dir is None:
            print('If a list is provided, then an output directory must be provided.')
            sys.exit()
        elif single_loc_name is None:
            single_loc_name = single_output_dir.split('/')[-2]
        df_input.loc[len(df_input.index)] = ['list',single_output_dir,3]
    else:
        df_list = None
    
    # if machine_name == 'b':
    #     tmp_dir = tmp_dir.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
    #     gsw_dir = gsw_dir.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
    #     if df_list is not None:
    #         df_list.strip = [s.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/') for s in df_list.strip]
    # elif machine_name == 'local':
    #     tmp_dir = tmp_dir.replace('/BhaltosMount/Bhaltos/EDUARD/','/home/heijkoop/Desktop/Projects/')
    #     gsw_dir = gsw_dir.replace('/BhaltosMount/Bhaltos/EDUARD/DATA_REPOSITORY/','/media/heijkoop/DATA/').replace('Extent/','')

    POLYGON_AREA_THRESHOLD = config.getfloat('MOSAIC_CONSTANTS','POLYGON_AREA_THRESHOLD') #in m^2
    POLYGON_SIMPLIFY_VALUE = config.getfloat('MOSAIC_CONSTANTS','POLYGON_SIMPLIFY_VALUE') #in m^2
    STRIP_AREA_THRESHOLD = config.getfloat('MOSAIC_CONSTANTS','STRIP_AREA_THRESHOLD') #in m^2
    GSW_POCKET_THRESHOLD = config.getfloat('MOSAIC_CONSTANTS','GSW_POCKET_THRESHOLD') #in %
    GSW_CRS_TRANSFORM_THRESHOLD = config.getfloat('MOSAIC_CONSTANTS','GSW_CRS_TRANSFORM_THRESHOLD') #in %
    GSW_OVERLAP_THRESHOLD = config.getfloat('MOSAIC_CONSTANTS','GSW_OVERLAP_THRESHOLD') #in %
    STRIP_TOTAL_AREA_PERCENTAGE_THRESHOLD = config.getfloat('MOSAIC_CONSTANTS','STRIP_TOTAL_AREA_PERCENTAGE_THRESHOLD') #in %
    STRIP_CONTAINMENT_THRESHOLD = config.getfloat('MOSAIC_CONSTANTS','STRIP_CONTAINMENT_THRESHOLD') #in %
    STRIP_DELTA_TIME_THRESHOLD = config.getint('MOSAIC_CONSTANTS','STRIP_DELTA_TIME_THRESHOLD') #in days
    STRIP_CLOUD_THRESHOLD = config.getfloat('MOSAIC_CONSTANTS','STRIP_CLOUD_THRESHOLD') #in %
    STRIP_WATER_THRESHOLD = config.getfloat('MOSAIC_CONSTANTS','STRIP_WATER_THRESHOLD') #in %
    N_STRIPS_CONTAINMENT = config.getint('MOSAIC_CONSTANTS','N_STRIPS_CONTAINMENT') #[-]
    AREA_OVERLAP_THRESHOLD = config.getfloat('MOSAIC_CONSTANTS','AREA_OVERLAP_THRESHOLD') #in m^2
    GSW_INTERSECTION_THRESHOLD = config.getfloat('MOSAIC_CONSTANTS','GSW_INTERSECTION_THRESHOLD') #in %
    X_SPACING = config.getfloat('MOSAIC_CONSTANTS','X_SPACING') #in m
    Y_SPACING = config.getfloat('MOSAIC_CONSTANTS','Y_SPACING') #in m
    X_MAX_SEARCH = config.getfloat('MOSAIC_CONSTANTS','X_MAX_SEARCH') #in m
    Y_MAX_SEARCH = config.getfloat('MOSAIC_CONSTANTS','y_MAX_SEARCH') #in m
    MOSAIC_TILE_SIZE = config.getfloat('MOSAIC_CONSTANTS','MOSAIC_TILE_SIZE') #in m^2   
    if no_gsw_flag == True:
        GSW_OVERLAP_THRESHOLD = 1.0
        GSW_INTERSECTION_THRESHOLD = 1.0

    for i in range(len(df_input)):
        loc_dir = df_input.loc_dirs[i]
        output_dir = df_input.output_dirs[i]
        input_type = df_input.input_types[i]
        if loc_dir[len(loc_dir)-1] != '/' and loc_dir != 'list': #force the directories to end on a slash
            loc_dir = loc_dir + '/'
        if output_dir[len(output_dir)-1] != '/':
            output_dir = output_dir + '/'
        if output_dir.split('/')[-2].lower() == 'mosaic':
            mosaic_dir = output_dir
            output_dir = '/'.join(output_dir.split('/')[:-2]) + '/'
        else:
            mosaic_dir = f'{output_dir}Mosaic/'
        if not subprocess.os.path.isdir(output_dir):
            subprocess.os.mkdir(output_dir)
        if not subprocess.os.path.isdir(mosaic_dir):
            subprocess.os.mkdir(mosaic_dir)
        if loc_dir != 'list':
            loc_name = loc_dir.split('/')[-2]
        else:
            loc_name = single_loc_name
        t_start = datetime.datetime.now()
        print('Working on ' + loc_name)
        if loc_name is not None:
            output_name = loc_name
        else:
            output_name = output_dir.split('/')[-2]
        if loc_name != output_name:
            print('Warning! Output name and location name not the same. Continuing...')
            print(f'Calling everything {output_name} now.')
        if input_type == 3:
            full_strip_list = np.asarray(df_list.strip)
        else:
            full_strip_list = get_strip_list(loc_dir,input_type,corrected_flag,dir_structure)
        if cloud_water_filter_flag == 'default':
            cloud_water_filter_file = None
        elif cloud_water_filter_flag is None:
            cloud_water_filter_file = glob.glob(f'{loc_dir}*Threshold_Exceedance_Values.txt')
            if len(cloud_water_filter_file) == 0:
                print('No cloud/water filter file found! Skipping...')
                cloud_water_filter_file = None
            else:
                cloud_water_filter_file = cloud_water_filter_file[0]
                df_cloud_water = pd.read_csv(cloud_water_filter_file)
                full_strip_list = np.asarray(df_cloud_water.Strip)
        else:
            cloud_water_filter_file = cloud_water_filter_flag
            df_cloud_water = pd.read_csv(cloud_water_filter_file)
            full_strip_list = np.asarray(df_cloud_water.Strip)
        full_epsg_list = np.asarray([osr.SpatialReference(wkt=gdal.Open(s,gdalconst.GA_ReadOnly).GetProjection()).GetAttrValue('AUTHORITY',1) for s in full_strip_list])
        unique_epsg_list = np.unique(full_epsg_list)

        for epsg_code in unique_epsg_list:
            print(f'EPSG:{epsg_code}')
            idx_epsg = full_epsg_list == epsg_code
            strip_list = full_strip_list[idx_epsg]
            if strip_list.size == 0:
                print('No strips found!')
                continue
            lon_min_strips,lon_max_strips,lat_min_strips,lat_max_strips = 180,-180,90,-90
            for strip in strip_list:
                lon_min_single_strip,lon_max_single_strip,lat_min_single_strip,lat_max_single_strip = get_strip_extents(strip)
                lon_min_strips = np.min((lon_min_strips,lon_min_single_strip))
                lon_max_strips = np.max((lon_max_strips,lon_max_single_strip))
                lat_min_strips = np.min((lat_min_strips,lat_min_single_strip))
                lat_max_strips = np.max((lat_max_strips,lat_max_single_strip))
            if gsw_file is None:
                gsw_main_sea_only,gsw_output_shp_file_main_sea_only_clipped_transformed = get_gsw(output_dir,tmp_dir,gsw_dir,epsg_code,lon_min_strips,lon_max_strips,lat_min_strips,lat_max_strips,loc_name,GSW_POCKET_THRESHOLD,GSW_CRS_TRANSFORM_THRESHOLD)
            else:
                gsw_main_sea_only = gpd.read_file(gsw_file)
                gsw_main_sea_only = gsw_main_sea_only.to_crs(f'EPSG:{epsg_code}')

            if gsw_main_sea_only is not None:
                gsw_main_sea_only_buffered = gsw_main_sea_only.buffer(0)
            else:
                gsw_main_sea_only_buffered = None

            strip_shp_data = gpd.GeoDataFrame()
            strip_idx = np.ones(len(strip_list),dtype=bool)
            print('Loading strips...')
            for j,strip in enumerate(strip_list):
                sys.stdout.write('\r')
                n_progressbar = (j + 1) / len(strip_list)
                sys.stdout.write("[%-20s] %d%%" % ('='*int(20*n_progressbar), 100*n_progressbar))
                sys.stdout.flush()
                wv_strip_shp = get_strip_shp(strip,tmp_dir)
                wv_strip_shp_filtered_gsw = filter_strip_gsw(wv_strip_shp,gsw_main_sea_only,STRIP_AREA_THRESHOLD,POLYGON_AREA_THRESHOLD,GSW_OVERLAP_THRESHOLD,STRIP_TOTAL_AREA_PERCENTAGE_THRESHOLD)
                if wv_strip_shp_filtered_gsw is None:
                    strip_idx[j] = False
                    continue
                tmp_mp = shapely.ops.unary_union([Polygon(g) for g in wv_strip_shp_filtered_gsw.geometry.exterior])
                df_strip = pd.DataFrame({'strip':[strip]})
                tmp_gdf = gpd.GeoDataFrame(df_strip,geometry=[tmp_mp],crs='EPSG:'+epsg_code)
                strip_shp_data = gpd.GeoDataFrame(pd.concat([strip_shp_data,tmp_gdf],ignore_index=True),crs='EPSG:'+epsg_code)
            
            strip_list = strip_list[strip_idx]
            if cloud_water_filter_file is not None:
                print('\nApplying cloud/water filter...')
                strip_shp_data = find_cloud_water(strip_shp_data,df_cloud_water)
                idx_cloud_water = np.logical_or(strip_shp_data['Percent Exceedance']>STRIP_CLOUD_THRESHOLD,strip_shp_data['Percent Water']>STRIP_WATER_THRESHOLD)
                strip_shp_data = strip_shp_data[~idx_cloud_water].reset_index(drop=True)
                strip_list = strip_list[~idx_cloud_water]
            else:
                print('\n')
            output_strips_shp_file = f'{output_dir}{output_name}_Strips_{epsg_code}.shp'
            output_strips_shp_file_dissolved = f'{output_dir}{output_name}_Strips_{epsg_code}_Dissolved.shp'
            output_strips_shp_file_filtered = f'{output_dir}{output_name}_Strips_{epsg_code}_Filtered.shp'
            output_strips_shp_file_filtered_dissolved = f'{output_dir}{output_name}_Strips_{epsg_code}_Filtered_Dissolved.shp'
            # print(output_strips_shp_file)
            if simplify_flag == True:
                strip_shp_data.geometry = strip_shp_data.geometry.simplify(tolerance=POLYGON_SIMPLIFY_VALUE)
            
            strip_dates = np.asarray([int(s.split('/')[-1][5:13]) for s in strip_list])
            idx_date = np.argsort(-strip_dates)
            strip_dates = strip_dates[idx_date]
            strip_list = strip_list[idx_date]
            strip_shp_data = strip_shp_data.iloc[idx_date].reset_index(drop=True)

            strip_shp_data.to_file(output_strips_shp_file)
            subprocess.run('ogr2ogr ' + output_strips_shp_file_dissolved + ' ' + output_strips_shp_file + ' -dialect sqlite -sql \'SELECT ST_Union("geometry") FROM "' + os.path.basename(output_strips_shp_file).replace('.shp','') + '"\'',shell=True)

            if all_strips_flag == False:
                idx_contained = get_contained_strips(strip_shp_data,strip_dates,epsg_code,STRIP_CONTAINMENT_THRESHOLD,STRIP_DELTA_TIME_THRESHOLD,N_STRIPS_CONTAINMENT)
                strip_dates = strip_dates[idx_contained]
                strip_list = strip_list[idx_contained]
                strip_shp_data = strip_shp_data.iloc[idx_contained].reset_index(drop=True)

            strip_shp_data.to_file(output_strips_shp_file_filtered)
            subprocess.run('ogr2ogr ' + output_strips_shp_file_filtered_dissolved + ' ' + output_strips_shp_file_filtered + ' -dialect sqlite -sql \'SELECT ST_Union("geometry") FROM "' + os.path.basename(output_strips_shp_file_filtered).replace('.shp','') + '"\'',shell=True)
            
            valid_strip_overlaps = get_valid_strip_overlaps(strip_shp_data,gsw_main_sea_only_buffered,AREA_OVERLAP_THRESHOLD,GSW_INTERSECTION_THRESHOLD)
            mst_array,mst_weighted_array = get_minimum_spanning_tree(valid_strip_overlaps,strip_dates)
            #Need to weight mst_array by delta time (and overlapping area?)
            
            mosaic_dict,singles_dict = find_mosaic(strip_shp_data,mst_weighted_array,strip_dates)
            for mosaic_number in range(len(mosaic_dict)):
                merge_mosaic_output_file = build_mosaic(strip_shp_data,gsw_main_sea_only_buffered,landmask_c_file,mosaic_dict[mosaic_number],mosaic_dir,tmp_dir,output_name,mosaic_number,epsg_code,horizontal_flag,dir_structure,X_SPACING,Y_SPACING,X_MAX_SEARCH,Y_MAX_SEARCH,MOSAIC_TILE_SIZE,N_cpus)
            singles_list = copy_single_strips(strip_shp_data,singles_dict,mosaic_dir,output_name,epsg_code)
            t_end = datetime.datetime.now()
            dt = t_end - t_start
            dt_min, dt_sec = divmod(dt.seconds,60)
            dt_hour, dt_min = divmod(dt_min,60)
            print(f'Finished with {output_name} in EPSG:{epsg_code}.')
            print('It took:')
            print("%d hours, %d minutes, %d.%d seconds" %(dt_hour,dt_min,dt_sec,dt.microseconds%1000000))
            print('')
        
if __name__ == '__main__':
    main()
