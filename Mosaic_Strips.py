import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import shapely
from osgeo import gdal,gdalconst,osr
import os,sys
import argparse
import subprocess
import datetime
import warnings
import configparser

from dem_utils import get_strip_list,get_strip_extents
from dem_utils import get_gsw,get_strip_shp,filter_strip_gsw
from dem_utils import get_contained_strips,get_valid_strip_overlaps,get_minimum_spanning_tree
from dem_utils import find_mosaic,build_mosaic,copy_single_strips

def main():
    warnings.simplefilter(action='ignore')
    config_file = 'dem_config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',default=config.get('MOSAIC_PATHS','input_file'),help='path to dir containing strips')
    parser.add_argument('--list',default=None,help='path to list of strips to mosaic')
    parser.add_argument('--output_dir',default=None,help='path to output directory')
    parser.add_argument('--loc_name',default=None,help='name of location')
    parser.add_argument('--horizontal',default=False,help='Incorperate horizontal alignment in mosaic?',action='store_true')
    parser.add_argument('--machine',default='t',help='Machine to run on (t, b or local)')
    parser.add_argument('--corrected',default=False,help='Find corrected strips instead?',action='store_true')
    parser.add_argument('--all_strips',default=False,help='Mosaic all strips in directory? (No geometry filtering.)',action='store_true')
    parser.add_argument('--dir_structure',default='sealevel',help='Directory structure of input strips (sealevel or simple)')
    parser.add_argument('--cpus',help='Number of cpus to use',default=1)
    args = parser.parse_args()
    input_file = args.input_file
    list_file = args.list
    single_output_dir = args.output_dir
    single_loc_name = args.loc_name
    horizontal_flag = args.horizontal
    machine_name = args.machine
    corrected_flag = args.corrected
    all_strips_flag = args.all_strips
    dir_structure = args.dir_structure
    N_cpus = int(args.cpus)

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
    
    if machine_name == 'b':
        tmp_dir = tmp_dir.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
        gsw_dir = gsw_dir.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
        if df_list is not None:
            df_list.strip = [s.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/') for s in df_list.strip]
    elif machine_name == 'local':
        tmp_dir = tmp_dir.replace('/BhaltosMount/Bhaltos/EDUARD/','/home/heijkoop/Desktop/Projects/')
        gsw_dir = gsw_dir.replace('/BhaltosMount/Bhaltos/EDUARD/DATA_REPOSITORY/','/media/heijkoop/DATA/').replace('Extent/','')

    POLYGON_AREA_THRESHOLD = config.getfloat('MOSAIC_CONSTANTS','POLYGON_AREA_THRESHOLD') #in m^2
    STRIP_AREA_THRESHOLD = config.getfloat('MOSAIC_CONSTANTS','STRIP_AREA_THRESHOLD') #in m^2
    GSW_POCKET_THRESHOLD = config.getfloat('MOSAIC_CONSTANTS','GSW_POCKET_THRESHOLD') #in %
    GSW_CRS_TRANSFORM_THRESHOLD = config.getfloat('MOSAIC_CONSTANTS','GSW_CRS_TRANSFORM_THRESHOLD') #in %
    GSW_OVERLAP_THRESHOLD = config.getfloat('MOSAIC_CONSTANTS','GSW_OVERLAP_THRESHOLD') #in %
    STRIP_TOTAL_AREA_PERCENTAGE_THRESHOLD = config.getfloat('MOSAIC_CONSTANTS','STRIP_TOTAL_AREA_PERCENTAGE_THRESHOLD') #in %
    STRIP_CONTAINMENT_THRESHOLD = config.getfloat('MOSAIC_CONSTANTS','STRIP_CONTAINMENT_THRESHOLD') #in %
    STRIP_DELTA_TIME_THRESHOLD = config.getint('MOSAIC_CONSTANTS','STRIP_DELTA_TIME_THRESHOLD') #in days
    N_STRIPS_CONTAINMENT = config.getint('MOSAIC_CONSTANTS','N_STRIPS_CONTAINMENT') #[-]
    AREA_OVERLAP_THRESHOLD = config.getfloat('MOSAIC_CONSTANTS','AREA_OVERLAP_THRESHOLD') #in m^2
    GSW_INTERSECTION_THRESHOLD = config.getfloat('MOSAIC_CONSTANTS','GSW_INTERSECTION_THRESHOLD') #in %
    X_SPACING = config.getfloat('MOSAIC_CONSTANTS','X_SPACING') #in m
    Y_SPACING = config.getfloat('MOSAIC_CONSTANTS','Y_SPACING') #in m
    X_MAX_SEARCH = config.getfloat('MOSAIC_CONSTANTS','X_MAX_SEARCH') #in m
    Y_MAX_SEARCH = config.getfloat('MOSAIC_CONSTANTS','y_MAX_SEARCH') #in m
    MOSAIC_TILE_SIZE = config.getfloat('MOSAIC_CONSTANTS','MOSAIC_TILE_SIZE') #in m^2   
    
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
        full_epsg_list = np.asarray([osr.SpatialReference(wkt=gdal.Open(s,gdalconst.GA_ReadOnly).GetProjection()).GetAttrValue('AUTHORITY',1) for s in full_strip_list])
        unique_epsg_list = np.unique(full_epsg_list)

        for epsg_code in unique_epsg_list:
            print(f'EPSG:{epsg_code}')
            idx_epsg = full_epsg_list == epsg_code
            strip_list = full_strip_list[idx_epsg]
            if strip_list.size == 0:
                print('No strips found!')
                continue
            strip_shp_data = gpd.GeoDataFrame()
            lon_min_strips,lon_max_strips,lat_min_strips,lat_max_strips = 180,-180,90,-90
            for strip in strip_list:
                lon_min_single_strip,lon_max_single_strip,lat_min_single_strip,lat_max_single_strip = get_strip_extents(strip)
                lon_min_strips = np.min((lon_min_strips,lon_min_single_strip))
                lon_max_strips = np.max((lon_max_strips,lon_max_single_strip))
                lat_min_strips = np.min((lat_min_strips,lat_min_single_strip))
                lat_max_strips = np.max((lat_max_strips,lat_max_single_strip))

            gsw_main_sea_only,gsw_output_shp_file_main_sea_only_clipped_transformed = get_gsw(output_dir,tmp_dir,gsw_dir,epsg_code,lon_min_strips,lon_max_strips,lat_min_strips,lat_max_strips,loc_name,GSW_POCKET_THRESHOLD,GSW_CRS_TRANSFORM_THRESHOLD)
            if gsw_main_sea_only is not None:
                gsw_main_sea_only_buffered = gsw_main_sea_only.buffer(0)
            else:
                gsw_main_sea_only_buffered = None
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
            output_strips_shp_file = f'{output_dir}{output_name}_Strips_{epsg_code}.shp'
            output_strips_shp_file_dissolved = f'{output_dir}{output_name}_Strips_{epsg_code}_Dissolved.shp'
            output_strips_shp_file_filtered = f'{output_dir}{output_name}_Strips_{epsg_code}_Filtered.shp'
            output_strips_shp_file_filtered_dissolved = f'{output_dir}{output_name}_Strips_{epsg_code}_Filtered_Dissolved.shp'
            print('\n')
            print(output_strips_shp_file)
            
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
                merge_mosaic_output_file = build_mosaic(strip_shp_data,gsw_main_sea_only_buffered,landmask_c_file,mosaic_dict[mosaic_number],mosaic_dir,tmp_dir,output_name,mosaic_number,epsg_code,horizontal_flag,X_SPACING,Y_SPACING,X_MAX_SEARCH,Y_MAX_SEARCH,MOSAIC_TILE_SIZE,N_cpus)
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
