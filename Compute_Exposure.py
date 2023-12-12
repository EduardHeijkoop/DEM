import numpy as np
import geopandas as gpd
import configparser
import argparse
import warnings
from osgeo import gdal,gdalconst,osr,ogr
import subprocess
import shapely
import os
import glob
import geopandas as gpd
import shapely

from dem_utils import compress_raster,resample_raster,get_raster_extents


def get_population_data(lon_min,lon_max,lat_min,lat_max,tmp_dir,year=2020):
    '''
    Given a lon/lat extent, download EU population data files and merges them.
    '''
    data_site = f'https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss/V1-0/'
    shp_outline_location = 'https://ghsl.jrc.ec.europa.eu/download/GHSL_data_4326_shapefile.zip'
    shp_outline_file = shp_outline_location.split('/')[-1]
    download_outline_command = f'wget --quiet {shp_outline_location} -P {tmp_dir}'
    subprocess.run(download_outline_command,shell=True)
    unzip_outline_command = f'unzip -qq {tmp_dir}{shp_outline_file} -d {tmp_dir}EU_tile_schema'
    subprocess.run(unzip_outline_command,shell=True)
    subprocess.run(f'rm {tmp_dir}{shp_outline_file}',shell=True)

    dlon = lon_max - lon_min
    dlat = lat_max - lat_min
    nlon = int(np.floor(dlon/10)+2)
    nlat = int(np.floor(dlat/10)+2)
    lon = np.linspace(lon_min,lon_max,nlon)
    lat = np.linspace(lat_min,lat_max,nlat)
    lon_mesh,lat_mesh = np.meshgrid(lon,lat)
    lon_mesh = lon_mesh.flatten()
    lat_mesh = lat_mesh.flatten()

    gdf_outline = gpd.read_file(f'{tmp_dir}EU_tile_schema/WGS84_tile_schema.shp')
    idx_contained = []
    for i in range(len(lon_mesh)):
        tmp = np.argwhere(np.asarray(gdf_outline.contains(shapely.geometry.Point(lon_mesh[i],lat_mesh[i]))))
        if len(tmp) == 0:
            continue
        else:
            idx_contained.append(tmp[0][0])
    tile_list = np.unique(np.asarray(gdf_outline['tile_id'].iloc[idx_contained]))
    merge_command = f'gdal_merge.py -q -o {tmp_dir}tmp_merged.tif '
    for tile in tile_list:
        download_command = f'wget --no-check-certificate --quiet {data_site}tiles/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss_V1_0_{tile}.zip -P {tmp_dir}'
        subprocess.run(download_command,shell=True)
        unzip_command = f'unzip -qq {tmp_dir}GHS_POP_E{year}_GLOBE_R2023A_4326_3ss_V1_0_{tile}.zip -d {tmp_dir}EU_tiles'
        subprocess.run(unzip_command,shell=True)
        subprocess.run(f'rm {tmp_dir}GHS_POP_E{year}_GLOBE_R2023A_4326_3ss_V1_0_{tile}.zip',shell=True)
        subprocess.run(f'rm {tmp_dir}EU_tiles/*xlsx {tmp_dir}EU_tiles/*pdf',shell=True)
        merge_command += f'{tmp_dir}EU_tiles/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss_V1_0_{tile}.tif '
    subprocess.run(merge_command,shell=True)
    warp_command = f'gdalwarp -q -te {lon_min} {lat_min} {lon_max} {lat_max} {tmp_dir}tmp_merged.tif {tmp_dir}GHS_POP_merged_clipped.tif'
    subprocess.run(warp_command,shell=True)
    subprocess.run(f'rm {tmp_dir}tmp_merged.tif',shell=True)
    subprocess.run(f'rm -rf {tmp_dir}EU_tile_schema {tmp_dir}EU_tiles',shell=True)
    merged_file = f'{tmp_dir}GHS_POP_merged_clipped.tif'
    compress_raster(merged_file,nodata=None,quiet_flag=True)
    return merged_file

def compute_exposure(input_raster,tmp_dir,population_file=None,value_threshold=None,keep_population_file=True,nodata=0):
    lon_min,lon_max,lat_min,lat_max = get_raster_extents(input_raster,'global')
    if population_file is None:
        population_file = get_population_data(lon_min,lon_max,lat_min,lat_max,tmp_dir)
    input_raster_resampled = f'{os.path.splitext(input_raster)[0]}_resampled_GHS_POP{os.path.splitext(input_raster)[1]}'
    resample_raster(input_raster,population_file,input_raster_resampled,quiet_flag=True)
    population_subset = f'{os.path.splitext(population_file)[0]}_subset_{os.path.splitext(os.path.basename(input_raster))[0]}{os.path.splitext(population_file)[1]}'
    if value_threshold is not None:
        calc_command = f'gdal_calc.py -A {population_file} -B {input_raster_resampled} --calc="A*logical_and(B<{value_threshold},B!={nodata})" --outfile={population_subset} --NoDataValue=0 --co "COMPRESS=LZW" --quiet --co "BIGTIFF=IF_SAFER"'
    else:
        calc_command = f'gdal_calc.py -A {population_file} -B {input_raster_resampled} --calc="A*(B!={nodata})" --outfile={population_subset} --NoDataValue=0 --co "COMPRESS=LZW" --quiet --co "BIGTIFF=IF_SAFER"'
    subprocess.run(calc_command,shell=True)
    src_population = gdal.Open(population_subset,gdalconst.GA_ReadOnly)
    population_array = src_population.GetRasterBand(1).ReadAsArray()
    population_count = int(np.sum(population_array))
    if keep_population_file is False:
        subprocess.run(f'rm {population_file}',shell=True)
    subprocess.run(f'rm {input_raster_resampled} {population_subset}',shell=True)
    return population_count

def rasterize_polygon(input_polygon,output_raster,resolution,epsg_code,quiet_flag=False):
    '''
    Simple polygon to raster tool for variable resolution. Produces a raster of 0s and 1s.
    '''
    epsg_code = str(epsg_code).upper().replace('EPSG:','')
    rasterize_command = f'gdal_rasterize -burn 1 -ot Byte -a_srs EPSG:{epsg_code} -l {os.path.splitext(os.path.basename(input_polygon))[0]} -tr {resolution} {resolution} {input_polygon} {output_raster}'
    if quiet_flag == True:
        rasterize_command = rasterize_command.replace('gdal_rasterize','gdal_rasterize -q')
    subprocess.run(rasterize_command,shell=True)
    return None

def main():
    warnings.simplefilter(action='ignore')
    config_file = 'dem_config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)

    parser = argparse.ArgumentParser()
    parser.add_argument('--vlm',help='Path to VLM file.',default=None)
    parser.add_argument('--vlm_threshold',help='Threshold value for VLM.',type=float,default=0.0)
    parser.add_argument('--vlm_inverse',help='Use inverse sign for threshold? Default is less than.',action='store_true',default=False)
    parser.add_argument('--inundation',help='Path to inundation file.',default=None,nargs='+')
    parser.add_argument('--population',help='Path to population dataset.',default=None)
    parser.add_argument('--keep_pop',help='Keep population data.',action='store_true',default=False)
    parser.add_argument('--pop_divider',help='Divider for population data.',type=float,default=1.0,choices=[1E1,1E3,1E6])
    parser.add_argument('--machine',help='Machine to run on.',default='t',choices=['t','b','local'])

    args = parser.parse_args()
    vlm_file = args.vlm
    vlm_threshold = args.vlm_threshold
    vlm_inverse_flag = args.vlm_inverse
    inundation_file_list = args.inundation
    population_file = args.population
    keep_pop_flag = args.keep_pop
    population_divider = args.pop_divider
    machine_name = args.machine

    tmp_dir = config.get('GENERAL_PATHS','tmp_dir')
    inundation_resolution = config.get('INUNDATION_CONSTANTS','INUNDATION_GRIDDING_RESOLUTION')

    if machine_name == 'b':
        tmp_dir = tmp_dir.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
    elif machine_name == 'local':
        tmp_dir = tmp_dir.replace('/BhaltosMount/Bhaltos/EDUARD/','/home/heijkoop/Desktop/Projects/')

    if population_divider == 1E1:
        pop_suffix = ''
        pop_format_spec = ''
    elif population_divider == 1E3:
        pop_suffix = 'k'
        pop_format_spec = '.1f'
    elif population_divider == 1E6:
        pop_suffix = 'M'
        pop_format_spec = '.1f'

    lon_min,lon_max,lat_min,lat_max = 180,-180,90,-90 #inverted on purpose

    if vlm_file is not None:
        lon_min_vlm,lon_max_vlm,lat_min_vlm,lat_max_vlm = get_raster_extents(vlm_file,'global')
        lon_min = np.min((lon_min,lon_min_vlm))
        lon_max = np.max((lon_max,lon_max_vlm))
        lat_min = np.min((lat_min,lat_min_vlm))
        lat_max = np.max((lat_max,lat_max_vlm))

    if inundation_file_list is not None:
        if '*' in inundation_file_list:
            inundation_file_list = np.asarray(sorted(glob.glob(inundation_file_list)))
        else:
            inundation_file_list = np.atleast_1d(inundation_file_list)
        for inundation_file in inundation_file_list:
            src_inundation_vec = ogr.Open(inundation_file)
            epsg_code_inundation = src_inundation_vec.GetLayer().GetSpatialRef().GetAttrValue('AUTHORITY',1)
            x_min,x_max,y_min,y_max = src_inundation_vec.GetLayer().GetExtent()
            bbox = shapely.geometry.Polygon(((x_min,y_max),(x_max,y_max),(x_max,y_min),(x_min,y_min),(x_min,y_max)))
            gdf_bbox = gpd.GeoDataFrame(geometry=[bbox],crs=f'EPSG:{epsg_code_inundation}')
            if epsg_code_inundation != '4326':
                gdf_bbox_4326 = gdf_bbox.to_crs('EPSG:4326')
                lon_min_inundation,lon_max_inundation,lat_min_inundation,lat_max_inundation = list(gdf_bbox_4326.bounds.to_numpy()[0])
            else:
                lon_min_inundation,lon_max_inundation,lat_min_inundation,lat_max_inundation = x_min,x_max,y_min,y_max
            # lon_min_inundation,lon_max_inundation,lat_min_inundation,lat_max_inundation = list(gdf_bbox_4326.bounds.to_numpy()[0])
            lon_min = np.min((lon_min,lon_min_inundation))
            lon_max = np.max((lon_max,lon_max_inundation))
            lat_min = np.min((lat_min,lat_min_inundation))
            lat_max = np.max((lat_max,lat_max_inundation))

    if population_file is None:
        print('Downloading population data...')
        population_file = get_population_data(lon_min,lon_max,lat_min,lat_max,tmp_dir)
    
    if vlm_file is not None:
        vlm_count = compute_exposure(vlm_file,tmp_dir,population_file=population_file,value_threshold=None)
        print(f'Number of people exposed to VLM: {vlm_count}')
        if vlm_threshold is not None:
            vlm_count_threshold = compute_exposure(vlm_file,tmp_dir,population_file=population_file,value_threshold=vlm_threshold)
            if vlm_inverse_flag == True:
                print(f'Number of people exposed to VLM > {1000*vlm_threshold:.1f} mm/yr: {(vlm_count-vlm_count_threshold)/population_divider:{pop_format_spec}}{pop_suffix}')
            else:
                print(f'Number of people exposed to VLM < {1000*vlm_threshold:.1f} mm/yr: {vlm_count_threshold/population_divider:{pop_format_spec}}{pop_suffix}')

    if inundation_file_list is not None:
        for inundation_file in inundation_file_list:
            print(f'Working on: {os.path.splitext(os.path.basename(inundation_file))[0]}')
            inundation_raster = f'{tmp_dir}{os.path.splitext(os.path.basename(inundation_file))[0]}.tif'
            # inundation_raster_resampled_pop = inundation_raster.replace('.tif','resampled_GHS_POP.tif')
            src_inundation_vec = ogr.Open(inundation_file)
            epsg_code_inundation = src_inundation_vec.GetLayer().GetSpatialRef().GetAttrValue('AUTHORITY',1)
            rasterize_polygon(inundation_file,inundation_raster,inundation_resolution,epsg_code_inundation,quiet_flag=True)
            if epsg_code_inundation != '4326':
                inundation_raster_4326 = f'{tmp_dir}{os.path.splitext(os.path.basename(inundation_file))[0]}_EPSG4326.tif'
                warp_4326_command = f'gdalwarp -q -s_srs EPSG:{epsg_code_inundation} -t_srs EPSG:4326 {inundation_raster} {inundation_raster_4326}'
                subprocess.run(warp_4326_command,shell=True)
                subprocess.run(f'rm {inundation_raster}',shell=True)
            else:
                inundation_raster_4326 = inundation_raster
            # resample_raster(inundation_raster,population_file,inundation_raster_resampled_pop,quiet_flag=True)
            inundation_count = compute_exposure(inundation_raster_4326,tmp_dir,population_file=population_file,value_threshold=None)
            print(f'Number of people exposed to inundation: {inundation_count/population_divider:{pop_format_spec}}{pop_suffix}')
            subprocess.run(f'rm {inundation_raster_4326}',shell=True)

    if keep_pop_flag == False:
        subprocess.run(f'rm {population_file}',shell=True)


if __name__ == '__main__':
    main()