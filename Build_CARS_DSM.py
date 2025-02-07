# import cars
import numpy as np
import argparse
import warnings
import configparser
import subprocess
import rasterio as rio
import geopandas as gpd
import pandas as pd
import glob
import xml.etree.ElementTree as ET
import shapely
import os
import shutil
import affine
import datetime
import json
from osgeo import gdal,gdalconst,osr
import skimage
from scipy import ndimage

from max_rect import get_maximal_rectangle
from dem_utils import raster_to_geotiff_w_src,resample_raster


def get_window_from_roi(src, roi):
    '''
    get window from region of interest
    '''
    coordinates = []
    transformer = rio.transform.RPCTransformer(src.rpcs)
    for x in [roi[0], roi[2]]:
        for y in [roi[1], roi[3]]:
            coordinates.append(transformer.rowcol(x, y))
    coordinates = np.array(coordinates)
    (left, bottom), (right, top) = np.amin(coordinates, axis=0), np.amax(coordinates, axis=0)
    return left, bottom, right, top

def create_geomodel_file(src,geomodel_file):
    '''
    create geomodel file from ntf file src
    '''
    rpcs_as_dict = src.rpcs.to_dict()
    with open(geomodel_file, "w") as writer:
        for key in rpcs_as_dict:
            if isinstance(rpcs_as_dict[key], list):
                for idx, coef in enumerate(rpcs_as_dict[key]):
                    writer.write(": ".join([key+"_%02d"%idx, str(rpcs_as_dict[key][idx])]) +"\n")
            else:
                writer.write(": ".join([key, str(rpcs_as_dict[key])])+"\n")
        writer.write("type:  ossimRpcModel\n")
        writer.write("polynomial_format:  B\n")

def create_extents_file(ntf_file,output_dir,roi_bounds):
    '''
    Given the area of interest/overlap, subsets the input ntf files into .tif files
    Function courtesy of David Youssefi (CNES)
    '''
    extents_file = os.path.join(*[output_dir,'PAN_EXT',f'{os.path.splitext(os.path.basename(ntf_file))[0]}_extents.tif'])
    geomodel_file = os.path.join(*[output_dir,'PAN_EXT',f'{os.path.splitext(os.path.basename(ntf_file))[0]}_extents.geom'])
    src = rio.open(ntf_file)
    row_start, col_start, row_stop, col_stop = get_window_from_roi(src, roi_bounds)
    window = rio.windows.Window.from_slices((row_start, row_stop), (col_start, col_stop))
    array = src.read(1, window=window)
    profile = src.profile
    profile["driver"] = "GTiff"
    profile["width"] = window.width
    profile["height"] = window.height
    profile["transform"] = profile["transform"] * affine.Affine.translation(window.col_off, window.row_off)
    with rio.open(extents_file, "w", **profile) as dst:
        dst.write(array, 1)
    create_geomodel_file(src, geomodel_file)
    return extents_file

def create_config_file(cars_config_file,extents_file_list,output_dir,config_dict):
    '''
    Uses cars-starter to create config .json file that will be used to run CARS
    Omits the "--full" command so user can configure better
    '''
    extents_file_list_str = ' '.join(extents_file_list)
    cars_starter_command = f'cars-starter -il {extents_file_list_str} -out {output_dir} > {cars_config_file}'
    subprocess.run(cars_starter_command,shell=True)
    with open(cars_config_file,'r') as f:
        cars_config_data = json.load(f)
    cars_config_data['output']['resolution'] = config_dict['resolution']
    if config_dict['N_overlap'] > 1:
        cars_config_output_file = cars_config_file.replace('.json',f'_{config_dict["i_overlap"]}.json')
    else:
        cars_config_output_file = cars_config_file
    # if config_dict['bulldozer_flag'] == True:
    #     cars_config_data['applications'] = {}
    #     cars_config_data['applications']['dsm_filling'] = {}
    #     cars_config_data['applications']['dsm_filling']['method'] = 'bulldozer'
    #     cars_config_data['applications']['dsm_filling']['activated'] = True
    #     cars_config_data['applications']['dsm_filling']['save_intermediate_data'] = True  #or True to hang on to DSM? Does it delete DSM??
    # if config_dict['a_priori_flag'] == True:
        # cars_config_data[] = {}
    with open(cars_config_output_file,'w') as f:
        json.dump(cars_config_data,f,indent=4)
    return cars_config_output_file

def get_outline(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    ullon = float(root.find('IMD').find('BAND_P').find('ULLON').text)
    ullat = float(root.find('IMD').find('BAND_P').find('ULLAT').text)
    urlon = float(root.find('IMD').find('BAND_P').find('URLON').text)
    urlat = float(root.find('IMD').find('BAND_P').find('URLAT').text)
    lrlon = float(root.find('IMD').find('BAND_P').find('LRLON').text)
    lrlat = float(root.find('IMD').find('BAND_P').find('LRLAT').text)
    lllon = float(root.find('IMD').find('BAND_P').find('LLLON').text)
    lllat = float(root.find('IMD').find('BAND_P').find('LLLAT').text)
    outline = shapely.geometry.Polygon([(ullon,ullat),(urlon,urlat),(lrlon,lrlat),(lllon,lllat),(ullon,ullat)])
    return outline

def get_outline_geom(df_input):
    '''
    Grabs corner points of image from corresponding xml file and returns GeoDataFrame with them all
    '''
    gdf_outline_geom = gpd.GeoDataFrame(geometry=[get_outline(xml_file) for xml_file in df_input['xml_file']],crs='EPSG:4326')
    return gdf_outline_geom

def get_overlap(gdf_geometry,extents):
    '''
    Using an input file with image paths, get the overlapping area
    If extents is a set of lon_min,lon_max,lat_min,lat_max, that will be used to narrow down overlap,
    otherwise, the overlap of all images will be used.
    If multiple overlapping areas exist, they will be returned in sequence in the GeoDataFrame
    '''
    geom_intersection = shapely.intersection_all(gdf_geometry['geometry'])
    if geom_intersection.geom_type == 'MultiPolygon':
        gdf_overlap = gpd.GeoDataFrame(geometry=[geom for geom in geom_intersection],crs='EPSG:4326')
    elif geom_intersection.geom_type == 'Polygon':
        gdf_overlap = gpd.GeoDataFrame(geometry=[geom_intersection],crs='EPSG:4326')
    if extents == 'overlap':
        pass
    elif len(extents) == 4:
        lon_min,lon_max,lat_min,lat_max = extents
        geom_extents = shapely.box(lon_min,lat_min,lon_max,lat_max)
        gdf_overlap = gpd.GeoDataFrame(geometry=[geom.intersection(geom_extents) for geom in gdf_overlap['geometry'] if geom.intersects(geom_extents)],crs='EPSG:4326')
    if len(gdf_overlap) == 0:
        raise Exception('No overlapping area found!')
    elif len(gdf_overlap) > 1:
        print('Multiple overlapping areas found!')
    return gdf_overlap

def get_xml_list(df_input):
    '''
    Get the XML file associated with each NTF file
    '''
    ntf_file_list = df_input['ntf_file'].tolist()
    xml_file_list = [s.replace('.NTF','.XML').replace('.ntf','.xml') for s in ntf_file_list]
    xml_exist = [os.path.exists(s) for s in xml_file_list]
    for i,(f,e) in enumerate(zip(xml_file_list,xml_exist)):
        if not e:
            xml_swapped = f'{os.path.splitext(f)[0]}{os.path.splitext(f)[1].swapcase()}'
            if os.path.exists(xml_swapped):
                xml_file_list[i] = xml_swapped
                xml_exist[i] = True
    if not all(xml_exist):
        raise Exception('XML files do not exist for all NTF files.')
    return xml_file_list

def get_sensor_list(df_input):
    '''
    Get the sensor (WorldView-1, GeoEye, etc) associated with each NTF file
    '''
    ntf_file_list = df_input['ntf_file'].tolist()
    sensor_list = [os.path.basename(s).split('_')[0] for s in ntf_file_list]
    return sensor_list

def get_acq_date_list(df_input):
    '''
    Get the acquisition date associated with each NTF file
    We only care about unique acquisition dates, not time, hence the [:8]
    '''
    ntf_file_list = df_input['ntf_file'].tolist()
    acq_date_list = [os.path.basename(s).split('_')[1][:8] for s in ntf_file_list]
    return acq_date_list

def get_short_name_list(df_input):
    '''
    Get the short name associated with each NTF file
    '''
    ntf_file_list = df_input['ntf_file'].tolist()
    short_name_list = []
    for i in range(len(ntf_file_list)):
        sensor = df_input.sensor[i]
        acq_date = df_input.acq_date[i]
        acq_code = os.path.basename(df_input.ntf_file[i]).split("_")[2]
        img_number = os.path.splitext(os.path.basename(df_input.ntf_file[i]).split("_")[-1])[0]
        short_name = '_'.join([sensor,acq_date,acq_code,img_number])
        short_name_list.append(short_name)
    return short_name_list

def expand_input_list(df_input):
    '''
    Takes input DataFrame and expands columns to include:
    '''
    df_input['xml_file'] = get_xml_list(df_input)
    df_input['sensor'] = get_sensor_list(df_input)
    df_input['acq_date'] = get_acq_date_list(df_input)
    df_input['short_name'] = get_short_name_list(df_input)
    return df_input

def find_clumps(binary_arr,max_size,invert=False,remove_largest=False,fill_holes=False):
    '''
    Takes an array of 1s and 0s, finds the clumps and returns the array with clumps smaller/larger than threshold removed
    max_size is a percentage of the total array size
    Optionally fills holes in the clumps
    '''
    label,num_label = ndimage.label(binary_arr == 1)
    size = np.bincount(label.ravel())
    label_IDs = np.arange(len(size))
    if invert == True:
        label_IDs_select = label_IDs[size <= max_size]
    else:
        label_IDs_select = label_IDs[size >= max_size]
    if remove_largest == True:
        idx_largest = np.atleast_1d(np.argwhere(size==np.max(size)).squeeze())
        label_IDs_select = np.delete(label_IDs_select,idx_largest)
    clump_arr = np.isin(label,label_IDs_select)
    if fill_holes == True:
        clump_arr = ndimage.binary_fill_holes(clump_arr)
    return clump_arr.astype(int)

def arr_float_to_int(arr):
    '''
    Converts array of floats to [0,255] integers
    '''
    arr_int = arr.copy()
    arr_int -= np.nanmin(arr_int)
    arr_int /= np.nanmax(arr_int)
    arr_int *= 255
    arr_int[np.isnan(arr_int)] = 0
    arr_int = arr_int.astype(np.uint8)
    return arr_int

def entropy_filter(arr,entropy_threshold,size_threshold):
    '''
    Uses scikit-image to compute the entropy of the DSM (normalized to [0,255])
    Only entropy above a certain threshold is considered
    Patches larger than a certain size (user-defined) are considered clouds
    These patches are filled with scipy.ndimage.binary_fill_holes
    '''
    arr_int = arr_float_to_int(arr)
    entropy = skimage.filters.rank.entropy(arr_int,skimage.morphology.disk(10))
    entropy_binary = entropy >= entropy_threshold
    entr_clump_filled = find_clumps(entropy_binary,size_threshold,remove_largest=True,fill_holes=True)
    arr_filtered = arr.copy()
    arr_filtered[entr_clump_filled == 1] = np.nan
    return arr_filtered

def cloud_filter(dsm_file,cloud_filter_dict):
    '''
    ENTROPY FILTER N_PIXEL_THRESHOLD IS NOT N_PIXELS BUT PCT_PIXELS
        CHANGE NEEDS TO BE MADE
    '''
    cloud_filter_entropy_threshold = cloud_filter_dict['cloud_filter_entropy_threshold']
    cloud_filter_size_threshold = cloud_filter_dict['cloud_filter_size_threshold']
    src = gdal.Open(dsm_file, gdalconst.GA_ReadOnly)
    dsm_nodata = src.GetRasterBand(1).GetNoDataValue()
    dsm_arr = np.array(src.GetRasterBand(1).ReadAsArray())
    dsm_arr[dsm_arr == dsm_nodata] = np.nan
    dsm_arr_filtered = entropy_filter(dsm_arr,cloud_filter_entropy_threshold,cloud_filter_size_threshold)
    dsm_filtered_file = dsm_file.replace('.tif','_filtered.tif')
    raster_to_geotiff_w_src(src,dsm_arr_filtered,dsm_filtered_file)
    return dsm_filtered_file

def get_diff_a_priori(input_file,a_priori_file,return_arr=True,remove_file=False,centering=True,nodata=-9999):
    '''
    Computes difference between input and a priori files, by resampling a priori to input
    If return_arr is True, returns array of difference, optionally centered to the mean diff
    else returns the filename of the difference file
    '''
    a_priori_resampled_file = a_priori_file.replace('.tif','_resampled.tif')
    diff_file = os.path.join(os.path.dirname(a_priori_file),f'{os.path.basename(input_file).replace(".tif","_diff_a_priori.tif")}')
    resample_raster(a_priori_file,input_file,a_priori_resampled_file,resample_method='bilinear',compress=True,nodata=nodata,quiet_flag=True)
    diff_command = f'gdal_calc.py --quiet --overwrite -A {input_file} -B {a_priori_resampled_file} --outfile={diff_file} --calc="A-B" --creation-option "COMPRESS=LZW" --creation-option "BIGTIFF=IF_SAFER" --NoDataValue={nodata}'
    subprocess.run(diff_command,shell=True)
    if return_arr == True:
        src_diff = gdal.Open(diff_file, gdalconst.GA_ReadOnly)
        arr_diff = np.array(src_diff.GetRasterBand(1).ReadAsArray())
        if centering == True:
            arr_diff[arr_diff==nodata] = np.nan
            mean_offset = np.nanmean(arr_diff)
            arr_diff -= mean_offset
            offset_calc_command = f'gdal_calc.py --quiet --overwrite -A {diff_file} --outfile={diff_file.replace(".tif","_offset.tif")} --calc="A-{mean_offset:.3f}" --creation-option "COMPRESS=LZW" --creation-option "BIGTIFF=IF_SAFER" --NoDataValue={nodata}'
            subprocess.run(offset_calc_command,shell=True)
            shutil.move(diff_file.replace('.tif','_offset.tif'),diff_file)
        if remove_file == True:
            os.remove(a_priori_resampled_file)
            os.remove(diff_file)
        return arr_diff
    else:
        return diff_file

def interpolate_holes(dsm_file,config_dict,interpolation_dict):
    '''
    CARS tends to create a DSM with some "holes" (pockets of strongly relatively negative values)
    This function identifies these holes using the Copernicus DEM as a priori data
    and removes them by interpolating across them
    '''
    src_dsm = gdal.Open(dsm_file, gdalconst.GA_ReadOnly)
    arr_dsm = np.array(src_dsm.GetRasterBand(1).ReadAsArray())
    dsm_file_filtered = dsm_file.replace('.tif','_filtered.tif')
    dsm_file_interpolated = dsm_file.replace('.tif','_interpolated.tif')
    cars_nodata_value = config_dict['cars_nodata_value']
    copernicus_file = interpolation_dict['copernicus_file']
    interpolation_vertical_threshold = interpolation_dict['interpolation_vertical_threshold']
    interpolation_size_threshold = interpolation_dict['interpolation_size_threshold']
    interpolation_n_dilations = interpolation_dict['interpolation_n_dilations']
    interpolation_max_distance = interpolation_dict['interpolation_max_distance']
    copernicus_diff_arr = get_diff_a_priori(input_file=dsm_file,a_priori_file=copernicus_file,return_arr=True,centering=True,nodata=cars_nodata_value)
    copernicus_diff_arr_binary = (copernicus_diff_arr < -interpolation_vertical_threshold).astype(int)
    copernicus_diff_arr_clumps = find_clumps(copernicus_diff_arr_binary,interpolation_size_threshold,invert=True,remove_largest=False,fill_holes=False)
    copernicus_diff_arr_clumps_file = copernicus_file.replace('.tif','_resampled_diff_clumps.tif')
    raster_to_geotiff_w_src(src_dsm,copernicus_diff_arr_clumps,copernicus_diff_arr_clumps_file,dtype=gdal.GDT_Byte)
    arr_dsm_filtered = arr_dsm.copy()
    arr_dsm_filtered[copernicus_diff_arr_clumps == 1] = cars_nodata_value
    raster_to_geotiff_w_src(src_dsm,arr_dsm_filtered,dsm_file_filtered)
    set_nodata_command = f'gdal_edit.py -a_nodata {cars_nodata_value} {dsm_file_filtered}'
    subprocess.run(set_nodata_command,shell=True)
    arr_dsm_mask = (arr_dsm_filtered != cars_nodata_value).astype(int)
    interpolation_command = f'gdal_fillnodata.py -q -md {interpolation_max_distance} {dsm_file_filtered} {dsm_file_interpolated}'
    arr_binary_dilated = ndimage.binary_dilation(arr_dsm_mask,structure=ndimage.generate_binary_structure(2,2),iterations=interpolation_n_dilations)
    arr_binary_dilated_filled = ndimage.binary_fill_holes(arr_binary_dilated).astype(int)
    arr_mask_file = dsm_file.replace('.tif','_valid_mask.tif')
    raster_to_geotiff_w_src(src_dsm,arr_binary_dilated_filled,arr_mask_file,dtype=gdal.GDT_Byte)
    tmp_file = os.path.join(os.path.dirname(dsm_file),'tmp.tif')
    mask_command = f'gdal_calc.py --quiet -A {dsm_file_interpolated} -B {arr_mask_file} --calc="A*B" --outfile={tmp_file} --NoDataValue=0 --co="COMPRESS=LZW"'
    subprocess.run(interpolation_command,shell=True)
    subprocess.run(mask_command,shell=True)
    shutil.move(tmp_file,dsm_file_interpolated)
    os.remove(arr_mask_file)
    os.remove(copernicus_diff_arr_clumps_file)
    os.remove(dsm_file_filtered)
    return dsm_file_interpolated

def build_dirs(output_dir,project_name):
    '''
    Makes directories if they don't exist yet
    Forces output_dir to be named project name
        e.g. if original output_dir is '/data/DEM/Europe/' and project_name is 'Netherlands'
        then output_dir will be '/data/DEM/Europe/Netherlands/'
    '''
    if output_dir[-1] != '/':
        output_dir = f'{output_dir}/'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if output_dir.split('/')[-1] != project_name:
        output_dir = os.path.join(output_dir,project_name)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(os.path.join(output_dir,'PAN_EXT')):
        os.makedirs(os.path.join(output_dir,'PAN_EXT'))
    return output_dir

def write_dsm_metadata(config_dict,df_select,dsm_name):
    '''
    Writes a metadata file for the DSM
    '''
    output_dir = config_dict['output_dir']
    metadata_file = os.path.join(*[output_dir,'dsm',f'{dsm_name}_metadata.json'])
    metadata_dict = {
        'project_name':config_dict['project_name'],
        'dsm_name':dsm_name,
        'overlap_number':config_dict['i_overlap'],
        'overlap_images':df_select['short_name'].to_list(),
        'overlap_sensors':df_select['sensor'].to_list(),
        'overlap_acquisition_dates':df_select['acq_date'].to_list(),
        'overlap_NTF_files':df_select['ntf_file'].to_list(),
    }
    with open(metadata_file,'w') as f:
        json.dump(metadata_dict,f,indent=4)

def create_dsm_name(config_dict,df_select):
    '''
    Takes input from df_select and config_dict and turns that into a filename
    '''
    output_dir = config_dict['output_dir']
    project_name = config_dict['project_name']
    i_overlap = config_dict['i_overlap']
    dsm_file = os.path.join(*[output_dir,'dsm','dsm.tif'])
    sensor_date_list = ['_'.join([s,a]) for s,a in zip(df_select['sensor'],df_select['acq_date'])]
    unique_sensor_date_list = np.unique(sensor_date_list)
    acq_code_img_number_list = ['_'.join(s.split('_')[-2:]) for s in df_select.short_name]
    if len(df_select) == 2:
        if len(unique_sensor_date_list) == 1:
            dsm_name = f'{project_name}_DSM_{unique_sensor_date_list[0]}_{"_".join(acq_code_img_number_list)}'
        else:
            dsm_name = f'{project_name}_DSM_{"_".join(df_select['short_name'].to_list())}'
    else:
        if config_dict['N_overlap'] == 1:
            dsm_name = f'{project_name}_DSM'
        else:
            dsm_name = f'{project_name}_DSM_{i_overlap}'
    write_dsm_metadata(config_dict,df_select,dsm_name)
    dsm_name = f'{dsm_name}.tif'
    return dsm_name

def move_rename_dsm(config_dict,df_select):
    '''
    Cleans up directory of CARS output and renames dsm.tif file based on input NTFs
    '''
    output_dir = config_dict['output_dir']
    project_name = config_dict['project_name']
    config_file = config_dict['config_file']
    i_overlap = config_dict['i_overlap']
    metadata_dir = os.path.join(output_dir,f'METADATA')
    if config_dict['N_overlap'] > 1:
        config_file = config_file.replace('.json',f'_{i_overlap}.json')
        metadata_dir = metadata_dir.replace('METADATA',f'METADATA_{i_overlap}')
    if not os.path.isdir(metadata_dir):
        os.makedirs(metadata_dir)
    dsm_dir = os.path.join(output_dir,'DSM')
    if not os.path.isdir(dsm_dir):
        os.makedirs(dsm_dir)
    dsm_file = os.path.join(*[output_dir,'dsm','dsm.tif'])
    dsm_name_new = create_dsm_name(config_dict,df_select)
    dsm_file_new = os.path.join(dsm_dir,dsm_name_new)
    shutil.move(dsm_file,dsm_file_new)
    shutil.move(os.path.join(output_dir,'dsm'),metadata_dir)
    shutil.move(os.path.join(output_dir,'logs'),metadata_dir)
    shutil.move(os.path.join(output_dir,'PAN_EXT'),metadata_dir)
    shutil.move(os.path.join(output_dir,'metadata.json'),metadata_dir)
    shutil.move(os.path.join(output_dir,'used_conf.json'),metadata_dir)
    shutil.move(os.path.join(output_dir,config_file),metadata_dir)
    return dsm_file_new

def main():
    '''
    Builds on example on GitHub to create DSM from multiple .ntf files
    Takes image paths in a file and lon/lat bbox and turns into a DSM
    Adds optional filtering/interpolation/smoothing to correct for small gaps/artifacts
    1. Builds geometry files from .NTF/.xml files
    2. Builds config.json file
    3. Calls CARS to build DSM
    Input file (no header line) must have full paths to NTF files, that must be in the format:
    WV**_YYYYMMDDhhmmss_*_YYMMDDhhmmss-P1BS-*_*_01_P0**.ntf
    e.g. WV02_20241001015424_1030010106AA5C00_24OCT01015424-P1BS-016474886010_01_P003.ntf
    '''
    warnings.simplefilter(action='ignore')
    input_config_file = 'dem_config.ini'
    config = configparser.ConfigParser()
    config.read(input_config_file)

    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name',help='Project name? Output files will be use this.',default=f'CARS_Run_{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")}')
    parser.add_argument('--input_file',help='File with full paths of image NTFs to build DSM from.')
    parser.add_argument('--extents',help='Extents (lon_min,lon_max,lat_min,lat_max) or overlapping (overlap) area to build DSM.',nargs='*',default='overlap')
    parser.add_argument('--output_dir',help='Output directory for DSM.')
    parser.add_argument('--resolution',help='Resolution of DSM.',default=2.0,type=float)
    # Processing options
    parser.add_argument('--coast',help='Use coastline file to clip DSM.',default=None)
    parser.add_argument('--crop_coast',help='Crop DSM extents to that of coastline?',action='store_true',default=False)
    parser.add_argument('--cloud_filter',help='Use entropy-based cloud filter to remove cloudy patches.',action='store_true',default=False)
    parser.add_argument('--interpolation',help='Apply interpolation to holes?',action='store_true',default=False)
    # To implement next:
    parser.add_argument('--a_priori',help='Use Copernicus DEM as a priori.',action='store_true',default=False)
    # parser.add_argument('--optimal_pairs',help='Find optimal pairs from input file to create DSMs with.',action='store_true',default=False)
    # parser.add_argument('--bulldozer',help='Use bulldozer method for DSM to DTM conversion.',action='store_true',default=False)

    args = parser.parse_args()
    project_name = args.project_name
    input_file = args.input_file
    extents = args.extents
    output_dir = args.output_dir
    output_resolution = args.resolution
    coast_file = args.coast
    crop_coast_flag = args.crop_coast #this just adds the crop_to_cutline flag to gdalwarp
    cloud_filter_flag = args.cloud_filter
    interpolation_flag = args.interpolation
    a_priori_flag = args.a_priori
    # optimal_pairs_flag = args.optimal_pairs
    # bulldozer_flag = args.bulldozer

    output_dir = build_dirs(output_dir,project_name)

    if len(extents) == 0:
        extents = extents[0]
        
    if extents != 'overlap':
        extents = [float(e) for e in extents.replace(',',' ').split()]
        if len(extents) != 4:
            raise Exception('Extents must be in format lon_min,lon_max,lat_min,lat_max')

    cars_config_file = os.path.join(output_dir,f'{project_name}_cars_config_{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")}.json')

    df_input = pd.read_csv(input_file,header=None,names=['ntf_file'])
    df_input = expand_input_list(df_input)
    gdf_geometry = get_outline_geom(df_input)
    gdf_overlap = get_overlap(gdf_geometry,extents)

    if a_priori_flag == True or interpolation_flag == True:
        from Global_DEMs import download_copernicus
        tmp_dir = config.get('GENERAL_PATHS','tmp_dir')
        egm2008_file = config.get('GENERAL_PATHS','EGM2008_path')
        output_copernicus_file = os.path.join(tmp_dir,f'{project_name}_Copernicus_WGS84_0.tif')

    config_dict = {
        'output_dir':output_dir,
        'project_name':project_name,
        'config_file':cars_config_file,
        'resolution':output_resolution,
        'interpolation_flag':interpolation_flag,
        'a_priori_flag':a_priori_flag,
        # 'bulldozer_flag':bulldozer_flag,
        'N_overlap':len(gdf_overlap),
        'i_overlap':0,
        'cars_nodata_value':config.getint('CARS_CONSTANTS','CARS_NODATA'),
    }

    if cloud_filter_flag == True:
        cloud_filter_dict = {
            'cloud_filter_entropy_threshold':config.getfloat('CARS_CONSTANTS','cloud_filter_entropy_threshold'),
            'cloud_filter_size_threshold':config.getfloat('CARS_CONSTANTS','cloud_filter_size_threshold')/output_resolution**2,
        }
    
    if interpolation_flag == True:
        interpolation_dict = {
            'interpolation_vertical_threshold':config.getfloat('CARS_CONSTANTS','interpolation_vertical_threshold'),
            'interpolation_size_threshold':config.getfloat('CARS_CONSTANTS','interpolation_size_threshold')/output_resolution**2,
            'interpolation_n_dilations':config.getint('CARS_CONSTANTS','interpolation_n_dilations'),
            'interpolation_max_distance':config.getint('CARS_CONSTANTS','interpolation_max_distance'),
        }

    gdalwarp_compress = '-co "COMPRESS=LZW" -co "BIGTIFF=IF_SAFER" -co "TILED=YES"'
    
    #Build DSMs for each overlapping area
    for i in range(len(gdf_overlap)):
        config_dict['i_overlap'] = i
        extents_file_list = []
        idx_select = [geom.contains(gdf_overlap.geometry[i].buffer(-1e-10)) for geom in gdf_geometry.geometry]
        df_select = df_input.loc[idx_select]
        if extents == 'overlap':
            roi_bounds = get_maximal_rectangle(gdf_overlap.geometry[i])
        else:
            roi_bounds = [extents[0],extents[2],extents[1],extents[3]]
        for j in range(len(df_select)):
            ntf_file = df_select['ntf_file'].iloc[j]
            extents_file = create_extents_file(ntf_file,output_dir,roi_bounds)
            extents_file_list.append(extents_file)
        if a_priori_flag == True or interpolation_flag == True:
            output_copernicus_file = output_copernicus_file.replace(f'_{i-1}.tif',f'_{i}.tif')
            download_copernicus(roi_bounds[0],roi_bounds[2],roi_bounds[1],roi_bounds[3],egm2008_file,tmp_dir,output_copernicus_file,copy_nan_flag=False)
            interpolation_dict['copernicus_file'] = output_copernicus_file
        cars_config_file_new = create_config_file(cars_config_file,extents_file_list,output_dir,config_dict)
        cars_run_command = f'cars {cars_config_file_new}'
        subprocess.run(cars_run_command,shell=True)
        dsm_file = move_rename_dsm(config_dict,df_select)
        if cloud_filter_flag is not None:
            dsm_file = cloud_filter(dsm_file,cloud_filter_dict)
        if interpolation_flag == True:
            dsm_file = interpolate_holes(dsm_file,config_dict,interpolation_dict)
        if coast_file is not None:
            if coast_file == 'osm':
                osm_coastline_file = config.get('GENERAL_PATHS','osm_shp_file')
                coast_file = os.path.join(output_dir,f'{project_name}_OSM_Coast.shp')
                if not os.path.exists(coast_file):
                    osm_subset_command = f'ogr2ogr -f "ESRI Shapefile" {coast_file} {osm_coastline_file} -clipsrc {gdf_overlap.bounds.minx.min()} {gdf_overlap.bounds.miny.min()} {gdf_overlap.bounds.maxx.max()} {gdf_overlap.bounds.maxy.max()}'
                    subprocess.run(osm_subset_command,shell=True)
            clip_command = f'gdalwarp -q -cutline {coast_file} {dsm_file} {dsm_file.replace(".tif","_clipped.tif")} {gdalwarp_compress}'
            if crop_coast_flag == True:
                clip_command = clip_command.replace('gdalwarp','gdalwarp -crop_to_cutline')
            subprocess.run(clip_command,shell=True)
            dsm_file = dsm_file.replace('.tif','_clipped.tif')
        if len(gdf_overlap) > 1:
            shutil.move(output_dir,output_dir.replace(project_name,f'{project_name}_{i}'))


    ##################
    # To do:
    # - Use flags to modify config file. Start with simple, i.e. no --full flag on cars-starter
    ##################


    ##################
    # Needs flags like a priori data
    # Do I need an a priori geoid as well when I toggle a priori dem?
    # config file structure for variables:
    # a priori dem : config_json['inputs']['initial_elevation']['dem']
    # a priori geoid : config_json['inputs']['initial_elevation']['geoid']  #necessary??
    # resolution : config_json['output']['resolution']
    # Bulldozing : 
    #   config_json['applications']['dsm_filling']['method'] = 'bulldozer'
    #   config_json['applications']['dsm_filling']['activated'] = True
    #   config_json['applications']['dsm_filling']['save_intermediate_data'] = False  #or True to hang on to DSM? Does it delete DSM??
    ##################

    ##################
    # Test cases:
    # - Compare to SETSM (both lsf and unsmoothed) of same date
    # - Include Copernicus as a priori DEM
    #   - With and without geoid
    # - Run CARS with two (or more) separate images (e.g. two nadir on two different dates, not on same date)
    ##################


    # subprocess.run(f'cars {cars_config_file}')

if __name__ == '__main__':
    main()