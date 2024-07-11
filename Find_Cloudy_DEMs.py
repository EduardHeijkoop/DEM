import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import subprocess
import argparse
import configparser
import warnings
import os
import sys
from osgeo import gdal,gdalconst,osr
import multiprocessing
import itertools
import datetime
import getpass
from scipy import ndimage

from Global_DEMs import download_srtm,download_aster,download_copernicus
from dem_utils import get_strip_list,get_list_extents,get_strip_extents,resample_raster,get_strip_shp,raster_to_geotiff

def find_cloudy_DEMs(strip,cloud_water_dict):
    #Read from dictionary
    a_priori_dem = cloud_water_dict['a_priori_dem']
    a_priori_filename = cloud_water_dict['a_priori_filename']
    coastline_file = cloud_water_dict['coastline_file']
    tmp_dir = cloud_water_dict['tmp_dir']
    quiet_flag = cloud_water_dict['quiet_flag']
    keep_diff_flag = cloud_water_dict['keep_diff_flag']
    diff_threshold = cloud_water_dict['diff_threshold']
    intermediate_res = cloud_water_dict['intermediate_res']
    remove_flag = cloud_water_dict['remove_flag']
    N_dilations = cloud_water_dict['N_dilations']
    strip_name = os.path.splitext(os.path.basename(strip))[0]
    # if quiet_flag == False:
    #     print(f'Working on {strip_name}...')
    warp_comp_bigtiff = '-co "COMPRESS=LZW" -co "BIGTIFF=IF_SAFER"'
    calc_comp_bigtiff = '--co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER"'
    #Define new names:
    a_priori_subset = f'{os.path.splitext(a_priori_filename)[0]}_Subset_{strip_name}{os.path.splitext(a_priori_filename)[1]}'
    a_priori_clipped = f'{os.path.splitext(a_priori_filename)[0]}_Subset_{strip_name}_Clipped{os.path.splitext(a_priori_filename)[1]}'
    strip_resampled = f'{tmp_dir}{strip_name}_Resampled_to_{a_priori_dem}{os.path.splitext(strip)[1]}'
    strip_resampled_intermediate = f'{tmp_dir}{strip_name}_Resampled_to_{intermediate_res}m{os.path.splitext(strip)[1]}'
    strip_resampled_intermediate_nodata_removed = f'{tmp_dir}{strip_name}_Resampled_to_{intermediate_res}m_NoData_Removed{os.path.splitext(strip)[1]}'
    strip_resampled_intermediate_4326 = f'{tmp_dir}{strip_name}_Resampled_to_{intermediate_res}m_4326{os.path.splitext(strip)[1]}'
    strip_ones = f'{tmp_dir}{strip_name}_Ones{os.path.splitext(strip)[1]}'
    strip_ones_clipped = f'{tmp_dir}{strip_name}_Ones_Clipped{os.path.splitext(strip)[1]}'
    strip_outline = f'{tmp_dir}{strip_name}_Outline.shp'
    strip_resampled_clipped = f'{tmp_dir}{strip_name}_Resampled_to_{a_priori_dem}_Clipped{os.path.splitext(strip)[1]}'
    strip_cloud_removed = f'{os.path.splitext(strip)[0]}_Clouds_Removed{os.path.splitext(strip)[1]}'
    diff_file = f'{os.path.splitext(a_priori_filename)[0]}_Minus_{strip_name}{os.path.splitext(strip)[1]}'
    diff_file_threshold = f'{os.path.splitext(a_priori_filename)[0]}_Minus_{strip_name}_Threshold_{diff_threshold}m{os.path.splitext(strip)[1]}'
    diff_file_threshold_buffered = f'{os.path.splitext(a_priori_filename)[0]}_Minus_{strip_name}_Threshold_{diff_threshold}m_Buffered{os.path.splitext(strip)[1]}'
    diff_file_threshold_buffered_upsampled = f'{os.path.splitext(a_priori_filename)[0]}_Minus_{strip_name}_Threshold_{diff_threshold}m_Buffered_Upsampled{os.path.splitext(strip)[1]}'
    delete_list = [a_priori_subset,a_priori_clipped,strip_resampled,strip_resampled_intermediate,
                    strip_resampled_intermediate_4326,strip_resampled_clipped,strip_outline.replace('.shp','.*'),
                    strip_ones,strip_ones_clipped,strip_resampled_intermediate_nodata_removed]
    if keep_diff_flag == False:
        delete_list.append(diff_file)
    if remove_flag == True:
        delete_list.extend([diff_file_threshold,diff_file_threshold_buffered,diff_file_threshold_buffered_upsampled])
    #Subset a priori DEM to strip extents:
    lon_min_strip,lon_max_strip,lat_min_strip,lat_max_strip = get_strip_extents(strip,round_flag=True,N_round=4)
    a_priori_subset_command = f'gdalwarp -q -te {lon_min_strip} {lat_min_strip} {lon_max_strip} {lat_max_strip} {warp_comp_bigtiff} {a_priori_filename} {a_priori_subset}'
    subprocess.run(a_priori_subset_command,shell=True)
    #Resample strip to intermediate resolution, then warp to EPSG:4326, then resample to match a priori DEM:
    epsg_code = osr.SpatialReference(wkt=gdal.Open(strip,gdalconst.GA_ReadOnly).GetProjection()).GetAttrValue('AUTHORITY',1)
    intermediate_resample_command = f'gdalwarp -q -tr {intermediate_res} {intermediate_res} -r bilinear -of GTiff {warp_comp_bigtiff} {strip} {strip_resampled_intermediate}'
    subprocess.run(intermediate_resample_command,shell=True)
    #Remove NoData values from intermediate resampled strip, but only on perimeter, not interior pixels:
    src_strip_resampled_intermediate = gdal.Open(strip_resampled_intermediate,gdalconst.GA_ReadOnly)
    ones_array = np.ones((src_strip_resampled_intermediate.RasterYSize,src_strip_resampled_intermediate.RasterXSize))
    x_ones = np.linspace(src_strip_resampled_intermediate.GetGeoTransform()[0] + 0.5*src_strip_resampled_intermediate.GetGeoTransform()[1],
                            src_strip_resampled_intermediate.GetGeoTransform()[0] - 0.5*src_strip_resampled_intermediate.GetGeoTransform()[1] + src_strip_resampled_intermediate.GetGeoTransform()[1]*src_strip_resampled_intermediate.RasterXSize,
                            src_strip_resampled_intermediate.RasterXSize)
    y_ones = np.linspace(src_strip_resampled_intermediate.GetGeoTransform()[3] + 0.5*src_strip_resampled_intermediate.GetGeoTransform()[5],
                            src_strip_resampled_intermediate.GetGeoTransform()[3] - 0.5*src_strip_resampled_intermediate.GetGeoTransform()[5] + src_strip_resampled_intermediate.GetGeoTransform()[5]*src_strip_resampled_intermediate.RasterYSize,
                            src_strip_resampled_intermediate.RasterYSize)
    y_ones = np.flip(y_ones)
    raster_to_geotiff(x_ones,y_ones,ones_array,epsg_code,strip_ones)
    gdf_strip = get_strip_shp(strip_resampled_intermediate,tmp_dir)
    if len(gdf_strip) == 0:
        if quiet_flag == False:
            print(f'{strip_name}: Cannot compute outline!')
        return np.nan,np.nan
    mp_unary_union = shapely.ops.unary_union([shapely.geometry.Polygon(g) for g in gdf_strip.geometry.exterior])
    gdf_strip_filtered = gpd.GeoDataFrame(pd.DataFrame({'strip':[strip]}),geometry=[mp_unary_union],crs=f'EPSG:{epsg_code}')
    gdf_strip_filtered.to_file(strip_outline)
    ones_clip_command = f'gdalwarp -q -cutline {strip_outline} -cl {os.path.splitext(os.path.basename(strip_outline))[0]} -dstnodata 0 {strip_ones} {strip_ones_clipped}'
    unset_nodata_command = f'gdal_edit.py -unsetnodata {strip_resampled_intermediate}'
    multiply_command = f'gdal_calc.py --quiet -A {strip_resampled_intermediate} -B {strip_ones_clipped} --outfile={strip_resampled_intermediate_nodata_removed} --calc="A*B" {calc_comp_bigtiff}'
    subprocess.run(ones_clip_command,shell=True)
    subprocess.run(unset_nodata_command,shell=True)
    subprocess.run(multiply_command,shell=True)
    warp_4326_command = f'gdalwarp -q -s_srs EPSG:{epsg_code} -t_srs EPSG:4326 {warp_comp_bigtiff} {strip_resampled_intermediate_nodata_removed} {strip_resampled_intermediate_4326}'
    edit_command = f'gdal_edit.py -a_nodata 0 {strip_resampled}'
    subprocess.run(warp_4326_command,shell=True)
    resample_raster(strip_resampled_intermediate_4326,a_priori_subset,strip_resampled,resample_method='bilinear',compress=True,nodata=0,quiet_flag=True)
    subprocess.run(edit_command,shell=True)
    #Clip a priori and strip to coastline:
    a_priori_coastline_clip_command = f'gdalwarp -q -s_srs EPSG:4326 -t_srs EPSG:4326 -of GTiff -cutline {coastline_file} -cl {os.path.splitext(os.path.basename(coastline_file))[0]} -dstnodata 0 {a_priori_subset} {a_priori_clipped} {warp_comp_bigtiff}'
    strip_coastline_clip_command = f'gdalwarp -q -s_srs EPSG:4326 -t_srs EPSG:4326 -of GTiff -cutline {coastline_file} -cl {os.path.splitext(os.path.basename(coastline_file))[0]} -dstnodata 0 {strip_resampled} {strip_resampled_clipped} {warp_comp_bigtiff}'
    subprocess.run(a_priori_coastline_clip_command,shell=True)
    subprocess.run(strip_coastline_clip_command,shell=True)
    #Subtract a priori from strip:
    calc_command = f'gdal_calc.py --quiet -A {a_priori_clipped} -B {strip_resampled_clipped} --calc="A-B" --outfile={diff_file} --NoDataValue=0 {calc_comp_bigtiff}'
    subprocess.run(calc_command,shell=True)
    #Optionally, remove clouds from strip:
    if remove_flag == True:
        #Compute threshold of -diff_threshold because clouds are anomalously high so will appear as negative in a_priori - strip:
        diff_threshold_command = f'gdal_calc.py --quiet -A {diff_file} --outfile={diff_file_threshold} --calc="A < -{diff_threshold}" --NoDataValue=0 {calc_comp_bigtiff}'
        subprocess.run(diff_threshold_command,shell=True)
        #Dilate clouds to make sure to fully cover them:
        #Reshape 1s to 0s and 0s to 1s so mask is 1s where clouds are *not*:
        src_diff_threshold = gdal.Open(diff_file_threshold,gdalconst.GA_ReadOnly)
        diff_threshold_array = np.asarray(src_diff_threshold.GetRasterBand(1).ReadAsArray()).astype(int)
        dilation_structure = ndimage.generate_binary_structure(2,2)
        diff_threshold_array_dilated = ndimage.binary_dilation(diff_threshold_array,structure=dilation_structure,iterations=N_dilations).astype(int)
        diff_threshold_array_dilated = 1 - diff_threshold_array_dilated
        #Write to file:
        geotransform_diff_threshold = src_diff_threshold.GetGeoTransform()
        proj_diff_threshold = src_diff_threshold.GetProjection()
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(diff_file_threshold_buffered,diff_threshold_array_dilated.shape[1],diff_threshold_array_dilated.shape[0],1,gdal.GDT_UInt16)
        dataset.SetGeoTransform(geotransform_diff_threshold)
        dataset.SetProjection(proj_diff_threshold)
        dataset.GetRasterBand(1).WriteArray(diff_threshold_array_dilated)
        dataset.GetRasterBand(1).SetNoDataValue(0)
        dataset.FlushCache()
        dataset = None
        #Upsample cloud mask to strip resolution (uses nearest neighbor for speed):
        resample_raster(diff_file_threshold_buffered,strip,diff_file_threshold_buffered_upsampled,resample_method='nearest',compress=True,nodata=None,quiet_flag=True,dtype='int')
        #Mask strip with dilated cloud mask:
        strip_cloud_mask_command = f'gdal_calc.py --quiet -A {strip} -B {diff_file_threshold_buffered_upsampled} --outfile={strip_cloud_removed} --calc="A*B" --NoDataValue=-9999 {calc_comp_bigtiff}'
        # set_nodata_command = f'gdal_calc.py --overwrite --quiet -A {strip_cloud_removed} --outfile={strip_cloud_removed} --calc="A(A == 0)-9999" --NoDataValue=-9999 {calc_comp_bigtiff}'
        set_nodata_command = f'gdal_calc.py --overwrite --quiet -A {strip_cloud_removed} --outfile={tmp_dir}tmp_cloudfree.tif --calc="A - 9999*(A==0)" --NoDataValue=-9999 {calc_comp_bigtiff}'
        subprocess.run(strip_cloud_mask_command,shell=True)
        subprocess.run(set_nodata_command,shell=True)

    #Load difference file and calculate statistics:
    src_diff = gdal.Open(diff_file,gdalconst.GA_ReadOnly)
    diff_array = np.asarray(src_diff.GetRasterBand(1).ReadAsArray())
    diff_array[diff_array == 0] = np.nan
    if np.sum(np.isnan(diff_array)) / (diff_array.shape[0]*diff_array.shape[1]) > 0.95:
        pct_exceedance = 1.0
        pct_water = 1.0
    else:
        pct_exceedance = np.sum(np.abs(diff_array) > diff_threshold) / np.sum(~np.isnan(diff_array))
        src_unclipped = gdal.Open(strip_resampled,gdalconst.GA_ReadOnly)
        src_clipped = gdal.Open(strip_resampled_clipped,gdalconst.GA_ReadOnly)
        unclipped_array = np.asarray(src_unclipped.GetRasterBand(1).ReadAsArray())
        clipped_array = np.asarray(src_clipped.GetRasterBand(1).ReadAsArray())
        unclipped_array[unclipped_array == 0] = np.nan
        clipped_array[clipped_array == 0] = np.nan
        pct_nan_unclipped = np.sum(np.isnan(unclipped_array)) / (unclipped_array.shape[0] * unclipped_array.shape[1])
        pct_nan_clipped = np.sum(np.isnan(clipped_array)) / (clipped_array.shape[0] * clipped_array.shape[1])
        pct_water = pct_nan_clipped - pct_nan_unclipped
    subprocess.run(f'rm {" ".join(delete_list)}',shell=True)
    if quiet_flag == False:
        print(f'{strip_name}: {100*pct_exceedance:.1f}% cloudy, {100*pct_water:.1f}% over water.')
    return pct_exceedance,pct_water

def single_str(v,s):
    if v == 1:
        s = s.replace('days','day').replace('hours','hour').replace('minutes','minute').replace('seconds','second')
    return s

def main():
    warnings.simplefilter(action='ignore')
    config_file = 'dem_config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',default=None,help='Path to input file to run in batch mode.')
    parser.add_argument('--input_dir',default=None,help='Path to directory containing strips.')
    parser.add_argument('--list',default=None,help='Path to list of strips to detect clouds in.')
    parser.add_argument('--loc_name',default=None,help='Location name.')
    parser.add_argument('--machine',default='t',help='Machine to run on.',choices=['t','b','local'])
    parser.add_argument('--dir_structure',default='sealevel',help='Directory structure of input strips',choices=['sealevel','simple'])
    parser.add_argument('--a_priori',default='copernicus',help='A priori DEM to use.',choices=['srtm','aster','copernicus'])
    parser.add_argument('--coastline',default=config.get('GENERAL_PATHS','osm_shp_file'),help='Path to coastline shapefile')
    parser.add_argument('--vertical_threshold',default=50,type=float,help='Vertical threshold for exceedance.')
    parser.add_argument('--N_cpus',help='Number of CPUs to use.',default=1,type=int)
    parser.add_argument('--keep_diff',default=False,help='Keep DEM differences?',action='store_true')
    parser.add_argument('--remove_clouds',default=False,help='Remove cloudy parts of DEMs?',action='store_true')
    parser.add_argument('--N_dilations',default=2,type=int,help='Number of dilation iterations when removing clouds.')
    parser.add_argument('--quiet',default=False,help='Suppress output.',action='store_true')

    args = parser.parse_args()
    input_file = args.input_file
    input_dir = args.input_dir
    list_file = args.list
    loc_name = args.loc_name
    a_priori_dem = args.a_priori
    coastline_file = args.coastline
    diff_threshold = args.vertical_threshold
    machine_name = args.machine
    dir_structure = args.dir_structure
    N_cpus = args.N_cpus
    keep_diff_flag = args.keep_diff
    remove_flag = args.remove_clouds
    N_dilations = args.N_dilations
    quiet_flag = args.quiet

    if input_dir is None and list_file is None and input_file is None:
        raise ValueError('Must specify either input_dir, list_file or input_file.')
    if np.sum((list_file is not None,input_file is not None,input_dir is not None)) > 1:
        raise ValueError('Cannot specify more than one of input_dir, list_file or input_file.')
    
    EGM96_file = config.get('GENERAL_PATHS','EGM96_path')
    EGM2008_file = config.get('GENERAL_PATHS','EGM2008_path')
    tmp_dir = config.get('GENERAL_PATHS','tmp_dir')
    default_coastline = config.get('GENERAL_PATHS','osm_shp_file')
    intermediate_res = 10
    if np.logical_or(a_priori_dem == 'srtm',a_priori_dem == 'aster'):
        username = config.get('GENERAL_CONSTANTS','earthdata_username')
        pw = getpass.getpass()

    if machine_name == 'b':
        EGM96_file = EGM96_file.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
        EGM2008_file = EGM2008_file.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
        tmp_dir = tmp_dir.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
        coastline_file = coastline_file.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
        default_coastline = default_coastline.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
    elif machine_name == 'local':
        EGM96_file = EGM96_file.replace('/BhaltosMount/Bhaltos/EDUARD/DATA_REPOSITORY/','/media/heijkoop/DATA/')
        EGM2008_file = EGM2008_file.replace('/BhaltosMount/Bhaltos/EDUARD/DATA_REPOSITORY/','/media/heijkoop/DATA/')
        tmp_dir = tmp_dir.replace('/BhaltosMount/Bhaltos/','/home/heijkoop/Desktop/Projects/tmp/')
        coastline_file = coastline_file.replace('/BhaltosMount/Bhaltos/EDUARD/DATA_REPOSITORY/','/media/heijkoop/DATA/')
        default_coastline = default_coastline.replace('/BhaltosMount/Bhaltos/EDUARD/DATA_REPOSITORY/','/media/heijkoop/DATA/')
    
    if input_file is not None:
        df_input = pd.read_csv(input_file)
        input_dir_array = np.asarray([f'{o}/' if o[-1] != '/' else o for o in df_input.Location])
        loc_name_array = np.asarray([o.split('/')[-2] for o in input_dir_array])
        output_file_array = np.asarray([f'{o}{loc}_Threshold_Exceedance_Values.txt' for o,loc in zip(input_dir_array,loc_name_array)])
        coastline_array = np.asarray(df_input.Coastline)
    elif list_file is not None:
        if loc_name is None:
            loc_name_array = np.atleast_1d(os.path.splitext(os.path.basename(list_file))[0])
        else:
            loc_name_array = np.atleast_1d(loc_name)
        output_file_array = np.atleast_1d(list_file.replace(os.path.splitext(list_file)[1],f'_Threshold_Exceedance_Values{os.path.splitext(list_file)[1]}'))
        coastline_array = np.atleast_1d(coastline_file)
    elif input_dir is not None:
        if input_dir[-1] != '/':
            input_dir += '/'
        input_dir_array = np.atleast_1d(input_dir)
        if loc_name is None:
            loc_name_array = np.atleast_1d(os.path.basename(os.path.dirname(input_dir)))
        else:
            loc_name_array = np.atleast_1d(loc_name)
        output_file_array = np.atleast_1d(f'{input_dir}{loc_name}_Threshold_Exceedance_Values.txt')
        coastline_array = np.atleast_1d(coastline_file)
    
    if input_file is not None:
        error_log_file = input_file.replace(os.path.splitext(input_file)[1],f'_error_log.txt')
    elif list_file is not None:
        error_log_file = list_file.replace(os.path.splitext(list_file)[1],f'_error_log.txt')
    else:
        error_log_file = None

    t_start_full = datetime.datetime.now()
    for input_dir,output_file,loc_name,coast in zip(input_dir_array,output_file_array,loc_name_array,coastline_array):
        t_start_loc = datetime.datetime.now()
        print(f'Working on {loc_name}.')
        if list_file is not None:
            df_list = pd.read_csv(list_file,header=None,names=['strip'],dtype={'strip':'str'})
            strip_list = np.asarray(df_list.strip)
        else:
            strip_list = get_strip_list(input_dir,input_type=0,corrected_flag=False,dir_structure=dir_structure)
        if len(strip_list) == 0:
            print('No strips found. Skipping!\n')
            continue
        #Remove duplicate strips from consideration.
        strip_filenames = np.asarray([s.split('/')[-1].split('_dem')[0] for s in strip_list])
        if len(np.unique(strip_filenames)) != len(strip_filenames):
            print('Duplicate strip names found. Only doing unique ones.')
            if error_log_file is not None:
                with open(error_log_file,'a') as f:
                    f.write(f'{loc_name}: Duplicate strip names found. Only doing unique ones.\n')
            unique_strips = np.unique(strip_filenames)
            keep_list = np.empty([0,1],dtype=int)
            discard_list = np.empty([0,1],dtype=int)
            for us in unique_strips:
                strip_idx = np.where(strip_filenames == us)[0]
                if len(strip_idx) == 1:
                    keep_list = np.append(keep_list,strip_idx)
                elif len(strip_idx) == 2:
                    find_utm = np.asarray(['/UTM' in s for s in strip_list[strip_idx]])
                    if len(find_utm) - np.sum(find_utm) == 1:
                        discard_list = np.append(discard_list,strip_idx[np.argwhere(find_utm).squeeze()])
                        keep_list = np.append(keep_list,strip_idx[np.argwhere(~find_utm).squeeze()])
                    else:
                        discard_list = np.append(discard_list,strip_idx[0])
                        keep_list = np.append(keep_list,strip_idx[1])
                else:
                    keep_list = np.append(keep_list,strip_idx[0])
                    discard_list = np.append(discard_list,strip_idx[1:])
            strip_list = strip_list[keep_list]
        a_priori_filename = f'{tmp_dir}{loc_name}_{a_priori_dem}_WGS84.tif'
        lon_min,lon_max,lat_min,lat_max = get_list_extents(strip_list)

        if np.logical_or(coast == default_coastline,pd.isnull(coast)):
            print('Using default coastline file. Clipping it to DEM extents...')
            new_coastline_file = f'{tmp_dir}{loc_name}_coastline.shp'
            coastline_clip_command = f'ogr2ogr {new_coastline_file} {coastline_file} -clipsrc {lon_min} {lat_min} {lon_max} {lat_max}'
            subprocess.run(coastline_clip_command,shell=True)
            coast = new_coastline_file

        print(f'Downloading {a_priori_dem}...')
        if a_priori_dem == 'srtm':
            download_srtm(lon_min,lon_max,lat_min,lat_max,username,pw,EGM96_file,tmp_dir,a_priori_filename)
        elif a_priori_dem == 'aster':
            download_aster(lon_min,lon_max,lat_min,lat_max,username,pw,EGM96_file,tmp_dir,a_priori_filename)
        elif a_priori_dem == 'copernicus':
            download_copernicus(lon_min,lon_max,lat_min,lat_max,EGM2008_file,tmp_dir,a_priori_filename)
        print('Download complete.')

        cloud_water_dict = {
            'a_priori_dem':a_priori_dem,
            'a_priori_filename':a_priori_filename,
            'coastline_file':coast,
            'tmp_dir':tmp_dir,
            'quiet_flag':quiet_flag,
            'keep_diff_flag':keep_diff_flag,
            'diff_threshold':diff_threshold,
            'intermediate_res':intermediate_res,
            'remove_flag':remove_flag,
            'N_dilations':N_dilations
        }

        ir = itertools.repeat
        p = multiprocessing.Pool(np.min((N_cpus,len(strip_list))))
        pct_exceedance_water = p.starmap(find_cloudy_DEMs,zip(
            strip_list,
            ir(cloud_water_dict)
            ))
        p.close()
        pct_exceedance = np.asarray([pct[0] for pct in pct_exceedance_water])
        pct_water = np.asarray([pct[1] for pct in pct_exceedance_water])
        idx_nan = np.isnan(pct_exceedance)
        pct_exceedance = pct_exceedance[~idx_nan]
        pct_water = pct_water[~idx_nan]
        strip_list = strip_list[~idx_nan]
        subprocess.run(f'rm {a_priori_filename}',shell=True)
        if tmp_dir in coast:
            subprocess.run(f'rm {coast.replace(os.path.splitext(coast)[1],".*")}',shell=True)
        np.savetxt(output_file,np.c_[strip_list.astype(object),pct_exceedance,pct_water],fmt='%s,%.3f,%.3f',header='Strip,Percent Exceedance,Percent Water',comments='',delimiter=',')
        t_end_loc = datetime.datetime.now()
        dt_loc = (t_end_loc - t_start_loc).seconds
        minutes_loc,seconds_loc = divmod(dt_loc,60)
        print(f'Finished {loc_name} in {minutes_loc} {single_str(minutes_loc,"minutes")}, {seconds_loc} {single_str(seconds_loc,"seconds")}.')
        print('\n')
    t_end_full = datetime.datetime.now()
    dt_full = (t_end_full - t_start_full).seconds
    hours_full,remainder_full = divmod(dt_full,3600)
    minutes_full,seconds_full = divmod(remainder_full,60)
    print(f'Finished all locations in {hours_full} {single_str(hours_full,"hours")}, {minutes_full} {single_str(minutes_full,"minutes")}, {seconds_full} {single_str(seconds_full,"seconds")}.')

if __name__ == "__main__":
    main()