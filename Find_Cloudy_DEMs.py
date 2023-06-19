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
    diff_file = f'{os.path.splitext(a_priori_filename)[0]}_Minus_{strip_name}{os.path.splitext(strip)[1]}'
    delete_list = [a_priori_subset,a_priori_clipped,strip_resampled,strip_resampled_intermediate,
                    strip_resampled_intermediate_4326,strip_resampled_clipped,strip_outline.replace('.shp','.*'),
                    strip_ones,strip_ones_clipped,strip_resampled_intermediate_nodata_removed]
    if keep_diff_flag == False:
        delete_list.append(diff_file)
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
    #Load difference file and calculate statistics:
    src_diff = gdal.Open(diff_file,gdalconst.GA_ReadOnly)
    diff_array = np.asarray(src_diff.GetRasterBand(1).ReadAsArray())
    diff_array[diff_array == 0] = np.nan
    if np.sum(np.isnan(diff_array)) / (diff_array.shape[0]*diff_array.shape[1]) > 0.95:
        pct_exceedance = 1.0
        pct_water = 1.0
        # if quiet_flag == False:
            # print(f'{strip_name} is likely entirely over water. Setting both values to 100%.')
    else:
        pct_exceedance = np.sum(np.abs(diff_array) > diff_threshold) / np.sum(~np.isnan(diff_array))
    # if quiet_flag == False:
        # print(f'{pct_exceedance[i]*100:.2f}% of pixels exceed {diff_threshold}m.')
        # if pct_exceedance[i] > exceedance_threshold:
            # print(f'{strip_name} exceeds difference threshold of {exceedance_threshold*100:.2f}%!')
        src_unclipped = gdal.Open(strip_resampled,gdalconst.GA_ReadOnly)
        src_clipped = gdal.Open(strip_resampled_clipped,gdalconst.GA_ReadOnly)
        unclipped_array = np.asarray(src_unclipped.GetRasterBand(1).ReadAsArray())
        clipped_array = np.asarray(src_clipped.GetRasterBand(1).ReadAsArray())
        unclipped_array[unclipped_array == 0] = np.nan
        clipped_array[clipped_array == 0] = np.nan
        pct_nan_unclipped = np.sum(np.isnan(unclipped_array)) / (unclipped_array.shape[0] * unclipped_array.shape[1])
        pct_nan_clipped = np.sum(np.isnan(clipped_array)) / (clipped_array.shape[0] * clipped_array.shape[1])
        pct_water = pct_nan_clipped - pct_nan_unclipped
    # if quiet_flag == False:
    #     print(f'{100*pct_water:.1f}% over water.')
    #     if pct_water > water_threshold:
    #         print(f'{strip_name} exceeds water threshold of {water_threshold*100:.1f}%!')
    subprocess.run(f'rm {" ".join(delete_list)}',shell=True)
    if quiet_flag == False:
        print(f'{strip_name}: {100*pct_exceedance:.1f}% over clouds, {100*pct_water:.1f}% over water.')
    return pct_exceedance,pct_water

def main():
    warnings.simplefilter(action='ignore')
    config_file = 'dem_config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',default=None,help='Path to dir containing strips.')
    parser.add_argument('--list',default=None,help='Path to list of strips to mosaic.')
    parser.add_argument('--loc_name',default=None,help='Location name.')
    parser.add_argument('--batch',default=None,help='Path to file to run in batch mode.')
    parser.add_argument('--a_priori',default='copernicus',help='A priori DEM to use.',choices=['srtm','aster','copernicus'])
    parser.add_argument('--coastline',default=config.get('GENERAL_PATHS','osm_shp_file'),help='Path to coastline shapefile')
    # parser.add_argument('--pct_threshold',default=config.getfloat('MOSAIC_CONSTANTS','STRIP_CLOUD_THRESHOLD'),type=float,help='Threshold exceedance value.')
    parser.add_argument('--vertical_threshold',default=50,type=float,help='Vertical threshold for exceedance.')
    parser.add_argument('--machine',default='t',help='Machine to run on.',choices=['t','b','local'])
    parser.add_argument('--dir_structure',default='sealevel',help='Directory structure of input strips',choices=['sealevel','simple'])
    parser.add_argument('--cpus',help='Number of CPUs to use.',default=1,type=int)
    parser.add_argument('--keep_diff',default=False,help='Keep DEM differences.',action='store_true')
    # parser.add_argument('--water_threshold',default=config.getfloat('MOSAIC_CONSTANTS','STRIP_WATER_THRESHOLD'),type=float,help='Water percentage threshold.')
    parser.add_argument('--quiet',default=False,help='Suppress output.',action='store_true')

    args = parser.parse_args()
    input_dir = args.input_dir
    list_file = args.list
    loc_name = args.loc_name
    batch_file = args.batch
    a_priori_dem = args.a_priori
    coastline_file = args.coastline
    # exceedance_threshold = args.pct_threshold
    diff_threshold = args.vertical_threshold
    machine_name = args.machine
    dir_structure = args.dir_structure
    N_cpus = args.cpus
    keep_diff_flag = args.keep_diff
    # water_threshold = args.water_threshold
    quiet_flag = args.quiet

    # gdal.SetConfigOption('GDAL_NUM_THREADS', f'{N_cpus}')

    if input_dir is None and list_file is None and batch_file is None:
        raise ValueError('Must specify either input_dir, list_file or batch file.')
    if input_dir is not None and list_file is not None:
        raise ValueError('Cannot specify both input_dir and list_file.')
    
    if list_file is not None:
        df_list = pd.read_csv(list_file,header=None,names=['strip'],dtype={'strip':'str'})
        strip_list = np.asarray(df_list.strip)
        if loc_name is None:
            loc_name = os.path.splitext(os.path.basename(list_file))[0]
        output_file = list_file.replace(os.path.splitext(list_file)[1],f'_Threshold_Exceedance_Values{os.path.splitext(list_file)[1]}')
    elif input_dir is not None:
        if input_dir[-1] != '/':
            input_dir += '/'
        strip_list = get_strip_list(input_dir,input_type=0,corrected_flag=False,dir_structure=dir_structure)
        if loc_name is None:
            loc_name = os.path.basename(os.path.dirname(input_dir))
        output_file = f'{input_dir}{loc_name}_Threshold_Exceedance_Values.txt'
    
    EGM96_file = config.get('GENERAL_PATHS','EGM96_path')
    EGM2008_file = config.get('GENERAL_PATHS','EGM2008_path')
    tmp_dir = config.get('GENERAL_PATHS','tmp_dir')
    default_coastline = config.get('GENERAL_PATHS','osm_shp_file')
    intermediate_res = 10
    # pct_exceedance = np.zeros(len(strip_list))
    # pct_water = np.zeros(len(strip_list))

    a_priori_filename = f'{tmp_dir}{loc_name}_{a_priori_dem}_WGS84.tif'
    lon_min,lon_max,lat_min,lat_max = get_list_extents(strip_list)

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
    
    if coastline_file == default_coastline:
        print('Using default coastline file. Clipping it to DEM extents...')
        new_coastline_file = f'{tmp_dir}{loc_name}_coastline.shp'
        coastline_clip_command = f'ogr2ogr {new_coastline_file} {coastline_file} -clipsrc {lon_min} {lat_min} {lon_max} {lat_max}'
        subprocess.run(coastline_clip_command,shell=True)
        coastline_file = new_coastline_file

    # gdf_coast = gpd.read_file(coastline_file)
    # epsg_list = [osr.SpatialReference(wkt=gdal.Open(s,gdalconst.GA_ReadOnly).GetProjection()).GetAttrValue('AUTHORITY',1) for s in strip_list]
    # if len(np.unique(epsg_list)) == 1:
    #     gdf_coast_epsg = gdf_coast.to_crs(f'EPSG:{epsg_list[0]}')
    # else:
    #     gdf_coast_epsg = None

    print(f'Working on {loc_name}.')
    print(f'Downloading {a_priori_dem}...')
    if a_priori_dem == 'srtm':
        download_srtm(lon_min,lon_max,lat_min,lat_max,EGM96_file,tmp_dir,a_priori_filename)
    elif a_priori_dem == 'aster':
        download_aster(lon_min,lon_max,lat_min,lat_max,EGM96_file,tmp_dir,a_priori_filename)
    elif a_priori_dem == 'copernicus':
        download_copernicus(lon_min,lon_max,lat_min,lat_max,EGM2008_file,tmp_dir,a_priori_filename)
    print('Download complete.')

    cloud_water_dict = {
        'a_priori_dem':a_priori_dem,
        'a_priori_filename':a_priori_filename,
        'coastline_file':coastline_file,
        'tmp_dir':tmp_dir,
        'quiet_flag':quiet_flag,
        'keep_diff_flag':keep_diff_flag,
        'diff_threshold':diff_threshold,
        'intermediate_res':intermediate_res,
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

    subprocess.run(f'rm {a_priori_filename}',shell=True)
    if tmp_dir in coastline_file:
        subprocess.run(f'rm {coastline_file.replace(os.path.splitext(coastline_file)[1],".*")}',shell=True)
    
    np.savetxt(output_file,np.c_[strip_list.astype(object),pct_exceedance,pct_water],fmt='%s,%.3f',header='Strip,Percent Exceedance,Percent Water',comments='',delimiter=',')

if __name__ == "__main__":
    main()