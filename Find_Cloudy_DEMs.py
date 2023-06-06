import numpy as np
import pandas as pd
import subprocess
import argparse
import configparser
import warnings
import os
import sys
from osgeo import gdal,gdalconst,osr


from Global_DEMs import download_srtm,download_aster,download_copernicus
from dem_utils import get_strip_list,get_list_extents,get_strip_extents,resample_raster



def main():
    warnings.simplefilter(action='ignore')
    config_file = 'dem_config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',default=None,help='Path to dir containing strips.')
    parser.add_argument('--list',default=None,help='path to list of strips to mosaic.')
    parser.add_argument('--loc_name',default=None,help='Location name.')
    parser.add_argument('--a_priori',default='copernicus',help='A priori DEM to use.',choices=['srtm','aster','copernicus'])
    parser.add_argument('--coastline',default=config.get('GENERAL_PATHS','osm_shp_file'),help='Path to coastline shapefile')
    parser.add_argument('--pct_threshold',default=0.05,type=float,help='Threshold exceedance value.')
    parser.add_argument('--vertical_threshold',default=50,type=float,help='Vertical threshold for exceedance.')
    parser.add_argument('--machine',default='t',help='Machine to run on.',choices=['t','b','local'])
    parser.add_argument('--dir_structure',default='sealevel',help='Directory structure of input strips',choices=['sealevel','simple'])
    parser.add_argument('--cpus',help='Number of CPUs to use.',default=1,type=int)
    parser.add_argument('--keep_diff',help='Keep DEM differences.',action='store_true')
    parser.add_argument('--find_water',default=False,help='Find percentage of water in DEM too.',action='store_true')
    parser.add_argument('--water_threshold',default=0.3,type=float,help='Water percentage threshold.')
    parser.add_argument('--quiet',default=False,help='Suppress output.',action='store_true')

    args = parser.parse_args()
    input_dir = args.input_dir
    list_file = args.list
    loc_name = args.loc_name
    a_priori_dem = args.a_priori
    coastline_file = args.coastline
    exceedance_threshold = args.pct_threshold
    diff_threshold = args.vertical_threshold
    machine_name = args.machine
    dir_structure = args.dir_structure
    N_cpus = args.cpus
    keep_diff_flag = args.keep_diff
    find_water_flag = args.find_water
    water_threshold = args.water_threshold
    quiet_flag = args.quiet

    gdal.SetConfigOption('GDAL_NUM_THREADS', f'{N_cpus}')

    if input_dir is None and list_file is None:
        raise ValueError('Must specify either input_dir or list_file')
    if input_dir is not None and list_file is not None:
        raise ValueError('Cannot specify both input_dir and list_file')
    
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
    pct_exceedance = np.zeros(len(strip_list))

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

    warp_comp_bigtiff = '-co "COMPRESS=LZW" -co "BIGTIFF=IF_SAFER"'
    calc_comp_bigtiff = '--co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER"'

    for i,strip in enumerate(strip_list):
        strip_name = os.path.splitext(os.path.basename(strip))[0]
        if quiet_flag == False:
            print(f'Working on {strip_name}...')
        else:
            sys.stdout.write('\r')
            n_progressbar = (i + 1) / len(strip_list)
            sys.stdout.write("[%-20s] %d%%" % ('='*int(20*n_progressbar), 100*n_progressbar))
            sys.stdout.flush()
        #Define new names:
        a_priori_subset = f'{os.path.splitext(a_priori_filename)[0]}_Subset_{strip_name}{os.path.splitext(a_priori_filename)[1]}'
        a_priori_clipped = f'{os.path.splitext(a_priori_filename)[0]}_Subset_{strip_name}_Clipped{os.path.splitext(a_priori_filename)[1]}'
        strip_resampled = f'{tmp_dir}{strip_name}_Resampled_to_{a_priori_dem}{os.path.splitext(strip)[1]}'
        strip_resampled_intermediate = f'{tmp_dir}{strip_name}_Resampled_to_{intermediate_res}m{os.path.splitext(strip)[1]}'
        strip_resampled_intermediate_4326 = f'{tmp_dir}{strip_name}_Resampled_to_{intermediate_res}m_4326{os.path.splitext(strip)[1]}'
        strip_resampled_clipped = f'{tmp_dir}{strip_name}_Resampled_to_{a_priori_dem}_Clipped{os.path.splitext(strip)[1]}'
        diff_file = f'{os.path.splitext(a_priori_filename)[0]}_Minus_{strip_name}{os.path.splitext(strip)[1]}'
        #Subset a priori DEM to strip extents:
        lon_min_strip,lon_max_strip,lat_min_strip,lat_max_strip = get_strip_extents(strip,round_flag=True,N_round=4)
        a_priori_subset_command = f'gdalwarp -q -te {lon_min_strip} {lat_min_strip} {lon_max_strip} {lat_max_strip} {warp_comp_bigtiff} {a_priori_filename} {a_priori_subset}'
        subprocess.run(a_priori_subset_command,shell=True)
        #Resample strip to intermediate resolution, then warp to EPSG:4326, then resample to match a priori DEM:
        epsg_code = osr.SpatialReference(wkt=gdal.Open(strip,gdalconst.GA_ReadOnly).GetProjection()).GetAttrValue('AUTHORITY',1)
        intermediate_resample_command = f'gdalwarp -q -tr {intermediate_res} {intermediate_res} -r bilinear -of GTiff {warp_comp_bigtiff} {strip} {strip_resampled_intermediate}'
        warp_4326_command = f'gdalwarp -q -s_srs EPSG:{epsg_code} -t_srs EPSG:4326 {warp_comp_bigtiff} {strip_resampled_intermediate} {strip_resampled_intermediate_4326}'
        edit_command = f'gdal_edit.py -a_nodata 0 {strip_resampled}'
        subprocess.run(intermediate_resample_command,shell=True)
        subprocess.run(warp_4326_command,shell=True)
        resample_raster(strip_resampled_intermediate_4326,a_priori_subset,strip_resampled,resample_method='bilinear',compress=True,nodata=-9999,quiet_flag=True)
        subprocess.run(edit_command,shell=True)
        #Clip a priori and strip to coastline:
        a_priori_coastline_clip_command = f'gdalwarp -q -s_srs EPSG:4326 -t_srs EPSG:4326 -of GTiff -cutline {coastline_file} -cl {os.path.splitext(os.path.basename(coastline_file))[0]} -dstnodata -9999 {a_priori_subset} {a_priori_clipped} {warp_comp_bigtiff}'
        strip_coastline_clip_command = f'gdalwarp -q -s_srs EPSG:4326 -t_srs EPSG:4326 -of GTiff -cutline {coastline_file} -cl {os.path.splitext(os.path.basename(coastline_file))[0]} -dstnodata -9999 {strip_resampled} {strip_resampled_clipped} {warp_comp_bigtiff}'
        subprocess.run(a_priori_coastline_clip_command,shell=True)
        subprocess.run(strip_coastline_clip_command,shell=True)
        #Subtract a priori from strip:
        calc_command = f'gdal_calc.py --quiet -A {a_priori_clipped} -B {strip_resampled_clipped} --calc="A-B" --outfile={diff_file} --NoDataValue=-9999 {calc_comp_bigtiff}'
        subprocess.run(calc_command,shell=True)
        #Load difference file and calculate statistics:
        src_diff = gdal.Open(diff_file,gdalconst.GA_ReadOnly)
        diff_array = np.asarray(src_diff.GetRasterBand(1).ReadAsArray())
        diff_array[diff_array == -9999] = np.nan
        pct_exceeding = np.sum(np.abs(diff_array) > diff_threshold) / np.sum(~np.isnan(diff_array))
        pct_exceedance[i] = pct_exceeding
        if quiet_flag == False:
            print(f'{pct_exceeding*100:.2f}% of pixels exceed {diff_threshold}m.')
            if pct_exceeding > exceedance_threshold:
                print(f'{strip_name} exceeds difference threshold of {exceedance_threshold*100:.2f}%!')
        if find_water_flag == True:
            src_unclipped = gdal.Open(strip_resampled,gdalconst.GA_ReadOnly)
            src_clipped = gdal.Open(strip_resampled_clipped,gdalconst.GA_ReadOnly)
            unclipped_array = np.asarray(src_unclipped.GetRasterBand(1).ReadAsArray())
            clipped_array = np.asarray(src_clipped.GetRasterBand(1).ReadAsArray())
            unclipped_array[unclipped_array == 0] = np.nan
            clipped_array[clipped_array == -9999] = np.nan
            pct_nan_unclipped = np.sum(np.isnan(unclipped_array)) / (unclipped_array.shape[0] * unclipped_array.shape[1])
            pct_nan_clipped = np.sum(np.isnan(clipped_array)) / (clipped_array.shape[0] * clipped_array.shape[1])
            pct_water = pct_nan_clipped - pct_nan_unclipped
            if quiet_flag == False:
                print(f'{100*pct_water:.1f}% over water.')
                if pct_water > water_threshold:
                    print(f'{strip_name} exceeds water threshold of {water_threshold*100:.1f}%!')
        delete_list = [a_priori_subset,a_priori_clipped,strip_resampled,strip_resampled_intermediate,strip_resampled_intermediate_4326,strip_resampled_clipped]
        if keep_diff_flag == False:
            delete_list.append(diff_file)
        subprocess.run(f'rm {" ".join(delete_list)}',shell=True)
    
    subprocess.run(f'rm {a_priori_filename}',shell=True)
    if tmp_dir in coastline_file:
        subprocess.run(f'rm {coastline_file.replace(os.path.splitext(coastline_file)[1],".*")}',shell=True)
    
    np.savetxt(output_file,np.c_[strip_list.astype(object),pct_exceedance],fmt='%s,%.3f',header='Strip,Percent Exceedance',comments='',delimiter=',')
    if find_water_flag == True:
        np.savetxt(f'{tmp_dir}tmp_water.txt',np.c_[pct_water],fmt='%.3f',header='Percent Water',comments='',delimiter=',')
        subprocess.run(f'paste -d , {output_file} {tmp_dir}tmp_water.txt > {tmp_dir}tmp.txt',shell=True)
        subprocess.run(f'mv {tmp_dir}tmp.txt {output_file}',shell=True)
        subprocess.run(f'rm {tmp_dir}tmp_water.txt',shell=True)

if __name__ == "__main__":
    main()