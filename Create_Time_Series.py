import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import subprocess
import datetime
import sys
import os
import shapely
import warnings
import argparse
warnings.simplefilter(action='ignore')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',help='Path to shapefile containing strips.')
    parser.add_argument('--xy',help='X and Y coordinates of where you want the time series.',nargs=2)
    parser.add_argument('--shp',help='Path to shapefile of geometry of where you want the time series.')
    parser.add_argument('--threshold',help='Vertical threshold (in meters) for the time series.')
    parser.add_argument('--area_threshold',help='Area threshold for polygons.',nargs='*')
    args = parser.parse_args()
    input_file = args.input_file
    xy = args.xy
    shp = args.shp
    threshold = int(args.threshold)
    area_threshold = args.area_threshold
    area_threshold = [int(a) for a in np.atleast_1d(area_threshold)]
    if xy is not None:
        x_input,y_input = xy
        x_input = float(x_input)
        y_input = float(y_input)
        input_point = shapely.geometry.Point(x_input,y_input)
    if shp is not None:
        shp = gpd.read_file(shp)
    
    time_series_dir = f'{"/".join(input_file.split("/")[0:-1])}/Time_Series_Strips/'
    mosaic_dir = f'{"/".join(input_file.split("/")[0:-1])}/Mosaic/'
    if os.path.isdir(time_series_dir) == False:
        os.mkdir(time_series_dir)
    loc_name = input_file.split('/')[-2]

    strip_shp_data = gpd.read_file(input_file)

    file_list = glob.glob(f'{time_series_dir}*align.tif')
    file_list.sort()
    file_list = np.asarray(file_list)
    strip_dates = np.asarray([int(s.split('/')[-1][5:13]) for s in file_list])
    unique_dates = np.unique(strip_dates)
    idx_dates_sorted = np.argsort(strip_dates) #in this case we want old -> new
    strip_dates_sorted = strip_dates[idx_dates_sorted]
    strip_dates_sorted_datetime = np.asarray([datetime.datetime(year=int(s[0:4]),month=int(s[4:6]),day=int(s[6:8])) for s in strip_dates_sorted.astype(str)])
    file_list_sorted = file_list[idx_dates_sorted]

    file_list_sorted_simple = np.asarray([f.split('/')[-1].split('_sampled_mosaic')[0] for f in file_list_sorted])
    strip_list_simple = np.asarray([s.split('/')[-1].split('.tif')[0] for s in strip_shp_data.strip])
    idx_resort = [np.argwhere(f == strip_list_simple)[0][0] for f in file_list_sorted_simple]
    strip_shp_data = strip_shp_data.iloc[idx_resort].reset_index(drop=True)
    #shp will overrule xy point
    if xy is not None:
        idx_contains = [s.contains(input_point) for s in strip_shp_data.geometry]
    if shp is not None:
        idx_contains_shp = np.asarray([s.contains(shp.geometry[0]) for s in strip_shp_data.geometry])
        idx_overlaps_shp = np.asarray([s.intersects(shp.geometry[0]) for s in strip_shp_data.geometry])
        idx_contains = np.any((idx_contains_shp,idx_overlaps_shp),axis=0)
    
    output_file = f'{time_series_dir}Coregistered_Strips.txt'
    np.savetxt(output_file,np.c_[file_list_sorted],fmt='%s')
    subprocess.run(f'gdalbuildvrt -separate -overwrite -input_file_list {output_file} {time_series_dir}{loc_name}_Time_Series.vrt',shell=True)

    idx_old_strips = np.argwhere(idx_contains).squeeze()[:-1]
    idx_new_strips = np.argwhere(idx_contains).squeeze()[1:]

    for i in range(len(idx_old_strips)):
        sys.stdout.write('\r')
        n_progressbar = (i + 1) / len(idx_old_strips)
        sys.stdout.write("[%-20s] %d%%" % ('='*int(20*n_progressbar), 100*n_progressbar))
        sys.stdout.flush()
        old_strip = file_list_sorted[idx_old_strips[i]].split('/')[-1].split('_dem')[0]
        new_strip = file_list_sorted[idx_new_strips[i]].split('/')[-1].split('_dem')[0]
        dt = (strip_dates_sorted_datetime[idx_new_strips[i]] - strip_dates_sorted_datetime[idx_old_strips[i]]).days
        dem_diff_file = f'{time_series_dir}Diff_{new_strip}_minus_{old_strip}_dt_{dt}_days.tif'
        dem_diff_threshold_file = dem_diff_file.replace('.tif',f'_gt_{threshold}m.tif')
        dem_diff_shp_file = dem_diff_threshold_file.replace('.tif','.shp')
        diff_calc_command = f'gdal_calc.py -A {time_series_dir}{loc_name}_Time_Series.vrt --A_band={idx_new_strips[i]+1} -B {time_series_dir}{loc_name}_Time_Series.vrt --B_band={idx_old_strips[i]+1} --outfile={dem_diff_file} --calc="A-B" --format=GTiff --co="COMPRESS=LZW" --quiet'
        threshold_calc_command = f'gdal_calc.py -A {dem_diff_file} --calc="numpy.abs(A)>{threshold}" --outfile={dem_diff_threshold_file} --format=GTiff --co="COMPRESS=LZW" --quiet'
        nodata_translate_command = f'gdal_translate -q -a_nodata 0 {dem_diff_threshold_file} {time_series_dir}tmp.tif'
        polygonize_command = f'gdal_polygonize.py -q {time_series_dir}tmp.tif -f "ESRI Shapefile" {dem_diff_shp_file}'
        subprocess.run(diff_calc_command,shell=True)
        subprocess.run(threshold_calc_command,shell=True)
        subprocess.run(nodata_translate_command,shell=True)
        subprocess.run(polygonize_command,shell=True)
        subprocess.run(f'rm {time_series_dir}tmp.tif',shell=True)
        dem_diff_shp_data = gpd.read_file(dem_diff_shp_file)
        for area in area_threshold:
            dem_diff_shp_file_area = dem_diff_shp_file.replace('.shp',f'_gt_{area}_m2.shp')
            idx_area = dem_diff_shp_data.geometry.area > area
            dem_diff_shp_data_area = dem_diff_shp_data.loc[idx_area].reset_index(drop=True)
            dem_diff_shp_data_area.to_file(dem_diff_shp_file_area)
            del dem_diff_shp_file_area,dem_diff_shp_data_area
    print('\n')

if '__main__' == __name__:
    main()