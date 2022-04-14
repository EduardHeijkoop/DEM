import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import subprocess
import datetime
import sys
import shapely
import warnings
warnings.simplefilter(action='ignore')

x_input, y_input = (463000,2905000)
input_point = shapely.geometry.Point(x_input,y_input)
threshold = 10

time_series_dir = '/BhaltosMount/Bhaltos/EDUARD/Projects/DEM/MiddleEast/Bahrain_Manama/Time_Series_Strips/'
loc_name = 'Bahrain_Manama'

#file_list = glob.glob(f'{time_series_dir}*align.tif')
#[subprocess.run('mv ' + f + ' ' + f.split('2m_')[0] + f.split('/')[-1].split(f.split('/')[-1].split('2m_')[0])[2],shell=True) for f in file_list]
file_list = glob.glob(f'{time_series_dir}*align.tif')
file_list.sort()
file_list = np.asarray(file_list)
strip_dates = np.asarray([int(s.split('/')[-1][5:13]) for s in file_list])
unique_dates = np.unique(strip_dates)
idx_dates_sorted = np.argsort(strip_dates) #in this case we want old -> new
strip_dates_sorted = strip_dates[idx_dates_sorted]
strip_dates_sorted_datetime = np.asarray([datetime.datetime(year=int(s[0:4]),month=int(s[4:6]),day=int(s[6:8])) for s in strip_dates_sorted.astype(str)])
file_list_sorted = file_list[idx_dates_sorted]

strip_shp_data = gpd.read_file('/BhaltosMount/Bhaltos/EDUARD/Projects/DEM/MiddleEast/Bahrain_Manama/Bahrain_Manama_Strips_32639.shp')

#sort strip_shp_data to match file_list_sorted
file_list_sorted_simple = np.asarray([f.split('/')[-1].split('_sampled_mosaic')[0] for f in file_list_sorted])
strip_list_simple = np.asarray([s.split('/')[-1].split('.tif')[0] for s in strip_shp_data.strip])
idx_resort = [np.argwhere(f == strip_list_simple)[0][0] for f in file_list_sorted_simple]
strip_shp_data = strip_shp_data.iloc[idx_resort].reset_index(drop=True)

idx_contains_point = [s.contains(input_point) for s in strip_shp_data.geometry]

output_file = f'{time_series_dir}Coregistered_Strips.txt'
np.savetxt(output_file,np.c_[file_list_sorted],fmt='%s')
subprocess.run(f'gdalbuildvrt -separate -overwrite -input_file_list {output_file} {time_series_dir}{loc_name}_Time_Series.vrt',shell=True)

idx_old_strips = np.argwhere(idx_contains_point).squeeze()[:-1]
idx_new_strips = np.argwhere(idx_contains_point).squeeze()[1:]

for i in range(len(idx_old_strips)):
    sys.stdout.write('\r')
    n_progressbar = (i + 1) / len(idx_old_strips)
    sys.stdout.write("[%-20s] %d%%" % ('='*int(20*n_progressbar), 100*n_progressbar))
    sys.stdout.flush()
    old_strip = file_list_sorted[idx_old_strips[i]].split('/')[-1].split('_dem')[0]
    new_strip = file_list_sorted[idx_new_strips[i]].split('/')[-1].split('_dem')[0]
    dt = (strip_dates_sorted_datetime[idx_new_strips[i]] - strip_dates_sorted_datetime[idx_old_strips[i]]).days
    dem_diff_file = f'{time_series_dir}Diff_{new_strip}_minus_{old_strip}_dt_{dt}_days.tif'
    dem_diff_threshold_file = dem_diff_file.replace('.tif','_gt_'+str(threshold)+'m.tif')
    dem_diff_shp_file = dem_diff_threshold_file.replace('.tif','.shp')
    dem_diff_shp_file_gt_1000 = dem_diff_shp_file.replace('.shp','_gt_1000_m2.shp')
    dem_diff_shp_file_gt_5000 = dem_diff_shp_file.replace('.shp','_gt_5000_m2.shp')
    diff_calc_command = f'gdal_calc.py -A {time_series_dir}{loc_name}_Time_Series.vrt --A_band={idx_new_strips[i]+1} -B {time_series_dir}{loc_name}_Time_Series.vrt --B_band={idx_old_strips[i]+1} --outfile={dem_diff_file} --calc="A-B" --format=GTiff --co="COMPRESS=LZW" --quiet'
    threshold_calc_command = f'gdal_calc.py -A {dem_diff_file} --calc="numpy.abs(A)>{threshold}" --outfile={dem_diff_threshold_file} --format=GTiff --co="COMPRESS=LZW" --quiet'
    nodata_translate_command = f'gdal_translate -q -a_nodata 0 {dem_diff_threshold_file} tmp.tif'
    polygonize_command = f'gdal_polygonize.py -q tmp.tif -f "ESRI Shapefile" {dem_diff_shp_file}'
    subprocess.run(diff_calc_command,shell=True)
    subprocess.run(threshold_calc_command,shell=True)
    subprocess.run(nodata_translate_command,shell=True)
    subprocess.run(polygonize_command,shell=True)
    subprocess.run('rm tmp.tif',shell=True)
    dem_diff_shp_data = gpd.read_file(dem_diff_shp_file)
    idx_1000 = dem_diff_shp_data.geometry.area > 1000
    idx_5000 = dem_diff_shp_data.geometry.area > 5000
    dem_diff_shp_data_1000 = dem_diff_shp_data[idx_1000].reset_index(drop=True)
    dem_diff_shp_data_5000 = dem_diff_shp_data[idx_5000].reset_index(drop=True)
    dem_diff_shp_data_1000.to_file(dem_diff_shp_file_gt_1000)
    dem_diff_shp_data_5000.to_file(dem_diff_shp_file_gt_5000)
    del dem_diff_shp_data,dem_diff_shp_data_1000,dem_diff_shp_data_5000
print('\n')

# gdal_calc.py -A Diff.tif --calc="A>10" --outfile=Diff_threshold.tif --format=GTiff --co="COMPRESS=LZW" --quiet

'''
for i in range(len(unique_dates)-1):
    sys.stdout.write('\r')
    n_progressbar = (i + 1) / len(unique_dates)
    sys.stdout.write("[%-20s] %d%%" % ('='*int(20*n_progressbar), 100*n_progressbar))
    sys.stdout.flush()
    idx_old_dates = strip_dates_sorted == unique_dates[i]
    idx_new_dates = strip_dates_sorted == unique_dates[i+1]
    for j in np.atleast_1d(np.argwhere(idx_old_dates).squeeze()):
        for k in np.atleast_1d(np.argwhere(idx_new_dates).squeeze()):
            print(j)
            print(k)
            old_strip = file_list_sorted[j].split('/')[-1].split('_dem')[0]
            new_strip = file_list_sorted[k].split('/')[-1].split('_dem')[0]
            dt = (strip_dates_sorted_datetime[k] - strip_dates_sorted_datetime[j]).days
            dem_diff_file = f'{time_series_dir}Diff_{new_strip}_minus_{old_strip}_dt_{dt}_days.tif'
            calc_command = f'gdal_calc.py -A {time_series_dir}{loc_name}_Time_Series.vrt --A_band={k+1} -B {time_series_dir}{loc_name}_Time_Series.vrt --B_band={j+1} --outfile={dem_diff_file} --calc="A-B" --format=GTiff --co="COMPRESS=LZW" --quiet'
            print(calc_command)
            print(' ')
'''
