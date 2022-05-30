import numpy as np
import pandas as pd
import netCDF4 as nc
import glob
from GIS_functions import resample_dem
import os, sys
import datetime

#############
#DEFINITIONS#
#############

main_dir = '/BhaltosMount/Bhaltos/EDUARD/Projects/DEM/ICESat-2/Canada_Vancouver/'

SROCC_dir = '/BhaltosMount/Bhaltos/NASA_SEALEVEL/DATABASE/SROCC_DATA/'

coastline_shp = '/BhaltosMount/Bhaltos/EDUARD/DATA_REPOSITORY/Coast/land-polygons-complete-4326/land_polygons.shp'
output_shp = main_dir + 'Canada_Vancouver.shp'

input_dir = main_dir + 'Input/'
dem_dir = main_dir + 'DEM/'
inundation_dir = main_dir + 'Inundation/'
vlm_dir = main_dir + 'VLM/'
other_dir = main_dir + 'Other/'
gsw_dir = '/BhaltosMount/Bhaltos/EDUARD/DATA_REPOSITORY/Global_Surface_Water/Occurrence/'

input_file_dtu18 = input_dir + 'Canada_Vancouver_DTU18_MSL.txt'
#input_file_icesat2 = input_dir + 'India_Mumbai_ICESat-2_MSL.txt'
input_file_tides = input_dir + 'Canada_Vancouver_tides.txt'


df_dtu18 = pd.read_csv(input_file_dtu18,header=None,names=['location','MSL'],dtype={'location':'str','MSL':'float'})
#df_icesat2 = pd.read_csv(input_file_icesat2,header=None,names=['location','MSL'],dtype={'location':'str','MSL':'float'})
df_tides = pd.read_csv(input_file_tides,header=0,names=['location','tide'],dtype={'location':'str','tide':'float'})
#vlm_file = vlm_dir + 'MSBAS_LINEAR_RATE_LOS_0.7.tif'
vlm_file = '/BhaltosMount/Bhaltos/EDUARD/Projects/DEM/ICESat-2/Canada_Vancouver/VLM/Canada_Vancouver_DInSAR_Dec2021_noInf.tif'



dem_list = glob.glob('/BhaltosMount/Bhaltos/EDUARD/Projects/DEM/ICESat-2/Canada_Vancouver/Mosaic/Canada_Vancouver_Full_Mosaic_City_Only_dem_align/Canada_Vancouver_Full_Mosaic_City_Only_Canada_Vancouver_ATL03-DEM_nuth_x-6.99_y+5.45_z-1.96_align_clipped_LZW.tif')
#dem_list = glob.glob(dem_dir + '*04_02*align.tif')

gsw_threshold = 65
gsw_threshold_str = str(gsw_threshold/100)
gsw_threshold_str = gsw_threshold_str.replace('.','p')


#years = np.asarray([2030,2050,2100])
years = np.asarray([2020,2040,2060,2080,2100])
#years = np.asarray([2020])
sea_level_rise = np.asarray([0])


#For 2030,2050,2100
#RCP8.5
#SROCC_slr_md = [0.034,0.147,0.698]
#SROCC_slr_he = [0.072,0.209,0.897]
#SROCC_slr_le = [0,0.091,0.533]

#RCP4.5
#SROCC_slr_md = [0.028,0.117,0.395]
#SROCC_slr_he = [0.059,0.177,0.533]
#SROCC_slr_le = [0,0.061,0.273]

#RCP2.6
#SROCC_slr_md = [0.042,0.109,0.280]
#SROCC_slr_he = [0.079,0.164,0.400]
#SROCC_slr_le = [0.008,0.059,0.175]

#For 2040,2060-2090
#SROCC_slr_md = [0.089,0.230,0.327,0.433,0.548]
#SROCC_slr_he = [0.138,0.314,0.434,0.532,0.713]
#SROCC_slr_le = [0.043,0.155,0.233,0.321,0.407]

#For 2040,2060,2080,2100
SROCC_slr_md = [0,0.080,0.210,0.382,0.639]
SROCC_slr_he = [0,0.133,0.296,0.513,0.834]
SROCC_slr_le = [0,0.031,0.132,0.268,0.466]



#SROCC_slr_md = [0.0]
#SROCC_slr_he = [0.0]
#SROCC_slr_le = [0.0]


msl_dtu18 = df_dtu18.MSL[0]
#msl_icesat2 = df_icesat2.MSL[0]
tide = df_tides.tide[0]


#############
#CALCULATIONS
#############

#opposite so it will always overwrite
lon_min_total = 180
lon_max_total = -180
lat_min_total = 90
lat_max_total = -90

'''
#loop once to get global lon/lat min/max
#gdalinfo is not computationally intensive
for dem in dem_list:
    os.system('Get_tif_lonlat.py ' + dem + ' > ' + main_dir + 'tmp_out.txt')
    for line in open(main_dir + 'tmp_out.txt'):
        tmp = 0 #placeholder

    line = line.replace('\n','')
    tmp_lonlat_dem = line.split(',')
    tmp_lon_min_dem = float(tmp_lonlat_dem[0])
    tmp_lon_max_dem = float(tmp_lonlat_dem[1])
    tmp_lat_min_dem = float(tmp_lonlat_dem[2])
    tmp_lat_max_dem = float(tmp_lonlat_dem[3])

    lon_min_total = min(lon_min_total,tmp_lon_min_dem)
    lon_max_total = max(lon_max_total,tmp_lon_max_dem)
    lat_min_total = min(lat_min_total,tmp_lat_min_dem)
    lat_max_total = max(lat_max_total,tmp_lat_max_dem)


lonlat_str_total = str("%.3f" % (lon_min_total-0.1)) + ' ' + str("%.3f" % (lat_min_total-0.1)) + ' ' + str("%.3f" % (lon_max_total+0.1)) + ' ' + str("%.3f" % (lat_max_total+0.1))

#shp_command = 'ogr2ogr ' + output_shp + ' ' + coastline_shp + ' -clipsrc ' + lonlat_str_total
#os.system(shp_command)


lon_min_dem_rounded_gsw = int(np.floor(lon_min_total/10)*10)
lon_max_dem_rounded_gsw = int(np.floor(lon_max_total/10)*10)
lat_min_dem_rounded_gsw = int(np.ceil(lat_min_total/10)*10)
lat_max_dem_rounded_gsw = int(np.ceil(lat_max_total/10)*10)

lon_gsw_range = range(lon_min_dem_rounded_gsw,lon_max_dem_rounded_gsw+10,10)
lat_gsw_range = range(lat_min_dem_rounded_gsw,lat_max_dem_rounded_gsw+10,10)

gsw_output_file = other_dir + 'GSW_India_Mumbai.tif'
#gsw_output_file_resized = gsw_output_file.replace('.tif','_resized.tif')
gsw_output_file_threshold = gsw_output_file.replace('.tif','_lt_'+gsw_threshold_str + '.tif')
#gsw_output_file_resized_threshold_shp = gsw_output_file_resized_threshold.replace('.tif','.shp')

gsw_merge_command = 'gdal_merge.py -o ' + gsw_output_file + ' -co COMPRESS=LZW '
for lon in lon_gsw_range:
    for lat in lat_gsw_range:
        if lon>=0:
            EW_str = 'E'
        else:
            EW_str = 'W'
        if lat>=0:
            NS_str = 'N'
        else:
            NS_str = 'S'
        gsw_file = gsw_dir + 'occurrence_' + str(lon) + EW_str + '_' + str(lat) + NS_str + 'v1_1_2019.tif '
        gsw_merge_command = gsw_merge_command + gsw_file

#print(gsw_merge_command)
os.system(gsw_merge_command)
gsw_calc_command = 'gdal_calc.py -A ' + gsw_output_file + ' --calc="A<'+str(gsw_threshold)+'" --outfile=' + gsw_output_file_threshold + ' --format=GTiff --quiet --co="COMPRESS=LZW"'
#print(gsw_calc_command)
os.system(gsw_calc_command)
'''

for dem in dem_list:
    dem_file = dem.split('/')
    dem_file.reverse()
    dem_file = dem_file[0]
    dem_clipped_file = dem
    #gsw_filtered_dem_file = dem_clipped_file.replace('.tif','gsw_filtered.tif')

    
    utm_code = 'Canada_Vancouver'


    
    #tmp_clipped_coast = dem_dir + utm_code + '_dem_clipped_coast_tmp.tif'

    #dem_clipped_coast = tmp_clipped_coast.replace('_tmp.tif','.tif')
    #dem_clipped_gsw = dem_dir + utm_code + '_dem_clipped_coast_gsw.tif'
    '''
    os.system('Get_tif_lonlat.py ' + dem + ' > ' + main_dir + 'tmp_out.txt')
    for line in open(main_dir + 'tmp_out.txt'):
        tmp = 0 #placeholder
    os.system('rm ' + main_dir + 'tmp_out.txt')

    line = line.replace('\n','')
    lonlat_dem = line.split(',')
    lon_min_dem = float(lonlat_dem[0])
    lon_max_dem = float(lonlat_dem[1])
    lat_min_dem = float(lonlat_dem[2])
    lat_max_dem = float(lonlat_dem[3])

    gsw_dem_output_file_threshold = other_dir + 'GSW_lt_' + gsw_threshold_str + '_' + utm_code + '.tif'
    #gsw_dem_output_file_threshold_shp = gsw_dem_output_file_threshold.replace('.tif','.shp')
    gsw_dem_output_file_threshold_resampled = gsw_dem_output_file_threshold.replace('.tif','_resampled.tif')
    gsw_dem_output_file_threshold_resampled_uncompressed = gsw_dem_output_file_threshold_resampled.replace('.tif','_uncompressed.tif')

    lonlat_str = str("%.3f" % (lon_min_dem-0.1)) + ' ' + str("%.3f" % (lat_min_dem-0.1)) + ' ' + str("%.3f" % (lon_max_dem+0.1)) + ' ' + str("%.3f" % (lat_max_dem+0.1))
    shp_command = 'ogr2ogr ' + coastline_dem_shp + ' ' + coastline_shp + ' -clipsrc ' + lonlat_str
    #print(shp_command)
    os.system(shp_command)





    gsw_warp_command = 'gdalwarp -te ' + lonlat_str + ' ' + gsw_output_file_threshold + ' ' + gsw_dem_output_file_threshold
    #print(gsw_warp_command)
    os.system(gsw_warp_command)
    

    #gsw_polygonize_command = 'gdal_polygonize.py ' + gsw_dem_output_file_threshold + ' -f "ESRI Shapefile" ' + gsw_dem_output_file_threshold_shp
    #print(gsw_polygonize_command)
    ##os.system(gsw_polygonize_command)

    coastline_cut1_command = 'gdalwarp -of GTiff -cutline ' + coastline_dem_shp + ' -cl ' + utm_code + ' -crop_to_cutline -dstnodata 0.0 ' + dem + ' ' + tmp_clipped_coast
    coastline_cut2_command = 'gdal_translate -co compress=LZW -co BIGTIFF=YES ' + tmp_clipped_coast + ' ' + dem_clipped_coast

    os.system(coastline_cut1_command)
    os.system(coastline_cut2_command)
    os.system('rm ' + tmp_clipped_coast)

    resample_dem(gsw_dem_output_file_threshold,dem_clipped_coast,gsw_dem_output_file_threshold_resampled_uncompressed)
    compress_gsw_command = 'gdal_translate -co compress=LZW ' + gsw_dem_output_file_threshold_resampled_uncompressed + ' ' + gsw_dem_output_file_threshold_resampled
    os.system(compress_gsw_command)
    os.system('rm ' + gsw_dem_output_file_threshold_resampled_uncompressed)


    gsw_mask_command = 'gdal_calc.py -A ' + dem_clipped_coast + ' -B ' + gsw_dem_output_file_threshold_resampled + ' --calc="A*(B==1)" --outfile=' + dem_clipped_gsw + ' --quiet --NoDataValue=0 --co="COMPRESS=LZW"'
    os.system(gsw_mask_command)
    '''
    vlm_resampled = vlm_file.replace('.tif','_2m.tif')
    
    vlm_tmp = vlm_dir + 'tmp_vlm_uncompressed.tif'
    print('Resampling VLM...')
    resample_dem(vlm_file,dem_clipped_file,vlm_tmp)
    vlm_compress_command = 'gdal_translate -co compress=LZW -co BIGTIFF=YES ' + vlm_tmp + ' ' + vlm_resampled
    t_start = datetime.datetime.now()
    os.system(vlm_compress_command)
    os.system('rm ' + vlm_tmp)
    t_end = datetime.datetime.now()
    print('Resize complete.')
    dt = t_end - t_start
    dt_min, dt_sec = divmod(dt.seconds,60)
    dt_hour, dt_min = divmod(dt_min,60)
    print('It took:')
    print("%d hours, %d minutes, %d.%d seconds" %(dt_hour,dt_min,dt_sec,dt.microseconds%1000000))
    
    #######
    #TO DO#
    #######
    

    #Clip DEM to coastline, before GSW extents
    #    Clip *that* DEM to GSW
    #Add VLM to fully clipped DEM based on time, but not when absolute SLR
    #    Assume now is 2020 
    #    Check if it adds values to clipped out areas
    #Do inundation based on sea level (+tides!)
    #    Sea level needs to be added to with SLR rate + (time-2020)
    #        This one needs VLM-corrected DEM to be used
    #    Sea level needs 0, 30, 50, 100, 150 cm added
    #        This one needs original DEM







    print(' ')
    for j in range(len(years)):
        yr = years[j]
        slr_md = SROCC_slr_md[j]
        slr_he = SROCC_slr_he[j]
        slr_le = SROCC_slr_le[j]

        delta_year = yr - 2020
        delta_year_str = str(delta_year)
        dem_year = dem_clipped_file.replace('.tif','_in_'+str(yr)+'.tif')
        dem_year_command = 'gdal_calc.py -A ' + dem_clipped_file + ' -B ' + vlm_resampled + ' --calc="A+B*'+delta_year_str+'" --outfile='+dem_year+' --quiet --NoDataValue=0 --co="COMPRESS=LZW" --co="BIGTIFF=YES"'
        print('Updating DEM to ' + str(yr) + '...')
        t_start = datetime.datetime.now()
        os.system(dem_year_command)
        t_end = datetime.datetime.now()
        print('Update complete.')
        dt = t_end - t_start
        dt_min, dt_sec = divmod(dt.seconds,60)
        dt_hour, dt_min = divmod(dt_min,60)
        print('It took:')
        print("%d hours, %d minutes, %d.%d seconds" %(dt_hour,dt_min,dt_sec,dt.microseconds%1000000))

        
        
        sea_level_year_md = msl_dtu18 + slr_md + tide
        sea_level_year_md_str = str("%.2f" % sea_level_year_md)
        inundation_year_md_file = inundation_dir + utm_code + '_Inundation_md_in_' + str(yr) + '_RCP8p5.tif'
        calc_md_str = '(A<'+sea_level_year_md_str+')*('+sea_level_year_md_str+'-A)'
        inundation_md_year_command = 'gdal_calc.py -A ' + dem_year + ' --calc="'+calc_md_str+'" --outfile=' + inundation_year_md_file + ' --format=GTiff --quiet --co="COMPRESS=LZW" --co="BIGTIFF=YES"'
        
        sea_level_year_he = msl_dtu18 + slr_he + tide
        sea_level_year_he_str = str("%.2f" % sea_level_year_he)
        inundation_year_he_file = inundation_dir + utm_code + '_Inundation_he_in_' + str(yr) + '_RCP8p5.tif'
        calc_he_str = '(A<'+sea_level_year_he_str+')*('+sea_level_year_he_str+'-A)'
        inundation_he_year_command = 'gdal_calc.py -A ' + dem_year + ' --calc="'+calc_he_str+'" --outfile=' + inundation_year_he_file + ' --format=GTiff --quiet --co="COMPRESS=LZW" --co="BIGTIFF=YES"'

        sea_level_year_le = msl_dtu18 + slr_le + tide
        sea_level_year_le_str = str("%.2f" % sea_level_year_le)
        inundation_year_le_file = inundation_dir + utm_code + '_Inundation_le_in_' + str(yr) + '_RCP8p5.tif'
        calc_le_str = '(A<'+sea_level_year_le_str+')*('+sea_level_year_le_str+'-A)'
        inundation_le_year_command = 'gdal_calc.py -A ' + dem_year + ' --calc="'+calc_le_str+'" --outfile=' + inundation_year_le_file + ' --format=GTiff --quiet --co="COMPRESS=LZW" --co="BIGTIFF=YES"'


        print('Inundating DEM in the year ' + str(yr) + '...')
        t_start = datetime.datetime.now()
        os.system(inundation_md_year_command)
        #os.system(inundation_he_year_command)
        #os.system(inundation_le_year_command)
        t_end = datetime.datetime.now()
        print('Inundating complete.')
        dt = t_end - t_start
        dt_min, dt_sec = divmod(dt.seconds,60)
        dt_hour, dt_min = divmod(dt_min,60)
        print('It took:')
        print("%d hours, %d minutes, %d.%d seconds" %(dt_hour,dt_min,dt_sec,dt.microseconds%1000000))


    '''
    for slr in sea_level_rise:
        inundation_str = "%3.1f" % slr
        inundation_str = inundation_str.replace('.','p')
        inundation_slr_file = inundation_dir + utm_code + '_Inundation_SLR_' + inundation_str + '.tif'
        sea_level_slr = msl_dtu18 + slr + tide
        sea_level_slr_str = str("%.2f" % sea_level_slr)

        calc_str = '(A<'+sea_level_slr_str+')*('+sea_level_slr_str+'-A)'
        inundation_slr_command = 'gdal_calc.py -A ' + dem_clipped_gsw + ' --calc="'+calc_str+'" --outfile=' + inundation_slr_file + ' --format=GTiff --quiet --co="COMPRESS=LZW"'
        print('Inundating ' + utm_code + ' with ' + str(slr) + ' m of SLR...')
        t_start = datetime.datetime.now()
        os.system(inundation_slr_command)
        t_end = datetime.datetime.now()
        print('Inundating complete.')
        dt = t_end - t_start
        dt_min, dt_sec = divmod(dt.seconds,60)
        dt_hour, dt_min = divmod(dt_min,60)
        print('It took:')
        print("%d hours, %d minutes, %d.%d seconds" %(dt_hour,dt_min,dt_sec,dt.microseconds%1000000))
    '''

    



    print(' ')
    print(' ')






