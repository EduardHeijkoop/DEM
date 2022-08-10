import numpy as np
import pandas as pd
import netCDF4 as nc
import geopandas as gpd
import glob
from osgeo import gdal,osr,gdalconst
import os, sys
import datetime
import argparse
import subprocess
import matplotlib.pyplot as plt
import warnings
import configparser
from dem_utils import get_lonlat_gdf,find_corner_points_gdf
from dem_utils import get_raster_extents,resample_raster,get_gsw
from inundation_utils import create_icesat2_grid, interpolate_grid, interpolate_points,get_codec,csv_to_grid
from inundation_utils import upscale_SROCC_grid


def main():
    warnings.simplefilter(action='ignore')
    config_file = 'dem_config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dem',help='Path to input DEM to run inundation on.')
    parser.add_argument('--loc_name',help='Name of location to run inundation on.')
    parser.add_argument('--geoid',help='Path to geoid file to calculate orthometric heights with.')
    parser.add_argument('--vlm',help='Path to VLM file to propagate input file in time.')
    parser.add_argument('--clip_vlm',help='Clip DEM to VLM extents?',default=False,action='store_true')
    parser.add_argument('--icesat2',help='Path to ICESat-2 file to calculate coastal sea level with.')
    parser.add_argument('--sealevel_grid',help='Path to sea level grid to calculate coastal sea level with.')
    parser.add_argument('--grid_extents',help='Extents of grid to be used in calculation (x_min x_max y_min y_max)',nargs=4)
    parser.add_argument('--coastline',help='Path to coastline file to calculate coastal sea level on.')
    parser.add_argument('--clip_coast',help='Clip DEM to coastline?',default=False,action='store_true')
    parser.add_argument('--years',help='Years to compute inundation for.',nargs='*',default='2020')
    parser.add_argument('--rcp',help='RCP to use.',default='4.5')
    parser.add_argument('--t0',help='Time to use as t0 to zero SLR.',default='2020')
    parser.add_argument('--connectivity',help='Calculate inundation connectivity to sea?',default=False,action='store_true')
    args = parser.parse_args()

    dem_file = args.dem
    loc_name = args.loc_name
    geoid_file = args.geoid
    vlm_file = args.vlm
    clip_vlm_flag = args.clip_vlm
    icesat2_file = args.icesat2
    sl_grid_file = args.sealevel_grid
    sl_grid_extents = args.grid_extents
    coastline_file = args.coastline
    clip_coast_flag = args.clip_coast
    years = args.years
    years = [int(yr) for yr in np.atleast_1d(years)]
    rcp = args.rcp
    t0 = int(args.t0)
    connectivity_flag = args.connectivity

    if icesat2_file is not None and sl_grid_file is not None:
        print('ICESat-2 file and sea level grid given, cannot handle both!')
        sys.exit()
    if vlm_file is None:
        print('No VLM file supplied to propagate in time!')
        print('Still running inundation with sea level rise.')
    if sl_grid_file is not None and sl_grid_extents is None:
        print('Warning, selecting whole grid as input!')
        src_sl_grid = gdal.Open(sl_grid_file,gdalconst.GA_ReadOnly)
        sl_grid_extents = get_raster_extents(sl_grid_file,'global')
    if vlm_file is None and clip_vlm_flag == True:
        print('No VLM file supplied, but clipping desired!')
        sys.exit()

    if os.path.dirname(os.path.abspath(dem_file)).split('/')[-1] == 'Mosaic':
        inundation_dir = f'{"/".join(os.path.dirname(os.path.abspath(dem_file)).split("/")[:-1])}/Inundation/'
    else:
        inundation_dir = f'{os.path.dirname(os.path.abspath(dem_file))}/Inundation/'

    if not os.path.exists(inundation_dir):
        os.mkdir(inundation_dir)

    
    SROCC_dir = config.get('INUNDATION_PATHS','SROCC_dir')
    CODEC_file = config.get('INUNDATION_PATHS','CODEC_file')
    tmp_dir = config.get('GENERAL_PATHS','tmp_dir')
    gsw_dir = config.get('GENERAL_PATHS','gsw_dir')
    VLM_NODATA = config.getfloat('VLM_CONSTANTS','VLM_NODATA')
    N_PTS = config.getint('INUNDATION_CONSTANTS','N_PTS')
    INTERPOLATE_METHOD = config.get('INUNDATION_CONSTANTS','INTERPOLATE_METHOD')
    ICESAT2_GRID_RESOLUTION = config.getfloat('INUNDATION_CONSTANTS','ICESAT2_GRID_RESOLUTION')
    GRID_ALGORITHM = config.get('INUNDATION_CONSTANTS','GRID_ALGORITHM')
    GRID_NODATA = config.getint('INUNDATION_CONSTANTS','GRID_NODATA')
    GRID_SMOOTHING = config.getfloat('INUNDATION_CONSTANTS','GRID_SMOOTHING')
    GRID_POWER = config.getfloat('INUNDATION_CONSTANTS','GRID_POWER')
    GRID_MAX_PTS = config.getint('INUNDATION_CONSTANTS','GRID_MAX_PTS')
    GRID_NUM_THREADS = config.getint('INUNDATION_CONSTANTS','GRID_NUM_THREADS')
    GRID_INTERMEDIATE_RES = config.getint('INUNDATION_CONSTANTS','GRID_INTERMEDIATE_RES')
    RETURN_PERIOD = config.getint('INUNDATION_CONSTANTS','RETURN_PERIOD')
    INUNDATION_NODATA = config.getfloat('INUNDATION_CONSTANTS','INUNDATION_NODATA')
    GSW_BUFFER = config.getfloat('INUNDATION_CONSTANTS','GSW_BUFFER')

    algorithm_dict = {'grid_algorithm':GRID_ALGORITHM,
        'grid_nodata':GRID_NODATA,
        'grid_smoothing':GRID_SMOOTHING,
        'grid_power':GRID_POWER,
        'grid_max_pts':GRID_MAX_PTS,
        'grid_num_threads':GRID_NUM_THREADS,
        'grid_res':GRID_INTERMEDIATE_RES}

    if loc_name is None:
        loc_name = '_'.join(dem_file.split('/')[-1].split('_')[0:2])
    src = gdal.Open(dem_file,gdalconst.GA_ReadOnly)
    dem_nodata = src.GetRasterBand(1).GetNoDataValue()
    epsg_code = osr.SpatialReference(wkt=src.GetProjection()).GetAttrValue('AUTHORITY',1)

    gdf_coast = gpd.read_file(coastline_file)
    epsg_coastline = gdf_coast.crs.to_epsg()
    lon_coast,lat_coast = get_lonlat_gdf(gdf_coast)
    idx_corners = find_corner_points_gdf(lon_coast,lat_coast,gdf_coast)
    lon_coast[idx_corners] = np.nan
    lat_coast[idx_corners] = np.nan
    gdf_coast = gdf_coast.to_crs(f'EPSG:{epsg_code}')
    x_coast,y_coast = get_lonlat_gdf(gdf_coast)
    x_coast_orig = x_coast
    y_coast_orig = y_coast

    print(f'Working on {loc_name}.')

    if vlm_file is not None:
        t_start = datetime.datetime.now()
        print('Resampling VLM...')
        lon_vlm_min,lon_vlm_max,lat_vlm_min,lat_vlm_max = get_raster_extents(vlm_file,'global')
        src_vlm = gdal.Open(vlm_file,gdalconst.GA_ReadOnly)
        epsg_vlm_file = osr.SpatialReference(wkt=src_vlm.GetProjection()).GetAttrValue('AUTHORITY',1)
        if clip_vlm_flag == True:
            dem_clipped_to_vlm_file = dem_file.replace('.tif','_clipped_to_vlm.tif')
            clip_dem_to_vlm_command = f'gdal_translate -projwin {lon_vlm_min} {lat_vlm_max} {lon_vlm_max} {lat_vlm_min} -projwin_srs EPSG:{epsg_vlm_file} {dem_file} {dem_clipped_to_vlm_file} -co "COMPRESS=LZW"'
            subprocess.run(clip_dem_to_vlm_command,shell=True)
            dem_file = dem_clipped_to_vlm_file
            src = gdal.Open(dem_file,gdalconst.GA_ReadOnly)
        vlm_resampled_file = vlm_file.replace('.tif',f'_resampled.tif')
        resample_raster(vlm_file,dem_file,vlm_resampled_file,VLM_NODATA)
        t_end = datetime.datetime.now()
        delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
        delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
        print(f'Resampling VLM took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')


    if geoid_file is not None:
        t_start = datetime.datetime.now()
        print('Resampling geoid...')
        geoid_name = geoid_file.split('/')[-1].split('.')[0]
        geoid_resampled_file = f'{os.path.dirname(os.path.abspath(dem_file))}/{loc_name}_{geoid_name}_resampled.tif'
        resample_raster(geoid_file,dem_file,geoid_resampled_file,None)
        dem_file_orthometric = dem_file.replace('.tif','_orthometric.tif')
        orthometric_command = f'gdal_calc.py --quiet -A {dem_file} -B {geoid_resampled_file} --outfile={dem_file_orthometric} --calc="A-B" --NoDataValue={dem_nodata} --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --co "TILED=YES"'
        subprocess.run(orthometric_command,shell=True)
        dem_file = dem_file_orthometric
        src = gdal.Open(dem_file,gdalconst.GA_ReadOnly)
        t_end = datetime.datetime.now()
        delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
        delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
        print(f'Resampling geoid took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')

    if clip_coast_flag == True:
        t_start = datetime.datetime.now()
        print('Clipping DEM to coastline...')
        dem_file_clipped = dem_file.replace('.tif','_clipped.tif')
        clip_command = f'gdalwarp -s_srs EPSG:{epsg_code} -t_srs EPSG:{epsg_code} -of GTiff -cutline {coastline_file} -cl {os.path.splitext(os.path.basename(coastline_file))[0]} -dstnodata {GRID_NODATA} {dem_file} {dem_file_clipped} -overwrite -co "COMPRESS=LZW" -co "BIGTIFF=IF_SAFER" -co "TILED=YES"'
        subprocess.run(clip_command,shell=True)
        dem_file = dem_file_clipped
        src = gdal.Open(dem_file,gdalconst.GA_ReadOnly)
        t_end = datetime.datetime.now()
        delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
        delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
        print(f'Clipping took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')
    
    lon_dem_min,lon_dem_max,lat_dem_min,lat_dem_max = get_raster_extents(dem_file,'global')
    lon_center_dem = (lon_dem_min + lon_dem_max)/2
    lat_center_dem = (lat_dem_min + lat_dem_max)/2

    t_start = datetime.datetime.now()
    print('Resampling DEM to {GRID_INTERMEDIATE_RES} meters.')
    dem_resampled_file = dem_file.replace('.tif',f'_resampled_{GRID_INTERMEDIATE_RES}m.tif')
    resample_dem_command = f'gdalwarp -overwrite -tr {GRID_INTERMEDIATE_RES} {GRID_INTERMEDIATE_RES} -r bilinear {dem_file} {dem_resampled_file}'
    subprocess.run(resample_dem_command,shell=True)
    src_resampled = gdal.Open(dem_resampled_file,gdalconst.GA_ReadOnly)
    t_end = datetime.datetime.now()
    delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
    delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
    print(f'Resampling DEM took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')

    dem_x_size = src.RasterXSize
    dem_y_size = src.RasterYSize
    dem_resampled_x_size = src_resampled.RasterXSize
    dem_resampled_y_size = src_resampled.RasterYSize
    xres_dem_resampled,yres_dem_resampled = src_resampled.GetGeoTransform()[1],-src_resampled.GetGeoTransform()[5]
    x_dem_resampled_min,x_dem_resampled_max,y_dem_resampled_min,y_dem_resampled_max = get_raster_extents(dem_resampled_file,'local')
    dx_dem_resampled = np.abs(x_dem_resampled_max - x_dem_resampled_min)
    dy_dem_resampled = np.abs(y_dem_resampled_max - y_dem_resampled_min)
    grid_max_dist = np.max((dx_dem_resampled,dy_dem_resampled))
    algorithm_dict['grid_max_dist'] = grid_max_dist


    t_start = datetime.datetime.now()
    print('Generating coastal sea level grid...')
    if sl_grid_extents is not None:
        h_coast = interpolate_grid(lon_coast,lat_coast,sl_grid_file,sl_grid_extents,loc_name,tmp_dir,GRID_NODATA)
        idx_fillvalue = h_coast==GRID_NODATA
        idx_keep = ~np.logical_or(idx_fillvalue,np.isnan(h_coast))
        x_coast = x_coast[idx_keep]
        y_coast = y_coast[idx_keep]
        h_coast = h_coast[idx_keep]
        if loc_name in sl_grid_file:
            output_file_coastline = f'{tmp_dir}{os.path.basename(sl_grid_file).replace(".tif","_subset_interpolated_coastline.csv")}'
        else:
            output_file_coastline = f'{tmp_dir}{loc_name}_{os.path.basename(sl_grid_file).replace(".tif","_subset_interpolated_coastline.csv")}'
        if geoid_file is not None:
            h_geoid = interpolate_grid(lon_coast,lat_coast,geoid_file,None,loc_name,tmp_dir,GRID_NODATA)
            h_geoid = h_geoid[idx_keep]
            h_coast = h_coast - h_geoid
            output_file_coastline = output_file_coastline.replace('.csv','_orthometric.csv')
    elif icesat2_file is not None:
        df_icesat2 = pd.read_csv(icesat2_file,header=None,names=['lon','lat','height','time'],dtype={'lon':'float','lat':'float','height':'float','time':'str'})
        x_icesat2_grid_array,y_icesat2_grid_array,h_icesat2_grid_array = create_icesat2_grid(df_icesat2,epsg_code,geoid_file,tmp_dir,loc_name,ICESAT2_GRID_RESOLUTION,N_PTS)
        h_coast = interpolate_points(x_icesat2_grid_array,y_icesat2_grid_array,h_icesat2_grid_array,x_coast,y_coast,INTERPOLATE_METHOD)
        idx_keep = ~np.isnan(x_coast)
        x_coast = x_coast[idx_keep]
        y_coast = y_coast[idx_keep]
        if loc_name in icesat2_file:
            output_file_coastline = f'{tmp_dir}{os.path.basename(icesat2_file).replace(".txt",f"_subset_{INTERPOLATE_METHOD}BivariateSpline_coastline.csv")}'
        else:
            output_file_coastline = f'{tmp_dir}{loc_name}_{os.path.basename(icesat2_file).replace(".txt",f"_subset_{INTERPOLATE_METHOD}BivariateSpline_coastline.csv")}'
        if geoid_file is not None:
            output_file_coastline = output_file_coastline.replace('.csv','_orthometric.csv')
    t_end = datetime.datetime.now()
    delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
    delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
    print(f'Generating coastal sea level took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')


    
    t_start = datetime.datetime.now()
    print(f'Finding CoDEC sea level extremes for return period of {RETURN_PERIOD} years...')
    rps_coast = get_codec(lon_coast,lat_coast,CODEC_file,RETURN_PERIOD)
    output_file_codec = f'{tmp_dir}{loc_name}_CoDEC_{RETURN_PERIOD}_yrs_coastline.csv'
    np.savetxt(output_file_codec,np.c_[x_coast,y_coast,rps_coast],fmt='%f',delimiter=',',comments='')
    codec_grid_intermediate_res = csv_to_grid(output_file_codec,algorithm_dict,x_dem_resampled_min,x_dem_resampled_max,xres_dem_resampled,y_dem_resampled_min,y_dem_resampled_max,yres_dem_resampled,epsg_code)
    codec_grid_full_res = codec_grid_intermediate_res.replace(f'_{GRID_INTERMEDIATE_RES}m','')
    resample_raster(codec_grid_intermediate_res,dem_file,codec_grid_full_res)
    # t_SROCC,slr_md_closest_minus_t0,slr_he_closest_minus_t0,slr_le_closest_minus_t0 = get_SROCC_data(SROCC_dir,dem_file,rcp,t0)
    t_end = datetime.datetime.now()
    delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
    delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
    print(f'Generating CoDEC sea level extremes took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')

    if connectivity_flag == True:
        gdf_gsw_main_sea_only,gsw_output_shp_file_main_sea_only_clipped_transformed = get_gsw(inundation_dir,tmp_dir,gsw_dir,epsg_code,lon_dem_min,lon_dem_max,lat_dem_min,lat_dem_max,loc_name)
        if loc_name not in os.path.basename(gsw_output_shp_file_main_sea_only_clipped_transformed):
            gsw_output_shp_file_main_sea_only_clipped_transformed = f'{os.path.dirname(gsw_output_shp_file_main_sea_only_clipped_transformed)}/{loc_name}_{os.path.basename(gsw_output_shp_file_main_sea_only_clipped_transformed)}'
        gsw_output_shp_file_main_sea_only_clipped_transformed_buffered = gsw_output_shp_file_main_sea_only_clipped_transformed.replace('.shp',f"_buffered_{int(GSW_BUFFER)}m.shp")
        gdf_gsw_main_sea_only_buffered = gdf_gsw_main_sea_only.buffer(GSW_BUFFER)
        gdf_gsw_main_sea_only_buffered.to_file(gsw_output_shp_file_main_sea_only_clipped_transformed_buffered)



    for yr in years:
        print(f'Creating inundation in {yr}...')
        t_start = datetime.datetime.now()
        if geoid_file is not None:
            output_inundation_file = f'{inundation_dir}{loc_name}_Orthometric_Inundation_{yr}_RCP_{str(rcp).replace(".","p")}_RP_{RETURN_PERIOD}_yrs.tif'
        else:
            output_inundation_file = f'{inundation_dir}{loc_name}_Inundation_{yr}_RCP_{str(rcp).replace(".","p")}_RP_{RETURN_PERIOD}_yrs.tif'
        lon_SROCC_t_select,lat_SROCC_t_select,slr_SROCC_t_select = upscale_SROCC_grid(SROCC_dir,dem_file,rcp,t0,yr)
        h_SROCC_coast = interpolate_points(lon_SROCC_t_select,lat_SROCC_t_select,slr_SROCC_t_select,x_coast,y_coast,INTERPOLATE_METHOD)
        h_coast_yr = h_coast + h_SROCC_coast
        output_file_coastline_yr = output_file_coastline.replace('.csv',f'_{yr}.csv')
        np.savetxt(output_file_coastline_yr,np.c_[x_coast,y_coast,h_coast_yr],fmt='%f',delimiter=',',comments='')
        sl_grid_file_intermediate_res = csv_to_grid(output_file_coastline_yr,algorithm_dict,x_dem_resampled_min,x_dem_resampled_max,xres_dem_resampled,y_dem_resampled_min,y_dem_resampled_max,yres_dem_resampled,epsg_code)
        sl_grid_file_full_res = sl_grid_file_intermediate_res.replace(f'_{GRID_INTERMEDIATE_RES}m','')
        resample_raster(sl_grid_file_intermediate_res,dem_file,sl_grid_file_full_res)
        dt = int(yr - t0)
        inundation_command = f'gdal_calc.py --quiet -A {dem_file} -B {vlm_resampled_file} -C {sl_grid_file_full_res} -D {codec_grid_full_res} --outfile={output_inundation_file} --calc="A+B*{dt} < C+D" --NoDataValue={INUNDATION_NODATA} --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --co "TILED=YES"'
        subprocess.run(inundation_command,shell=True)
        t_end = datetime.datetime.now()
        delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
        delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
        print(f'Inundation creation took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')
        if connectivity_flag == True:
            print('Computing connectivity to the ocean...')
            t_start = datetime.datetime.now()
            output_inundation_shp_file = output_inundation_file.replace('.tif','.shp')
            output_inundation_shp_file_connected = output_inundation_shp_file.replace('.shp','_connected_GSW.shp')
            polygonize_command = f'gdal_polygonize.py -f "ESRI Shapefile" {output_inundation_file} {output_inundation_shp_file}'
            subprocess.run(polygonize_command,shell=True)
            gdf_inundation = gpd.read_file(output_inundation_shp_file)
            if len(gdf_gsw_main_sea_only_buffered) == 1:
                idx_intersects = np.asarray([gdf_gsw_main_sea_only_buffered.geometry[0].intersects(geom) for geom in gdf_inundation.geometry])
                idx_contains = np.asarray([gdf_gsw_main_sea_only_buffered.geometry[0].contains(geom) for geom in gdf_inundation.geometry])
                idx_connected = np.any((idx_intersects,idx_contains),axis=0)
            else:
                idx_intersects = np.zeros((len(gdf_inundation),len(gdf_gsw_main_sea_only_buffered)),dtype=bool)
                idx_contains = np.zeros((len(gdf_inundation),len(gdf_gsw_main_sea_only_buffered)),dtype=bool)
                for i,gsw_geom in enumerate(gdf_gsw_main_sea_only_buffered.geometry):
                    idx_intersects[i,:] = np.asarray([gsw_geom.intersects(geom) for geom in gdf_inundation.geometry])
                    idx_contains[i,:] = np.asarray([gsw_geom.contains(geom) for geom in gdf_inundation.geometry])
                idx_intersects = np.any(idx_intersects,axis=0)
                idx_contains = np.any(idx_contains,axis=0)
                idx_connected = np.any((idx_intersects,idx_contains),axis=0)
            gdf_inundation_connected = gdf_inundation[idx_connected]
            gdf_inundation_connected.to_file(output_inundation_shp_file_connected)
            t_end = datetime.datetime.now()
            delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
            delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
            print(f'Connectivity took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')
        subprocess.run(f'rm {output_file_coastline_yr}',shell=True)
        subprocess.run(f'rm {output_file_coastline_yr.replace(".csv",".vrt")}',shell=True)
        subprocess.run(f'rm {sl_grid_file_intermediate_res}',shell=True)
        subprocess.run(f'rm {sl_grid_file_full_res}',shell=True)
    
    subprocess.run(f'rm {output_file_coastline}',shell=True)
    subprocess.run(f'rm {output_file_codec}',shell=True)
    subprocess.run(f'rm {output_file_codec.replace(".csv",".vrt")}',shell=True)
    subprocess.run(f'rm {codec_grid_intermediate_res}',shell=True)
    subprocess.run(f'rm {codec_grid_full_res}',shell=True)
    if connectivity_flag == True:
        subprocess.run(f'rm {gdf_gsw_main_sea_only_buffered.replace(".shp",".*")}',shell=True)
    print(f'Finished with {loc_name} at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.')

if __name__ == '__main__':
    main()