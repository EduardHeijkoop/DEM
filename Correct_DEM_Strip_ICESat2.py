import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal,gdalconst,osr
import argparse
import subprocess
import matplotlib.pyplot as plt
import warnings
import configparser
import os
import sys
import scipy.optimize
import xml.etree.ElementTree as ET
import multiprocessing
import itertools
import getpass

from scipy.interpolate import SmoothBivariateSpline,LSQBivariateSpline

from dem_utils import deg2utm,get_raster_extents,resample_raster,sample_raster
from dem_utils import get_strip_list,get_strip_extents,raster_to_geotiff
from Global_DEMs import download_copernicus

def filter_outliers(dh,mean_median_mode='mean',n_sigma_filter=2):
    dh_mean = np.nanmean(dh)
    dh_std = np.nanstd(dh)
    dh_median = np.nanmedian(dh)
    if mean_median_mode == 'mean':
        dh_mean_filter = dh_mean
    elif mean_median_mode == 'median':
        dh_mean_filter = dh_median
    dh_filter = np.abs(dh-dh_mean_filter) < n_sigma_filter*dh_std
    return dh_filter

def calculate_shift(df_sampled,mean_median_mode='mean',n_sigma_filter=2,vertical_shift_iterative_threshold=0.05):
    count = 0
    cumulative_shift = 0
    original_len = len(df_sampled)
    height_icesat2_original = np.asarray(df_sampled.height_icesat2)
    height_dsm_original = np.asarray(df_sampled.height_dsm)
    dh_original = height_icesat2_original - height_dsm_original
    rmse_original = np.sqrt(np.sum(dh_original**2)/len(dh_original))
    while True:
        count = count + 1
        height_icesat2 = np.asarray(df_sampled.height_icesat2)
        height_dsm = np.asarray(df_sampled.height_dsm)
        dh = height_icesat2 - height_dsm
        dh_filter = filter_outliers(dh,mean_median_mode,n_sigma_filter)
        if mean_median_mode == 'mean':
            incremental_shift = np.mean(dh[dh_filter])
        elif mean_median_mode == 'median':
            incremental_shift = np.median(dh[dh_filter])
        df_sampled = df_sampled[dh_filter].reset_index(drop=True)
        df_sampled.height_dsm = df_sampled.height_dsm + incremental_shift
        cumulative_shift = cumulative_shift + incremental_shift
        # if printing == True:
            # print(f'Iteration        : {count}')
            # print(f'Incremental shift: {incremental_shift:.2f} m\n')
        if np.abs(incremental_shift) <= vertical_shift_iterative_threshold:
            break
        if count == 15:
            break
    height_icesat2_filtered = np.asarray(df_sampled.height_icesat2)
    height_dsm_filtered = np.asarray(df_sampled.height_dsm)
    dh_filtered = height_icesat2_filtered - height_dsm_filtered
    rmse_filtered = np.sqrt(np.sum(dh_filtered**2)/len(dh_filtered))
    stats_dict = {
        'N_iterations':count,
        'N_points_before':original_len,
        'N_points_after':len(df_sampled),
        'cumulative_shift':cumulative_shift,
        'rmse_original':rmse_original,
        'rmse_coregistered':rmse_filtered
    }
    # if printing == True:
    #     print(f'Number of iterations: {count}')
    #     print(f'Number of points before filtering: {original_len}')
    #     print(f'Number of points after filtering: {len(df_sampled)}')
    #     print(f'Retained {len(df_sampled)/original_len*100:.1f}% of points.')
    #     print(f'Cumulative shift: {cumulative_shift:.2f} m')
    #     print(f'RMSE before filtering: {rmse_original:.2f} m')
    #     print(f'RMSE after filtering: {rmse_filtered:.2f} m')
    return df_sampled,cumulative_shift,stats_dict


def vertical_shift_raster(raster_path,df_sampled,mean_median_mode='mean',n_sigma_filter=2,vertical_shift_iterative_threshold=0.05,new_dir=None):
    src = gdal.Open(raster_path,gdalconst.GA_Update)
    raster_nodata = src.GetRasterBand(1).GetNoDataValue()
    df_sampled_filtered,vertical_shift,stats_dict = calculate_shift(df_sampled,mean_median_mode,n_sigma_filter,vertical_shift_iterative_threshold)
    raster_base,raster_ext = os.path.splitext(raster_path)
    if new_dir is not None:
        raster_base = os.path.join(new_dir,os.path.basename(raster_base))
    if 'Shifted' in raster_base:
        if 'Shifted_x' in raster_base:
            if '_z_' in raster_base:
                #case: input is Shifted_x_0.00m_y_0.00m_z_0.00m*.tif
                original_shift = float(raster_base.split('Shifted')[1].split('_z_')[1].split('_')[0].replace('p','.').replace('neg','-').replace('m',''))
                original_shift_str = f'{original_shift}'.replace(".","p").replace("-","neg")
                new_shift = original_shift + vertical_shift
                new_shift_str = f'{new_shift:.2f}'.replace('.','p').replace('-','neg')
                raster_shifted = f'{raster_base}{raster_ext}'.replace(original_shift_str,new_shift_str)
            else:
                #case: input is Shifted_x_0.00m_y_0.00m*.tif
                vertical_shift_str = f'{vertical_shift:.2f}'.replace('.','p').replace('-','neg')
                post_string_fill = "_".join(raster_base.split("_y_")[1].split("_")[1:])
                if len(post_string_fill) == 0:
                    raster_shifted = f'{raster_base}{raster_ext}'.replace(raster_ext,f'_z_{vertical_shift_str}m{raster_ext}')
                else:
                    raster_shifted = f'{raster_base.split(post_string_fill)[0]}z_{vertical_shift_str}m_{post_string_fill}{raster_ext}'
        elif 'Shifted_z' in raster_base:
            #case: input is Shifted_z_0.00m*.tif
            original_shift = float(raster_base.split('Shifted')[1].split('_z_')[1].split('_')[0].replace('p','.').replace('neg','-').replace('m',''))
            new_shift = original_shift + vertical_shift
            raster_shifted = f'{raster_base.split("Shifted")[0]}Shifted_z_{"{:.2f}".format(new_shift).replace(".","p").replace("-","neg")}m{raster_ext}'
    else:
        #case: input is *.tif
        raster_shifted = f'{raster_base}_Shifted_z_{"{:.2f}".format(vertical_shift).replace(".","p").replace("-","neg")}m{raster_ext}'
    shift_command = f'gdal_calc.py --quiet -A {raster_path} --outfile={raster_shifted} --calc="A+{vertical_shift:.2f}" --NoDataValue={raster_nodata} --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --co "TILED=YES"'
    subprocess.run(shift_command,shell=True)
    return df_sampled_filtered,raster_shifted,stats_dict

def fit_sine(x,a,b):
    return a*np.sin(b*x)

def fit_sine_slope(x,a,b,c):
    return a*np.sin(b*x) + c*x

def fit_plane(xy,a,b,c):
    x = xy[:,0]
    y = xy[:,1]
    dh = a*x + b*y + c
    return dh

def fit_jitter(xy,A,c,p,k,x0,y0):
    x = xy[:,0]
    y = xy[:,1]
    dh = A*np.sin(2*np.math.pi*(y-y0)/p + c*(x-x0)) * (1-k)*(x/x0)
    return dh

def compute_plane_correction(df_sampled,dem_file):
    '''
    Checks if DEM has plane effect in it and corrects for it, using ICESat-2 as truth
    '''
    x_icesat2 = np.asarray(df_sampled.x_local)
    y_icesat2 = np.asarray(df_sampled.y_local)
    height_icesat2 = np.asarray(df_sampled.height_icesat2)
    time_icesat2 = np.asarray(df_sampled.time)
    height_dsm = np.asarray(df_sampled.height_dsm)
    dh = height_icesat2 - height_dsm
    try:
        params_plane,params_covariance_plane = scipy.optimize.curve_fit(fit_plane,np.stack((x_icesat2,y_icesat2),axis=1),dh)
    except ValueError:
        print('Plane fit failed. Skipping plane correction.')
        return None,None,None
    src_dem = gdal.Open(dem_file,gdalconst.GA_Update)
    src_dem_proj = src_dem.GetProjection()
    src_dem_geotrans = src_dem.GetGeoTransform()
    src_dem_epsg = osr.SpatialReference(wkt=src_dem_proj).GetAttrValue('AUTHORITY',1)
    dem_x_size = src_dem.RasterXSize
    dem_y_size = src_dem.RasterYSize
    xres_dem,yres_dem = src_dem.GetGeoTransform()[1],-src_dem.GetGeoTransform()[5]
    x_dem_min,x_dem_max,y_dem_min,y_dem_max = get_raster_extents(dem_file,'local')
    dx_dem = np.abs(x_dem_max - x_dem_min)
    dy_dem = np.abs(y_dem_max - y_dem_min)
    xres = 100
    yres = 100
    x_grid = np.arange(np.floor(x_dem_min/xres)*xres,np.ceil(x_dem_max/xres)*xres,xres)
    y_grid = np.arange(np.floor(y_dem_min/yres)*yres,np.ceil(y_dem_max/yres)*yres,yres)
    x_mesh,y_mesh = np.meshgrid(x_grid,y_grid)
    x_mesh_array = np.reshape(x_mesh,x_mesh.shape[0]*x_mesh.shape[1])
    y_mesh_array = np.reshape(y_mesh,y_mesh.shape[0]*y_mesh.shape[1])
    xy_mesh_array = np.stack((x_mesh_array,y_mesh_array),axis=1)
    dh_plane = fit_plane(xy_mesh_array,params_plane[0],params_plane[1],params_plane[2])
    dh_grid = np.reshape(dh_plane,x_mesh.shape)
    if np.max(np.abs(dh_grid)) < 0.1:
        print('Plane is too flat. Skipping plane correction.')
        return None,None,None
    return x_grid,y_grid,dh_grid

# def compute_jitter_correction(df_sampled,dem_file,N_segments_x=8,N_segments_y=1000):
#     '''
#     Checks if DEM has jitter effect in it and corrects for it, using ICESat-2 as truth
#     '''
#     x_icesat2 = np.asarray(df_sampled.x_local)
#     y_icesat2 = np.asarray(df_sampled.y_local)
#     height_icesat2 = np.asarray(df_sampled.height_icesat2)
#     # time_icesat2 = np.asarray(df_sampled.time)
#     height_dsm = np.asarray(df_sampled.height_dsm)
#     dh = height_icesat2 - height_dsm
#     x_segments = np.linspace(np.min(x_icesat2),np.max(x_icesat2),N_segments_x+1)
#     y_segments = np.linspace(np.min(y_icesat2),np.max(y_icesat2),N_segments_y)
#     src_dem = gdal.Open(dem_file,gdalconst.GA_Update)
#     src_dem_proj = src_dem.GetProjection()
#     src_dem_geotrans = src_dem.GetGeoTransform()
#     src_dem_epsg = osr.SpatialReference(wkt=src_dem_proj).GetAttrValue('AUTHORITY',1)
#     dem_x_size = src_dem.RasterXSize
#     dem_y_size = src_dem.RasterYSize
#     xres_dem,yres_dem = src_dem.GetGeoTransform()[1],-src_dem.GetGeoTransform()[5]
#     x_dem_min,x_dem_max,y_dem_min,y_dem_max = get_raster_extents(dem_file,'local')
#     dx_dem = np.abs(x_dem_max - x_dem_min)
#     dy_dem = np.abs(y_dem_max - y_dem_min)
#     grid_max_dist = np.max((dx_dem,dy_dem))
#     x_segments_array = np.empty([0,1],dtype=float)
#     y_segments_array = np.empty([0,1],dtype=float)
#     dh_segments_array = np.empty([0,1],dtype=float)
#     p0_estimate = 15133
#     count_error = 0
#     for i in range(N_segments_x):
#         idx_x = np.logical_and(x_icesat2 >= x_segments[i],x_icesat2 <= x_segments[i+1])
#         x_segment_middle = np.mean(x_segments[i:i+2])
#         x_segment = x_icesat2[idx_x]
#         y_segment = y_icesat2[idx_x]
#         dh_segment = dh[idx_x]
#         try:
#             params_segment,params_covariance_segment = scipy.optimize.curve_fit(fit_sine,y_segment,dh_segment,p0=[1, 2*np.math.pi/p0_estimate],bounds=((0,-np.inf),(np.max(dh_segment),np.inf)))
#         except ValueError:
#             count_error += 1
#             continue
#         dh_segment_sine = fit_sine(y_segments,params_segment[0],params_segment[1])
#         x_segments_array = np.append(x_segments_array,x_segment_middle*np.ones([len(y_segments),1]).squeeze())
#         y_segments_array = np.append(y_segments_array,y_segments)
#         dh_segments_array = np.append(dh_segments_array,dh_segment_sine)
#     if count_error >= N_segments_x/2:
#         # print('Jitter correction failed')
#         return None,None,None
#     xy_segments_array = np.stack((x_segments_array,y_segments_array),axis=1)
#     xres = 100
#     yres = 100
#     A_jitter = 0.6
#     c_jitter = 0
#     p_jitter = 15133
#     k_jitter = 0
#     x_grid = np.arange(np.floor(x_dem_min/xres)*xres,np.ceil(x_dem_max/xres)*xres,xres)
#     y_grid = np.arange(np.floor(y_dem_min/yres)*yres,np.ceil(y_dem_max/yres)*yres,yres)
#     x_mesh,y_mesh = np.meshgrid(x_grid,y_grid)
#     x_mesh_array = np.reshape(x_mesh,x_mesh.shape[0]*x_mesh.shape[1])
#     y_mesh_array = np.reshape(y_mesh,y_mesh.shape[0]*y_mesh.shape[1])
#     xy_mesh_array = np.stack((x_mesh_array,y_mesh_array),axis=1)
#     try:
#         params_jitter,params_covariance_jitter = scipy.optimize.curve_fit(fit_jitter,xy_segments_array,dh_segments_array,
#             p0=[A_jitter,c_jitter,p_jitter,k_jitter,x_dem_min,y_dem_min],
#             bounds=((-np.max(np.abs(dh_segments_array)),-0.0005,0.9*p_jitter,-2,0.8*x_dem_min,0.8*y_dem_min),(1.5*np.max(np.abs(dh_segments_array)),0.0005,1.1*p_jitter,2,1.1*x_dem_max,1.1*y_dem_max)))
#     except ValueError:
#         # print('Jitter correction failed')
#         return None,None,None
#     dh_jitter_orig = fit_jitter(xy_segments_array,params_jitter[0],params_jitter[1],params_jitter[2],params_jitter[3],params_jitter[4],params_jitter[5])
#     dh_jitter = fit_jitter(xy_mesh_array,params_jitter[0],params_jitter[1],params_jitter[2],params_jitter[3],params_jitter[4],params_jitter[5])
#     dh_grid = np.reshape(dh_jitter,x_mesh.shape)
#     if np.max(np.abs(dh_grid)) < 0.1:
#         # print('Jitter correction is too flat. Skipping jitter correction.')
#         return None,None,None
#     return x_grid,y_grid,dh_grid

def compute_jitter_correction(df_sampled,dem_file):
    x_icesat2 = np.asarray(df_sampled.x_local)
    y_icesat2 = np.asarray(df_sampled.y_local)
    xy_icesat2 = np.stack((x_icesat2,y_icesat2),axis=1)
    height_icesat2 = np.asarray(df_sampled.height_icesat2)
    height_dsm = np.asarray(df_sampled.height_dsm)
    dh = height_icesat2 - height_dsm
    src_dem = gdal.Open(dem_file,gdalconst.GA_Update)
    x_dem_min,x_dem_max,y_dem_min,y_dem_max = get_raster_extents(dem_file,'local')
    xres = 100
    yres = 100
    A_jitter = 0.6
    c_jitter = 0
    p_jitter = 15133
    k_jitter = 0
    x_grid = np.arange(np.floor(x_dem_min/xres)*xres,np.ceil(x_dem_max/xres)*xres,xres)
    y_grid = np.arange(np.floor(y_dem_min/yres)*yres,np.ceil(y_dem_max/yres)*yres,yres)
    x_mesh,y_mesh = np.meshgrid(x_grid,y_grid)
    x_mesh_array = np.reshape(x_mesh,x_mesh.shape[0]*x_mesh.shape[1])
    y_mesh_array = np.reshape(y_mesh,y_mesh.shape[0]*y_mesh.shape[1])
    xy_mesh_array = np.stack((x_mesh_array,y_mesh_array),axis=1)
    try:
        params_jitter,params_covariance_jitter = scipy.optimize.curve_fit(fit_jitter,xy_icesat2,dh,
            p0=[A_jitter,c_jitter,p_jitter,k_jitter,x_dem_min,y_dem_min],
            bounds=((-np.max(np.abs(dh)),-0.0005,0.9*p_jitter,-2,0.8*x_dem_min,0.8*y_dem_min),(np.max(np.abs(dh)),0.0005,1.1*p_jitter,2,1.1*x_dem_max,1.1*y_dem_max)))
    except ValueError:
        return None,None,None
    dh_jitter = fit_jitter(xy_mesh_array,*params_jitter)
    dh_grid = np.reshape(dh_jitter,x_mesh.shape)
    if np.max(np.abs(dh_grid)) < 0.1:
        return None,None,None
    return x_grid,y_grid,dh_grid



def interpolate_points(x_input,y_input,h_input,x_output,y_output,interpolate_method='Smooth',k_select=3):
    '''
    Inter/Extrapolates points (e.g. ICESat-2) onto a new set of points (e.g. coastline).
    '''
    x_output = x_output[~np.isnan(x_output)]
    y_output = y_output[~np.isnan(y_output)]
    if interpolate_method == 'Smooth':
        interp_func = SmoothBivariateSpline(x_input,y_input,h_input,kx=k_select, ky=k_select)
    elif interpolate_method == 'LSQ':
        interp_func = LSQBivariateSpline(x_input,y_input,h_input,x_output,y_output,kx=k_select, ky=k_select)
    h_output = interp_func(x_output,y_output,grid=False)
    return h_output


def create_csv_vrt(vrt_name,file_name,layer_name):
    ogr_vrt_data_source = ET.Element('OGRVRTDataSource')
    ogr_vrt_layer = ET.SubElement(ogr_vrt_data_source,'OGRVRTLayer',name=layer_name)
    src_data_source = ET.SubElement(ogr_vrt_layer,'SrcDataSource').text=file_name
    geometry_type = ET.SubElement(ogr_vrt_layer,'GeometryType').text='wkbPoint'
    geometry_field = ET.SubElement(ogr_vrt_layer,'GeometryField encoding="PointFromColumns" x="field_1" y="field_2" z="field_3"').text=''
    #geometry_field = ET.SubElement(ogr_vrt_layer,'GeometryField').text='encoding="PointFromColumns" x="field_1" y="field_2" z="field_3"'
    # geometry_field = ET.SubElement(ogr_vrt_layer,'GeometryField').text='encoding="PointFromColumns" x="Easting" y="Northing" z="Elevation"'
    tree = ET.ElementTree(ogr_vrt_data_source)
    ET.indent(tree, '    ')
    tree.write(vrt_name)
    return None

def csv_to_grid(csv_file,algorithm_dict,xmin,xmax,xres,ymin,ymax,yres,epsg_code):
    '''
    Turn a csv file of x/y/z into a grid with gdal_grid.
    X_MIN X_MAX Y_MAX Y_MIN is correct to get pixel size to N, -N
    This is because the origin is top-left (i.e. Northwest)
    '''
    grid_algorithm = algorithm_dict['grid_algorithm']
    grid_nodata = algorithm_dict['grid_nodata']
    grid_smoothing = algorithm_dict['grid_smoothing']
    grid_power = algorithm_dict['grid_power']
    grid_max_pts = algorithm_dict['grid_max_pts']
    grid_max_dist = algorithm_dict['grid_max_dist']
    grid_num_threads = algorithm_dict['grid_num_threads']
    grid_res = algorithm_dict['grid_res']
    vrt_file = f'{os.path.splitext(csv_file)[0]}.vrt'
    grid_file = vrt_file.replace('.vrt',f'_{grid_algorithm}_grid_{grid_res}m.tif')
    layer_name = os.path.splitext(os.path.basename(csv_file))[0]
    vrt_flag = create_csv_vrt(vrt_file,csv_file,layer_name)
    if grid_algorithm == 'nearest':
        build_grid_command = f'gdal_grid -q -a {grid_algorithm}:nodata={grid_nodata} -txe {xmin} {xmax} -tye {ymax} {ymin} -tr {xres} {yres} -a_srs EPSG:{epsg_code} -of GTiff -ot Float32 -l {layer_name} {vrt_file} {grid_file} --config GDAL_NUM_THREADS {grid_num_threads} -co "COMPRESS=LZW"'
    elif grid_algorithm == 'invdist':
        build_grid_command = f'gdal_grid -q -a {grid_algorithm}:nodata={grid_nodata}:smoothing={grid_smoothing}:power={grid_power} -txe {xmin} {xmax} -tye {ymax} {ymin} -tr {xres} {yres} -a_srs EPSG:{epsg_code} -of GTiff -ot Float32 -l {layer_name} {vrt_file} {grid_file} --config GDAL_NUM_THREADS {grid_num_threads} -co "COMPRESS=LZW"'
    elif grid_algorithm == 'invdistnn':
        build_grid_command = f'gdal_grid -q -a {grid_algorithm}:nodata={grid_nodata}:smoothing={grid_smoothing}:power={grid_power}:max_points={grid_max_pts}:radius={grid_max_dist} -txe {xmin} {xmax} -tye {ymax} {ymin} -tr {xres} {yres} -a_srs EPSG:{epsg_code} -of GTiff -ot Float32 -l {layer_name} {vrt_file} {grid_file} --config GDAL_NUM_THREADS {grid_num_threads} -co "COMPRESS=LZW"'
    subprocess.run(build_grid_command,shell=True)
    return grid_file

def parallel_corrections(dem,df_icesat2,icesat2_file,mean_median_mode,n_sigma_filter,vertical_shift_iterative_threshold,N_coverage_minimum,N_photons_minimum,tmp_dir,keep_flag,print_flag,copernicus_dict):
    lon_icesat2 = np.asarray(df_icesat2.lon)
    lat_icesat2 = np.asarray(df_icesat2.lat)
    height_icesat2 = np.asarray(df_icesat2.height_icesat2)
    # time_icesat2 = np.asarray(df_icesat2.time)
    icesat2_base,icesat2_ext = os.path.splitext(os.path.basename(icesat2_file))
    dem_base,dem_ext = os.path.splitext(os.path.basename(dem))
    print(f'Processing {dem_base}...')
    sampled_original_file = f'{tmp_dir}{icesat2_base}_Sampled_{dem_base}{icesat2_ext}'
    plane_correction_file_coarse = f'{tmp_dir}{dem_base}_coarse_plane_correction{dem_ext}'
    plane_correction_file = f'{tmp_dir}{dem_base}_plane_correction{dem_ext}'
    jitter_correction_file_coarse = f'{tmp_dir}{dem_base}_coarse_jitter_correction{dem_ext}'
    jitter_correction_file = f'{tmp_dir}{dem_base}_jitter_correction{dem_ext}'

    src_dem = gdal.Open(dem,gdalconst.GA_Update)
    src_dem_proj = src_dem.GetProjection()
    src_dem_geotrans = src_dem.GetGeoTransform()
    src_dem_epsg = osr.SpatialReference(wkt=src_dem_proj).GetAttrValue('AUTHORITY',1)
    lon_min_dem,lon_max_dem,lat_min_dem,lat_max_dem = get_raster_extents(dem)

    a_priori_flag = copernicus_dict['a_priori_flag']

    if a_priori_flag == True:
        copernicus_wgs84_file = copernicus_dict['copernicus_wgs84_file']
        faulty_pixel_height_threshold = copernicus_dict['diff_threshold']
        faulty_pixel_pct_threshold = copernicus_dict['pct_threshold']
        coastline_file = copernicus_dict['coastline_file']
        copernicus_dir = os.path.dirname(copernicus_wgs84_file)
        copernicus_base,copernicus_ext = os.path.splitext(os.path.basename(copernicus_wgs84_file))
        copernicus_wgs84_clipped_file = f'{copernicus_dir}/{copernicus_base}_{dem_base}{copernicus_ext}'
        copernicus_wgs84_clipped_coastline_file = f'{copernicus_dir}/{copernicus_base}_{dem_base}_clipped_coastline{copernicus_ext}'
        dem_resampled = f'{tmp_dir}{dem_base}_resampled_COPERNICUS{dem_ext}'
        dem_resampled_coastline = f'{tmp_dir}{dem_base}_resampled_COPERNICUS_clipped_coastline{dem_ext}'
        copernicus_dem_threshold_file = f'{tmp_dir}{dem_base}_COPERNICUS_diff_threshold{dem_ext}'
        clip_lonlat_command = f'gdalwarp -q -te {lon_min_dem} {lat_min_dem} {lon_max_dem} {lat_max_dem} {copernicus_wgs84_file} {copernicus_wgs84_clipped_file}'
        clip_copernicus_coastline_command = f'gdalwarp -q -cutline {coastline_file} {copernicus_wgs84_clipped_file} {copernicus_wgs84_clipped_coastline_file}'
        clip_dem_coastline_command = f'gdalwarp -q -cutline {coastline_file} {dem_resampled} {dem_resampled_coastline}'
        diff_threshold_command = f'gdal_calc.py -A {dem_resampled_coastline} -B {copernicus_wgs84_clipped_coastline_file} --outfile={copernicus_dem_threshold_file} --calc="A-B>{faulty_pixel_height_threshold}" --quiet --NoDataValue -9999'
        subprocess.run(clip_lonlat_command,shell=True)
        resample_raster(dem,copernicus_wgs84_clipped_file,dem_resampled,quiet_flag=True)
        subprocess.run(clip_copernicus_coastline_command,shell=True)
        subprocess.run(clip_dem_coastline_command,shell=True)
        subprocess.run(diff_threshold_command,shell=True)
        src_diff_threshold = gdal.Open(copernicus_dem_threshold_file,gdalconst.GA_Update)
        diff_threshold_array = np.array(src_diff_threshold.GetRasterBand(1).ReadAsArray())
        if np.sum(diff_threshold_array==1) / np.sum(np.logical_or(diff_threshold_array==0,diff_threshold_array==1)) > faulty_pixel_pct_threshold:
            print(f'Too many outliers wrt Copernicus DEM, probably too cloudy. Skipping {dem_base}.')
            return None

    idx_lon = np.logical_and(lon_icesat2 >= lon_min_dem,lon_icesat2 <= lon_max_dem)
    idx_lat = np.logical_and(lat_icesat2 >= lat_min_dem,lat_icesat2 <= lat_max_dem)
    idx_lonlat = np.logical_and(idx_lon,idx_lat)
    if not (np.sum(idx_lonlat) / len(idx_lonlat) >= N_coverage_minimum or np.sum(idx_lonlat) >= N_photons_minimum):
        print(f'Not enough ICESat-2 coverage over {dem_base}! Skipping.')
        print(f'ICESat-2 coverage: {100*np.sum(idx_lonlat) / len(idx_lonlat):.2f}%')
        print(f'ICESat-2 photons: {np.sum(idx_lonlat)}')
        return None
    lon_icesat2 = lon_icesat2[idx_lonlat]
    lat_icesat2 = lat_icesat2[idx_lonlat]
    height_icesat2 = height_icesat2[idx_lonlat]
    # time_icesat2 = time_icesat2[idx_lonlat]
    if np.sum(idx_lonlat)/len(idx_lonlat) < 0.9:
        # print('ICESat-2 file covers more than just the DEM.')
        # print('Subsetting into new file.')
        icesat2_file = f'{tmp_dir}{os.path.splitext(os.path.basename(icesat2_file))[0]}_Subset_{os.path.splitext(os.path.basename(dem))[0]}{os.path.splitext(icesat2_file)[1]}'
        df_icesat2[idx_lonlat].to_csv(icesat2_file,index=False,float_format='%.6f',sep=',')
        subset_flag = True
    else:
        subset_flag = False

    sample_code = sample_raster(dem,icesat2_file,sampled_original_file,header='height_dsm')
    df_sampled_original = pd.read_csv(sampled_original_file)
    if (src_dem_epsg[:3] == '326' or src_dem_epsg[:3] == '327') and len(src_dem_epsg) == 5:
        x_sampled_original,y_sampled_original,zone_sampled_original = deg2utm(df_sampled_original.lon,df_sampled_original.lat)
    else:
        gdf = gpd.GeoDataFrame(df_sampled_original,geometry=gpd.points_from_xy(df_sampled_original.lon,df_sampled_original.lat),crs='EPSG:4326')
        gdf_local = gdf.to_crs(f'EPSG:{src_dem_epsg}')
        x_sampled_original = gdf_local.geometry.x
        y_sampled_original = gdf_local.geometry.y
    df_sampled_original['x_local'] = x_sampled_original
    df_sampled_original['y_local'] = y_sampled_original
    df_sampled_coregistered,raster_shifted,raster_stats_dict = vertical_shift_raster(dem,df_sampled_original,mean_median_mode,n_sigma_filter,vertical_shift_iterative_threshold,new_dir=tmp_dir)
    raster_shifted_base,raster_shifted_ext = os.path.splitext(os.path.basename(raster_shifted))
    sampled_coregistered_file = f'{tmp_dir}{icesat2_base}_Sampled_{dem_base}_Coregistered{icesat2_ext}'
    df_sampled_coregistered.to_csv(sampled_coregistered_file,index=False,sep=',',float_format='%.6f')
    dh_coregistered = df_sampled_coregistered.height_icesat2 - df_sampled_coregistered.height_dsm
    rmse_coregistered = np.sqrt(np.sum(dh_coregistered**2)/len(dh_coregistered))
    # print(f'RMSE of co-registered DEM: {rmse_coregistered:.2f} m')

    x_plane,y_plane,dh_plane = compute_plane_correction(df_sampled_coregistered,raster_shifted)
    if x_plane is not None:
        write_code = raster_to_geotiff(x_plane,y_plane,dh_plane,src_dem_epsg,plane_correction_file_coarse)
        resample_code = resample_raster(plane_correction_file_coarse,raster_shifted,plane_correction_file,quiet_flag=True)
        subprocess.run(f'rm {plane_correction_file_coarse}',shell=True)
        plane_corrected_dem = f'{tmp_dir}{raster_shifted_base}_Plane_Corrected{raster_shifted_ext}'
        plane_corrected_dem_base,plane_corrected_dem_ext = os.path.splitext(os.path.basename(plane_corrected_dem))
        plane_correction_command = f'gdal_calc.py --quiet -A {raster_shifted} -B {plane_correction_file} --outfile={plane_corrected_dem} --calc="A+B" --NoDataValue={src_dem.GetRasterBand(1).GetNoDataValue()} --overwrite'
        subprocess.run(plane_correction_command,shell=True)
        sampled_plane_corrected_file = f'{tmp_dir}{icesat2_base}_Sampled_{plane_corrected_dem_base}{icesat2_ext}'
        sample_code = sample_raster(plane_corrected_dem,sampled_coregistered_file,sampled_plane_corrected_file,header='height_dsm_plane_corrected')
        df_sampled_plane_corrected = pd.read_csv(sampled_plane_corrected_file)
        dh_plane_corrected = df_sampled_plane_corrected.height_icesat2 - df_sampled_plane_corrected.height_dsm_plane_corrected
        rmse_plane_corrected = np.sqrt(np.sum(dh_plane_corrected**2)/len(dh_plane_corrected))
        # print(f'RMSE of plane-corrected DEM: {rmse_plane_corrected:.2f} m')
        raster_stats_dict['plane_improvement'] = True
        if rmse_plane_corrected > rmse_coregistered:
            # print('Plane correction did not improve RMSE. Keeping co-registered DEM.')
            raster_stats_dict['plane_improvement'] = False
            rmse_plane_corrected = rmse_coregistered
            plane_corrected_dem = raster_shifted
            df_sampled_plane_corrected = df_sampled_coregistered
        if keep_flag == False:
            if os.path.exists(sampled_plane_corrected_file):
                subprocess.run(f'rm {sampled_plane_corrected_file}',shell=True)
            if os.path.exists(plane_correction_file):
                subprocess.run(f'rm {plane_correction_file}',shell=True)
    else:
        plane_corrected_dem = raster_shifted
        plane_corrected_dem_base,plane_corrected_dem_ext = os.path.splitext(os.path.basename(plane_corrected_dem))
        df_sampled_plane_corrected = df_sampled_coregistered

    x_jitter,y_jitter,dh_jitter = compute_jitter_correction(df_sampled_plane_corrected,plane_corrected_dem)
    if x_jitter is not None:
        write_code = raster_to_geotiff(x_jitter,y_jitter,dh_jitter,src_dem_epsg,jitter_correction_file_coarse)
        resample_code = resample_raster(jitter_correction_file_coarse,raster_shifted,jitter_correction_file,quiet_flag=True)
        subprocess.run(f'rm {jitter_correction_file_coarse}',shell=True)
        jitter_corrected_dem = f'{tmp_dir}{plane_corrected_dem_base}_Jitter_Corrected{plane_corrected_dem_ext}'
        jitter_corrected_dem = jitter_corrected_dem.replace('Plane_Corrected_Jitter_Corrected','Plane_Jitter_Corrected')
        jitter_corrected_dem_base,jitter_corrected_dem_ext = os.path.splitext(os.path.basename(jitter_corrected_dem))
        jitter_correction_command = f'gdal_calc.py --quiet -A {plane_corrected_dem} -B {jitter_correction_file} --outfile={jitter_corrected_dem} --calc="A+B" --NoDataValue={src_dem.GetRasterBand(1).GetNoDataValue()} --overwrite'
        subprocess.run(jitter_correction_command,shell=True)
        sampled_jitter_corrected_file = f'{tmp_dir}{icesat2_base}_Sampled_{jitter_corrected_dem_base}{icesat2_ext}'
        sample_code = sample_raster(jitter_corrected_dem,sampled_coregistered_file,sampled_jitter_corrected_file,header='height_dsm_jitter_corrected')
        df_sampled_jitter_corrected = pd.read_csv(sampled_jitter_corrected_file)
        dh_jitter_corrected = df_sampled_jitter_corrected.height_icesat2 - df_sampled_jitter_corrected.height_dsm_jitter_corrected
        rmse_jitter_corrected = np.sqrt(np.sum(dh_jitter_corrected**2)/len(dh_jitter_corrected))
        # print(f'RMSE of jitter-corrected DEM: {rmse_jitter_corrected:.2f} m')
        raster_stats_dict['jitter_improvement'] = True
        if rmse_jitter_corrected > rmse_plane_corrected:
            # print('Jitter correction did not improve RMSE. Keeping plane-corrected DEM.')
            raster_stats_dict['jitter_improvement'] = False
        if keep_flag == False:
            if os.path.exists(sampled_jitter_corrected_file):
                subprocess.run(f'rm {sampled_jitter_corrected_file}',shell=True)
            if os.path.exists(jitter_correction_file):
                subprocess.run(f'rm {jitter_correction_file}',shell=True)
    else:
        jitter_corrected_dem = plane_corrected_dem
    subprocess.run(f'mv {jitter_corrected_dem} {os.path.dirname(dem)}/',shell=True)

    if print_flag == True:
        print(f'Finished correcting {dem_base}.')
        print(f'RMSE of original DEM: {raster_stats_dict["rmse_original"]:.2f} m')
        print(f'RMSE of co-registered DEM: {raster_stats_dict["rmse_coregistered"]:.2f} m')
        print(f'Coregistration converged in {raster_stats_dict["N_iterations"]} iterations.')
        print(f'Retained {raster_stats_dict["N_points_after"]} of {raster_stats_dict["N_points_before"]} points ({100*raster_stats_dict["N_points_after"]/raster_stats_dict["N_points_before"]:.1f}%).')
        if x_plane is None:
            print(f'Plane correction was too small and was not applied.')
        elif raster_stats_dict['plane_improvement'] == False:
            print(f'Plane correction did not improve RMSE. Did not apply correction.')
        else:
            print(f'RMSE of plane-corrected DEM: {rmse_plane_corrected:.2f} m')
        
        if x_jitter is None:
            print(f'Jitter correction failed or was too small and was not applied.')
        elif raster_stats_dict['jitter_improvement'] == False:
            print(f'Jitter correction did not improve RMSE. Did not apply correction.')
        else:
            print(f'RMSE of jitter-corrected DEM: {rmse_jitter_corrected:.2f} m')
    if subset_flag == True:
        subprocess.run(f'rm {icesat2_file}',shell=True)

    if keep_flag == False:
        if os.path.exists(sampled_original_file):
            subprocess.run(f'rm {sampled_original_file}',shell=True)
        if os.path.exists(sampled_coregistered_file):
            subprocess.run(f'rm {sampled_coregistered_file}',shell=True)
        if os.path.exists(raster_shifted):
            subprocess.run(f'rm {raster_shifted}',shell=True)
        if os.path.exists(plane_corrected_dem):
            subprocess.run(f'rm {plane_corrected_dem}',shell=True)
        if os.path.exists(jitter_corrected_dem):
            subprocess.run(f'rm {jitter_corrected_dem}',shell=True)

def get_batch_lonlat_extents(strip_list):
    lon_min_strips,lon_max_strips,lat_min_strips,lat_max_strips = 180,-180,90,-90
    for strip in strip_list:
        lon_min_single_strip,lon_max_single_strip,lat_min_single_strip,lat_max_single_strip = get_strip_extents(strip)
        lon_min_strips = np.min((lon_min_strips,lon_min_single_strip))
        lon_max_strips = np.max((lon_max_strips,lon_max_single_strip))
        lat_min_strips = np.min((lat_min_strips,lat_min_single_strip))
        lat_max_strips = np.max((lat_max_strips,lat_max_single_strip))
    return lon_min_strips,lon_max_strips,lat_min_strips,lat_max_strips


def main():
    warnings.simplefilter(action='ignore')
    config_file = 'dem_config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--location',default=None,help='Location of DEM(s) to correct.')
    parser.add_argument('--dem',default=None,help='Path to input DEM to correct.')
    parser.add_argument('--dem_list',default=None,help='Path to input list of DEMs to correct.')
    parser.add_argument('--icesat2',default=None,help='Path to ICESat-2 file used to correct DEM(s).')
    parser.add_argument('--mean',default=False,action='store_true')
    parser.add_argument('--median',default=False,action='store_true')
    parser.add_argument('--sigma', nargs='?', type=int, default=2)
    parser.add_argument('--threshold', nargs='?', type=float, default=0.02)
    parser.add_argument('--print',default=False,action='store_true')
    parser.add_argument('--keep_files',default=False,action='store_true')
    parser.add_argument('--cpus',help='Number of CPUs to use',default=1,type=int)
    parser.add_argument('--machine',default='t',help='Machine to run on (t, b or local)')
    parser.add_argument('--dir_structure',default='sealevel',help='Directory structure of input strips (sealevel or simple)')
    parser.add_argument('--a_priori',default=False,help='Filter with a priori DEM (COPERNICUS)?',action='store_true')
    parser.add_argument('--coastline',default=None,help='Coastline file to filter DEM & COPERNICUS')
    args = parser.parse_args()

    tmp_dir = config.get('GENERAL_PATHS','tmp_dir')
    N_coverage_minimum = config.getfloat('CORRECTIONS_CONSTANTS','N_coverage_minimum')
    N_photons_minimum = config.getint('CORRECTIONS_CONSTANTS','N_photons_minimum')

    loc_dir = args.location
    dem_file = args.dem
    dem_list_file = args.dem_list
    icesat2_file = args.icesat2
    mean_mode = args.mean
    median_mode = args.median
    n_sigma_filter = args.sigma
    vertical_shift_iterative_threshold = args.threshold
    print_flag = args.print
    keep_flag = args.keep_files
    N_cpus = args.cpus
    machine_name = args.machine
    dir_structure = args.dir_structure
    a_priori_flag = args.a_priori
    coastline_file = args.coastline

    if machine_name == 'b':
        tmp_dir = tmp_dir.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
    elif machine_name == 'local':
        tmp_dir = tmp_dir.replace('/BhaltosMount/Bhaltos/EDUARD/','/home/heijkoop/Desktop/Projects/')

    if dem_file is not None and dem_list_file is not None:
        print('Only doing files in list.')
    
    if loc_dir is None and dem_file is None and dem_list_file is None:
        print('Please provide a location, a single DEM or a list of DEMs.')
        sys.exit()
    
    if dem_list_file is not None:
        df_list = pd.read_csv(dem_list_file,header=None,names=['dem_file'],dtype={'dem_file':'str'})
        dem_array = np.asarray(df_list.dem_file)
    elif dem_file is not None:
        dem_array = np.atleast_1d(dem_file)
    elif loc_dir is not None:
        if loc_dir[-1] != '/':
            loc_dir = f'{loc_dir}/'
        dem_array = get_strip_list(loc_dir,input_type=2,corrected_flag=False,dir_structure=dir_structure)
    
    if np.logical_xor(mean_mode,median_mode) == True:
        if mean_mode == True:
            mean_median_mode = 'mean'
        elif median_mode == True:
            mean_median_mode = 'median'
    else:
        print('Please choose exactly one mode: mean or median.')
        sys.exit()
    
    if a_priori_flag == True:
        username = config.get('GENERAL_CONSTANTS','earthdata_username')
        egm2008_file = config.get('GENERAL_PATHS','EGM2008_path')
        faulty_pixel_height_threshold = config.getfloat('CORRECTIONS_CONSTANTS','faulty_pixel_height_threshold')
        faulty_pixel_pct_threshold = config.getfloat('CORRECTIONS_CONSTANTS','faulty_pixel_pct_threshold')
        if dem_list_file is not None:
            loc_name = dem_list_file.split('/')[-1].lower().split('_strip_list')[0]
            loc_name = '_'.join([''.join([l[i].capitalize() if i == 0 else l for i,l in enumerate(t)]) for t in loc_name.split('_')])
        elif dem_file is not None:
            loc_name = '_'.join(dem_file.split('/')[-1].split('_')[:4])
        elif loc_dir is not None:
            loc_name = loc_dir.split('/')[-2]
        copernicus_wgs84_file = f'{tmp_dir}{loc_name}_COPERNICUS_WGS84.tif'
        
        if machine_name == 'b':
            egm96_file = egm96_file.replace('/BhaltosMount/Bhaltos/','/Bhaltos/willismi/')
        elif machine_name == 'local':
            egm96_file = egm96_file.replace('/BhaltosMount/Bhaltos/EDUARD/DATA_REPOSITORY/','/media/heijkoop/DATA/GEOID/')
        pw = getpass.getpass()
        lon_min,lon_max,lat_min,lat_max = get_batch_lonlat_extents(dem_array)
        download_copernicus(lon_min,lon_max,lat_min,lat_max,egm2008_file,tmp_dir,copernicus_wgs84_file)
        copernicus_dict = {'a_priori_flag':True,
                      'copernicus_wgs84_file':copernicus_wgs84_file,
                      'diff_threshold':faulty_pixel_height_threshold,
                      'pct_threshold':faulty_pixel_pct_threshold,
                      'coastline_file':coastline_file}
    else:
        copernicus_wgs84_file = None
        copernicus_dict = {'a_priori_flag':False}
    '''
    Change EPSG of icesat-2 to match dem. If EPSG==326XX or 327XX, then use deg2utm, otherwise use:
        pd = df.read_csv()
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat),crs='EPSG:4326')
        gdf_3413 = gdf.to_crs('EPSG:3413') #3413 as an example
        df_3413 = pd.DataFrame({'x':gdf_3413.geometry.x,'y':gdf_3413.geometry.y,'height':df.height,'time':df.time})
    '''
    
    print('Loading ICESat-2...')    
    df_icesat2 = pd.read_csv(icesat2_file)
    print('Loading done.')

    ir = itertools.repeat
    p = multiprocessing.Pool(np.min((N_cpus,len(dem_array))))
    p.starmap(parallel_corrections,zip(
        dem_array,
        ir(df_icesat2),ir(icesat2_file),ir(mean_median_mode),ir(n_sigma_filter),ir(vertical_shift_iterative_threshold),
        ir(N_coverage_minimum),ir(N_photons_minimum),ir(tmp_dir),ir(keep_flag),ir(print_flag),ir(copernicus_dict)
        ))
    p.close()
        
    
if __name__ == "__main__":
    main()