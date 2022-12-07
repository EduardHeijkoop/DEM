import numpy as np
import pandas as pd
from osgeo import gdal,gdalconst,osr
import argparse
import argparse
import subprocess
import matplotlib.pyplot as plt
import warnings
import configparser
import os
import sys
import scipy.optimize
import xml.etree.ElementTree as ET

from scipy.interpolate import SmoothBivariateSpline,LSQBivariateSpline,griddata

from dem_utils import deg2utm,get_raster_extents,resample_raster


def sample_raster(raster_path, csv_path, output_file,projection='wgs84'):
    cat_command = f"cat {csv_path} | cut -d, -f1-2 | sed 's/,/ /g' | gdallocationinfo -valonly -{projection} {raster_path} > tmp.txt"
    subprocess.run(cat_command,shell=True)
    fill_nan_command = f"awk '!NF{{$0=\"NaN\"}}1' tmp.txt > tmp2.txt"
    subprocess.run(fill_nan_command,shell=True)
    paste_command = f"paste -d , {csv_path} tmp2.txt > {output_file}"
    subprocess.run(paste_command,shell=True)
    subprocess.run(f"sed -i '/-9999/d' {output_file}",shell=True)
    subprocess.run(f"sed -i '/NaN/d' {output_file}",shell=True)
    subprocess.run(f"sed -i '/nan/d' {output_file}",shell=True)
    subprocess.run(f"rm tmp.txt tmp2.txt",shell=True)
    return None

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

def calculate_shift(df_sampled,mean_median_mode='mean',n_sigma_filter=2,vertical_shift_iterative_threshold=0.05,printing=False):
    count = 0
    cumulative_shift = 0
    original_len = len(df_sampled)
    height_icesat2_original = np.asarray(df_sampled.height_icesat2)
    height_dem_original = np.asarray(df_sampled.height_dem)
    dh_original = height_icesat2_original - height_dem_original
    rmse_original = np.sqrt(np.sum(dh_original**2)/len(dh_original))
    while True:
        count = count + 1
        height_icesat2 = np.asarray(df_sampled.height_icesat2)
        height_dem = np.asarray(df_sampled.height_dem)
        dh = height_icesat2 - height_dem
        dh_filter = filter_outliers(dh,mean_median_mode,n_sigma_filter)
        if mean_median_mode == 'mean':
            incremental_shift = np.mean(dh[dh_filter])
        elif mean_median_mode == 'median':
            incremental_shift = np.median(dh[dh_filter])
        df_sampled = df_sampled[dh_filter].reset_index(drop=True)
        df_sampled.height_dem = df_sampled.height_dem + incremental_shift
        cumulative_shift = cumulative_shift + incremental_shift
        if printing == True:
            print(f'Iteration        : {count}')
            print(f'Incremental shift: {incremental_shift:.2f} m\n')
        if np.abs(incremental_shift) <= vertical_shift_iterative_threshold:
            break
        if count == 15:
            break
    height_icesat2_filtered = np.asarray(df_sampled.height_icesat2)
    height_dem_filtered = np.asarray(df_sampled.height_dem)
    dh_filtered = height_icesat2_filtered - height_dem_filtered
    rmse_filtered = np.sqrt(np.sum(dh_filtered**2)/len(dh_filtered))
    if printing == True:
        print(f'Number of iterations: {count}')
        print(f'Number of points before filtering: {original_len}')
        print(f'Number of points after filtering: {len(df_sampled)}')
        print(f'Retained {len(df_sampled)/original_len*100:.1f}% of points.')
        print(f'Cumulative shift: {cumulative_shift:.2f} m')
        print(f'RMSE before filtering: {rmse_original:.2f} m')
        print(f'RMSE after filtering: {rmse_filtered:.2f} m')
    return df_sampled,cumulative_shift


def vertical_shift_raster(raster_path,df_sampled,mean_median_mode='mean',n_sigma_filter=2,vertical_shift_iterative_threshold=0.05,corrections_only_flag=False,print_flag=False,new_dir=None):
    src = gdal.Open(raster_path,gdalconst.GA_Update)
    raster_nodata = src.GetRasterBand(1).GetNoDataValue()
    df_sampled_filtered,vertical_shift = calculate_shift(df_sampled,mean_median_mode,n_sigma_filter,vertical_shift_iterative_threshold,print_flag)
    raster_base,raster_ext = os.path.splitext(raster_path)
    if new_dir is not None:
        raster_base = os.path.join(new_dir,os.path.basename(raster_base))
    if corrections_only_flag == False:
        raster_shifted = f'{raster_base}_Shifted_{"{:.2f}".format(vertical_shift).replace(".","p").replace("-","neg")}m{raster_ext}'
        shift_command = f'gdal_calc.py --quiet -A {raster_path} --outfile={raster_shifted} --calc="A+{vertical_shift:.2f}" --NoDataValue={raster_nodata} --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --co "TILED=YES"'
        subprocess.run(shift_command,shell=True)
    else:
        raster_shifted = raster_path
    return df_sampled_filtered,raster_shifted

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
    lon_icesat2 = np.asarray(df_sampled.lon)
    lat_icesat2 = np.asarray(df_sampled.lat)
    height_icesat2 = np.asarray(df_sampled.height_icesat2)
    time_icesat2 = np.asarray(df_sampled.time)
    height_dem = np.asarray(df_sampled.height_dem)
    dh = height_icesat2 - height_dem
    x_icesat2,y_icesat2,zone_icesat2 = deg2utm(lon_icesat2,lat_icesat2)
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

def compute_jitter_correction(df_sampled,dem_file,N_segments_x=8,N_segments_y=1000):
    '''
    Checks if DEM has jitter effect in it and corrects for it, using ICESat-2 as truth
    '''
    lon_icesat2 = np.asarray(df_sampled.lon)
    lat_icesat2 = np.asarray(df_sampled.lat)
    height_icesat2 = np.asarray(df_sampled.height_icesat2)
    time_icesat2 = np.asarray(df_sampled.time)
    height_dem = np.asarray(df_sampled.height_dem)
    dh = height_icesat2 - height_dem
    x_icesat2,y_icesat2,zone_icesat2 = deg2utm(lon_icesat2,lat_icesat2)
    x_segments = np.linspace(np.min(x_icesat2),np.max(x_icesat2),N_segments_x+1)
    y_segments = np.linspace(np.min(y_icesat2),np.max(y_icesat2),N_segments_y)
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
    grid_max_dist = np.max((dx_dem,dy_dem))
    x_segments_array = np.empty([0,1],dtype=float)
    y_segments_array = np.empty([0,1],dtype=float)
    dh_segments_array = np.empty([0,1],dtype=float)
    p0_estimate = 15133
    count_error = 0
    for i in range(N_segments_x):
        idx_x = np.logical_and(x_icesat2 >= x_segments[i],x_icesat2 <= x_segments[i+1])
        x_segment_middle = np.mean(x_segments[i:i+2])
        x_segment = x_icesat2[idx_x]
        y_segment = y_icesat2[idx_x]
        dh_segment = dh[idx_x]
        try:
            params_segment,params_covariance_segment = scipy.optimize.curve_fit(fit_sine,y_segment,dh_segment,p0=[1, 2*np.math.pi/p0_estimate],bounds=((0,-np.inf),(np.max(dh_segment),np.inf)))
        except ValueError:
            count_error += 1
            continue
        dh_segment_sine = fit_sine(y_segments,params_segment[0],params_segment[1])
        x_segments_array = np.append(x_segments_array,x_segment_middle*np.ones([len(y_segments),1]).squeeze())
        y_segments_array = np.append(y_segments_array,y_segments)
        dh_segments_array = np.append(dh_segments_array,dh_segment_sine)
    if count_error >= N_segments_x/2:
        print('Jitter correction failed')
        return None,None,None
    xy_segments_array = np.stack((x_segments_array,y_segments_array),axis=1)
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
        params_jitter,params_covariance_jitter = scipy.optimize.curve_fit(fit_jitter,xy_segments_array,dh_segments_array,
            p0=[A_jitter,c_jitter,p_jitter,k_jitter,x_dem_min,y_dem_min],
            bounds=((-np.max(np.abs(dh_segments_array)),-0.0005,0.9*p_jitter,-2,0.8*x_dem_min,0.8*y_dem_min),(1.5*np.max(np.abs(dh_segments_array)),0.0005,1.1*p_jitter,2,1.1*x_dem_max,1.1*y_dem_max)))
    except ValueError:
        print('Jitter correction failed')
        return None,None,None
    dh_jitter_orig = fit_jitter(xy_segments_array,params_jitter[0],params_jitter[1],params_jitter[2],params_jitter[3],params_jitter[4],params_jitter[5])
    dh_jitter = fit_jitter(xy_mesh_array,params_jitter[0],params_jitter[1],params_jitter[2],params_jitter[3],params_jitter[4],params_jitter[5])
    dh_grid = np.reshape(dh_jitter,x_mesh.shape)
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
        build_grid_command = f'gdal_grid -a {grid_algorithm}:nodata={grid_nodata} -txe {xmin} {xmax} -tye {ymax} {ymin} -tr {xres} {yres} -a_srs EPSG:{epsg_code} -of GTiff -ot Float32 -l {layer_name} {vrt_file} {grid_file} --config GDAL_NUM_THREADS {grid_num_threads} -co "COMPRESS=LZW"'
    elif grid_algorithm == 'invdist':
        build_grid_command = f'gdal_grid -a {grid_algorithm}:nodata={grid_nodata}:smoothing={grid_smoothing}:power={grid_power} -txe {xmin} {xmax} -tye {ymax} {ymin} -tr {xres} {yres} -a_srs EPSG:{epsg_code} -of GTiff -ot Float32 -l {layer_name} {vrt_file} {grid_file} --config GDAL_NUM_THREADS {grid_num_threads} -co "COMPRESS=LZW"'
    elif grid_algorithm == 'invdistnn':
        build_grid_command = f'gdal_grid -a {grid_algorithm}:nodata={grid_nodata}:smoothing={grid_smoothing}:power={grid_power}:max_points={grid_max_pts}:radius={grid_max_dist} -txe {xmin} {xmax} -tye {ymax} {ymin} -tr {xres} {yres} -a_srs EPSG:{epsg_code} -of GTiff -ot Float32 -l {layer_name} {vrt_file} {grid_file} --config GDAL_NUM_THREADS {grid_num_threads} -co "COMPRESS=LZW"'
    subprocess.run(build_grid_command,shell=True)
    return grid_file

def raster_to_geotiff(x,y,arr,epsg_code,output_file):
    '''
    given numpy array and x and y coordinates, produces a geotiff in the right epsg code
    '''
    arr = np.flipud(arr)
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    xres = x[1] - x[0]
    yres = y[1] - y[0]
    geotransform = (xmin-xres/2,xres,0,ymax+yres/2,0,-yres)
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_file,arr.shape[1],arr.shape[0],1,gdal.GDT_Float32)
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(f'EPSG:{epsg_code}')
    dataset.GetRasterBand(1).WriteArray(arr)
    dataset.FlushCache()
    dataset = None
    return None

def main():
    warnings.simplefilter(action='ignore')
    config_file = 'dem_config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)
    '''
    dem_file = '/media/heijkoop/DATA/DEM/Accuracy_Assessment/Mosaic/WV01_20190126_1020010082E41B00_1020010083282500/strips/WV01_20190126_1020010082E41B00_1020010083282500_2m_lsf_seg4_dem.tif'
    icesat2_file = '/media/heijkoop/DATA/DEM/Accuracy_Assessment/Strip/Rural/US_Savannah_ATL03_Rural_Strip_Filtered_NDVI_NDWI.txt'
    dem_file = '/media/heijkoop/DATA/DEM/Accuracy_Assessment/Mosaic/WV01_20190126_10200100827B1600_102001007F04DC00/strips/WV01_20190126_10200100827B1600_102001007F04DC00_2m_lsf_seg3_dem.tif'
    icesat2_file = '/media/heijkoop/DATA/DEM/Accuracy_Assessment/Mosaic/WV01_20190126_10200100827B1600_102001007F04DC00/strips/US_Savannah_ATL03_high_conf_masked_SRTM_filtered_threshold_10p0_m_subset.txt'
    dem_file = '/media/heijkoop/DATA/DEM/Locations/Bolivia_SalarDeUyuni/Accuracy_Assessment/WV02_20141208_103001003B2BD300_103001003B5B7600_2m_lsf_seg1_dem.tif'
    icesat2_file = '/media/heijkoop/DATA/DEM/Locations/Bolivia_SalarDeUyuni/Accuracy_Assessment/Salar_de_Uyuni_ATL03_WV02_20141208.txt'
    mean_mode = True
    median_mode = False
    n_sigma_filter = 2
    vertical_shift_iterative_threshold = 0.02
    corrections_only_flag = False
    print_flag = True
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dem',help='Path to input DEM to correct.')
    parser.add_argument('--icesat2',help='Return period of CoDEC in years')
    parser.add_argument('--mean',default=False,action='store_true')
    parser.add_argument('--median',default=False,action='store_true')
    parser.add_argument('--sigma', nargs='?', type=int, default=2)
    parser.add_argument('--threshold', nargs='?', type=float, default=0.02)
    parser.add_argument('--corrections_only',help='Corrections only, no co-registration?',default=False,action='store_true')
    parser.add_argument('--print',default=False,action='store_true')
    parser.add_argument('--keep_files',default=False,action='store_true')
    args = parser.parse_args()

    tmp_dir = config.get('GENERAL_PATHS','tmp_dir')

    dem_file = args.dem
    icesat2_file = args.icesat2
    mean_mode = args.mean
    median_mode = args.median
    n_sigma_filter = args.sigma
    vertical_shift_iterative_threshold = args.threshold
    corrections_only_flag = args.corrections_only
    print_flag = args.print
    keep_flag = args.keep_files

    
    if np.logical_xor(mean_mode,median_mode) == True:
        if mean_mode == True:
            mean_median_mode = 'mean'
        elif median_mode == True:
            mean_median_mode = 'median'
    else:
        print('Please choose exactly one mode: mean or median.')
        sys.exit()

    icesat2_base,icesat2_ext = os.path.splitext(os.path.basename(icesat2_file))
    dem_base,dem_ext = os.path.splitext(os.path.basename(dem_file))
    sampled_original_file = f'{tmp_dir}{icesat2_base}_Sampled_{dem_base}{icesat2_ext}'
    plane_correction_file_coarse = f'{tmp_dir}{dem_base}_coarse_plane_correction{dem_ext}'
    plane_correction_file = f'{tmp_dir}{dem_base}_plane_correction{dem_ext}'
    jitter_correction_file_coarse = f'{tmp_dir}{dem_base}_coarse_jitter_correction{dem_ext}'
    jitter_correction_file = f'{tmp_dir}{dem_base}_jitter_correction{dem_ext}'

    df_icesat2 = pd.read_csv(icesat2_file,header=None,names=['lon','lat','height_icesat2','time'],dtype={'lon':'float','lat':'float','height_icesat2':'float','time':'str'})
    lon_icesat2 = np.asarray(df_icesat2.lon)
    lat_icesat2 = np.asarray(df_icesat2.lat)
    height_icesat2 = np.asarray(df_icesat2.height_icesat2)
    time_icesat2 = np.asarray(df_icesat2.time)

    src_dem = gdal.Open(dem_file,gdalconst.GA_Update)
    src_dem_proj = src_dem.GetProjection()
    src_dem_geotrans = src_dem.GetGeoTransform()
    src_dem_epsg = osr.SpatialReference(wkt=src_dem_proj).GetAttrValue('AUTHORITY',1)
    lon_min_dem,lon_max_dem,lat_min_dem,lat_max_dem = get_raster_extents(dem_file)

    idx_lon = np.logical_and(lon_icesat2 >= lon_min_dem,lon_icesat2 <= lon_max_dem)
    idx_lat = np.logical_and(lat_icesat2 >= lat_min_dem,lat_icesat2 <= lat_max_dem)
    idx_lonlat = np.logical_and(idx_lon,idx_lat)
    lon_icesat2 = lon_icesat2[idx_lonlat]
    lat_icesat2 = lat_icesat2[idx_lonlat]
    height_icesat2 = height_icesat2[idx_lonlat]
    time_icesat2 = time_icesat2[idx_lonlat]
    if np.sum(idx_lonlat)/len(idx_lonlat) < 0.9:
        print('ICESat-2 file covers more than just the DEM.')
        print('Subsetting into new file.')
        icesat2_file = f'{tmp_dir}{os.path.splitext(os.path.basename(icesat2_file))[0]}_Subset_{os.path.splitext(os.path.basename(dem_file))[0]}{os.path.splitext(icesat2_file)[1]}'
        np.savetxt(icesat2_file,np.c_[lon_icesat2,lat_icesat2,height_icesat2,time_icesat2.astype(object)],fmt='%f,%f,%f,%s',delimiter=',')

    sample_code = sample_raster(dem_file,icesat2_file,sampled_original_file)
    df_sampled_original = pd.read_csv(sampled_original_file,header=None,names=['lon','lat','height_icesat2','time','height_dem'],dtype={'lon':'float','lat':'float','height_icesat2':'float','time':'str','height_dem':'float'})
    df_sampled_coregistered,raster_shifted = vertical_shift_raster(dem_file,df_sampled_original,mean_median_mode,n_sigma_filter,vertical_shift_iterative_threshold,corrections_only_flag,print_flag,new_dir=tmp_dir)
    raster_shifted_base,raster_shifted_ext = os.path.splitext(os.path.basename(raster_shifted))
    sampled_coregistered_file = f'{tmp_dir}{icesat2_base}_Sampled_{dem_base}_Coregistered{icesat2_ext}'
    df_sampled_coregistered.to_csv(sampled_coregistered_file,header=None,index=False,sep=',',float_format='%.6f')
    dh_coregistered = df_sampled_coregistered.height_icesat2 - df_sampled_coregistered.height_dem
    rmse_coregistered = np.sqrt(np.sum(dh_coregistered**2)/len(dh_coregistered))
    print(f'RMSE of co-registered DEM: {rmse_coregistered:.2f} m')





    x_plane,y_plane,dh_plane = compute_plane_correction(df_sampled_coregistered,raster_shifted)
    if x_plane is not None:
        write_code = raster_to_geotiff(x_plane,y_plane,dh_plane,src_dem_epsg,plane_correction_file_coarse)
        resample_code = resample_raster(plane_correction_file_coarse,raster_shifted,plane_correction_file)
        subprocess.run(f'rm {plane_correction_file_coarse}',shell=True)
        plane_corrected_dem = f'{tmp_dir}{raster_shifted_base}_Plane_Corrected{raster_shifted_ext}'
        plane_corrected_dem_base,plane_corrected_dem_ext = os.path.splitext(os.path.basename(plane_corrected_dem))
        plane_correction_command = f'gdal_calc.py --quiet -A {raster_shifted} -B {plane_correction_file} --outfile={plane_corrected_dem} --calc="A+B" --NoDataValue={src_dem.GetRasterBand(1).GetNoDataValue()} --overwrite'
        subprocess.run(plane_correction_command,shell=True)
        sampled_plane_corrected_file = f'{tmp_dir}{icesat2_base}_Sampled_{plane_corrected_dem_base}{icesat2_ext}'
        sample_code = sample_raster(plane_corrected_dem,sampled_coregistered_file,sampled_plane_corrected_file)
        df_sampled_plane_corrected = pd.read_csv(sampled_plane_corrected_file,header=None,names=['lon','lat','height_icesat2','time','height_dem','height_dem_plane_corrected'],dtype={'lon':'float','lat':'float','height_icesat2':'float','time':'str','height_dem':'float','height_dem__plane_corrected':'float'})
        dh_plane_corrected = df_sampled_plane_corrected.height_icesat2 - df_sampled_plane_corrected.height_dem_plane_corrected
        rmse_plane_corrected = np.sqrt(np.sum(dh_plane_corrected**2)/len(dh_plane_corrected))
        print(f'RMSE of plane-corrected DEM: {rmse_plane_corrected:.2f} m')
    else:
        plane_corrected_dem = raster_shifted

    x_jitter,y_jitter,dh_jitter = compute_jitter_correction(df_sampled_plane_corrected,plane_corrected_dem)
    if x_jitter is not None:
        write_code = raster_to_geotiff(x_jitter,y_jitter,dh_jitter,src_dem_epsg,jitter_correction_file_coarse)
        resample_code = resample_raster(jitter_correction_file_coarse,raster_shifted,jitter_correction_file)
        subprocess.run(f'rm {jitter_correction_file_coarse}',shell=True)
        jitter_corrected_dem = f'{tmp_dir}{plane_corrected_dem_base}_Jitter_Corrected{plane_corrected_dem_ext}'
        jitter_corrected_dem = jitter_corrected_dem.replace('Plane_Corrected_Jitter_Corrected','Plane_Jitter_Corrected')
        jitter_corrected_dem_base,jitter_corrected_dem_ext = os.path.splitext(os.path.basename(jitter_corrected_dem))
        jitter_correction_command = f'gdal_calc.py --quiet -A {plane_corrected_dem} -B {jitter_correction_file} --outfile={jitter_corrected_dem} --calc="A+B" --NoDataValue={src_dem.GetRasterBand(1).GetNoDataValue()} --overwrite'
        subprocess.run(jitter_correction_command,shell=True)
        sampled_jitter_corrected_file = f'{tmp_dir}{icesat2_base}_Sampled_{jitter_corrected_dem_base}{icesat2_ext}'
        sample_code = sample_raster(jitter_corrected_dem,sampled_coregistered_file,sampled_jitter_corrected_file)
        df_sampled_jitter_corrected = pd.read_csv(sampled_jitter_corrected_file,header=None,names=['lon','lat','height_icesat2','time','height_dem','height_dem_jitter_corrected'],dtype={'lon':'float','lat':'float','height_icesat2':'float','time':'str','height_dem':'float','height_dem_jitter_corrected':'float'})
        dh_jitter_corrected = df_sampled_jitter_corrected.height_icesat2 - df_sampled_jitter_corrected.height_dem_jitter_corrected
        rmse_jitter_corrected = np.sqrt(np.sum(dh_jitter_corrected**2)/len(dh_jitter_corrected))
        print(f'RMSE of jitter-corrected DEM: {rmse_jitter_corrected:.2f} m')
    else:
        jitter_corrected_dem = plane_corrected_dem
    subprocess.run(f'mv {jitter_corrected_dem} {os.path.dirname(dem_file)}/',shell=True)

    if keep_flag == False:
        if os.path.exists(sampled_original_file):
            subprocess.run(f'rm {sampled_original_file}',shell=True)
        if os.path.exists(sampled_coregistered_file):
            subprocess.run(f'rm {sampled_coregistered_file}',shell=True)
        if os.path.exists(sampled_plane_corrected_file):
            subprocess.run(f'rm {sampled_plane_corrected_file}',shell=True)
        if os.path.exists(sampled_jitter_corrected_file):
            subprocess.run(f'rm {sampled_jitter_corrected_file}',shell=True)
        if os.path.exists(raster_shifted):
            subprocess.run(f'rm {raster_shifted}',shell=True)
        if os.path.exists(plane_correction_file):
            subprocess.run(f'rm {plane_correction_file}',shell=True)
        if os.path.exists(jitter_correction_file):
            subprocess.run(f'rm {jitter_correction_file}',shell=True)
        if os.path.exists(plane_corrected_dem):
            subprocess.run(f'rm {plane_corrected_dem}',shell=True)
        if os.path.exists(jitter_corrected_dem):
            subprocess.run(f'rm {jitter_corrected_dem}',shell=True)
        
    
if __name__ == "__main__":
    main()