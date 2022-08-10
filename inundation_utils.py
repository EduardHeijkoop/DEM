import numpy as np
import pandas as pd
import geopandas as gpd
import netCDF4 as nc
from osgeo import gdal,gdalconst,osr
from dem_utils import get_raster_extents,great_circle_distance,lonlat2epsg,deg2utm
from dem_utils import utm2epsg
import os,sys
import xml.etree.ElementTree as ET
import subprocess

# from pykrige.ok import OrdinaryKriging
# from pykrige.uk import UniversalKriging

from scipy.interpolate import RegularGridInterpolator,SmoothBivariateSpline,LSQBivariateSpline,LinearNDInterpolator


def get_SROCC_data(SROCC_dir,raster,rcp,t0):
    '''
    Get SROCC data for a raster
    '''
    lon_min,lon_max,lat_min,lat_max = get_raster_extents(raster,'global')
    lon_input = np.mod(np.mean([lon_min,lon_max]),360)
    lat_input = np.mean([lat_min,lat_max])
    rcp = str(rcp)
    if SROCC_dir[-1] != '/':
        SROCC_dir = SROCC_dir + '/'
    SROCC_file = f'{SROCC_dir}rsl_ts_{rcp.replace(".","")}.nc'
    SROCC_data = nc.Dataset(SROCC_file)
    lon_SROCC = np.asarray(SROCC_data['x'][:])
    lat_SROCC = np.asarray(SROCC_data['y'][:])
    slr_md = np.asarray(SROCC_data['slr_md'][:])
    slr_he = np.asarray(SROCC_data['slr_he'][:])
    slr_le = np.asarray(SROCC_data['slr_le'][:])
    t_SROCC = np.asarray(SROCC_data['time'][:])
    idx_t0 = np.atleast_1d(np.argwhere(t_SROCC==t0).squeeze())

    idx_closest_lon = np.argmin(np.abs(lon_SROCC-lon_input))
    idx_closest_lat = np.argmin(np.abs(lat_SROCC-lat_input))
    if np.isnan(slr_md[0,idx_closest_lat,idx_closest_lon]) == True:
        lon_meshgrid,lat_meshgrid = np.meshgrid(lon_SROCC[idx_closest_lon-1:idx_closest_lon+2],lat_SROCC[idx_closest_lat-1:idx_closest_lat+2])
        idx_lon_meshgrid,idx_lat_meshgrid = np.meshgrid(np.arange(idx_closest_lon-1,idx_closest_lon+2),np.arange(idx_closest_lat-1,idx_closest_lat+2))
        lon_meshgrid = lon_meshgrid.reshape(3*3,1).squeeze()
        lat_meshgrid = lat_meshgrid.reshape(3*3,1).squeeze()
        idx_lon_meshgrid = idx_lon_meshgrid.reshape(3*3,1).squeeze()
        idx_lat_meshgrid = idx_lat_meshgrid.reshape(3*3,1).squeeze()
        dist = great_circle_distance(lon_input,lat_input,lon_meshgrid,lat_meshgrid)
        idx_dist_sorted = np.argsort(dist)
        dist_cond = False
        dist_count = 0
        while dist_cond == False:
            if np.isnan(slr_md[0,idx_lat_meshgrid[idx_dist_sorted[dist_count]],idx_lon_meshgrid[idx_dist_sorted[dist_count]]]) == False:
                dist_cond = True
                idx_closest_lon = idx_lon_meshgrid[idx_dist_sorted[dist_count]]
                idx_closest_lat = idx_lat_meshgrid[idx_dist_sorted[dist_count]]
                break
            dist_count += 1
    slr_md_closest = slr_md[:,idx_closest_lat,idx_closest_lon]
    slr_md_closest_minus_t0 = slr_md_closest - slr_md_closest[idx_t0]
    slr_he_closest = slr_he[:,idx_closest_lat,idx_closest_lon]
    slr_he_closest_minus_t0 = slr_he_closest - slr_he_closest[idx_t0]
    slr_le_closest = slr_le[:,idx_closest_lat,idx_closest_lon]
    slr_le_closest_minus_t0 = slr_le_closest - slr_le_closest[idx_t0]
    return t_SROCC,slr_md_closest_minus_t0,slr_he_closest_minus_t0,slr_le_closest_minus_t0

def upscale_SROCC_grid(SROCC_dir,raster,rcp,t0,t_select,high_med_low_select='md',upscaling_factor=10,search_radius=3.0):
    '''
    Get SROCC data for a raster
    '''
    lon_min,lon_max,lat_min,lat_max = get_raster_extents(raster,'global')
    lon_input = np.mod(np.mean([lon_min,lon_max]),360)
    lat_input = np.mean([lat_min,lat_max])
    rcp = str(rcp)
    if SROCC_dir[-1] != '/':
        SROCC_dir = SROCC_dir + '/'
    SROCC_file = f'{SROCC_dir}rsl_ts_{rcp.replace(".","")}.nc'
    SROCC_data = nc.Dataset(SROCC_file)
    lon_SROCC = np.asarray(SROCC_data['x'][:])
    lat_SROCC = np.asarray(SROCC_data['y'][:])
    slr_grid_select = np.asarray(SROCC_data[f'slr_{high_med_low_select}'][:])
    slr_grid_select = np.concatenate((slr_grid_select[:,:,-3:],slr_grid_select,slr_grid_select[:,:,:3]),axis=2)
    lon_SROCC = np.concatenate((lon_SROCC[-3:]-360,lon_SROCC,lon_SROCC[:3]),axis=0)
    t_SROCC = np.asarray(SROCC_data['time'][:])
    idx_t0 = np.atleast_1d(np.argwhere(t_SROCC==t0).squeeze())
    idx_t_select = np.atleast_1d(np.argwhere(t_SROCC==t_select).squeeze())
    slr_grid_select = slr_grid_select - slr_grid_select[idx_t0,:,:]
    lon_SROCC_meshgrid,lat_SROCC_meshgrid = np.meshgrid(lon_SROCC,lat_SROCC)
    slr_array_select = np.reshape(slr_grid_select[idx_t_select,:,:],slr_grid_select.shape[1]*slr_grid_select.shape[2])
    idx_nan = np.isnan(slr_array_select)
    lon_SROCC_array = np.reshape(lon_SROCC_meshgrid,lon_SROCC_meshgrid.shape[0]*lon_SROCC_meshgrid.shape[1])
    lat_SROCC_array = np.reshape(lat_SROCC_meshgrid,lat_SROCC_meshgrid.shape[0]*lat_SROCC_meshgrid.shape[1])
    lon_SROCC_array = lon_SROCC_array[~idx_nan]
    lat_SROCC_array = lat_SROCC_array[~idx_nan]
    slr_array_select = slr_array_select[~idx_nan]
    idx_lon = np.logical_and(lon_SROCC_array>=lon_min-search_radius,lon_SROCC_array<=lon_max+search_radius)
    idx_lat = np.logical_and(lat_SROCC_array>=lat_min-search_radius,lat_SROCC_array<=lat_max+search_radius)
    idx_tot = np.logical_and(idx_lon,idx_lat)
    lon_SROCC_search = lon_SROCC_array[idx_tot]
    lat_SROCC_search = lat_SROCC_array[idx_tot]
    slr_array_search = slr_array_select[idx_tot]
    lon_SROCC_high_res = np.linspace(np.min(lon_SROCC_search),np.max(lon_SROCC_search),int((np.max(lon_SROCC_search)-np.min(lon_SROCC_search))/(1/upscaling_factor))+1)
    lat_SROCC_high_res = np.linspace(np.min(lat_SROCC_search),np.max(lat_SROCC_search),int((np.max(lat_SROCC_search)-np.min(lat_SROCC_search))/(1/upscaling_factor))+1)
    interp_func = LinearNDInterpolator(list(zip(lon_SROCC_search, lat_SROCC_search)), slr_array_search)
    lon_SROCC_high_res_meshgrid,lat_SROCC_high_res_meshgrid = np.meshgrid(lon_SROCC_high_res,lat_SROCC_high_res)
    slr_high_res = interp_func(lon_SROCC_high_res_meshgrid,lat_SROCC_high_res_meshgrid)
    lon_pts = np.reshape(lon_SROCC_high_res_meshgrid,lon_SROCC_high_res_meshgrid.shape[0]*lon_SROCC_high_res_meshgrid.shape[1])
    lat_pts = np.reshape(lat_SROCC_high_res_meshgrid,lat_SROCC_high_res_meshgrid.shape[0]*lat_SROCC_high_res_meshgrid.shape[1])
    slr_pts = np.reshape(slr_high_res,slr_high_res.shape[0]*slr_high_res.shape[1])
    idx_has_ne = np.asarray([np.sum(np.logical_and(ln<=lon_SROCC_search,lt<=lat_SROCC_search)) > 0 for (ln,lt) in zip(lon_pts,lat_pts)])
    idx_has_nw = np.asarray([np.sum(np.logical_and(ln>=lon_SROCC_search,lt<=lat_SROCC_search)) > 0 for (ln,lt) in zip(lon_pts,lat_pts)])
    idx_has_se = np.asarray([np.sum(np.logical_and(ln<=lon_SROCC_search,lt>=lat_SROCC_search)) > 0 for (ln,lt) in zip(lon_pts,lat_pts)])
    idx_has_sw = np.asarray([np.sum(np.logical_and(ln>=lon_SROCC_search,lt>=lat_SROCC_search)) > 0 for (ln,lt) in zip(lon_pts,lat_pts)])
    idx_contained = np.all((idx_has_ne,idx_has_nw,idx_has_se,idx_has_sw),axis=0)
    lon_pts = lon_pts[idx_contained]
    lat_pts = lat_pts[idx_contained]
    slr_pts = slr_pts[idx_contained]
    return lon_pts,lat_pts,slr_pts


def reshape_grid_array(grid):
    array = np.reshape(grid,(grid.shape[0]*grid.shape[1]))
    array = array[~np.isnan(array)]
    return array

def create_icesat2_grid(df_icesat2,epsg_code,geoid_file,tmp_dir,loc_name,grid_res,N_pts):
    lon = np.asarray(df_icesat2.lon)
    lat = np.asarray(df_icesat2.lat)
    height = np.asarray(df_icesat2.height)
    if geoid_file is not None:
        '''
        clip geoid to icesat2 extents
        interpolate grid onto icesat2 points
        correct height to geoid height
        '''
        lon_min = np.min(lon)
        lon_max = np.max(lon)
        lat_min = np.min(lat)
        lat_max = np.max(lat)
        geoid_file_icesat2_extents = f'{tmp_dir}{loc_name}_{os.path.basename(geoid_file).replace(".tif","_clipped_icesat2.tif")}'
        geoid_clip_command = f'gdal_translate -projwin {lon_min-0.1} {lat_max+0.1} {lon_max+0.1} {lat_min-0.1} -projwin_srs EPSG:4326 {geoid_file} {geoid_file_icesat2_extents} -co "COMPRESS=LZW"'
        subprocess.run(geoid_clip_command,shell=True)
        h_geoid = interpolate_grid(lon,lat,geoid_file_icesat2_extents,None,loc_name,tmp_dir)
        height = height - h_geoid
        idx_9999 = h_geoid != -9999
        lon = lon[idx_9999]
        lat = lat[idx_9999]
        height = height[idx_9999]

    x,y,zone = deg2utm(lon,lat)
    epsg_zone = utm2epsg(zone)
    idx_epsg = epsg_zone == epsg_code
    x = x[idx_epsg]
    y = y[idx_epsg]
    height = height[idx_epsg]
    x_grid_min = np.floor(np.min(x)/grid_res)*grid_res
    x_grid_max = np.ceil(np.max(x)/grid_res)*grid_res
    y_grid_min = np.floor(np.min(y)/grid_res)*grid_res
    y_grid_max = np.ceil(np.max(y)/grid_res)*grid_res
    x_range = np.linspace(x_grid_min-grid_res/2,x_grid_max+grid_res/2,int((x_grid_max - x_grid_min)/grid_res+2))
    y_range = np.linspace(y_grid_min-grid_res/2,y_grid_max+grid_res/2,int((y_grid_max - y_grid_min)/grid_res+2))
    x_grid = np.linspace(x_grid_min,x_grid_max,int((x_grid_max - x_grid_min)/grid_res+1))
    y_grid = np.linspace(y_grid_min,y_grid_max,int((y_grid_max - y_grid_min)/grid_res+1))
    x_meshgrid,y_meshgrid = np.meshgrid(x_grid,y_grid)
    ocean_grid = np.empty([len(y_grid),len(x_grid)],dtype=float)
    for i in range(len(y_grid)):
        sys.stdout.write('\r')
        n_progressbar = (i+1)/len(y_grid)
        sys.stdout.write("[%-20s] %d%%" % ('='*int(20*n_progressbar), 100*n_progressbar))
        sys.stdout.flush()
        for j in range(len(x_grid)):
            idx_grid_x = np.logical_and(x >= x_range[j],x <= x_range[j+1])
            idx_grid_y = np.logical_and(y >= y_range[i],y <= y_range[i+1])
            idx_grid_tot = np.logical_and(idx_grid_x,idx_grid_y)
            if np.sum(idx_grid_tot) < N_pts:
                x_meshgrid[i,j] = np.nan
                y_meshgrid[i,j] = np.nan
                ocean_grid[i,j] = np.nan
            else:
                ocean_grid[i,j] = np.mean(height[idx_grid_tot])
    print('')
    x_meshgrid_array = reshape_grid_array(x_meshgrid)
    y_meshgrid_array = reshape_grid_array(y_meshgrid)
    ocean_grid_array = reshape_grid_array(ocean_grid)
    return x_meshgrid_array,y_meshgrid_array,ocean_grid_array

def interpolate_grid(lon_input,lat_input,grid_file,grid_extents,loc_name,tmp_dir,grid_nodata=-9999):
    '''
    Interpolates regular grid onto input lon/lat using scipy RegularGridInterpolator
    This more accurate than sampling the points onto the grid with gdallocationinfo,
    as that one uses nearest neighbor interpolation
    '''
    if grid_extents is not None:
        lon_min_grid,lon_max_grid,lat_min_grid,lat_max_grid = [float(e) for e in grid_extents]
        lon_min_grid -= 0.1
        lon_max_grid += 0.1
        lat_min_grid -= 0.1
        lat_max_grid += 0.1
        grid_subset_file = f'{tmp_dir}{loc_name}_{os.path.basename(grid_file).replace(".tif","_subset.tif")}'
        clip_grid_command = f'gdal_translate -q -projwin {lon_min_grid} {lat_max_grid} {lon_max_grid} {lat_min_grid} {grid_file} {grid_subset_file} -co "COMPRESS=LZW"'
        subprocess.run(clip_grid_command,shell=True)
        src_grid = gdal.Open(grid_subset_file,gdalconst.GA_ReadOnly)
    else:
        src_grid = gdal.Open(grid_file,gdalconst.GA_ReadOnly)
    grid = np.array(src_grid.GetRasterBand(1).ReadAsArray())
    lon_grid_array = np.linspace(src_grid.GetGeoTransform()[0] + 0.5*src_grid.GetGeoTransform()[1], src_grid.GetGeoTransform()[0] + src_grid.RasterXSize * src_grid.GetGeoTransform()[1] - 0.5*src_grid.GetGeoTransform()[1], src_grid.RasterXSize)
    lat_grid_array = np.linspace(src_grid.GetGeoTransform()[3] + 0.5*src_grid.GetGeoTransform()[5], src_grid.GetGeoTransform()[3] + src_grid.RasterYSize * src_grid.GetGeoTransform()[5] - 0.5*src_grid.GetGeoTransform()[5], src_grid.RasterYSize)
    interp_func = RegularGridInterpolator((lon_grid_array,lat_grid_array[::-1]),np.flipud(grid).T,bounds_error=False,fill_value=grid_nodata)
    z_interp = interp_func((lon_input,lat_input))
    if grid_extents is not None:
        subprocess.run(f'rm {grid_subset_file}',shell=True)
    return z_interp

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

def get_codec(lon,lat,codec_file,return_period=10):
    '''
    Returns sea level extremes from CoDEC dataset.
    '''
    lon = lon[~np.isnan(lon)]
    lat = lat[~np.isnan(lat)]
    lonlat_threshold = 2.0
    codec_data = nc.Dataset(codec_file)
    lon_codec = np.asarray(codec_data['station_x_coordinate'][:])
    lat_codec = np.asarray(codec_data['station_y_coordinate'][:])
    return_period_array = np.asarray(codec_data['return_periods'][:]) #2,5,10,25,50,100,250,500,1000 years
    rps = np.asarray(codec_data['RPS'][:])
    idx_return_period = np.atleast_1d(np.argwhere(return_period_array == return_period).squeeze())
    if len(idx_return_period) == 0:
        raise ValueError('Return period not found in CoDEC dataset.')
    else:
        idx_return_period = idx_return_period[0]
    idx_lon = np.logical_and(lon_codec > np.min(lon) - lonlat_threshold,lon_codec < np.max(lon) + lonlat_threshold)
    idx_lat = np.logical_and(lat_codec > np.min(lat) - lonlat_threshold,lat_codec < np.max(lat) + lonlat_threshold)
    idx_lonlat = np.logical_and(idx_lon,idx_lat)
    lon_codec_select = lon_codec[idx_lonlat]
    lat_codec_select = lat_codec[idx_lonlat]
    rps_select = rps[idx_lonlat,idx_return_period]
    rps_input = interpolate_points(lon_codec_select,lat_codec_select,rps_select,lon,lat,'Smooth',2)
    return rps_input

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