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

from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

from scipy.interpolate import RegularGridInterpolator


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

def reshape_grid_array(grid):
    array = np.reshape(grid,(grid.shape[0]*grid.shape[1]))
    array = array[~np.isnan(array)]
    return array

def create_icesat2_grid(df_icesat2,epsg_code,grid_res,N_pts):
    lon = np.asarray(df_icesat2.lon)
    lat = np.asarray(df_icesat2.lat)
    height = np.asarray(df_icesat2.height)
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

def create_sl_grid(sl_grid_file,sl_grid_extents,loc_name,epsg_code,tmp_dir):
    if sl_grid_extents is not None:
        lon_min_sl_grid,lon_max_sl_grid,lat_min_sl_grid,lat_max_sl_grid = [float(e) for e in sl_grid_extents]
        sl_grid_subset_file = f'{tmp_dir}{loc_name}_{os.path.basename(sl_grid_file).replace(".tif","_subset.tif")}'
        clip_sl_grid_command = f'gdal_translate -q -projwin {lon_min_sl_grid} {lat_max_sl_grid} {lon_max_sl_grid} {lat_min_sl_grid} {sl_grid_file} {sl_grid_subset_file} -co "COMPRESS=LZW"'
        subprocess.run(clip_sl_grid_command,shell=True)
        src_sl_grid = gdal.Open(sl_grid_subset_file,gdalconst.GA_ReadOnly)
    else:
        src_sl_grid = gdal.Open(sl_grid_file,gdalconst.GA_ReadOnly)
    sl_grid = np.array(src_sl_grid.GetRasterBand(1).ReadAsArray())
    lon_sl_grid_array = np.linspace(src_sl_grid.GetGeoTransform()[0] + 0.5*src_sl_grid.GetGeoTransform()[1], src_sl_grid.GetGeoTransform()[0] + src_sl_grid.RasterXSize * src_sl_grid.GetGeoTransform()[1] - 0.5*src_sl_grid.GetGeoTransform()[1], src_sl_grid.RasterXSize)
    lat_sl_grid_array = np.linspace(src_sl_grid.GetGeoTransform()[3] + 0.5*src_sl_grid.GetGeoTransform()[5], src_sl_grid.GetGeoTransform()[3] + src_sl_grid.RasterYSize * src_sl_grid.GetGeoTransform()[5] - 0.5*src_sl_grid.GetGeoTransform()[5], src_sl_grid.RasterYSize)
    lon_sl_meshgrid,lat_sl_meshgrid = np.meshgrid(lon_sl_grid_array,lat_sl_grid_array)
    lon_sl_meshgrid_array = reshape_grid_array(lon_sl_meshgrid)
    lat_sl_meshgrid_array = reshape_grid_array(lat_sl_meshgrid)
    sl_grid_array = reshape_grid_array(sl_grid)
    x,y,zone = deg2utm(lon_sl_meshgrid_array,lat_sl_meshgrid_array)
    epsg_zone = utm2epsg(zone)
    idx_epsg = epsg_zone == epsg_code
    x_sl_meshgrid_array = x[idx_epsg]
    y_sl_meshgrid_array = y[idx_epsg]
    sl_grid_array = sl_grid_array[idx_epsg]
    return x_sl_meshgrid_array,y_sl_meshgrid_array,sl_grid_array

def interpolate_grid(lon_input,lat_input,sl_grid_file,sl_grid_extents,loc_name,tmp_dir,GRID_NODATA=-9999):
    '''
    Interpolates regular grid onto input lon/lat using scipy RegularGridInterpolator
    This more accurate than sampling the points onto the grid with gdallocationinfo,
    as that one uses nearest neighbor interpolation
    '''
    if sl_grid_extents is not None:
        lon_min_sl_grid,lon_max_sl_grid,lat_min_sl_grid,lat_max_sl_grid = [float(e) for e in sl_grid_extents]
        sl_grid_subset_file = f'{tmp_dir}{loc_name}_{os.path.basename(sl_grid_file).replace(".tif","_subset.tif")}'
        clip_sl_grid_command = f'gdal_translate -q -projwin {lon_min_sl_grid} {lat_max_sl_grid} {lon_max_sl_grid} {lat_min_sl_grid} {sl_grid_file} {sl_grid_subset_file} -co "COMPRESS=LZW"'
        subprocess.run(clip_sl_grid_command,shell=True)
        src_sl_grid = gdal.Open(sl_grid_subset_file,gdalconst.GA_ReadOnly)
    else:
        src_sl_grid = gdal.Open(sl_grid_file,gdalconst.GA_ReadOnly)
    sl_grid = np.array(src_sl_grid.GetRasterBand(1).ReadAsArray())
    lon_sl_grid_array = np.linspace(src_sl_grid.GetGeoTransform()[0] + 0.5*src_sl_grid.GetGeoTransform()[1], src_sl_grid.GetGeoTransform()[0] + src_sl_grid.RasterXSize * src_sl_grid.GetGeoTransform()[1] - 0.5*src_sl_grid.GetGeoTransform()[1], src_sl_grid.RasterXSize)
    lat_sl_grid_array = np.linspace(src_sl_grid.GetGeoTransform()[3] + 0.5*src_sl_grid.GetGeoTransform()[5], src_sl_grid.GetGeoTransform()[3] + src_sl_grid.RasterYSize * src_sl_grid.GetGeoTransform()[5] - 0.5*src_sl_grid.GetGeoTransform()[5], src_sl_grid.RasterYSize)
    interp_func = RegularGridInterpolator((lon_sl_grid_array,lat_sl_grid_array[::-1]),np.flipud(sl_grid).T,bounds_error=False,fill_value=GRID_NODATA)
    z_interp = interp_func((lon_input,lat_input))
    return z_interp


def kriging_inundation(x_input,y_input,h_input,x_coast,y_coast,kriging_method='ordinary',variogram='linear'):
    x_coast_pts = x_coast[~np.isnan(x_coast)]
    y_coast_pts = y_coast[~np.isnan(y_coast)]
    idx_nan = np.argwhere(np.isnan(x_coast)).squeeze()

    if kriging_method == 'universal':
        UK = UniversalKriging(
            x_input,
            y_input,
            h_input,
            variogram_model=variogram,
            verbose=False,
            enable_plotting=False,
            drift_terms=['regional_linear'],
        )
        h_krig, var_krig = UK.execute("points", x_coast_pts,y_coast_pts)
    elif kriging_method == 'ordinary':
        OK = OrdinaryKriging(
            x_input,
            y_input,
            h_input,
            variogram_model=variogram,
            verbose=False,
            enable_plotting=False,
        )
        h_krig, var_krig = OK.execute("points", x_coast_pts,y_coast_pts)

    h_krig = h_krig.data
    var_krig = var_krig.data
    for idx in idx_nan:
        h_krig = np.concatenate((h_krig[idx:],[np.nan],h_krig[:idx]))
        var_krig = np.concatenate((var_krig[idx:],[np.nan],var_krig[:idx]))
    
    return h_krig,var_krig

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