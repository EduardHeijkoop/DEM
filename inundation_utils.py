import numpy as np
import pandas as pd
import geopandas as gpd
import netCDF4 as nc
from osgeo import gdal,gdalconst,osr
from dem_utils import get_lonlat_gdf, get_raster_extents,great_circle_distance,lonlat2epsg,deg2utm
from dem_utils import utm2epsg, landmask_dem,resample_raster
import os,sys
import xml.etree.ElementTree as ET
import subprocess
import multiprocessing
import itertools
import datetime

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

def upscale_ar6_data(AR6_dir,tmp_dir,landmask_c_file,raster,ssp,shp_file,t_select,quantile_select=0.5,conf_level='medium',upscaling_factor=10,search_radius=3.0):
    lon_min,lon_max,lat_min,lat_max = get_raster_extents(raster,'global')
    if not isinstance(ssp,str):
        ssp = f'{ssp:.2f}'.replace('.','')
    AR6_file = f'{AR6_dir}Regional/{conf_level}_confidence/ssp{ssp}/total_ssp{ssp}_{conf_level}_confidence_values.nc'
    if np.mod(t_select,10) != 0:
        print('Invalid time selected, must be ')
        return None
    if not os.path.isfile(AR6_file):
        print('No valid AR6 projection file found.')
        return None
    tmp_shp = f'{tmp_dir}tmp_shp.shp'
    shp_subset_command = f'ogr2ogr {tmp_shp} {shp_file} -f "ESRI Shapefile" -clipsrc {lon_min-search_radius} {lat_min-search_radius} {lon_max+search_radius} {lat_max+search_radius}'
    subprocess.run(shp_subset_command,shell=True)
    gdf_coast_large = gpd.read_file(tmp_shp)
    lon_coast_large,lat_coast_large = get_lonlat_gdf(gdf_coast_large)
    AR6_data = nc.Dataset(AR6_file)
    lon_AR6 = np.asarray(AR6_data['lon'][:])
    lat_AR6 = np.asarray(AR6_data['lat'][:])
    t_AR6 = np.asarray(AR6_data['years'][:])
    sl_change_AR6 = np.asarray(AR6_data['sea_level_change'][:])
    sl_change_AR6 = sl_change_AR6/1000
    quantiles_AR6 = np.asarray(AR6_data['quantiles'][:])

    idx_quantile = np.argwhere(quantiles_AR6==quantile_select).squeeze()
    idx_no_tg = np.arange(1030,len(lon_AR6))
    idx_t_select = np.atleast_1d(np.argwhere(t_AR6==t_select).squeeze())
    lon_AR6 = lon_AR6[idx_no_tg]
    lat_AR6 = lat_AR6[idx_no_tg]
    sl_change_AR6 = sl_change_AR6[idx_quantile,idx_t_select,idx_no_tg]
    idx_invalid = sl_change_AR6 == -32.768 #because divide by 1000
    lon_AR6 = lon_AR6[~idx_invalid]
    lat_AR6 = lat_AR6[~idx_invalid]
    sl_change_AR6 = sl_change_AR6[~idx_invalid]

    idx_lon = np.logical_and(lon_AR6>=lon_min-search_radius,lon_AR6<=lon_max+search_radius)
    idx_lat = np.logical_and(lat_AR6>=lat_min-search_radius,lat_AR6<=lat_max+search_radius)
    idx_tot = np.logical_and(idx_lon,idx_lat)
    lon_AR6_search = lon_AR6[idx_tot]
    lat_AR6_search = lat_AR6[idx_tot]
    sl_change_AR6_search = sl_change_AR6[idx_tot]

    landmask = landmask_dem(lon_AR6_search,lat_AR6_search,lon_coast_large,lat_coast_large,landmask_c_file,0)
    lon_AR6_search = lon_AR6_search[landmask]
    lat_AR6_search = lat_AR6_search[landmask]
    sl_change_AR6_search = sl_change_AR6_search[landmask]

    lon_AR6_high_res = np.linspace(np.min(lon_AR6_search),np.max(lon_AR6_search),int((np.max(lon_AR6_search)-np.min(lon_AR6_search))/(1/upscaling_factor))+1)
    lat_AR6_high_res = np.linspace(np.min(lat_AR6_search),np.max(lat_AR6_search),int((np.max(lat_AR6_search)-np.min(lat_AR6_search))/(1/upscaling_factor))+1)
    interp_func = LinearNDInterpolator(list(zip(lon_AR6_search, lat_AR6_search)), sl_change_AR6_search)
    lon_AR6_high_res_meshgrid,lat_AR6_high_res_meshgrid = np.meshgrid(lon_AR6_high_res,lat_AR6_high_res)
    slr_high_res = interp_func(lon_AR6_high_res_meshgrid,lat_AR6_high_res_meshgrid)
    lon_pts = np.reshape(lon_AR6_high_res_meshgrid,lon_AR6_high_res_meshgrid.shape[0]*lon_AR6_high_res_meshgrid.shape[1])
    lat_pts = np.reshape(lat_AR6_high_res_meshgrid,lat_AR6_high_res_meshgrid.shape[0]*lat_AR6_high_res_meshgrid.shape[1])
    slr_pts = np.reshape(slr_high_res,slr_high_res.shape[0]*slr_high_res.shape[1])
    idx_has_ne = np.asarray([np.sum(np.all((ln - lon_AR6_search >= 0,ln - lon_AR6_search < 1,lt - lat_AR6_search <= 0,lt - lat_AR6_search > -1),axis=0)) > 0 for (ln,lt) in zip(lon_pts,lat_pts)])
    idx_has_nw = np.asarray([np.sum(np.all((ln - lon_AR6_search <= 0,ln - lon_AR6_search > -1,lt - lat_AR6_search <= 0,lt - lat_AR6_search > -1),axis=0)) > 0 for (ln,lt) in zip(lon_pts,lat_pts)])
    idx_has_se = np.asarray([np.sum(np.all((ln - lon_AR6_search >= 0,ln - lon_AR6_search < 1,lt - lat_AR6_search >= 0,lt - lat_AR6_search < 1),axis=0)) > 0 for (ln,lt) in zip(lon_pts,lat_pts)])
    idx_has_sw = np.asarray([np.sum(np.all((ln - lon_AR6_search <= 0,ln - lon_AR6_search > -1,lt - lat_AR6_search >= 0,lt - lat_AR6_search < 1),axis=0)) > 0 for (ln,lt) in zip(lon_pts,lat_pts)])

    idx_contained = np.all((idx_has_ne,idx_has_nw,idx_has_se,idx_has_sw),axis=0)
    lon_pts = lon_pts[idx_contained]
    lat_pts = lat_pts[idx_contained]
    slr_pts = slr_pts[idx_contained]
    subprocess.run(f'rm {tmp_shp.replace(".shp",".*")}',shell=True)
    return lon_pts,lat_pts,slr_pts


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
    idx_has_ne = np.asarray([np.sum(np.all((ln - lon_SROCC_search >= 0,ln - lon_SROCC_search < 1,lt - lat_SROCC_search <= 0,lt - lat_SROCC_search > -1),axis=0)) > 0 for (ln,lt) in zip(lon_pts,lat_pts)])
    idx_has_nw = np.asarray([np.sum(np.all((ln - lon_SROCC_search <= 0,ln - lon_SROCC_search > -1,lt - lat_SROCC_search <= 0,lt - lat_SROCC_search > -1),axis=0)) > 0 for (ln,lt) in zip(lon_pts,lat_pts)])
    idx_has_se = np.asarray([np.sum(np.all((ln - lon_SROCC_search >= 0,ln - lon_SROCC_search < 1,lt - lat_SROCC_search >= 0,lt - lat_SROCC_search < 1),axis=0)) > 0 for (ln,lt) in zip(lon_pts,lat_pts)])
    idx_has_sw = np.asarray([np.sum(np.all((ln - lon_SROCC_search <= 0,ln - lon_SROCC_search > -1,lt - lat_SROCC_search >= 0,lt - lat_SROCC_search < 1),axis=0)) > 0 for (ln,lt) in zip(lon_pts,lat_pts)])
    idx_contained = np.all((idx_has_ne,idx_has_nw,idx_has_se,idx_has_sw),axis=0)
    lon_pts = lon_pts[idx_contained]
    lat_pts = lat_pts[idx_contained]
    slr_pts = slr_pts[idx_contained]
    return lon_pts,lat_pts,slr_pts

def resample_vlm(vlm_file,raster,clip_vlm_flag,vlm_nodata):
    t_start = datetime.datetime.now()
    print('Resampling VLM...')
    lon_vlm_min,lon_vlm_max,lat_vlm_min,lat_vlm_max = get_raster_extents(vlm_file,'global')
    src_vlm = gdal.Open(vlm_file,gdalconst.GA_ReadOnly)
    epsg_vlm_file = osr.SpatialReference(wkt=src_vlm.GetProjection()).GetAttrValue('AUTHORITY',1)
    if clip_vlm_flag == True:
        raster_clipped_to_vlm_file = raster.replace('.tif','_clipped_to_vlm.tif')
        clip_dem_to_vlm_command = f'gdal_translate -q -projwin {lon_vlm_min} {lat_vlm_max} {lon_vlm_max} {lat_vlm_min} -projwin_srs EPSG:{epsg_vlm_file} {raster} {raster_clipped_to_vlm_file} -co "COMPRESS=LZW"'
        subprocess.run(clip_dem_to_vlm_command,shell=True)
        raster = raster_clipped_to_vlm_file
        # src = gdal.Open(raster,gdalconst.GA_ReadOnly)
    vlm_resampled_file = vlm_file.replace('.tif',f'_resampled.tif')
    resample_raster(vlm_file,raster,vlm_resampled_file,nodata=vlm_nodata,quiet_flag=True)
    unset_nodata_command = f'gdal_edit.py -unsetnodata {vlm_resampled_file}'
    subprocess.run(unset_nodata_command,shell=True)
    t_end = datetime.datetime.now()
    delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
    delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
    print(f'Resampling VLM took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')
    return raster,vlm_resampled_file

def resample_geoid(geoid_file,raster,loc_name,raster_nodata):
    t_start = datetime.datetime.now()
    print('Resampling geoid...')
    geoid_name = geoid_file.split('/')[-1].split('.')[0]
    geoid_resampled_file = f'{os.path.dirname(os.path.abspath(raster))}/{loc_name}_{geoid_name}_resampled.tif'
    resample_raster(geoid_file,raster,geoid_resampled_file,nodata=None,quiet_flag=True)
    raster_orthometric = raster.replace('.tif','_orthometric.tif')
    orthometric_command = f'gdal_calc.py --quiet -A {raster} -B {geoid_resampled_file} --outfile={raster_orthometric} --calc="A-B" --NoDataValue={raster_nodata} --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --co "TILED=YES"'
    subprocess.run(orthometric_command,shell=True)
    raster = raster_orthometric
    src = gdal.Open(raster,gdalconst.GA_ReadOnly)
    t_end = datetime.datetime.now()
    delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
    delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
    print(f'Resampling geoid took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')
    return raster

def clip_coast(raster,coastline_file,epsg_code,grid_nodata):
    t_start = datetime.datetime.now()
    print('Clipping DEM to coastline...')
    raster_clipped = raster.replace('.tif','_clipped.tif')
    clip_command = f'gdalwarp -q -s_srs EPSG:{epsg_code} -t_srs EPSG:{epsg_code} -of GTiff -cutline {coastline_file} -cl {os.path.splitext(os.path.basename(coastline_file))[0]} -dstnodata {grid_nodata} {raster} {raster_clipped} -overwrite -co "COMPRESS=LZW" -co "BIGTIFF=IF_SAFER" -co "TILED=YES"'
    subprocess.run(clip_command,shell=True)
    raster = raster_clipped
    src = gdal.Open(raster,gdalconst.GA_ReadOnly)
    t_end = datetime.datetime.now()
    delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
    delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
    print(f'Clipping took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')
    return raster



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

def interpolate_grid(lon_input,lat_input,grid_file,grid_extents,loc_name,tmp_dir,grid_nodata=-9999,method='linear'):
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
    interp_func = RegularGridInterpolator((lon_grid_array,lat_grid_array[::-1]),np.flipud(grid).T,bounds_error=False,fill_value=grid_nodata,method=method)
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

def get_fes(lon,lat,fes2014_file,search_radius=3.0,mhhw_flag=False):
    lon = lon[~np.isnan(lon)]
    lat = lat[~np.isnan(lat)]
    df_fes = pd.read_csv(fes2014_file)
    lon[lon<0] += 360
    lon_fes = np.asarray(df_fes['lon'])
    lat_fes = np.asarray(df_fes['lat'])
    if mhhw_flag == True:
        max_tide_fes = np.asarray(df_fes['MHHW'])
    else:
        max_tide_fes = np.asarray(df_fes['tide_max'])
    idx_fes_lon_close = np.logical_and(lon_fes > np.min(lon) - search_radius,lon_fes < np.max(lon) + search_radius)
    idx_fes_lat_close = np.logical_and(lat_fes > np.min(lat) - search_radius,lat_fes < np.max(lat) + search_radius)
    idx_fes_lonlat_close = np.logical_and(idx_fes_lon_close,idx_fes_lat_close)
    if np.sum(idx_fes_lonlat_close) == 0:
        raise ValueError('No FES points found within search radius.')
    lon_fes = lon_fes[idx_fes_lonlat_close]
    lat_fes = lat_fes[idx_fes_lonlat_close]
    max_tide_fes = max_tide_fes[idx_fes_lonlat_close]
    height_fes_coast = interpolate_points(lon_fes,lat_fes,max_tide_fes,lon,lat,'Smooth',3)
    return height_fes_coast

def csv_to_grid(csv_file,algorithm_dict,raster_dict,epsg_code):
    '''
    Turn a csv file of x/y/z into a grid with gdal_grid.
    X_MIN X_MAX Y_MAX Y_MIN is correct to get pixel size to N, -N
    This is because the origin is top-left (i.e. Northwest)
    '''
    xmin = raster_dict['xmin']
    xmax = raster_dict['xmax']
    xres = raster_dict['xres']
    ymin = raster_dict['ymin']
    ymax = raster_dict['ymax']
    yres = raster_dict['yres']
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

def compute_connectivity(inundation_vec_file,gdf_surface_water):
    '''
    Compute connectivty of inundation to the sea (defined as the gdf_surface_water)
    Surface water should be buffered already, if desired
    '''
    gdf_inundation = gpd.read_file(inundation_vec_file)
    inundation_vec_file_base,inundation_vec_file_ext = os.path.splitext(inundation_vec_file)
    inundation_vec_file_connected = f'{inundation_vec_file_base}_connectivity{inundation_vec_file_ext}'
    if len(gdf_surface_water) == 1:
        idx_intersects = np.asarray([gdf_surface_water.geometry[0].intersects(geom) for geom in gdf_inundation.geometry])
        idx_contains = np.asarray([gdf_surface_water.geometry[0].contains(geom) for geom in gdf_inundation.geometry])
        idx_connected = np.any((idx_intersects,idx_contains),axis=0)
    else:
        idx_intersects = np.zeros((len(gdf_surface_water),len(gdf_inundation)),dtype=bool)
        idx_contains = np.zeros((len(gdf_surface_water),len(gdf_inundation)),dtype=bool)
        for i,gsw_geom in enumerate(gdf_surface_water.geometry):
            idx_intersects[i,:] = np.asarray([gsw_geom.intersects(geom) for geom in gdf_inundation.geometry])
            idx_contains[i,:] = np.asarray([gsw_geom.contains(geom) for geom in gdf_inundation.geometry])
        idx_intersects = np.any(idx_intersects,axis=0)
        idx_contains = np.any(idx_contains,axis=0)
        idx_connected = np.any((idx_intersects,idx_contains),axis=0)
    gdf_inundation_connected = gdf_inundation[idx_connected].reset_index(drop=True)
    gdf_inundation_connected.to_file(inundation_vec_file_connected)

def quantile_to_sigma(quantile):
    if quantile == 0.16 or quantile == 0.84:
        sigma = 1
    elif quantile == 0.02 or quantile == 0.98:
        sigma = 2
    elif quantile == 0.001 or quantile == 0.999:
        sigma = 3
    else:
        sigma = None
    return sigma

def sigma_to_quantiles(sigma,uncertainty_flag):
    if sigma is None:
        quantiles = [0.5]
    elif uncertainty_flag == True:
        if sigma == 1:
            quantiles = [0.16,0.5,0.84]
        elif sigma == 2:
            quantiles = [0.02,0.5,0.98]
        elif sigma == 3:
            quantiles = [0.001,0.5,0.999]
        else:
            quantiles = [0.5]
    else:
        quantiles = [0.5]
    return quantiles

def get_coastal_sealevel(loc_name,x_coast,y_coast,lon_coast,lat_coast,sl_grid_extents,sl_grid_file,icesat2_file,dir_dict,constants_dict,epsg_code,geoid_file):
    tmp_dir = dir_dict['tmp_dir']
    GRID_NODATA = constants_dict['GRID_NODATA']
    REGGRID_INTERPOLATE_METHOD = constants_dict['REGGRID_INTERPOLATE_METHOD']
    INTERPOLATE_METHOD = constants_dict['INTERPOLATE_METHOD']
    ICESAT2_GRID_RESOLUTION = constants_dict['ICESAT2_GRID_RESOLUTION']
    N_PTS = constants_dict['N_PTS']
    t_start = datetime.datetime.now()
    print('Generating coastal sea level grid...')
    if sl_grid_extents is not None:
        h_coast = interpolate_grid(lon_coast,lat_coast,sl_grid_file,sl_grid_extents,loc_name,tmp_dir,grid_nodata=GRID_NODATA,method=REGGRID_INTERPOLATE_METHOD)
        idx_fillvalue = h_coast==GRID_NODATA
        idx_keep = ~np.logical_or(idx_fillvalue,np.isnan(h_coast))
        x_coast = x_coast[idx_keep]
        y_coast = y_coast[idx_keep]
        h_coast = h_coast[idx_keep]
        lon_coast = lon_coast[idx_keep]
        lat_coast = lat_coast[idx_keep]
        if loc_name in sl_grid_file:
            output_file_coastline = f'{tmp_dir}{os.path.basename(sl_grid_file).replace(".tif","_subset_interpolated_coastline.csv")}'
        else:
            output_file_coastline = f'{tmp_dir}{loc_name}_{os.path.basename(sl_grid_file).replace(".tif","_subset_interpolated_coastline.csv")}'
        if geoid_file is not None:
            h_geoid = interpolate_grid(lon_coast,lat_coast,geoid_file,None,loc_name,tmp_dir,grid_nodata=GRID_NODATA,method=REGGRID_INTERPOLATE_METHOD)
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
    return x_coast,y_coast,lon_coast,lat_coast,h_coast,output_file_coastline

def get_sealevel_high(raster,high_tide,return_period,fes2014_flag,mhhw_flag,loc_name,epsg_code,
                      x_coast,y_coast,lon_coast,lat_coast,
                      dir_dict,constants_dict,dem_dict,algorithm_dict,resampled_dict):
    src_resampled = resampled_dict['src_resampled']
    dem_resampled_x_size = resampled_dict['dem_resampled_x_size']
    dem_resampled_y_size = resampled_dict['dem_resampled_y_size']
    src_resampled_geotransform = src_resampled.GetGeoTransform()
    src_resampled_proj = src_resampled.GetProjection()
    tmp_dir = dir_dict['tmp_dir']
    CODEC_file = constants_dict['CODEC_file']
    fes2014_file = constants_dict['fes2014_file']
    GRID_INTERMEDIATE_RES = constants_dict['GRID_INTERMEDIATE_RES']
    if high_tide is not None:
        '''
        Implement manual high tide value here
        Create a grid of high tide values at DEM resampled resolution and resample to full res
        '''
        t_start = datetime.datetime.now()
        print(f'Generating high tide file with a value of {high_tide:.2f} m...')
        high_tide_array = np.ones((dem_resampled_y_size,dem_resampled_x_size))*high_tide
        high_tide_full_res = f'{tmp_dir}{loc_name}_high_tide.tif'
        high_tide_intermediate_res = f'{tmp_dir}{loc_name}_high_tide_resampled.tif'
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(high_tide_intermediate_res,high_tide_array.shape[1],high_tide_array.shape[0],1,gdal.GDT_Float32)
        dataset.SetGeoTransform(src_resampled_geotransform)
        dataset.SetProjection(src_resampled_proj)
        dataset.GetRasterBand(1).WriteArray(high_tide_array)
        dataset.FlushCache()
        dataset = None
        resample_raster(high_tide_intermediate_res,raster,high_tide_full_res,quiet_flag=True)
        t_end = datetime.datetime.now()
        delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
        delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
        print(f'Generating high tide file took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')
        sealevel_csv_output = f'{tmp_dir}tmp.txt'
        sealevel_high_grid_intermediate_res = high_tide_intermediate_res
        sealevel_high_grid_full_res = high_tide_full_res
    elif return_period is not None:
        t_start = datetime.datetime.now()
        print(f'Finding CoDEC sea level extremes for return period of {return_period} years...')
        rps_coast = get_codec(lon_coast,lat_coast,CODEC_file,return_period)
        output_file_codec = f'{tmp_dir}{loc_name}_CoDEC_{return_period}_yrs_coastline.csv'
        np.savetxt(output_file_codec,np.c_[x_coast,y_coast,rps_coast],fmt='%f',delimiter=',',comments='')
        codec_grid_intermediate_res = csv_to_grid(output_file_codec,algorithm_dict,dem_dict,epsg_code)
        codec_grid_full_res = codec_grid_intermediate_res.replace(f'_{GRID_INTERMEDIATE_RES}m','')
        resample_raster(codec_grid_intermediate_res,raster,codec_grid_full_res,quiet_flag=True)
        t_end = datetime.datetime.now()
        delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
        delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
        print(f'Generating CoDEC sea level extremes took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')
        sealevel_csv_output = output_file_codec
        sealevel_high_grid_intermediate_res = codec_grid_intermediate_res
        sealevel_high_grid_full_res = codec_grid_full_res
    elif fes2014_flag == True:
        t_start = datetime.datetime.now()
        if mhhw_flag == False:
            print(f'Finding FES2014 max tidal heights...')
        else:
            print(f'Finding FES2014 MHHW values...')
        fes_heights_coast = get_fes(lon_coast,lat_coast,fes2014_file,mhhw_flag=mhhw_flag)
        output_file_fes = f'{tmp_dir}{loc_name}_FES2014_coastline.csv'
        np.savetxt(output_file_fes,np.c_[x_coast,y_coast,fes_heights_coast],fmt='%f',delimiter=',',comments='')
        fes_grid_intermediate_res = csv_to_grid(output_file_fes,algorithm_dict,dem_dict,epsg_code)
        fes_grid_full_res = fes_grid_intermediate_res.replace(f'_{GRID_INTERMEDIATE_RES}m','')
        resample_raster(fes_grid_intermediate_res,raster,fes_grid_full_res,quiet_flag=True)
        t_end = datetime.datetime.now()
        delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
        delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
        print(f'Generating FES2014 values took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')
        sealevel_csv_output = output_file_fes
        sealevel_high_grid_intermediate_res = fes_grid_intermediate_res
        sealevel_high_grid_full_res = fes_grid_full_res
    return sealevel_high_grid_full_res


def inundate_loc(raster,slr,years,quantiles,loc_name,high_tide,ssp,
                 x_coast,y_coast,h_coast,
                 dir_dict,flag_dict,constants_dict,dem_dict,algorithm_dict,vlm_dict,
                 output_file_coastline,epsg_code,gdf_surface_water,sealevel_high_grid_full_res,N_cpus):
    ir = itertools.repeat
    ip = itertools.product
    if slr is not None:
        p = multiprocessing.Pool(np.min((len(slr),N_cpus)))
        p.starmap(parallel_inundation_slr,zip(
            slr,ir(raster),ir(loc_name),ir(high_tide),ir(x_coast),ir(y_coast),ir(h_coast),
            ir(dir_dict),ir(flag_dict),ir(constants_dict),ir(dem_dict),ir(algorithm_dict),
            ir(output_file_coastline),ir(epsg_code),ir(gdf_surface_water),ir(sealevel_high_grid_full_res)
        ))
    else:
        p = multiprocessing.Pool(np.min((len(quantiles)*len(years),N_cpus)))
        p.starmap(parallel_inundation_ar6,zip(
            ip(years,quantiles),ir(ssp),ir(raster),ir(loc_name),ir(x_coast),ir(y_coast),ir(h_coast),
            ir(dir_dict),ir(flag_dict),ir(constants_dict),ir(dem_dict),ir(algorithm_dict),ir(vlm_dict),
            ir(output_file_coastline),ir(epsg_code),ir(gdf_surface_water),ir(sealevel_high_grid_full_res)
            ))
    p.close()


def parallel_inundation_slr(slr_value,raster,loc_name,x_coast,y_coast,h_coast,
                            dir_dict,flag_dict,constants_dict,dem_dict,algorithm_dict,
                            output_file_coastline,epsg_code,gdf_surface_water,sealevel_high_grid_full_res):
    inundation_dir = dir_dict['inundation_dir']
    return_period = flag_dict['return_period']
    fes2014_flag = flag_dict['fes2014']
    mhhw_flag = flag_dict['mhhw']
    connectivity_flag = flag_dict['connectivity']
    geoid_file = flag_dict['geoid']
    high_tide = flag_dict['high_tide']
    GRID_INTERMEDIATE_RES = constants_dict['GRID_INTERMEDIATE_RES']
    INUNDATION_NODATA = constants_dict['INUNDATION_NODATA']
    output_format = constants_dict['output_format']
    t_start = datetime.datetime.now()
    print(f'\nCreating inundation for {slr_value:.2f} m...')
    slr_value_str = f'SLR_{slr_value:.2f}m'.replace('.','p').replace('-','neg')
    if high_tide is not None:
        high_tide_str = f'{high_tide:.2f}m'.replace('.','p')
        output_inundation_file = f'{inundation_dir}{loc_name}_Inundation_{slr_value_str}_HT_{high_tide_str}.tif'
    elif return_period is not None:
        output_inundation_file = f'{inundation_dir}{loc_name}_Inundation_{slr_value_str}_CoDEC_RP_{return_period}_yrs.tif'
    elif fes2014_flag == True:
        output_inundation_file = f'{inundation_dir}{loc_name}_Inundation_{slr_value_str}_FES2014.tif'
        if mhhw_flag == True:
            output_inundation_file = output_inundation_file.replace('FES2014','FES2014_MHHW')
    if geoid_file is not None:
        output_inundation_file = output_inundation_file.replace('_Inundation_','_Orthometric_Inundation_')
    h_coast_slr = h_coast + slr_value
    output_file_coastline_slr = output_file_coastline.replace('.csv',f'_{slr_value_str}.csv')
    np.savetxt(output_file_coastline_slr,np.c_[x_coast,y_coast,h_coast_slr],fmt='%f',delimiter=',',comments='')
    sl_grid_file_intermediate_res = csv_to_grid(output_file_coastline_slr,algorithm_dict,dem_dict,epsg_code)
    sl_grid_file_full_res = sl_grid_file_intermediate_res.replace(f'_{GRID_INTERMEDIATE_RES}m','')
    resample_raster(sl_grid_file_intermediate_res,raster,sl_grid_file_full_res,quiet_flag=True)
    inundation_command = f'gdal_calc.py --quiet -A {raster} -C {sl_grid_file_full_res} -D {sealevel_high_grid_full_res} --outfile={output_inundation_file} --calc="A < C+D" --NoDataValue={INUNDATION_NODATA} --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --co "TILED=YES"'
    subprocess.run(inundation_command,shell=True)
    output_inundation_vec_file = output_inundation_file.replace('.tif',f'.{output_format}')
    polygonize_command = f'gdal_polygonize.py -q {output_inundation_file} {output_inundation_vec_file}'
    subprocess.run(polygonize_command,shell=True)
    t_end = datetime.datetime.now()
    delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
    delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
    print(f'Inundation creation took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')
    if connectivity_flag == True:
        print('Computing connectivity to the ocean...')
        t_start = datetime.datetime.now()
        compute_connectivity(output_inundation_vec_file,gdf_surface_water)
        t_end = datetime.datetime.now()
        delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
        delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
        print(f'Connectivity took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')
    subprocess.run(f'rm {output_file_coastline_slr}',shell=True)
    subprocess.run(f'rm {output_file_coastline_slr.replace(".csv",".vrt")}',shell=True)
    subprocess.run(f'rm {sl_grid_file_intermediate_res}',shell=True)
    subprocess.run(f'rm {sl_grid_file_full_res}',shell=True)


def parallel_inundation_ar6(year,quantile,ssp,raster,loc_name,x_coast,y_coast,h_coast,
                            dir_dict,flag_dict,constants_dict,dem_dict,algorithm_dict,vlm_dict,
                            output_file_coastline,epsg_code,gdf_surface_water,sealevel_high_grid_full_res):
    '''
    given raster, upscale AR6 data points with year and quantile_select
    interpolate onto coastal sea level points, and turn into grid
    '''
    tmp_dir = dir_dict['tmp_dir']
    inundation_dir = dir_dict['inundation_dir']
    AR6_dir = dir_dict['AR6_dir']
    return_period = flag_dict['return_period']
    fes2014_flag = flag_dict['fes2014']
    mhhw_flag = flag_dict['mhhw']
    connectivity_flag = flag_dict['connectivity']
    geoid_file = flag_dict['geoid']
    high_tide = flag_dict['high_tide']
    GRID_INTERMEDIATE_RES = constants_dict['GRID_INTERMEDIATE_RES']
    INUNDATION_NODATA = constants_dict['INUNDATION_NODATA']
    INTERPOLATE_METHOD = constants_dict['INTERPOLATE_METHOD']
    output_format = constants_dict['output_format']
    # projection_select = constants_dict['projection_select']
    landmask_c_file = constants_dict['landmask_c_file']
    osm_shp_file = constants_dict['osm_shp_file']
    t0 = vlm_dict['t0']
    vlm_rate = vlm_dict['vlm_rate']
    vlm_resampled_file = vlm_dict['vlm_resampled_file']
    sigma = quantile_to_sigma(quantile)

    t_start = datetime.datetime.now()
    if high_tide is not None:
        output_inundation_file = f'{inundation_dir}{loc_name}_Inundation_{year}_PROJECTION_METHOD_HT_{high_tide:.2f}.tif'.replace('.','p')
    elif return_period is not None:
        output_inundation_file = f'{inundation_dir}{loc_name}_Inundation_{year}_PROJECTION_METHOD_CoDEC_RP_{return_period}_yrs.tif'
    elif fes2014_flag == True:
        output_inundation_file = f'{inundation_dir}{loc_name}_Inundation_{year}_PROJECTION_METHOD_FES2014.tif'
        if mhhw_flag == True:
            output_inundation_file = output_inundation_file.replace('FES2014','FES2014_MHHW')
    output_inundation_file = output_inundation_file.replace('PROJECTION_METHOD',f'AR6_SSP_{ssp}')
    if quantile < 0.5:
        output_inundation_file = output_inundation_file.replace('_Inundation_',f'_Inundation_Minus_{sigma}sigma_')
        print(f'\nCreating inundation in {year} using SSP{ssp} (Median minus {sigma} sigma)...')
    elif quantile > 0.5:
        output_inundation_file = output_inundation_file.replace('_Inundation_',f'_Inundation_Plus_{sigma}sigma_')
        print(f'\nCreating inundation in {year} using SSP{ssp} (Median plus {sigma} sigma)...')
    else:
        print(f'\nCreating inundation in {year} using SSP{ssp}...')
    lon_projection,lat_projection,slr_projection = upscale_ar6_data(AR6_dir,tmp_dir,landmask_c_file,raster,ssp,osm_shp_file,year,quantile_select=quantile)
    if geoid_file is not None:
        output_inundation_file = output_inundation_file.replace('_Inundation_','_Orthometric_Inundation_')
    if vlm_resampled_file is None:
        output_inundation_file = output_inundation_file.replace('_Inundation_','_Inundation_No_VLM_')
    h_projection_coast = interpolate_points(lon_projection,lat_projection,slr_projection,x_coast,y_coast,INTERPOLATE_METHOD)
    h_coast_yr = h_coast + h_projection_coast
    output_file_coastline_yr = output_file_coastline.replace('.csv',f'_{year}.csv')
    np.savetxt(output_file_coastline_yr,np.c_[x_coast,y_coast,h_coast_yr],fmt='%f',delimiter=',',comments='')
    sl_grid_file_intermediate_res = csv_to_grid(output_file_coastline_yr,algorithm_dict,dem_dict,epsg_code)
    sl_grid_file_full_res = sl_grid_file_intermediate_res.replace(f'_{GRID_INTERMEDIATE_RES}m','')
    resample_raster(sl_grid_file_intermediate_res,raster,sl_grid_file_full_res,quiet_flag=True)
    if vlm_resampled_file is not None:
        dt = int(year - t0)
        inundation_command = f'gdal_calc.py --quiet -A {raster} -B {vlm_resampled_file} -C {sl_grid_file_full_res} -D {sealevel_high_grid_full_res} --outfile={output_inundation_file} --calc="A+B*{dt} < C+D" --NoDataValue={INUNDATION_NODATA} --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --co "TILED=YES"'
    elif vlm_rate is not None:
        inundation_command = f'gdal_calc.py --quiet -A {raster} -C {sl_grid_file_full_res} -D {sealevel_high_grid_full_res} --outfile={output_inundation_file} --calc="A+{vlm_rate}*{dt} < C+D" --NoDataValue={INUNDATION_NODATA} --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --co "TILED=YES"'
    else:
        inundation_command = f'gdal_calc.py --quiet -A {raster} -C {sl_grid_file_full_res} -D {sealevel_high_grid_full_res} --outfile={output_inundation_file} --calc="A < C+D" --NoDataValue={INUNDATION_NODATA} --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --co "TILED=YES"'
    subprocess.run(inundation_command,shell=True)
    output_inundation_vec_file = output_inundation_file.replace('.tif',f'.{output_format}')
    polygonize_command = f'gdal_polygonize.py -q {output_inundation_file} {output_inundation_vec_file}'
    subprocess.run(polygonize_command,shell=True)
    t_end = datetime.datetime.now()
    delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
    delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
    print(f'Inundation creation took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')
    if connectivity_flag == True:
        print('Computing connectivity to the ocean...')
        t_start = datetime.datetime.now()
        compute_connectivity(output_inundation_vec_file,gdf_surface_water)
        t_end = datetime.datetime.now()
        delta_time_mins = np.floor((t_end - t_start).total_seconds()/60).astype(int)
        delta_time_secs = np.mod((t_end - t_start).total_seconds(),60)
        print(f'Connectivity took {delta_time_mins} minutes, {delta_time_secs:.1f} seconds.')
    subprocess.run(f'rm {output_file_coastline_yr}',shell=True)
    subprocess.run(f'rm {output_file_coastline_yr.replace(".csv",".vrt")}',shell=True)
    subprocess.run(f'rm {sl_grid_file_intermediate_res}',shell=True)
    subprocess.run(f'rm {sl_grid_file_full_res}',shell=True)












