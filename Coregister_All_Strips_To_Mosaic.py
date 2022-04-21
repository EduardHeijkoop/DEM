import numpy as np
import pandas as pd
import geopandas as gpd
import subprocess
import glob
import os
import argparse
import ctypes as c
from osgeo import gdal,osr,gdalconst
import shapely
from itertools import compress

def get_lonlat_geometry(geom):
    '''
    Returns lon/lat of all exteriors and interiors of a Shapely geomery:
        -Polygon
        -MultiPolygon
        -GeometryCollection
    '''
    lon = np.empty([0,1],dtype=float)
    lat = np.empty([0,1],dtype=float)
    if geom.geom_type == 'Polygon':
        lon_geom,lat_geom = get_lonlat_polygon(geom)
        lon = np.append(lon,lon_geom)
        lat = np.append(lat,lat_geom)
    elif geom.geom_type == 'MultiPolygon':
        polygon_list = [p for p in geom.geoms if p.geom_type == 'Polygon']
        for polygon in polygon_list:
            lon_geom,lat_geom = get_lonlat_polygon(polygon)
            lon = np.append(lon,lon_geom)
            lat = np.append(lat,lat_geom)
    elif geom.geom_type == 'GeometryCollection':
        polygon_list = [p for p in geom.geoms if p.geom_type == 'Polygon']
        for polygon in polygon_list:
            lon_geom,lat_geom = get_lonlat_polygon(polygon)
            lon = np.append(lon,lon_geom)
            lat = np.append(lat,lat_geom)
    return lon,lat

def get_lonlat_polygon(polygon):
    lon = np.empty([0,1],dtype=float)
    lat = np.empty([0,1],dtype=float)
    exterior_xy = np.asarray(polygon.exterior.xy)
    lon = np.append(lon,exterior_xy[0,:])
    lon = np.append(lon,np.nan)
    lat = np.append(lat,exterior_xy[1,:])
    lat = np.append(lat,np.nan)
    for interior in polygon.interiors:
        interior_xy = np.asarray(interior.coords.xy)
        lon = np.append(lon,interior_xy[0,:])
        lon = np.append(lon,np.nan)
        lat = np.append(lat,interior_xy[1,:])
        lat = np.append(lat,np.nan)
    return lon,lat

def get_lonlat_gdf(gdf):
    '''
    Returns lon/lat of all exteriors and interiors of a GeoDataFrame
    '''
    lon = np.empty([0,1],dtype=float)
    lat = np.empty([0,1],dtype=float)
    for geom in gdf.geometry:
        lon_geom,lat_geom = get_lonlat_geometry(geom)
        lon = np.append(lon,lon_geom)
        lat = np.append(lat,lat_geom)
    return lon,lat

def get_strip_shp(strip,tmp_dir):
    if subprocess.os.path.exists(tmp_dir + 'tmp_strip_binary.tif'):
        subprocess.os.remove(tmp_dir + 'tmp_strip_binary.tif')
    if subprocess.os.path.exists(tmp_dir + 'tmp_strip_binary.shp'):
        for f in glob.glob(tmp_dir + 'tmp_strip_binary.*'):
            subprocess.os.remove(f)
    calc_command = 'gdal_calc.py -A ' + strip + ' --calc="A>-9999" --outfile=' + tmp_dir + 'tmp_strip_binary.tif --format=GTiff --co="COMPRESS=LZW" --quiet'
    subprocess.run(calc_command,shell=True)
    polygonize_command = 'gdal_polygonize.py -q ' + tmp_dir + 'tmp_strip_binary.tif -f "ESRI Shapefile" ' + tmp_dir + 'tmp_strip_binary.shp'
    subprocess.run(polygonize_command,shell=True)
    wv_strip_shp = gpd.read_file(tmp_dir + 'tmp_strip_binary.shp')
    if subprocess.os.path.exists(tmp_dir + 'tmp_strip_binary.tif'):
        subprocess.os.remove(tmp_dir + 'tmp_strip_binary.tif')
    if subprocess.os.path.exists(tmp_dir + 'tmp_strip_binary.shp'):
        for f in glob.glob(tmp_dir + 'tmp_strip_binary.*'):
            subprocess.os.remove(f)
    return wv_strip_shp

def epsg2proj4(epsg_code):
    epsg_code = str(epsg_code) #forces string, if input is int for example
    zone = epsg_code[3:5]
    if epsg_code[2] == '6':
        north_south = 'north'
    elif epsg_code[2] == '7':
        north_south = 'south'
    proj4 = f'+proj=utm +zone={zone} +{north_south} +datum=WGS84 +units=m +no_defs'
    return proj4

def filter_strip_gsw(wv_strip_shp,gsw_shp_data,STRIP_AREA_THRESHOLD,POLYGON_AREA_THRESHOLD,GSW_OVERLAP_THRESHOLD,STRIP_TOTAL_AREA_PERCENTAGE_THRESHOLD):
    if np.sum(wv_strip_shp.geometry.area) < STRIP_AREA_THRESHOLD:
        return None
    idx_polygon_area = wv_strip_shp.area > POLYGON_AREA_THRESHOLD
    wv_strip_shp_filtered_gsw = wv_strip_shp[idx_polygon_area].reset_index(drop=True)
    if len(gsw_shp_data) == 1:
        gsw_wv_joined_contains = gpd.sjoin(gsw_shp_data,wv_strip_shp_filtered_gsw,how='right',predicate='contains')
        idx_gsw_wv_contains = np.asarray(np.isnan(gsw_wv_joined_contains.DN_left).sort_index())
    elif len(gsw_shp_data) > 1:
        idx_gsw_wv_contains = np.zeros((len(gsw_shp_data),len(wv_strip_shp_filtered_gsw)),dtype=bool)
        for i in range(len(gsw_shp_data)):
            gsw_wv_joined_contains = gpd.sjoin(gsw_shp_data.iloc[[i]],wv_strip_shp_filtered_gsw,how='right',predicate='contains')
            idx_gsw_wv_contains[i,:] = np.asarray(np.isnan(gsw_wv_joined_contains.DN_left).sort_index())
        idx_gsw_wv_contains = np.any(idx_gsw_wv_contains,axis=0)
    if np.sum(idx_gsw_wv_contains) == 0:
        return None
    #Note: this will return a True if a polygon is NOT fully contained by the GSW dataset
    wv_strip_shp_filtered_gsw = wv_strip_shp_filtered_gsw.iloc[idx_gsw_wv_contains].reset_index(drop=True)

    if len(gsw_shp_data) == 1:
        gsw_wv_joined_intersects = gpd.sjoin(gsw_shp_data,wv_strip_shp_filtered_gsw,how='right',predicate='intersects')
        idx_no_intersect = np.asarray(np.isnan(gsw_wv_joined_intersects.DN_left))
        idx_some_intersect = np.asarray(gsw_wv_joined_intersects.DN_left==1)
        IDs_some_intersect = [j for j, x in enumerate(idx_some_intersect) if x]
        gsw_wv_overlay_intersection = gpd.overlay(wv_strip_shp_filtered_gsw[idx_some_intersect],gsw_shp_data,how='intersection')
        idx_overlay_intersect_threshold = gsw_wv_overlay_intersection.area/wv_strip_shp_filtered_gsw[idx_some_intersect].area.reset_index(drop=True) < GSW_OVERLAP_THRESHOLD
        idx_some_intersect[IDs_some_intersect] = idx_overlay_intersect_threshold
    elif len(gsw_shp_data) > 1:
        idx_some_intersect = np.zeros((len(gsw_shp_data),len(wv_strip_shp_filtered_gsw)),dtype=bool)
        for i in range(len(gsw_shp_data)):
            gsw_wv_joined_intersects = gpd.sjoin(gsw_shp_data.iloc[[i]],wv_strip_shp_filtered_gsw,how='right',predicate='intersects')
            idx_no_intersect = np.asarray(np.isnan(gsw_wv_joined_intersects.DN_left))
            idx_some_intersect[i,:] = np.asarray(gsw_wv_joined_intersects.DN_left==1)
            IDs_some_intersect = [j for j, x in enumerate(idx_some_intersect[i,:]) if x]
            if len(IDs_some_intersect) == 0:
                continue
            gsw_wv_overlay_intersection = gpd.overlay(wv_strip_shp_filtered_gsw[idx_some_intersect[i,:]],gsw_shp_data.iloc[[i]],how='intersection')
            idx_overlay_intersect_threshold = gsw_wv_overlay_intersection.area/wv_strip_shp_filtered_gsw[idx_some_intersect[i,:]].area.reset_index(drop=True) < GSW_OVERLAP_THRESHOLD
            idx_some_intersect[i,IDs_some_intersect] = idx_overlay_intersect_threshold
        idx_some_intersect = np.any(idx_some_intersect,axis=0)

    idx_intersection = np.logical_or(idx_no_intersect,idx_some_intersect)
    wv_strip_shp_filtered_gsw = wv_strip_shp_filtered_gsw[idx_intersection].reset_index(drop=True)
    # wv_strip_shp_filtered_gsw = wv_strip_shp_filtered_gsw.reset_index(drop=True)

    idx_total_area_percentage = np.asarray(wv_strip_shp_filtered_gsw.area/np.sum(wv_strip_shp_filtered_gsw.area) > STRIP_TOTAL_AREA_PERCENTAGE_THRESHOLD)
    wv_strip_shp_filtered_gsw = wv_strip_shp_filtered_gsw[idx_total_area_percentage].reset_index(drop=True)
    # wv_strip_shp_filtered_gsw = wv_strip_shp_filtered_gsw.reset_index(drop=True)

    wv_strip_shp_filtered_gsw = wv_strip_shp_filtered_gsw.buffer(0)
    if np.sum(idx_intersection) == 0:
        return None
    
    return wv_strip_shp_filtered_gsw

def populate_intersection(geom_intersection,gsw_main_sea_only_buffered,pnpoly_function,X_SPACING,Y_SPACING):
    '''
    Returns x and y inside a polygon
    We do the masking on each individual feature, rather than the whole to reduce the total number of points to be considered
        E.g., two polygons some distance away from each other, we first populate points around the two polygons,
        then we mask them, rather than populating the whole region and masking all of that.
    '''
    x = np.empty([0,1],dtype=float)
    y = np.empty([0,1],dtype=float)
    lon_gsw,lat_gsw = get_lonlat_gdf(gsw_main_sea_only_buffered)
    arrx_gsw = (c.c_float * len(lon_gsw))(*lon_gsw)
    arry_gsw = (c.c_float * len(lat_gsw))(*lat_gsw)

    if geom_intersection.geom_type == 'Polygon':
        geom_intersection_polygons = [geom_intersection]
    elif geom_intersection.geom_type == 'MultiPolygon':
        geom_intersection_polygons = [p for p in geom_intersection.geoms if p.geom_type == 'Polygon']
    elif geom_intersection.geom_type == 'GeometryCollection':
        geom_intersection_polygons = [p for p in geom_intersection.geoms if p.geom_type == 'Polygon']

    for feature in geom_intersection_polygons:
        x_min = feature.bounds[0]
        x_max = feature.bounds[2]
        y_min = feature.bounds[1]
        y_max = feature.bounds[3]

        x_min = X_SPACING*np.floor(x_min/X_SPACING)
        x_max = X_SPACING*np.ceil(x_max/X_SPACING)
        y_min = Y_SPACING*np.floor(y_min/Y_SPACING)
        y_max = Y_SPACING*np.ceil(y_max/Y_SPACING)

        x_sampling_range = np.arange(x_min,x_max,X_SPACING)
        y_sampling_range = np.arange(y_min,y_max,Y_SPACING)
        x_sampling_meshgrid,y_sampling_meshgrid = np.meshgrid(x_sampling_range,y_sampling_range)
        x_sampling = np.reshape(x_sampling_meshgrid,(len(x_sampling_range)*len(y_sampling_range),))
        y_sampling = np.reshape(y_sampling_meshgrid,(len(x_sampling_range)*len(y_sampling_range),))

        lon_intersection,lat_intersection = get_lonlat_geometry(feature)
        arrx_input = (c.c_float * len(x_sampling))(*x_sampling)
        arry_input = (c.c_float * len(y_sampling))(*y_sampling)
        arrx_intersection = (c.c_float * len(lon_intersection))(*lon_intersection)
        arry_intersection = (c.c_float * len(lat_intersection))(*lat_intersection)
        landmask_intersection = np.zeros(len(x_sampling),dtype=c.c_int)
        pnpoly_function.pnpoly(c.c_int(len(lon_intersection)),c.c_int(len(x_sampling)),arrx_intersection,arry_intersection,arrx_input,arry_input,c.c_void_p(landmask_intersection.ctypes.data))

        x_sampling_masked_intersection = x_sampling[landmask_intersection==1]
        y_sampling_masked_intersection = y_sampling[landmask_intersection==1]
        arrx_input_gsw = (c.c_float * len(x_sampling_masked_intersection))(*x_sampling_masked_intersection)
        arry_input_gsw = (c.c_float * len(y_sampling_masked_intersection))(*y_sampling_masked_intersection)

        landmask_gsw = np.zeros(len(x_sampling_masked_intersection),dtype=c.c_int)
        pnpoly_function.pnpoly(c.c_int(len(lon_gsw)),c.c_int(len(x_sampling_masked_intersection)),arrx_gsw,arry_gsw,arrx_input_gsw,arry_input_gsw,c.c_void_p(landmask_gsw.ctypes.data))

        x_sampling_masked_intersection_gsw = x_sampling_masked_intersection[landmask_gsw==0]
        y_sampling_masked_intersection_gsw = y_sampling_masked_intersection[landmask_gsw==0]
        x = np.append(x,x_sampling_masked_intersection_gsw)
        y = np.append(y,y_sampling_masked_intersection_gsw)

    return x,y


def coregister_to_mosaic(strip_data,mosaic_file,mosaic_shp_data,gsw_main_sea_only_buffered,pnpoly_function,time_series_dir,epsg_code,X_SPACING,Y_SPACING):
    '''
    Given a strip and mosaic, co-register the strip to the mosaic
    '''
    proj4_str = epsg2proj4(epsg_code)
    strip_data = strip_data.reset_index(drop=True)
    full_strip_path = strip_data.strip[0]
    strip_name = strip_data.strip[0].split('/')[-1]
    strip_basename = strip_name.replace('.tif','')
    geom_intersection_list = [strip_data.geometry[0].intersection(mos_geom) for mos_geom in mosaic_shp_data.geometry]
    idx_polygon = [geom.geom_type == 'Polygon' for geom in geom_intersection_list]
    idx_multipolygon = [geom.geom_type == 'MultiPolygon' for geom in geom_intersection_list]
    polygon_list = list(compress(geom_intersection_list,idx_polygon))
    multipolygon_list = list(compress(geom_intersection_list,idx_multipolygon))
    [polygon_list.extend(list(a.geoms)) for a in multipolygon_list]
    geom_intersection = shapely.geometry.MultiPolygon(polygon_list)
    x_masked_total,y_masked_total = populate_intersection(geom_intersection,gsw_main_sea_only_buffered,pnpoly_function,X_SPACING,Y_SPACING)
    output_xy_file = f'{time_series_dir}{strip_basename}_xy.txt'
    output_h_file = f'{time_series_dir}{strip_basename}_h_sampled_mosaic.txt'
    mosaic_sampled_file = f'{time_series_dir}{strip_basename}_sampled_mosaic_{epsg_code}.txt'
    mosaic_sampled_file_base = mosaic_sampled_file.replace('.txt','')
    np.savetxt(output_xy_file,np.c_[x_masked_total,y_masked_total],fmt='%10.5f',delimiter=' ')

    subprocess.run(f'cat {output_xy_file} | gdallocationinfo {mosaic_file} -geoloc -valonly > {output_h_file}',shell=True)
    subprocess.run('awk -i inplace \'!NF{$0="NaN"}1\' ' + output_h_file,shell=True)
    subprocess.run(f'tr -s \' \' \',\' <{output_xy_file} > {time_series_dir}tmp_xy.txt',shell=True)
    subprocess.run(f'paste -d , {time_series_dir}tmp_xy.txt {output_h_file} > {time_series_dir}tmp.txt',shell=True)
    subprocess.run(f'mv {time_series_dir}tmp.txt {mosaic_sampled_file}',shell=True)
    subprocess.run(f'sed -i \'/-9999/d\' {mosaic_sampled_file}',shell=True)
    subprocess.run(f'sed -i \'/NaN/d\' {mosaic_sampled_file}',shell=True)
    subprocess.run(f'rm {time_series_dir}tmp_xy.txt',shell=True)
    subprocess.run(f'rm {output_xy_file}',shell=True)
    subprocess.run(f'rm {output_h_file}',shell=True)

    point2dem_results_file = mosaic_sampled_file.replace('.txt','_'+epsg_code+'_point2dem_results.txt')
    point2dem_command = f'point2dem {mosaic_sampled_file} -o {mosaic_sampled_file_base} --nodata-value -9999 --tr 2 --csv-format \"1:easting 2:northing 3:height_above_datum\" --csv-proj4 \"{proj4_str}\" > {point2dem_results_file}'
    subprocess.run(point2dem_command,shell=True)
    mosaic_sampled_as_dem = mosaic_sampled_file.replace('.txt','-DEM.tif')
    align_results_file = mosaic_sampled_file.replace('.txt','_align_results.txt')
    align_command = f'dem_align.py -outdir {time_series_dir} -max_iter 15 -max_offset 2000 {mosaic_sampled_as_dem} {full_strip_path} > {align_results_file}'
    subprocess.run(align_command,shell=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mosaic',help='Path to mosaic')
    # parser.add_argument('--input_file',default='/home/eheijkoop/INPUTS/MOSAIC_Input.txt',help='path to dir containing strips')
    args = parser.parse_args()
    mosaic_file = args.mosaic

    tmp_dir = '/BhaltosMount/Bhaltos/EDUARD/tmp/'

    POLYGON_AREA_THRESHOLD = 250 #in m^2
    STRIP_AREA_THRESHOLD = 1e6 #in m^2
    GSW_POCKET_THRESHOLD = 0.01 #in %
    GSW_CRS_TRANSFORM_THRESHOLD = 0.05 #in %
    GSW_OVERLAP_THRESHOLD = 0.95 #in %
    STRIP_TOTAL_AREA_PERCENTAGE_THRESHOLD = 0.01 #in %
    STRIP_CONTAINMENT_THRESHOLD = 0.9 #in %
    STRIP_DELTA_TIME_THRESHOLD = 0 #in days
    N_STRIPS_CONTAINMENT = 2 #[-]
    AREA_OVERLAP_THRESHOLD = 2.5e5 #in m^2
    GSW_INTERSECTION_THRESHOLD = 0.667 #in %
    X_SPACING = 20.0 #in m
    Y_SPACING = 20.0 #in m
    MOSAIC_TILE_SIZE = 25000.0 #in m^2 

    subprocess.run('cc -fPIC -shared -o C_Code/pnpoly_function.so C_Code/pnpoly_function.c',shell=True)
    so_file = 'C_Code/pnpoly_function.so'
    pnpoly_function = c.cdll.LoadLibrary(so_file)

    mosaic_dir = os.path.dirname(mosaic_file)+'/'
    output_dir = '/'.join(mosaic_dir.split('/')[0:-2]) + '/'
    time_series_dir = output_dir + 'Time_Series_Strips/'
    if not subprocess.os.path.isdir(time_series_dir):
        subprocess.os.mkdir(time_series_dir)

    if os.path.exists(mosaic_file.replace('.tif','.shp')):
        mosaic_shp_data = gpd.read_file(mosaic_file.replace('.tif','.shp'))
    else:
        mosaic_shp_data = get_strip_shp(mosaic_file,tmp_dir)
        mosaic_shp_data.to_file(mosaic_file.replace('.tif','.shp'))

    epsg_code = osr.SpatialReference(wkt=gdal.Open(mosaic_file,gdalconst.GA_ReadOnly).GetProjection()).GetAttrValue('AUTHORITY',1)
    loc_name = mosaic_file.split('/')[-1].split('_Full_Mosaic')[0]

    strip_shp_file = f'{output_dir}{loc_name}_Strips_{epsg_code}.shp'
    strip_shp_data = gpd.read_file(strip_shp_file)

    gsw_main_sea_only_file = f'{output_dir}{loc_name}_GSW_merged_main_sea_only_clipped_transformed_{epsg_code}.shp'
    gsw_main_sea_only = gpd.read_file(gsw_main_sea_only_file)
    gsw_main_sea_only_buffered = gsw_main_sea_only.buffer(0)

    for i in range(len(strip_shp_data)):
        coregister_to_mosaic(strip_shp_data.iloc[[i]],mosaic_file,mosaic_shp_data,gsw_main_sea_only_buffered,pnpoly_function,time_series_dir,epsg_code,X_SPACING,Y_SPACING)

if __name__ == '__main__':
    main()