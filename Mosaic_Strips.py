# from sqlite3 import connect
# from typing_extensions import ParamSpecArgs
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon,MultiPolygon
import shapely
from osgeo import gdal,gdalconst,osr
from scipy import ndimage
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_tree
import glob
import os,sys
import argparse
import subprocess
import datetime
import ctypes as c
import warnings


def get_extent(gt,cols,rows):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]
    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
        yarr.reverse()
    return ext

def reproject_coords(coords,src_srs,tgt_srs):
    ''' Reproject a list of x,y coordinates.

        @type geom:     C{tuple/list}
        @param geom:    List of [[x,y],...[x,y]] coordinates
        @type src_srs:  C{osr.SpatialReference}
        @param src_srs: OSR SpatialReference object
        @type tgt_srs:  C{osr.SpatialReference}
        @param tgt_srs: OSR SpatialReference object
        @rtype:         C{tuple/list}
        @return:        List of transformed [[x,y],...[x,y]] coordinates
    '''
    trans_coords=[]
    transform = osr.CoordinateTransformation( src_srs, tgt_srs)
    for x,y in coords:
        y,x,z = transform.TransformPoint(x,y)
        trans_coords.append([x,y])
    return trans_coords

def utm2proj4(utm_code):
    if 'UTM' in utm_code:
        utm_code = utm_code.replace('UTM','')
    if 'utm' in utm_code:
        utm_code = utm_code.replace('utm','')
    north_south = ''
    zone = utm_code[0:2]
    lat_band_number = ord(utm_code[2])
    if lat_band_number >= 97 and lat_band_number <= 122:
        lat_band_number = lat_band_number - 96
    elif lat_band_number >= 65 and lat_band_number <= 90:
        lat_band_number = lat_band_number - 64
    
    if lat_band_number <= 13 and lat_band_number >= 3:
        north_south = 'south'
    elif lat_band_number <= 24 and lat_band_number >= 14:
        north_south = 'north'
    
    if len(north_south) == 0:
        print('Error! North/South not created!')
        return None

    proj4 = '+proj=utm +zone='+zone+' +'+north_south+' +datum=WGS84 +units=m +no_defs'
    return proj4

def epsg2proj4(epsg_code):
    epsg_code = str(epsg_code) #forces string, if input is int for example
    if epsg_code == '3413':
        proj4 = '+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
    else:
        zone = epsg_code[3:5]
        if epsg_code[2] == '6':
            north_south = 'north'
        elif epsg_code[2] == '7':
            north_south = 'south'
        proj4 = f'+proj=utm +zone={zone} +{north_south} +datum=WGS84 +units=m +no_defs'
    return proj4


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


def get_ortho_list(loc_dir):
    full_ortho_list = glob.glob(loc_dir + 'UTM*/*/strips/*ortho.tif')
    full_ortho_list.extend(glob.glob(loc_dir + '*/strips/*ortho.tif'))
    full_ortho_list = np.asarray(full_ortho_list)
    full_ortho_list.sort()
    return full_ortho_list


def get_strip_list(ortho_list,input_type):
    #find strips in a given folder
    strip_exist_old_type = np.asarray([subprocess.os.path.exists(o) for o in [d.replace('ortho.tif','dem_browse.tif') for d in ortho_list]])
    strip_exist_new_type = np.asarray([subprocess.os.path.exists(o) for o in [d.replace('ortho.tif','dem_10m.tif') for d in ortho_list]])
    
    strip_exist_old_type_full_res = np.asarray([subprocess.os.path.exists(o) for o in [d.replace('ortho.tif','dem_smooth.tif') for d in ortho_list]])
    strip_exist_new_type_full_res = np.asarray([subprocess.os.path.exists(o) for o in [d.replace('ortho.tif','dem.tif') for d in ortho_list]])

    strip_list_old = [s.replace('ortho.tif','dem_browse.tif') for s in ortho_list[strip_exist_old_type]]
    strip_list_old.extend([s.replace('ortho.tif','dem_smooth.tif') for s in ortho_list[np.logical_xor(strip_exist_old_type,strip_exist_old_type_full_res)]])

    strip_list_new = [s.replace('ortho.tif','dem_10m.tif') for s in ortho_list[strip_exist_new_type]]
    strip_list_new.extend([s.replace('ortho.tif','dem.tif') for s in ortho_list[np.logical_xor(strip_exist_new_type,strip_exist_new_type_full_res)]])

    if input_type == 0:
        strip_list_coarse = strip_list_old
        strip_list_coarse.extend(strip_list_new)
    elif input_type == 1:
        strip_list_coarse = strip_list_old
    elif input_type == 2:
        strip_list_coarse = strip_list_new
    strip_list_coarse = np.asarray(strip_list_coarse)
    strip_list_coarse.sort()

    strip_list_full_res = [s.replace('dem_10m.tif','dem.tif') for s in strip_list_coarse]
    strip_list_full_res = [s.replace('2m_dem_browse.tif','2m_dem_smooth.tif') for s in strip_list_full_res]
    strip_list_full_res = np.asarray(strip_list_full_res)
    return strip_list_coarse,strip_list_full_res

def get_strip_extents(strip):
    src = gdal.Open(strip)
    gt = src.GetGeoTransform()
    cols = src.RasterXSize
    rows = src.RasterYSize

    ext = get_extent(gt,cols,rows)
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(src.GetProjection())

    tgt_srs = osr.SpatialReference()
    tgt_srs.ImportFromEPSG(4326)

    geo_ext = reproject_coords(ext,src_srs,tgt_srs)
    #TEST IF IT'S LON/LAT OR LAT/LON WITH GDAL UPDATE
    lon_strip = [item[0] for item in geo_ext]
    lat_strip = [item[1] for item in geo_ext]

    lon_min = np.nanmin(lon_strip)
    lon_max = np.nanmax(lon_strip)
    lat_min = np.nanmin(lat_strip)
    lat_max = np.nanmax(lat_strip)
    return lon_min,lon_max,lat_min,lat_max


def get_gsw(output_dir,tmp_dir,gsw_dir,epsg_code,lon_min,lon_max,lat_min,lat_max,GSW_POCKET_THRESHOLD,GSW_CRS_TRANSFORM_THRESHOLD):
    epsg_code = str(epsg_code)
    output_name = output_dir.split('/')[-2]
    lon_min_rounded_gsw = int(np.floor(lon_min/10)*10)
    lon_max_rounded_gsw = int(np.floor(lon_max/10)*10)
    lat_min_rounded_gsw = int(np.ceil(lat_min/10)*10)
    lat_max_rounded_gsw = int(np.ceil(lat_max/10)*10)

    lon_gsw_range = range(lon_min_rounded_gsw,lon_max_rounded_gsw+10,10)
    lat_gsw_range = range(lat_min_rounded_gsw,lat_max_rounded_gsw+10,10)

    gsw_output_file = tmp_dir + 'GSW_merged.tif'
    gsw_output_file_clipped = tmp_dir + 'GSW_merged_clipped.tif'
    gsw_output_file_sea_only_clipped = tmp_dir + 'GSW_merged_sea_only_clipped.tif'
    gsw_output_file_sea_only_clipped_transformed = tmp_dir + 'GSW_merged_sea_only_clipped_transformed_' + epsg_code +'.tif'
    gsw_output_shp_file_sea_only_clipped_transformed = tmp_dir + 'GSW_merged_sea_only_clipped_transformed_' + epsg_code +'.shp'
    gsw_output_shp_file_main_sea_only_clipped_transformed = tmp_dir + output_name + '_GSW_merged_main_sea_only_clipped_transformed_' + epsg_code +'.shp'

    if subprocess.os.path.exists(gsw_output_file):
        subprocess.os.remove(gsw_output_file)
    if subprocess.os.path.exists(gsw_output_file_clipped):
        subprocess.os.remove(gsw_output_file_clipped)
    if subprocess.os.path.exists(gsw_output_file_sea_only_clipped):
        subprocess.os.remove(gsw_output_file_sea_only_clipped)
    if subprocess.os.path.exists(gsw_output_file_sea_only_clipped_transformed):
        subprocess.os.remove(gsw_output_file_sea_only_clipped_transformed)
    if subprocess.os.path.exists(gsw_output_shp_file_sea_only_clipped_transformed):
        for f1 in glob.glob(gsw_output_shp_file_sea_only_clipped_transformed.replace('.shp','.*')):
            subprocess.os.remove(f1)
    if subprocess.os.path.exists(gsw_output_shp_file_main_sea_only_clipped_transformed):
        for f1 in glob.glob(gsw_output_shp_file_main_sea_only_clipped_transformed.replace('.shp','.*')):
            subprocess.os.remove(f1)

    gsw_merge_command = 'gdal_merge.py -q -o ' + gsw_output_file + ' -co COMPRESS=LZW '
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
            gsw_file = gsw_dir + 'extent_' + str(np.abs(lon)) + EW_str + '_' + str(np.abs(lat)) + NS_str + '_v1_1.tif '
            gsw_merge_command = gsw_merge_command + gsw_file
    
    subprocess.run(gsw_merge_command,shell=True)
    lonlat_str = str(lon_min) + ' ' + str(lat_min) + ' ' + str(lon_max) + ' ' + str(lat_max)
    clip_command = 'gdalwarp -q -overwrite -te ' + lonlat_str + ' -co COMPRESS=LZW ' + gsw_output_file + ' ' + gsw_output_file_clipped
    subprocess.run(clip_command,shell=True)
    
    src_gsw = gdal.Open(gsw_output_file_clipped,gdalconst.GA_ReadOnly)
    src_gsw_proj = src_gsw.GetProjection()
    src_gsw_geotrans = src_gsw.GetGeoTransform()
    wide = src_gsw.RasterXSize
    high = src_gsw.RasterYSize
    gsw_array = np.array(src_gsw.GetRasterBand(1).ReadAsArray())
    gsw_array[gsw_array>0] = 1
    gsw_array = gsw_array.astype(int)
    gsw_area = gsw_array.shape[0]*gsw_array.shape[1]

    label, num_label = ndimage.label(gsw_array == 1)
    size = np.bincount(label.ravel())
    #sort the labels by size (descending, so biggest first), then remove label=0, because that one is land
    label_IDs_sorted = np.argsort(size)[::-1]
    label_IDs_sorted = label_IDs_sorted[label_IDs_sorted != 0]
    gsw_clump = np.zeros(gsw_array.shape,dtype=int)
    for label_id in label_IDs_sorted:
        if size[label_id]/gsw_area < GSW_POCKET_THRESHOLD:
            break
        gsw_clump = gsw_clump + np.asarray(label==label_id,dtype=int)

    # biggest_label = size[1:].argmax() + 1
    # gsw_clump = label == biggest_label

    dst = gdal.GetDriverByName('GTiff').Create(gsw_output_file_sea_only_clipped, wide, high, 1 , gdalconst.GDT_UInt16)
    outBand = dst.GetRasterBand(1)
    outBand.WriteArray(gsw_clump,0,0)
    outBand.FlushCache()
    outBand.SetNoDataValue(0)
    dst.SetProjection(src_gsw_proj)
    dst.SetGeoTransform(src_gsw_geotrans)
    del dst
    
    if epsg_code == '4326':
        gsw_output_file_sea_only_clipped_transformed = gsw_output_file_sea_only_clipped
    else:
        warp_command = 'gdalwarp -q -overwrite ' + gsw_output_file_sea_only_clipped + ' ' + gsw_output_file_sea_only_clipped_transformed + ' -s_srs EPSG:4326 -t_srs EPSG:' + epsg_code
        subprocess.run(warp_command,shell=True)

    gsw_polygonize_command = 'gdal_polygonize.py -q ' + gsw_output_file_sea_only_clipped_transformed + ' -f \"ESRI Shapefile\" ' + gsw_output_shp_file_sea_only_clipped_transformed
    subprocess.run(gsw_polygonize_command,shell=True)

    gsw_shp_data = gpd.read_file(gsw_output_shp_file_sea_only_clipped_transformed)
    if len(gsw_shp_data) == 0:
        print('No coast in this region!')
        return None,None
    # total_gsw_area = np.sum(gsw_shp_data.area)
    #only include areas that are >10% of the total to remove the clips introduced by the CRS transform
    idx_area = np.asarray(gsw_shp_data.area)/np.sum(gsw_shp_data.area) > GSW_CRS_TRANSFORM_THRESHOLD
    #most likely len(idx_area) == 1, but it may not be the case
    gsw_main_sea_only = gsw_shp_data[idx_area].reset_index(drop=True)

    gsw_main_sea_only.to_file(gsw_output_shp_file_main_sea_only_clipped_transformed)
    subprocess.run('mv ' + gsw_output_shp_file_main_sea_only_clipped_transformed.replace('.shp','.*') + ' ' + output_dir,shell=True)
    gsw_output_shp_file_main_sea_only_clipped_transformed = gsw_output_shp_file_main_sea_only_clipped_transformed.replace(tmp_dir,output_dir)
    if subprocess.os.path.exists(gsw_output_file):
        subprocess.os.remove(gsw_output_file)
    if subprocess.os.path.exists(gsw_output_file_clipped):
        subprocess.os.remove(gsw_output_file_clipped)
    if subprocess.os.path.exists(gsw_output_file_sea_only_clipped):
        subprocess.os.remove(gsw_output_file_sea_only_clipped)
    if subprocess.os.path.exists(gsw_output_file_sea_only_clipped_transformed):
        subprocess.os.remove(gsw_output_file_sea_only_clipped_transformed)
    if subprocess.os.path.exists(gsw_output_shp_file_sea_only_clipped_transformed):
        for f1 in glob.glob(gsw_output_shp_file_sea_only_clipped_transformed.replace('.shp','.*')):
            subprocess.os.remove(f1)
    return gsw_main_sea_only,gsw_output_shp_file_main_sea_only_clipped_transformed

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


def filter_strip_gsw(wv_strip_shp,gsw_shp_data,STRIP_AREA_THRESHOLD,POLYGON_AREA_THRESHOLD,GSW_OVERLAP_THRESHOLD,STRIP_TOTAL_AREA_PERCENTAGE_THRESHOLD):
    if np.sum(wv_strip_shp.geometry.area) < STRIP_AREA_THRESHOLD:
        return None
    idx_polygon_area = wv_strip_shp.area > POLYGON_AREA_THRESHOLD
    wv_strip_shp_filtered_gsw = wv_strip_shp[idx_polygon_area].reset_index(drop=True)
    if gsw_shp_data is not None:
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

def geometries_contained(geom_1,geom_2,epsg_code,containment_threshold=1.0):
    '''
    returns True/False if geom 2 is/isn't contained by geom 1
    '''
    containment = False
    if geom_1.contains(geom_2):
        containment = True
    elif geom_1.intersects(geom_2):
        geom_1_2_intersection = geom_1.intersection(geom_2)
        if geom_1_2_intersection.geom_type == 'Polygon':
            gdf_1_2_intersection = gpd.GeoDataFrame(geometry=[geom_1_2_intersection],crs='EPSG:'+epsg_code)
        elif geom_1_2_intersection.geom_type == 'MultiPolygon':
            gdf_1_2_intersection = gpd.GeoDataFrame(geometry=[geom for geom in geom_1_2_intersection.geoms if geom.geom_type=='Polygon'],crs='EPSG:'+epsg_code)
        elif geom_1_2_intersection.geom_type == 'GeometryCollection':
            gdf_1_2_intersection = gpd.GeoDataFrame(geometry=[geom for geom in geom_1_2_intersection.geoms if geom.geom_type=='Polygon'],crs='EPSG:'+epsg_code)
        elif geom_1_2_intersection.geom_type == 'LineString':
            return containment
        if np.sum(gdf_1_2_intersection.geometry.area) / geom_2.area > containment_threshold:
            containment = True
    return containment

def get_contained_strips(strip_shp_data,strip_dates,epsg_code,STRIP_CONTAINMENT_THRESHOLD=0.9,STRIP_DELTA_TIME_THRESHOLD=0,N_STRIPS_CONTAINMENT=2):
    '''
    Find strips that are fully contained by other strips AND are older than the one it is fully(/90%) contained by
    Then, find strips that are fully contained by the union of two(/three) other strips AND both(/all) of those are newer
    '''
    strip_dates_datetime = np.asarray([datetime.datetime(year=int(s[0:4]),month=int(s[4:6]),day=int(s[6:8])) for s in strip_dates.astype(str)])
    contain_dt_flag = np.ones(len(strip_shp_data),dtype=bool)
    for i in range(len(strip_shp_data)):
        idx_contained = np.asarray([geometries_contained(geom,strip_shp_data.geometry[i],epsg_code,STRIP_CONTAINMENT_THRESHOLD) for geom in strip_shp_data.geometry])
        idx_contained[i] = False #because a geometry is fully contained by itself
        idx_newer_strip = strip_dates_datetime[i] - strip_dates_datetime < datetime.timedelta(days=STRIP_DELTA_TIME_THRESHOLD)
        contain_dt_flag[i] = ~np.any(np.logical_and(idx_contained,idx_newer_strip))
        #skip it if it's already contained by one other
        if contain_dt_flag[i] == False:
            continue
        if N_STRIPS_CONTAINMENT < 2:
            continue
        idx_intersection = np.argwhere(np.asarray([strip_shp_data.geometry[i].intersects(geom) for geom in strip_shp_data.geometry])).squeeze()
        idx_intersection = np.delete(idx_intersection,np.where(idx_intersection==i))
        #need at least 2 intersecting strips to see if it's contained by union of 2 other strips
        if len(idx_intersection) < 2:
            continue
        idx_intersection_combinations = np.reshape(np.stack(np.meshgrid(idx_intersection,idx_intersection),-1),(len(idx_intersection)*len(idx_intersection),2))
        idx_intersection_combinations = np.delete(idx_intersection_combinations,idx_intersection_combinations[:,0]==idx_intersection_combinations[:,1],axis=0)
        idx_intersection_combinations = np.unique(np.sort(idx_intersection_combinations, axis=1), axis=0)
        idx_contained_combo = np.asarray([geometries_contained(strip_shp_data.geometry[combo[0]].union(strip_shp_data.geometry[combo[1]]),strip_shp_data.geometry[i],epsg_code,STRIP_CONTAINMENT_THRESHOLD) for combo in idx_intersection_combinations])
        idx_newer_strip = strip_dates_datetime[i] - strip_dates_datetime[idx_intersection_combinations] < datetime.timedelta(days=STRIP_DELTA_TIME_THRESHOLD)
        contain_dt_flag[i] = ~np.any(np.logical_and(idx_contained_combo,np.all(idx_newer_strip,axis=1)))
        #skip it if it's already contained by two others
        if contain_dt_flag[i] == False:
            continue
        if N_STRIPS_CONTAINMENT < 3:
            continue
        #need at least 3 intersecting strips to see if it's contained by union of 3 other strips
        if len(idx_intersection) < 3:
            continue
        idx_intersection_combinations = np.reshape(np.stack(np.meshgrid(idx_intersection,idx_intersection,idx_intersection),-1),(len(idx_intersection)*len(idx_intersection)*len(idx_intersection),3))
        idx_intersection_combinations = np.delete(idx_intersection_combinations,idx_intersection_combinations[:,0]==idx_intersection_combinations[:,1],axis=0)
        idx_intersection_combinations = np.delete(idx_intersection_combinations,idx_intersection_combinations[:,0]==idx_intersection_combinations[:,2],axis=0)
        idx_intersection_combinations = np.delete(idx_intersection_combinations,idx_intersection_combinations[:,1]==idx_intersection_combinations[:,2],axis=0)
        idx_intersection_combinations = np.unique(np.sort(idx_intersection_combinations, axis=1), axis=0)
        idx_contained_combo = np.asarray([geometries_contained(strip_shp_data.geometry[combo[0]].union(strip_shp_data.geometry[combo[1]]).union(strip_shp_data.geometry[combo[2]]),strip_shp_data.geometry[i],epsg_code,STRIP_CONTAINMENT_THRESHOLD) for combo in idx_intersection_combinations])
        idx_newer_strip = strip_dates_datetime[i] - strip_dates_datetime[idx_intersection_combinations] < datetime.timedelta(days=STRIP_DELTA_TIME_THRESHOLD)
        contain_dt_flag[i] = ~np.any(np.logical_and(idx_contained_combo,np.all(idx_newer_strip,axis=1)))

    return contain_dt_flag

def get_valid_strip_overlaps(strip_shp_data,gsw_main_sea_only_buffered,AREA_OVERLAP_THRESHOLD,GSW_INTERSECTION_THRESHOLD):
    '''
    for each strip, find overlap and perform checks:
        - is the overlapping area large enough?
        - is the overlapping area covered by water?
            - fully contained by GSW?
            - covered too much by GSW?
    '''
    if gsw_main_sea_only_buffered is not None:
        gsw_polygon = shapely.ops.unary_union(gsw_main_sea_only_buffered.geometry)
    valid_strip_overlaps = np.zeros((len(strip_shp_data),len(strip_shp_data)))
    for i in range(len(strip_shp_data)):
        idx_intersection = np.asarray([strip_shp_data.geometry[i].intersects(geom) for geom in strip_shp_data.geometry])
        idx_intersection[i] = False
        idx_area_threshold = [strip_shp_data.geometry[i].intersection(adjacent_geom).area > AREA_OVERLAP_THRESHOLD if idx_intersection[ii] == True else False for ii,adjacent_geom in enumerate(strip_shp_data.geometry)]
        if gsw_main_sea_only_buffered is not None:
            idx_gsw_threshold = [strip_shp_data.geometry[i].intersection(adjacent_geom).intersection(gsw_polygon).area / strip_shp_data.geometry[i].intersection(adjacent_geom).area < GSW_INTERSECTION_THRESHOLD if idx_intersection[ii] == True else False for ii,adjacent_geom in enumerate(strip_shp_data.geometry)]
        else:
            idx_gsw_threshold = idx_area_threshold
        valid_strip_overlaps[i,:] = np.logical_and(idx_area_threshold,idx_gsw_threshold).astype(int)
    return valid_strip_overlaps

def get_minimum_spanning_tree(connections_array,strip_dates):
    '''
    Given a weighted array of connections, build the MST
    '''
    strip_dates_datetime = np.asarray([datetime.datetime(year=int(s[0:4]),month=int(s[4:6]),day=int(s[6:8])) for s in strip_dates.astype(str)])
    dt_array = np.empty((len(strip_dates_datetime),len(strip_dates_datetime)),dtype=np.int16)
    for i in range(dt_array.shape[0]):
        for j in range(dt_array.shape[1]):
            dt_array[i,j] = np.abs((strip_dates_datetime[i] - strip_dates_datetime[j]).days)
    dt_array[dt_array==0] = 1
    Tcsr = minimum_spanning_tree(np.triu(connections_array)) #csr = compressed sparse graph
    mst_array = Tcsr.toarray().astype(int)
    Tcsr_weighted = minimum_spanning_tree(np.triu(connections_array*dt_array))
    mst_weighted_array = Tcsr_weighted.toarray().astype(int)
    mst_weighted_array[mst_weighted_array>0] = 1

    return mst_array,mst_weighted_array

def find_mosaic(strip_shp_data,mst_array,strip_dates):
    '''
    Given a list of strips and the minimum spanning tree, build mosaic(s)
    '''
    strip_dict = strip_shp_data.strip.to_dict()
    strip_ID_array = np.asarray(range(0,mst_array.shape[0]))
    strip_ID_array_full = np.asarray(range(0,mst_array.shape[0]))
    mosaic_count = 0
    group_count = 0
    single_count = 0
    groups_dict = {}
    singles_dict = {}
    mosaic_dict = {}
    groups_found = False
    singles_found = False
    while ~groups_found:
        ID_start = strip_shp_data.area[np.logical_and(strip_dates==strip_dates[strip_ID_array][0],np.in1d(strip_ID_array_full,strip_ID_array))].idxmax()
        mosaic_list = np.unique(np.where(depth_first_tree(mst_array,ID_start,directed=False).toarray().astype(int)))
        strip_ID_array = np.setdiff1d(strip_ID_array,mosaic_list)
        if len(mosaic_list) > 0:
            groups_dict[str(group_count)] = {
                'group':mosaic_list.tolist(),
                'ID_start':ID_start
                }
            group_count += 1
            if len(strip_ID_array) == 0:
                groups_found = True
                break
        elif len(mosaic_list) == 0:
            groups_found = True
            break
    for i in range(len(strip_ID_array)):
        single_count += 1
        singles_dict[str(i)] = {'single':strip_ID_array[i]}

    print(f'Building mosaic(s) from {group_count} group(s) & {single_count} single strip(s).')
    for i in range(len(groups_dict)):
        path_check = False
        mosaic_ID_start = groups_dict[str(i)]['ID_start']
        current_gen = np.asarray([mosaic_ID_start])
        next_gen = np.empty([0,1],dtype=np.int16)
        done_list = current_gen
        generation_count = 0
        ref_list = np.empty([0,1],dtype=int)
        src_list = np.empty([0,1],dtype=int)
        print(' ')
        print(f'Mosaic {i}:')
        print(f'Starting at: {mosaic_ID_start}')
        print(strip_dict[ID_start])

        while ~path_check:
            generation_count = generation_count+1
            gen_ref_list = np.empty([0,1],dtype=int)
            gen_src_list = np.empty([0,1],dtype=int)
            print(' ')
            print(f'Generation: {generation_count}')
            for src_ID in current_gen:
                tmp = mst_array[:,src_ID] + mst_array[src_ID,:]
                tmp_next_gen = np.squeeze(np.argwhere(tmp))
                tmp_next_gen = tmp_next_gen[~np.isin(tmp_next_gen,done_list)]
                next_gen = np.append(next_gen,tmp_next_gen)
                done_list = np.append(done_list,tmp_next_gen)
                for ref_ID in tmp_next_gen:
                    print(f'Strip {ref_ID} to strip {src_ID}')
                    ref_list = np.append(ref_list,ref_ID)
                    src_list = np.append(src_list,src_ID)
                    gen_ref_list = np.append(gen_ref_list,ref_ID)
                    gen_src_list = np.append(gen_src_list,src_ID)
            if np.array_equal(np.sort(done_list),groups_dict[str(i)]['group']):
                path_check = True
                break
            current_gen = next_gen
            next_gen = np.empty([0,1],dtype=int)
        mosaic_dict[str(i)] = {
            'ref':ref_list,
            'src':src_list
        }

    return mosaic_dict,singles_dict

def populate_intersection(geom_intersection,gsw_main_sea_only_buffered,pnpoly_function,X_SPACING,Y_SPACING):
    '''
    Returns x and y inside a polygon
    We do the masking on each individual feature, rather than the whole to reduce the total number of points to be considered
        E.g., two polygons some distance away from each other, we first populate points around the two polygons,
        then we mask them, rather than populating the whole region and masking all of that.
    '''
    x = np.empty([0,1],dtype=float)
    y = np.empty([0,1],dtype=float)
    if gsw_main_sea_only_buffered is not None:
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
        if gsw_main_sea_only_buffered is not None:
            arrx_input_gsw = (c.c_float * len(x_sampling_masked_intersection))(*x_sampling_masked_intersection)
            arry_input_gsw = (c.c_float * len(y_sampling_masked_intersection))(*y_sampling_masked_intersection)

            landmask_gsw = np.zeros(len(x_sampling_masked_intersection),dtype=c.c_int)
            pnpoly_function.pnpoly(c.c_int(len(lon_gsw)),c.c_int(len(x_sampling_masked_intersection)),arrx_gsw,arry_gsw,arrx_input_gsw,arry_input_gsw,c.c_void_p(landmask_gsw.ctypes.data))

            x_sampling_masked_intersection_gsw = x_sampling_masked_intersection[landmask_gsw==0]
            y_sampling_masked_intersection_gsw = y_sampling_masked_intersection[landmask_gsw==0]
        else:
            x_sampling_masked_intersection_gsw = x_sampling_masked_intersection
            y_sampling_masked_intersection_gsw = y_sampling_masked_intersection
        x = np.append(x,x_sampling_masked_intersection_gsw)
        y = np.append(y,y_sampling_masked_intersection_gsw)

    return x,y

def build_mosaic(strip_shp_data,gsw_main_sea_only_buffered,pnpoly_function,mosaic_dict,mosaic_dir,output_name,mosaic_number,epsg_code,X_SPACING,Y_SPACING,MOSAIC_TILE_SIZE):
    '''
    Build the mosaic, given list of indices of strips to co-register to each other
    Co-registering ref to src, by sampling src and treating that as truth
    '''
    mosaic_name = mosaic_dir + output_name + '_Full_Mosaic_' + str(mosaic_number) + '_' + epsg_code + '.tif'
    ref_list = mosaic_dict['ref']
    src_list = mosaic_dict['src']
    copy_check = False
    strip_list_coregistered = np.empty([0,1],dtype=str)
    for i in range(len(ref_list)):
        ref_strip_ID = ref_list[i]
        src_strip_ID = src_list[i]
        ref_strip = strip_shp_data.strip[ref_strip_ID]
        src_strip = strip_shp_data.strip[src_strip_ID]
        geom_intersection = strip_shp_data.geometry[src_strip_ID].intersection(strip_shp_data.geometry[ref_strip_ID])
        x_masked_total,y_masked_total = populate_intersection(geom_intersection,gsw_main_sea_only_buffered,pnpoly_function,X_SPACING,Y_SPACING)
        strip_sampled_file = mosaic_dir + output_name + f'_Mosaic_{mosaic_number}_{epsg_code}_sampled_{src_strip_ID}_for_coregistering_{ref_strip_ID}.txt'
        strip_sampled_file_base = strip_sampled_file.replace('.txt','')
        output_xy_file = strip_sampled_file.replace('.txt','_xy.txt')
        output_h_file = strip_sampled_file.replace('.txt','_h.txt')
        np.savetxt(strip_sampled_file,np.c_[x_masked_total,y_masked_total],fmt='%10.5f',delimiter=' ')
        if i in np.argwhere(src_list == src_list[0]):
            if not copy_check:
                subprocess.run(f'cp {src_strip} {mosaic_dir}',shell=True)
                strip_list_coregistered = np.append(strip_list_coregistered,mosaic_dir + src_strip.split('/')[-1])
                copy_check = True
            src_file = mosaic_dir + src_strip.split('/')[-1]
        else:
            src_file = glob.glob(mosaic_dir + src_strip.split('/')[-1].replace('.tif','') + f'_{output_name}_Mosaic_{mosaic_number}_{epsg_code}_sampled_{src_list[ref_list==src_strip_ID][0]}_for_coregistering_{src_strip_ID}-DEM*align.tif')[0]

        subprocess.run(f'cat {strip_sampled_file} | gdallocationinfo {src_file} -geoloc -valonly > {output_h_file}',shell=True)
        subprocess.run('awk -i inplace \'!NF{$0="NaN"}1\' ' + output_h_file,shell=True)
        subprocess.run(f'tr -s \' \' \',\' <{strip_sampled_file} > {output_xy_file}',shell=True)
        subprocess.run(f'paste -d , {output_xy_file} {output_h_file} > tmp.txt',shell=True)
        subprocess.run(f'mv tmp.txt {strip_sampled_file}',shell=True)
        subprocess.run(f'sed -i \'/-9999/d\' {strip_sampled_file}',shell=True)
        subprocess.run(f'sed -i \'/NaN/d\' {strip_sampled_file}',shell=True)
        subprocess.run(f'rm {output_xy_file}',shell=True)
        subprocess.run(f'rm {output_h_file}',shell=True)

        proj4_str = epsg2proj4(epsg_code)
        point2dem_results_file = strip_sampled_file.replace('.txt','_'+epsg_code+'_point2dem_results.txt')
        point2dem_command = f'point2dem {strip_sampled_file} -o {strip_sampled_file_base} --nodata-value -9999 --tr 2 --csv-format \"1:easting 2:northing 3:height_above_datum\" --csv-proj4 \"{proj4_str}\" > {point2dem_results_file}'
        subprocess.run(point2dem_command,shell=True)
        strip_sampled_as_dem = strip_sampled_file.replace('.txt','-DEM.tif')
        align_results_file = strip_sampled_file.replace('.txt','_'+epsg_code+'_align_results.txt')
        align_command = f'dem_align.py -outdir {mosaic_dir} -max_iter 15 -max_offset 2000 {strip_sampled_as_dem} {ref_strip} > {align_results_file}'
        subprocess.run(align_command,shell=True)

    print('')
    print('Mosaicing...')
    strip_list_coregistered = np.append(strip_list_coregistered,glob.glob(mosaic_dir + f'WV*Mosaic_{mosaic_number}*align.tif'))
    strip_list_coregistered = np.append(strip_list_coregistered,glob.glob(mosaic_dir + f'GE*Mosaic_{mosaic_number}*align.tif'))
    strip_list_coregistered_date = np.asarray([int(s.split('/')[-1][5:13]) for s in strip_list_coregistered])
    idx_date_coregistered_strip = np.argsort(-strip_list_coregistered_date)
    strip_list_coregistered_sorted = np.array(strip_list_coregistered)[idx_date_coregistered_strip.astype(int)]
    strip_list_coregistered_sorted_file = mosaic_dir + output_name + f'_Mosaic_{mosaic_number}_Coregistered_Strips_Sorted_Date_{epsg_code}.txt'
    np.savetxt(strip_list_coregistered_sorted_file,np.c_[strip_list_coregistered_sorted],fmt='%s')

    mosaic_results_file = mosaic_dir + output_name + '_'+epsg_code+'_mosaic_'+str(i)+'_results.txt'
    mosaic_command = f'dem_mosaic -l {strip_list_coregistered_sorted_file} --first --georef-tile-size {MOSAIC_TILE_SIZE} -o {mosaic_dir+output_name}_Mosaic_{mosaic_number}_{epsg_code} > {mosaic_results_file}'
    subprocess.run(mosaic_command,shell=True)

    merge_mosaic_output_file = mosaic_dir + output_name + f'_Full_Mosaic_{mosaic_number}_{epsg_code}.tif'
    merge_mosaic_output_file_vrt = merge_mosaic_output_file.replace('.tif','.vrt')
    merge_mosaic_vrt_command = f'gdalbuildvrt -q {merge_mosaic_output_file_vrt} {mosaic_dir+output_name}_Mosaic_{mosaic_number}_{epsg_code}-tile-*-first.tif'
    translate_mosaic_command = f'gdal_translate -co COMPRESS=LZW -co BIGTIFF=YES -q {merge_mosaic_output_file_vrt} {merge_mosaic_output_file}'
    
    subprocess.run(merge_mosaic_vrt_command,shell=True)
    subprocess.run(translate_mosaic_command,shell=True)
    print('')
    return merge_mosaic_output_file

def copy_single_strips(strip_shp_data,singles_dict,mosaic_dir,output_name,epsg_code):
    '''
    Copy single strips that are not part of a mosaic to the mosaic directory too
    '''
    singles_list = np.empty([0,1],dtype=str)
    singles_list_orig = np.empty([0,1],dtype=str)
    for i in range(len(singles_dict)):
        idx_single = singles_dict[str(i)]['single']
        subprocess.run(f'cp {strip_shp_data.strip[idx_single]} {mosaic_dir+output_name}_Single_Strip_{i}_{epsg_code}.tif',shell=True)
        singles_list = np.append(singles_list,f'{mosaic_dir+output_name}_Single_Strip_{i}_{epsg_code}.tif')
        singles_list_orig = np.append(singles_list_orig,strip_shp_data.strip[idx_single].split('/')[-1])
    singles_file = f'{mosaic_dir+output_name}_Single_Strips.txt'
    np.savetxt(singles_file,np.c_[singles_list,singles_list_orig],fmt='%s',delimiter=',')

    return singles_list
#CHECK GEOPANDAS VERSION REQUIREMENT

'''
given directory, find orthos in subdirectories
find unique EPSGs, may be more than one
for given EPSG, find strips based off of the orthos
find extents of strips to subset GSW tiles and then:
    merge GSW tiles
    clip merged GSW tile to match strips extent
    find sea only, i.e. GSW==1
    transform that from EPSG:4326 to the unique EPSG
    select main sea only, may be more than one
In this EPSG:
    load a strip, find the "real" borders (i.e. >-9999)
    turn that into a shapefile and load this multipolygon
    apply filters:
        smaller than 1 km^2 -> discard the whole strip
        within a strip, remove 1x1 pixel -> artifacts will just slow things down
        combine with GSW and find polygons of the strip that are >95% contained by GSW -> discard those
        "reset" strip and combine with GSW again, find intersection with GSW now, > threshold -> discard
    combine to produce final shapefile of all strips' "real" outlines:
        strip name and geometry (polygon or multipolygon) will be in there
    go through each strip, find strips that are older *and* fully contained by another -> discard that strip
    find overlap of each strip with each other strip
        find exceptions too:
            1. intersection of two strips is fully within GSW
            2. intersection of two strips is covered by GSW too much
            3. intersection of two strips is too small
    create minimum spanning tree, weighted by delta time
    given starting path (largest one of newest strips), find generations to link strips together
    given this path, "co-register" next strip to the current one:
        populate intersection with points that are sampled from the current strip
            regular grid, "land"masked by intersection polygon
        run point2dem on this to create a grid to which we can co-register with dem_align
    Once all strips have been co-registered to each other, we can create N>=1 unique mosaics with demmosaic, sorting by date (newest on top)
--->Output is in the form of tiles which are merged together with gdalbuildvrt
'''




def main():
    warnings.simplefilter(action='ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',default='/home/eheijkoop/INPUTS/MOSAIC_Input.txt',help='path to dir containing strips')
    args = parser.parse_args()
    input_file = args.input_file

    tmp_dir = '/BhaltosMount/Bhaltos/EDUARD/tmp/'
    gsw_dir = '/BhaltosMount/Bhaltos/EDUARD/DATA_REPOSITORY/Global_Surface_Water/Extent/'
    # tmp_dir = '/home/heijkoop/.tmp/'
    # gsw_dir = '/media/heijkoop/DATA/Global_Surface_Water/'

    subprocess.run('cc -fPIC -shared -o C_Code/pnpoly_function.so C_Code/pnpoly_function.c',shell=True)
    so_file = 'C_Code/pnpoly_function.so'
    pnpoly_function = c.cdll.LoadLibrary(so_file)

    df_input = pd.read_csv(input_file,header=0,names=['loc_dirs','output_dirs','input_types'],dtype={'loc_dirs':'str','output_dirs':'str','input_types':'object'})
    df_input.input_types = df_input.input_types.fillna('0')
    df_input.input_types = df_input.input_types.astype(int)

    POLYGON_AREA_THRESHOLD = 250 #in m^2
    STRIP_AREA_THRESHOLD = 4e6 #in m^2
    GSW_POCKET_THRESHOLD = 0.01 #in %
    GSW_CRS_TRANSFORM_THRESHOLD = 0.05 #in %
    GSW_OVERLAP_THRESHOLD = 0.95 #in %
    STRIP_TOTAL_AREA_PERCENTAGE_THRESHOLD = 0.01 #in %
    STRIP_CONTAINMENT_THRESHOLD = 0.75 #in %
    STRIP_DELTA_TIME_THRESHOLD = 0 #in days
    N_STRIPS_CONTAINMENT = 2 #[-]
    AREA_OVERLAP_THRESHOLD = 2.5e5 #in m^2
    GSW_INTERSECTION_THRESHOLD = 0.667 #in %
    X_SPACING = 20.0 #in m
    Y_SPACING = 20.0 #in m
    MOSAIC_TILE_SIZE = 25000.0 #in m^2   
    
    for i in range(len(df_input)):
        loc_dir = df_input.loc_dirs[i]
        output_dir = df_input.output_dirs[i]
        input_type = df_input.input_types[i]
        #force the directories to end on a slash
        if loc_dir[len(loc_dir)-1] != '/':
            loc_dir = loc_dir + '/'
        if output_dir[len(output_dir)-1] != '/':
            output_dir = output_dir + '/'
        mosaic_dir = output_dir + 'Mosaic/'
        if not subprocess.os.path.isdir(output_dir):
            subprocess.os.mkdir(output_dir)
        if not subprocess.os.path.isdir(mosaic_dir):
            subprocess.os.mkdir(mosaic_dir)
        loc_name = loc_dir.split('/')[-2]
        output_name = output_dir.split('/')[-2]
        t_start = datetime.datetime.now()
        print('Working on ' + loc_name)
        if loc_name != output_name:
            print('Warning! Output name and location name not the same. Continuing...')
            print(f'Calling everything {output_name} now.')
        full_ortho_list = get_ortho_list(loc_dir)
        full_epsg_list = np.asarray([osr.SpatialReference(wkt=gdal.Open(ortho,gdalconst.GA_ReadOnly).GetProjection()).GetAttrValue('AUTHORITY',1) for ortho in full_ortho_list])
        unique_epsg_list = np.unique(full_epsg_list)

        for epsg_code in unique_epsg_list:
            print(f'EPSG:{epsg_code}')
            idx_epsg = full_epsg_list == epsg_code
            ortho_list = full_ortho_list[idx_epsg]
            strip_list_coarse,strip_list_full_res = get_strip_list(ortho_list,input_type)
            if strip_list_coarse.size == 0:
                print('No strips found!')
                continue
            
            strip_shp_data = gpd.GeoDataFrame()
            lon_min_strips,lon_max_strips,lat_min_strips,lat_max_strips = 180,-180,90,-90
            for strip in strip_list_coarse:
                lon_min_single_strip,lon_max_single_strip,lat_min_single_strip,lat_max_single_strip = get_strip_extents(strip)
                lon_min_strips = np.min((lon_min_strips,lon_min_single_strip))
                lon_max_strips = np.max((lon_max_strips,lon_max_single_strip))
                lat_min_strips = np.min((lat_min_strips,lat_min_single_strip))
                lat_max_strips = np.max((lat_max_strips,lat_max_single_strip))

            gsw_main_sea_only,gsw_output_shp_file_main_sea_only_clipped_transformed = get_gsw(output_dir,tmp_dir,gsw_dir,epsg_code,lon_min_strips,lon_max_strips,lat_min_strips,lat_max_strips,GSW_POCKET_THRESHOLD,GSW_CRS_TRANSFORM_THRESHOLD)
            if gsw_main_sea_only is not None:
                gsw_main_sea_only_buffered = gsw_main_sea_only.buffer(0)
            else:
                gsw_main_sea_only_buffered = None
            strip_idx = np.ones(len(strip_list_coarse),dtype=bool)
            print('Loading strips...')
            for j,strip in enumerate(strip_list_full_res):
                sys.stdout.write('\r')
                n_progressbar = (j + 1) / len(strip_list_full_res)
                sys.stdout.write("[%-20s] %d%%" % ('='*int(20*n_progressbar), 100*n_progressbar))
                sys.stdout.flush()
                wv_strip_shp = get_strip_shp(strip,tmp_dir)
                wv_strip_shp_filtered_gsw = filter_strip_gsw(wv_strip_shp,gsw_main_sea_only,STRIP_AREA_THRESHOLD,POLYGON_AREA_THRESHOLD,GSW_OVERLAP_THRESHOLD,STRIP_TOTAL_AREA_PERCENTAGE_THRESHOLD)
                if wv_strip_shp_filtered_gsw is None:
                    strip_idx[j] = False
                    continue
                tmp_mp = shapely.ops.unary_union([Polygon(g) for g in wv_strip_shp_filtered_gsw.geometry.exterior])
                df_strip = pd.DataFrame({'strip':[strip]})
                tmp_gdf = gpd.GeoDataFrame(df_strip,geometry=[tmp_mp],crs='EPSG:'+epsg_code)
                strip_shp_data = gpd.GeoDataFrame(pd.concat([strip_shp_data,tmp_gdf],ignore_index=True),crs='EPSG:'+epsg_code)

            strip_list_coarse = strip_list_coarse[strip_idx]
            strip_list_full_res = strip_list_full_res[strip_idx]
            output_strips_shp_file = output_dir + output_name + '_Strips_' + epsg_code + '.shp'
            output_strips_shp_file_dissolved = output_dir + output_name + '_Strips_' + epsg_code + '_Dissolved.shp'
            output_strips_shp_file_filtered = output_dir + output_name + '_Strips_' + epsg_code + '_Filtered.shp'
            output_strips_shp_file_filtered_dissolved = output_dir + output_name + '_Strips_' + epsg_code + '_Filtered_Dissolved.shp'
            print('\n')
            print(output_strips_shp_file)
            
            strip_dates = np.asarray([int(s.split('/')[-1][5:13]) for s in strip_list_full_res])
            idx_date = np.argsort(-strip_dates)

            strip_dates = strip_dates[idx_date]
            strip_list_coarse = strip_list_coarse[idx_date]
            strip_list_full_res = strip_list_full_res[idx_date]
            strip_shp_data = strip_shp_data.iloc[idx_date].reset_index(drop=True)

            strip_shp_data.to_file(output_strips_shp_file)
            subprocess.run('ogr2ogr ' + output_strips_shp_file_dissolved + ' ' + output_strips_shp_file + ' -dialect sqlite -sql \'SELECT ST_Union("geometry") FROM "' + os.path.basename(output_strips_shp_file).replace('.shp','') + '"\'',shell=True)

            idx_contained = get_contained_strips(strip_shp_data,strip_dates,epsg_code,STRIP_CONTAINMENT_THRESHOLD,STRIP_DELTA_TIME_THRESHOLD,N_STRIPS_CONTAINMENT)
            strip_dates = strip_dates[idx_contained]
            strip_list_coarse = strip_list_coarse[idx_contained]
            strip_list_full_res = strip_list_full_res[idx_contained]
            strip_shp_data = strip_shp_data.iloc[idx_contained].reset_index(drop=True)

            strip_shp_data.to_file(output_strips_shp_file_filtered)
            subprocess.run('ogr2ogr ' + output_strips_shp_file_filtered_dissolved + ' ' + output_strips_shp_file_filtered + ' -dialect sqlite -sql \'SELECT ST_Union("geometry") FROM "' + os.path.basename(output_strips_shp_file_filtered).replace('.shp','') + '"\'',shell=True)
            
            valid_strip_overlaps = get_valid_strip_overlaps(strip_shp_data,gsw_main_sea_only_buffered,AREA_OVERLAP_THRESHOLD,GSW_INTERSECTION_THRESHOLD)
            mst_array,mst_weighted_array = get_minimum_spanning_tree(valid_strip_overlaps,strip_dates)
            #Need to weight mst_array by delta time (and overlapping area?)
            
            mosaic_dict,singles_dict = find_mosaic(strip_shp_data,mst_weighted_array,strip_dates)
            for mosaic_number in range(len(mosaic_dict)):
                merge_mosaic_output_file = build_mosaic(strip_shp_data,gsw_main_sea_only_buffered,pnpoly_function,mosaic_dict[str(mosaic_number)],mosaic_dir,output_name,mosaic_number,epsg_code,X_SPACING,Y_SPACING,MOSAIC_TILE_SIZE)
            singles_list = copy_single_strips(strip_shp_data,singles_dict,mosaic_dir,output_name,epsg_code)
            t_end = datetime.datetime.now()
            dt = t_end - t_start
            dt_min, dt_sec = divmod(dt.seconds,60)
            dt_hour, dt_min = divmod(dt_min,60)
            print(f'Finished with {output_name} in EPSG:{epsg_code}.')
            print('It took:')
            print("%d hours, %d minutes, %d.%d seconds" %(dt_hour,dt_min,dt_sec,dt.microseconds%1000000))
            print('')
        
if __name__ == '__main__':
    main()
