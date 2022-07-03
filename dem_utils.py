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
    '''
    Return list of corner coordinates from a geotransform
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
    '''
    Reproject a list of x,y coordinates.
    x and y are in src coordinates, going to tgt
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

def utm2epsg(utm_code,north_south_flag=False):
    utm_code = np.asarray([z.replace(' ','') for z in utm_code])
    lat_band_number = np.asarray([ord(u[2].upper()) for u in utm_code])
    if north_south_flag == True:
        hemisphere_ID = np.zeros(len(lat_band_number),dtype=int)
        hemisphere_ID[lat_band_number == 83] = 7 #south
        hemisphere_ID[lat_band_number == 78] = 6 #north
    else:
        hemisphere_ID = np.zeros(len(lat_band_number),dtype=int)
        hemisphere_ID[lat_band_number <= 77] = 7 #south
        hemisphere_ID[lat_band_number >= 78] = 6 #north
    epsg_code = np.asarray([f'32{a[1]}{a[0][0:2]}' for a in zip(utm_code,hemisphere_ID)])
    if len(epsg_code) == 1:
        epsg_code = epsg_code[0]
    return epsg_code

def lonlat2epsg(lon,lat):
    '''
    Finds the EPSG code for a given lon/lat coordinate.
    '''
    if lat >= 0:
        NS_code = '6'
    elif lat < 0:
        NS_code = '7'
    EW_code = f'{int(np.floor(lon/6.0))+31:02d}'
    epsg_code = f'32{NS_code}{EW_code}'
    return epsg_code

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

def get_raster_extents(raster,global_local_flag='global'):
    '''
    Get global or local extents of a raster
    '''
    src = gdal.Open(raster,gdalconst.GA_ReadOnly)
    gt = src.GetGeoTransform()
    cols = src.RasterXSize
    rows = src.RasterYSize
    local_ext = get_extent(gt,cols,rows)
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(src.GetProjection())
    tgt_srs = osr.SpatialReference()
    tgt_srs.ImportFromEPSG(4326)
    global_ext = reproject_coords(local_ext,src_srs,tgt_srs)
    x_local = [item[0] for item in local_ext]
    y_local = [item[1] for item in local_ext]
    x_min_local = np.nanmin(x_local)
    x_max_local = np.nanmax(x_local)
    y_min_local = np.nanmin(y_local)
    y_max_local = np.nanmax(y_local)
    x_global = [item[0] for item in global_ext]
    y_global = [item[1] for item in global_ext]
    x_min_global = np.nanmin(x_global)
    x_max_global = np.nanmax(x_global)
    y_min_global = np.nanmin(y_global)
    y_max_global = np.nanmax(y_global)
    if global_local_flag.lower() == 'global':
        return x_min_global,x_max_global,y_min_global,y_max_global
    elif global_local_flag.lower() == 'local':
        return x_min_local,x_max_local,y_min_local,y_max_local
    else:
        return None

def deg2rad(deg):
    rad = deg*np.math.pi/180
    return rad

def deg2utm(lon,lat):
    pi = np.math.pi
    n1 = np.asarray(lon).size
    n2 = np.asarray(lat).size
    if n1 != n2:
        print('Longitude and latitude vectors not equal in length.')
        print('Exiting')
        return
    lon_deg = np.atleast_1d(lon)
    lat_deg = np.atleast_1d(lat)
    lon_rad = lon*pi/180
    lat_rad = lat*pi/180
    cos_lat = np.cos(lat_rad)
    sin_lat = np.sin(lat_rad)
    tan_lat = np.tan(lat_rad)
    cos_lon = np.cos(lon_rad)
    sin_lon = np.sin(lon_rad)
    tan_lon = np.tan(lon_rad)
    x = np.empty([n1,1],dtype=float)
    y = np.empty([n2,1],dtype=float)
    zone_letter = [None]*n1
    semi_major_axis = 6378137.0
    semi_minor_axis = 6356752.314245
    second_eccentricity = np.sqrt(semi_major_axis**2 - semi_minor_axis**2)/semi_minor_axis
    second_eccentricity_squared = second_eccentricity**2
    c = semi_major_axis**2 / semi_minor_axis
    utm_number = np.fix(lon_deg/6 + 31)
    S = utm_number*6 - 183
    delta_S = lon_rad - S*pi/180
    epsilon = 0.5*np.log((1+cos_lat * np.sin(delta_S))/(1-cos_lat * np.sin(delta_S)))
    nu = np.arctan(tan_lat / np.cos(delta_S)) - lat_rad
    v = 0.9996 * c / np.sqrt(1+second_eccentricity_squared * cos_lat**2)
    tau = 0.5*second_eccentricity_squared * epsilon**2 * cos_lat**2
    a1 = np.sin(2*lat_rad)
    a2 = a1 * cos_lat**2
    j2 = lat_rad + 0.5*a1
    j4 = 0.25*(3*j2 + a2)
    j6 = (5*j4 + a2*cos_lat**2)/3
    alpha = 0.75*second_eccentricity_squared
    beta = (5/3) * alpha**2
    gamma = (35/27) * alpha**3
    Bm = 0.9996 * c * (lat_rad - alpha*j2 + beta*j4 - gamma*j6)
    x = epsilon * v * (1+tau/3) + 500000
    y = nu * v * (1+tau) + Bm
    idx_y = y<0
    if idx_y.any():
        y[idx_y] = y[idx_y] + 9999999
    for i in range(n1):
        if lat_deg[i]<-72:
            zone_letter[i] = ' C'
        elif lat_deg[i] < -64:
            zone_letter[i] = ' D'
        elif lat_deg[i] < -56:
            zone_letter[i] = ' E'
        elif lat_deg[i] < -48:
            zone_letter[i] = ' F'
        elif lat_deg[i] < -40:
            zone_letter[i] = ' G'
        elif lat_deg[i] < -32:
            zone_letter[i] = ' H'
        elif lat_deg[i] < -24:
            zone_letter[i] = ' J'
        elif lat_deg[i] < -16:
            zone_letter[i] = ' K'
        elif lat_deg[i] < -8:
            zone_letter[i] = ' L'
        elif lat_deg[i] < 0:
            zone_letter[i] = ' M'
        elif lat_deg[i] < 8:
            zone_letter[i] = ' N'
        elif lat_deg[i] < 16:
            zone_letter[i] = ' P'
        elif lat_deg[i] < 24:
            zone_letter[i] = ' Q'
        elif lat_deg[i] < 32:
            zone_letter[i] = ' R'
        elif lat_deg[i] < 40:
            zone_letter[i] = ' S'
        elif lat_deg[i] < 48:
            zone_letter[i] = ' T'
        elif lat_deg[i] < 56:
            zone_letter[i] = ' U'
        elif lat_deg[i] < 64:
            zone_letter[i] = ' V'
        elif lat_deg[i] < 72:
            zone_letter[i] = ' W'
        else:
            zone_letter[i] = ' X'
    utm_int = np.char.mod('%02d',utm_number.astype(int))
    utm_int_list = utm_int.tolist()
    utmzone = [s1 + s2 for s1, s2 in zip(utm_int_list, zone_letter)]
    return x, y, utmzone

def great_circle_distance(lon1,lat1,lon2,lat2,R=6378137.0):
    lon1 = deg2rad(lon1)
    lat1 = deg2rad(lat1)
    lon2 = deg2rad(lon2)
    lat2 = deg2rad(lat2)
    DL = np.abs(lon2 - lon1)
    DP = np.abs(lat2 - lat1)
    dsigma = 2*np.arcsin( np.sqrt( np.sin(0.5*DP)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(0.5*DL)**2))
    distance = R*dsigma
    return distance

def resample_raster(src_filename,match_filename,dst_filename,resample_method='bilinear'):
    '''
    src = what you want to resample
    match = resample to this one's resolution
    dst = output
    method = nearest neighbor, bilinear (default), cubic, cubic spline
    '''
    src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()
    src_geotrans = src.GetGeoTransform()
    src_espg = epsg_code = osr.SpatialReference(wkt=gdal.Open(src_filename,gdalconst.GA_ReadOnly).GetProjection()).GetAttrValue('AUTHORITY',1)

    match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize

    dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, 1, gdalconst.GDT_Float32)
    dst.SetGeoTransform( match_geotrans )
    dst.SetProjection( match_proj)
    if resample_method == 'nearest':
        gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_NearestNeighbour)
    elif resample_method == 'bilinear':
        gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)
    elif resample_method == 'cubic':
        gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Cubic)
    elif resample_method == 'cubicspline':
        gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_CubicSpline)
    del dst # Flush
    return None

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
    Returns lon/lat of all exteriors and interiors of a GeoDataFrame.
    '''
    lon = np.empty([0,1],dtype=float)
    lat = np.empty([0,1],dtype=float)
    for geom in gdf.geometry:
        lon_geom,lat_geom = get_lonlat_geometry(geom)
        lon = np.append(lon,lon_geom)
        lat = np.append(lat,lat_geom)
    return lon,lat

def buffer_gdf(gdf,buffer,AREA_THRESHOLD=1e6):
    '''
    Given an input gdf, will return a buffered gdf.
    '''
    lon_min,lat_min,lon_max,lat_max = gdf.total_bounds
    lon_center = np.mean([lon_min,lon_max])
    lat_center = np.mean([lat_min,lat_max])
    x_center,y_center,zone_center = deg2utm(lon_center,lat_center)
    epsg_center = utm2epsg(zone_center)
    gdf_utm = gdf.to_crs(f'EPSG:{epsg_center[0]}')
    gdf_utm = gdf_utm[gdf_utm.area>AREA_THRESHOLD].reset_index(drop=True)
    gdf_utm_buffered = gdf_utm.buffer(buffer)
    gdf_buffered = gdf_utm_buffered.to_crs('EPSG:4326')
    gdf_buffered = gpd.GeoDataFrame(geometry=[gdf_buffered.unary_union],crs='EPSG:4326')
    return gdf_buffered

def icesat2_df2array(df):
    '''
    Given an input df, will return a numpy array of the data.
    '''
    lon = np.asarray(df.lon)
    lat = np.asarray(df.lat)
    height = np.asarray(df.height)
    time = np.asarray(df.time)
    return lon,lat,height,time

def filter_utm(zone,epsg_code):
    '''
    Finds the indices of the utm zones that correspond to the given epsg code.
    '''
    epsg_zone = utm2epsg(zone)
    idx_epsg = epsg_code == epsg_zone
    return idx_epsg

def landmask_dem(lon,lat,lon_coast,lat_coast,landmask_c_file,inside_flag):
    #Given lon/lat of points, and lon/lat of coast (or any other boundary),
    #finds points inside the polygon. Boundary must be in the form of separate lon and lat arrays,
    #with polygons separated by NaNs
    c_float_p = c.POINTER(c.c_float)
    landmask_so_file = landmask_c_file.replace('.c','.so') #the .so file is created
    subprocess.run('cc -fPIC -shared -o ' + landmask_so_file + ' ' + landmask_c_file,shell=True)
    landmask_lib = c.cdll.LoadLibrary(landmask_so_file)
    arrx = (c.c_float * len(lon_coast))(*lon_coast)
    arry = (c.c_float * len(lat_coast))(*lat_coast)
    arrx_input = (c.c_float * len(lon))(*lon)
    arry_input = (c.c_float * len(lat))(*lat)
    landmask = np.zeros(len(lon),dtype=c.c_int)
    landmask_lib.pnpoly(c.c_int(len(lon_coast)),c.c_int(len(lon)),arrx,arry,arrx_input,arry_input,c.c_void_p(landmask.ctypes.data))
    landmask = landmask == inside_flag #just to be consistent and return Boolean array
    return landmask

def get_ortho_list(loc_dir):
    '''
    Get list of ortho images, using two different dir structures.
    '''
    full_ortho_list = glob.glob(loc_dir + 'UTM*/*/strips/*ortho.tif')
    full_ortho_list.extend(glob.glob(loc_dir + '*/strips/*ortho.tif'))
    full_ortho_list = np.asarray(full_ortho_list)
    full_ortho_list.sort()
    return full_ortho_list

def get_strip_list(ortho_list,input_type):
    '''
    Given an ortho list, find the corresponding strip list.
    Different input types:
        0: old and new methods
        1: old only (dem_browse.tif for 10 m and dem_smooth.tif for 2 m)
        2: new only (dem_10m.tif for 10 m and dem.tif for 2 m)
    '''
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
    '''
    Find extents of a given strip.
    Return will be lon/lat in EPSG:4326.
    '''
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
    lon_strip = [item[0] for item in geo_ext]
    lat_strip = [item[1] for item in geo_ext]
    lon_min = np.nanmin(lon_strip)
    lon_max = np.nanmax(lon_strip)
    lat_min = np.nanmin(lat_strip)
    lat_max = np.nanmax(lat_strip)
    return lon_min,lon_max,lat_min,lat_max


def get_gsw(output_dir,tmp_dir,gsw_dir,epsg_code,lon_min,lon_max,lat_min,lat_max,GSW_POCKET_THRESHOLD,GSW_CRS_TRANSFORM_THRESHOLD):
    '''
    Given lon/lat extents, find the biggest chunk(s) of the Global Surface Water dataset in that area.
    Extents might break up GSW into multiple chunks, GSW_POCKET_THRESHOLD sets the minimum size of a chunk.
    Returns a GeoDataFrame of the GSW and the filename of the shapefile it wrote to.
    '''
    epsg_code = str(epsg_code)
    output_name = output_dir.split('/')[-2]
    lon_min_rounded_gsw = int(np.floor(lon_min/10)*10)
    lon_max_rounded_gsw = int(np.floor(lon_max/10)*10)
    lat_min_rounded_gsw = int(np.ceil(lat_min/10)*10)
    lat_max_rounded_gsw = int(np.ceil(lat_max/10)*10)
    lon_gsw_range = range(lon_min_rounded_gsw,lon_max_rounded_gsw+10,10)
    lat_gsw_range = range(lat_min_rounded_gsw,lat_max_rounded_gsw+10,10)
    gsw_output_file = f'{tmp_dir}GSW_merged.tif'
    gsw_output_file_clipped = f'{tmp_dir}GSW_merged_clipped.tif'
    gsw_output_file_sea_only_clipped = f'{tmp_dir}GSW_merged_sea_only_clipped.tif'
    gsw_output_file_sea_only_clipped_transformed = f'{tmp_dir}GSW_merged_sea_only_clipped_transformed_{epsg_code}.tif'
    gsw_output_shp_file_sea_only_clipped_transformed = f'{tmp_dir}GSW_merged_sea_only_clipped_transformed_{epsg_code}.shp'
    gsw_output_shp_file_main_sea_only_clipped_transformed = f'{tmp_dir}{output_name}_GSW_merged_main_sea_only_clipped_transformed_{epsg_code}.shp'
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
    gsw_merge_command = f'gdal_merge.py -q -o {gsw_output_file} -co COMPRESS=LZW '
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
            gsw_file = f'{gsw_dir}extent_{np.abs(lon)}{EW_str}_{np.abs(lat)}{NS_str}_v1_1.tif '
            gsw_merge_command = gsw_merge_command + gsw_file
    subprocess.run(gsw_merge_command,shell=True)
    lonlat_str = f'{lon_min} {lat_min} {lon_max} {lat_max}'
    clip_command = f'gdalwarp -q -overwrite -te {lonlat_str} -co COMPRESS=LZW {gsw_output_file} {gsw_output_file_clipped}'
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
    label_IDs_sorted = np.argsort(size)[::-1] #sort the labels by size (descending, so biggest first), then remove label=0, because that one is land
    label_IDs_sorted = label_IDs_sorted[label_IDs_sorted != 0]
    gsw_clump = np.zeros(gsw_array.shape,dtype=int)
    for label_id in label_IDs_sorted:
        if size[label_id]/gsw_area < GSW_POCKET_THRESHOLD:
            break
        gsw_clump = gsw_clump + np.asarray(label==label_id,dtype=int)
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
        warp_command = f'gdalwarp -q -overwrite {gsw_output_file_sea_only_clipped} {gsw_output_file_sea_only_clipped_transformed} -s_srs EPSG:4326 -t_srs EPSG:{epsg_code}'
        subprocess.run(warp_command,shell=True)
    gsw_polygonize_command = f'gdal_polygonize.py -q {gsw_output_file_sea_only_clipped_transformed} -f \"ESRI Shapefile\" {gsw_output_shp_file_sea_only_clipped_transformed}'
    subprocess.run(gsw_polygonize_command,shell=True)
    gsw_shp_data = gpd.read_file(gsw_output_shp_file_sea_only_clipped_transformed)
    if len(gsw_shp_data) == 0:
        print('No coast in this region!')
        return None,None
    idx_area = np.asarray(gsw_shp_data.area)/np.sum(gsw_shp_data.area) > GSW_CRS_TRANSFORM_THRESHOLD
    gsw_main_sea_only = gsw_shp_data[idx_area].reset_index(drop=True)
    gsw_main_sea_only.to_file(gsw_output_shp_file_main_sea_only_clipped_transformed)
    subprocess.run(f'mv {gsw_output_shp_file_main_sea_only_clipped_transformed.replace(".shp",".*")} {output_dir}',shell=True)
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
    '''
    Returns the shapefile of the real outline of a strip,
    i.e. not the raster, but the actual area of valid data.
    '''
    if subprocess.os.path.exists(f'{tmp_dir}tmp_strip_binary.tif'):
        subprocess.os.remove(f'{tmp_dir}tmp_strip_binary.tif')
    if subprocess.os.path.exists(f'{tmp_dir}tmp_strip_binary.shp'):
        for fi in glob.glob(f'{tmp_dir}tmp_strip_binary.*'):
            subprocess.os.remove(fi)
    calc_command = f'gdal_calc.py -A {strip} --calc="A>-9999" --outfile={tmp_dir}tmp_strip_binary.tif --format=GTiff --co=\"COMPRESS=LZW\" --quiet'
    subprocess.run(calc_command,shell=True)
    polygonize_command = f'gdal_polygonize.py -q {tmp_dir}tmp_strip_binary.tif -f "ESRI Shapefile" {tmp_dir}tmp_strip_binary.shp'
    subprocess.run(polygonize_command,shell=True)
    wv_strip_shp = gpd.read_file(tmp_dir + 'tmp_strip_binary.shp')
    if subprocess.os.path.exists(tmp_dir + 'tmp_strip_binary.tif'):
        subprocess.os.remove(tmp_dir + 'tmp_strip_binary.tif')
    if subprocess.os.path.exists(tmp_dir + 'tmp_strip_binary.shp'):
        for fi in glob.glob(tmp_dir + 'tmp_strip_binary.*'):
            subprocess.os.remove(fi)
    return wv_strip_shp


def filter_strip_gsw(wv_strip_shp,gsw_shp_data,STRIP_AREA_THRESHOLD,POLYGON_AREA_THRESHOLD,GSW_OVERLAP_THRESHOLD,STRIP_TOTAL_AREA_PERCENTAGE_THRESHOLD):
    '''
    Filters the separate polygons of the strip outline with the GSW dataset.
    Removes the polygons that:
        are over fully over water
        are over water over a certain threshold
        are too small
    '''
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
        idx_total_area_percentage = np.asarray(wv_strip_shp_filtered_gsw.area/np.sum(wv_strip_shp_filtered_gsw.area) > STRIP_TOTAL_AREA_PERCENTAGE_THRESHOLD)
        wv_strip_shp_filtered_gsw = wv_strip_shp_filtered_gsw[idx_total_area_percentage].reset_index(drop=True)
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
        elif geom_1_2_intersection.geom_type == 'MultiLineString':
            return containment
        else:
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
        if contain_dt_flag[i] == False: #skip it if it's already contained by one other
            continue
        if N_STRIPS_CONTAINMENT < 2:
            continue
        idx_intersection = np.argwhere(np.asarray([strip_shp_data.geometry[i].intersects(geom) for geom in strip_shp_data.geometry])).squeeze()
        idx_intersection = np.delete(idx_intersection,np.where(idx_intersection==i))
        if len(idx_intersection) < 2: #need at least 2 intersecting strips to see if it's contained by union of 2 other strips
            continue
        idx_intersection_combinations = np.reshape(np.stack(np.meshgrid(idx_intersection,idx_intersection),-1),(len(idx_intersection)*len(idx_intersection),2))
        idx_intersection_combinations = np.delete(idx_intersection_combinations,idx_intersection_combinations[:,0]==idx_intersection_combinations[:,1],axis=0)
        idx_intersection_combinations = np.unique(np.sort(idx_intersection_combinations, axis=1), axis=0)
        idx_contained_combo = np.asarray([geometries_contained(strip_shp_data.geometry[combo[0]].union(strip_shp_data.geometry[combo[1]]),strip_shp_data.geometry[i],epsg_code,STRIP_CONTAINMENT_THRESHOLD) for combo in idx_intersection_combinations])
        idx_newer_strip = strip_dates_datetime[i] - strip_dates_datetime[idx_intersection_combinations] < datetime.timedelta(days=STRIP_DELTA_TIME_THRESHOLD)
        contain_dt_flag[i] = ~np.any(np.logical_and(idx_contained_combo,np.all(idx_newer_strip,axis=1)))
        if contain_dt_flag[i] == False: #skip it if it's already contained by two others
            continue
        if N_STRIPS_CONTAINMENT < 3:
            continue
        if len(idx_intersection) < 3: #need at least 3 intersecting strips to see if it's contained by union of 3 other strips
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

def populate_intersection(geom_intersection,gsw_main_sea_only_buffered,landmask_c_file,X_SPACING,Y_SPACING):
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
        landmask_intersection = landmask_dem(x_sampling,y_sampling,lon_intersection,lat_intersection,landmask_c_file,1)
        x_sampling_masked_intersection = x_sampling[landmask_intersection]
        y_sampling_masked_intersection = y_sampling[landmask_intersection]
        if gsw_main_sea_only_buffered is not None:
            lon_gsw,lat_gsw = get_lonlat_gdf(gsw_main_sea_only_buffered)
            landmask_gsw = landmask_dem(x_sampling_masked_intersection,y_sampling_masked_intersection,lon_gsw,lat_gsw,landmask_c_file,0)
            x_sampling_masked_intersection_gsw = x_sampling_masked_intersection[landmask_gsw]
            y_sampling_masked_intersection_gsw = y_sampling_masked_intersection[landmask_gsw]
        else:
            x_sampling_masked_intersection_gsw = x_sampling_masked_intersection
            y_sampling_masked_intersection_gsw = y_sampling_masked_intersection
        x = np.append(x,x_sampling_masked_intersection_gsw)
        y = np.append(y,y_sampling_masked_intersection_gsw)
    return x,y

def build_mosaic(strip_shp_data,gsw_main_sea_only_buffered,landmask_c_file,mosaic_dict,mosaic_dir,output_name,mosaic_number,epsg_code,X_SPACING,Y_SPACING,MOSAIC_TILE_SIZE):
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
        x_masked_total,y_masked_total = populate_intersection(geom_intersection,gsw_main_sea_only_buffered,landmask_c_file,X_SPACING,Y_SPACING)
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