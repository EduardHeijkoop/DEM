import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from osgeo import gdal,gdalconst,osr
from scipy import ndimage
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_tree
import glob
import os
import subprocess
import datetime
import ctypes as c
import multiprocessing
import itertools

def reinsert_nan(array_clean,array_nan):
    '''
    Reinserts nan values into a clean array, based on where they are in the nan array.
    '''
    array_clean = np.asarray(array_clean)
    array_nan = np.asarray(array_nan)
    idx_nan = np.atleast_1d(np.argwhere(np.isnan(array_nan)).squeeze())
    array_filled = array_clean
    for idx in idx_nan:
        array_filled = np.concatenate((array_filled[:idx],[np.nan],array_filled[idx:]))
    return array_filled

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
    if src_srs.GetAttrValue('AUTHORITY',1) == '4326':
        global_ext = local_ext
    else:
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

def rmse(x1,x2):
    if len(x1) != len(x2):
        print('Error: vectors must be same length')
        return None
    return np.sqrt(np.sum((x1-x2)**2/len(x1)))

def r_squared(x1,x2):
    if len(x1) != len(x2):
        print('Error: vectors must be same length')
        return None
    return 1 - np.sum((x1-x2)**2)/np.sum((x1-np.mean(x1))**2)

def mean_absolute_deviation(x):
    return np.mean(np.abs(x-np.mean(x)))

def sample_raster(raster_path, csv_path, output_file,nodata='-9999',header=None,proj='wgs84'):
    output_dir = os.path.dirname(output_file)
    raster_base = os.path.splitext(raster_path.split('/')[-1])[0]
    if header is not None:
        cat_command = f"tail -n+2 {csv_path} | cut -d, -f1-2 | sed 's/,/ /g' | gdallocationinfo -valonly -{proj} {raster_path} > tmp_{raster_base}.txt"
    else:
        cat_command = f"cat {csv_path} | cut -d, -f1-2 | sed 's/,/ /g' | gdallocationinfo -valonly -{proj} {raster_path} > tmp_{raster_base}.txt"
    subprocess.run(cat_command,shell=True,cwd=output_dir)
    fill_nan_command = f"awk '!NF{{$0=\"NaN\"}}1' tmp_{raster_base}.txt > tmp2_{raster_base}.txt"
    subprocess.run(fill_nan_command,shell=True,cwd=output_dir)
    if header is not None:
        subprocess.run(f"sed -i '1i {header}' tmp2_{raster_base}.txt",shell=True,cwd=output_dir)
    paste_command = f"paste -d , {csv_path} tmp2_{raster_base}.txt > {output_file}"
    subprocess.run(paste_command,shell=True,cwd=output_dir)
    subprocess.run(f"sed -i '/{nodata}/d' {output_file}",shell=True,cwd=output_dir)
    subprocess.run(f"sed -i '/NaN/d' {output_file}",shell=True,cwd=output_dir)
    subprocess.run(f"sed -i '/nan/d' {output_file}",shell=True,cwd=output_dir)
    subprocess.run(f"rm tmp_{raster_base}.txt tmp2_{raster_base}.txt",shell=True,cwd=output_dir)
    return None

def resample_raster(src_filename,match_filename,dst_filename,resample_method='bilinear',compress=True,nodata=-9999,quiet_flag=False):
    '''
    src = what you want to resample
    match = resample to this one's resolution
    dst = output
    method = nearest neighbor, bilinear (default), cubic, cubic spline
    '''
    src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()
    src_geotrans = src.GetGeoTransform()
    src_epsg = osr.SpatialReference(wkt=src_proj).GetAttrValue('AUTHORITY',1)
    match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize
    dst = gdal.GetDriverByName('GTiff').Create(dst_filename,wide,high,1,gdalconst.GDT_Float32)
    dst.SetGeoTransform(match_geotrans)
    dst.SetProjection(match_proj)
    if resample_method == 'nearest':
        gdal.ReprojectImage(src,dst,src_proj,match_proj,gdalconst.GRA_NearestNeighbour)
    elif resample_method == 'bilinear':
        gdal.ReprojectImage(src,dst,src_proj,match_proj,gdalconst.GRA_Bilinear)
    elif resample_method == 'cubic':
        gdal.ReprojectImage(src,dst,src_proj,match_proj,gdalconst.GRA_Cubic)
    elif resample_method == 'cubicspline':
        gdal.ReprojectImage(src,dst,src_proj,match_proj,gdalconst.GRA_CubicSpline)
    del dst # Flush
    if compress == True:
        compress_raster(dst_filename,nodata,quiet_flag)
    return None

def compress_raster(filename,nodata=-9999,quiet_flag = False):
    '''
    Compress a raster using gdal_translate
    '''
    file_ext = os.path.splitext(filename)[-1]
    tmp_filename = filename.replace(file_ext,f'_LZW{file_ext}')
    if nodata is not None:
        compress_command = f'gdal_translate -co "COMPRESS=LZW" -co "BIGTIFF=IF_SAFER" -a_nodata {nodata} {filename} {tmp_filename}'
    else:
        compress_command = f'gdal_translate -co "COMPRESS=LZW" -co "BIGTIFF=IF_SAFER" {filename} {tmp_filename}'
    if quiet_flag == True:
        compress_command = compress_command.replace('gdal_translate','gdal_translate -q')
    move_command = f'mv {tmp_filename} {filename}'
    subprocess.run(compress_command,shell=True)
    subprocess.run(move_command,shell=True)
    return None

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

def df_to_gdf(df,dt_threshold=0.01):
    '''
    dt_threshold given in seconds, then converted to ns
    '''
    dt_threshold = dt_threshold / 1e-9
    gdf = gpd.GeoDataFrame()
    df['date'] = [t[:10] for t in df.time]
    df['t_datetime'] = pd.to_datetime(df.time)
    unique_dates = np.unique(df.date)
    for ud in unique_dates:
        df_date = df[df.date == ud].copy()
        unique_beams = np.unique(df_date.beam)
        for beam in unique_beams:
            df_beam = df_date[df_date.beam == beam].copy()
            t_datetime = np.asarray(df_beam.t_datetime)
            dt = np.asarray(t_datetime[1:] - t_datetime[:-1])
            dt = np.append(0,dt).astype(int)
            idx_jump_orig = np.atleast_1d(np.argwhere(dt > dt_threshold).squeeze())
            idx_jump = np.concatenate([[0],idx_jump_orig,[len(df_beam)-1]])
            for k in range(len(idx_jump)-1):
                geom = shapely.geometry.LineString([(df_beam.lon.iloc[idx_jump[k]],df_beam.lat.iloc[idx_jump[k]]),(df_beam.lon.iloc[idx_jump[k+1]-1],df_beam.lat.iloc[idx_jump[k+1]-1])])
                tmp_gdf = gpd.GeoDataFrame(pd.DataFrame({'date':[ud],'beam':[beam]}),geometry=[geom],crs='EPSG:4326')
                gdf = pd.concat((gdf,tmp_gdf)).reset_index(drop=True)
    return gdf


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

def get_lonlat_gdf_center(gdf):
    lon_min,lat_min,lon_max,lat_max = gdf.total_bounds
    lon_center = (lon_min + lon_max) / 2
    lat_center = (lat_min + lat_max) / 2
    return lon_center,lat_center

def find_corner_points_gdf(lon,lat,gdf):
    lon_min,lat_min,lon_max,lat_max = gdf.total_bounds
    idx_ne = np.atleast_1d(np.argwhere(np.logical_and(lon == lon_max,lat == lat_max)).squeeze())
    idx_nw = np.atleast_1d(np.argwhere(np.logical_and(lon == lon_min,lat == lat_max)).squeeze())
    idx_se = np.atleast_1d(np.argwhere(np.logical_and(lon == lon_max,lat == lat_min)).squeeze())
    idx_sw = np.atleast_1d(np.argwhere(np.logical_and(lon == lon_min,lat == lat_min)).squeeze())
    idx_corners_points = np.concatenate([idx_ne,idx_nw,idx_se,idx_sw])
    return idx_corners_points

def buffer_gdf(gdf,buffer,AREA_THRESHOLD=1e6):
    '''
    Given an input gdf, will return a buffered gdf.
    '''
    lon_center,lat_center = get_lonlat_gdf_center(gdf)
    epsg_center = lonlat2epsg(lon_center,lat_center)
    gdf_utm = gdf.to_crs(f'EPSG:{epsg_center}')
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
    '''
    Given lon/lat of points, and lon/lat of coast (or any other boundary),
    finds points inside the polygon. Boundary must be in the form of separate lon and lat arrays,
    with polygons separated by NaNs
    '''
    c_float_p = c.POINTER(c.c_float)
    landmask_so_file = landmask_c_file.replace('.c','.so') #the .so file is created
    if not os.path.exists(landmask_so_file):
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

def parallel_landmask(lon_pts,lat_pts,lon_boundary,lat_boundary,landmask_c_file,inside_flag,N_cpus):
    '''
    Given lon/lat of points, and lon/lat of coast (or any other boundary),
    finds points inside the polygon. Boundary must be in the form of separate lon and lat arrays,
    with polygons separated by NaNs
    '''
    landmask_so_file = landmask_c_file.replace('.c','.so') #the .so file is created
    if not os.path.exists(landmask_so_file):
        subprocess.run('cc -fPIC -shared -o ' + landmask_so_file + ' ' + landmask_c_file,shell=True)
    landmask_lib = c.cdll.LoadLibrary(landmask_so_file)
    lon_split = np.array_split(lon_pts,N_cpus)
    lat_split = np.array_split(lat_pts,N_cpus)
    ir = itertools.repeat
    p = multiprocessing.Pool(N_cpus)
    landmask = p.starmap(parallel_pnpoly,zip(lon_split,lat_split,ir(lon_boundary),ir(lat_boundary),ir(landmask_so_file)))
    p.close()
    landmask = np.concatenate(landmask)
    landmask = landmask == inside_flag
    return landmask

def parallel_pnpoly(lon_pts,lat_pts,lon_boundary,lat_boundary,landmask_so_file):
    landmask_lib = c.cdll.LoadLibrary(landmask_so_file)
    arrx = (c.c_float * len(lon_boundary))(*lon_boundary)
    arry = (c.c_float * len(lat_boundary))(*lat_boundary)
    arrx_input = (c.c_float * len(lon_pts))(*lon_pts)
    arry_input = (c.c_float * len(lat_pts))(*lat_pts)
    landmask = np.zeros(len(lon_pts),dtype=c.c_int)
    landmask_lib.pnpoly(c.c_int(len(lon_boundary)),c.c_int(len(lon_pts)),arrx,arry,arrx_input,arry_input,c.c_void_p(landmask.ctypes.data))
    return landmask

def mask_csv_file(csv_file,shp_file,landmask_c_file,inside_flag=1,parallel_flag=True,N_cpus=1):
    '''
    Given an input CSV file and a shapefile, masks the points in/outside the shapefile.
    '''
    df = pd.read_csv(csv_file)
    csv_file_masked = f'{os.path.splitext(csv_file)[0]}_masked.{os.path.splitext(csv_file)[1]}'
    lon = np.asarray(df.lon)
    lat = np.asarray(df.lat)
    gdf = gpd.read_file(shp_file)
    lon_boundary,lat_boundary = get_lonlat_gdf(gdf)
    if parallel_flag == True:
        landmask = parallel_landmask(lon,lat,lon_boundary,lat_boundary,landmask_c_file,inside_flag,N_cpus)
    else:
        landmask = landmask_dem(lon,lat,lon_boundary,lat_boundary,landmask_c_file,inside_flag)
    df_masked = df[landmask].reset_index(drop=True)
    df_masked.to_csv(csv_file_masked,index=False,float_format='%.6f')
    return df_masked

def get_strip_list(loc_dir,input_type=0,corrected_flag=False,dir_structure='sealevel'):
    '''
    Different input types:
        0: old and new methods
        1: old only (dem_browse.tif for 10 m and dem_smooth.tif for 2 m)
        2: new only (dem_10m.tif for 10 m and dem.tif for 2 m)
        3: input is from a list of strips
    Dir structure:
        sealevel: finds strips in WV*/strips/ subdirectories
        simple: finds all *dem.tif strips in a given location
    '''
    if dir_structure == 'sealevel':
        if corrected_flag == True:
            strip_list_old = sorted(glob.glob(f'{loc_dir}*/strips/*dem_smooth_Shifted*.tif'))
            strip_list_old.extend(sorted(glob.glob(f'{loc_dir}UTM*/*/strips/*dem_smooth_Shifted*.tif')))
            strip_list_new = sorted(glob.glob(f'{loc_dir}*/strips/*dem_Shifted*.tif'))
            strip_list_new.extend(sorted(glob.glob(f'{loc_dir}UTM*/*/strips/*dem_Shifted*.tif')))
        else:
            strip_list_old = sorted(glob.glob(f'{loc_dir}*/strips/*dem_smooth.tif'))
            strip_list_old.extend(sorted(glob.glob(f'{loc_dir}UTM*/*/strips/*dem_smooth.tif')))
            strip_list_new = sorted(glob.glob(f'{loc_dir}*/strips/*dem.tif'))
            strip_list_new.extend(sorted(glob.glob(f'{loc_dir}UTM*/*/strips/*dem.tif')))
        if input_type == 0:
            strip_list = strip_list_old
            strip_list.extend(strip_list_new)
        elif input_type == 1:
            strip_list = strip_list_old
        elif input_type == 2:
            strip_list = strip_list_new
    elif dir_structure == 'simple':
        if corrected_flag == True:
            strip_list = sorted(glob.glob(f'{loc_dir}*dem_Shifted*.tif'))
        else:
            strip_list = sorted(glob.glob(f'{loc_dir}*dem.tif'))
    return np.asarray(strip_list)

def get_strip_extents(strip,round_flag=False,N_round=3):
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
    if round_flag == True:
        lon_min = np.floor(lon_min * 10**N_round) / 10**N_round
        lon_max = np.ceil(lon_max * 10**N_round) / 10**N_round
        lat_min = np.floor(lat_min * 10**N_round) / 10**N_round
        lat_max = np.ceil(lat_max * 10**N_round) / 10**N_round
    return lon_min,lon_max,lat_min,lat_max

def get_list_extents(file_list,round_flag=False,N_round=3):
    lon_min,lon_max,lat_min,lat_max = 180,-180,90,-90
    for f in file_list:
        lon_min_single_file,lon_max_single_file,lat_min_single_file,lat_max_single_file = get_strip_extents(f,round_flag=round_flag,N_round=N_round)
        lon_min = np.min((lon_min,lon_min_single_file))
        lon_max = np.max((lon_max,lon_max_single_file))
        lat_min = np.min((lat_min,lat_min_single_file))
        lat_max = np.max((lat_max,lat_max_single_file))
    return lon_min,lon_max,lat_min,lat_max

def get_gsw(output_dir,tmp_dir,gsw_dir,epsg_code,lon_min,lon_max,lat_min,lat_max,loc_name=None,gsw_pocket_threshold=0.01,gsw_crs_transform_threshold=0.05):
    '''
    Given lon/lat extents, find the biggest chunk(s) of the Global Surface Water dataset in that area.
    Extents might break up GSW into multiple chunks, GSW_POCKET_THRESHOLD sets the minimum size of a chunk.
    Returns a GeoDataFrame of the GSW and the filename of the shapefile it wrote to.
    '''
    epsg_code = str(epsg_code)
    if loc_name is None:
        output_name = output_dir.split('/')[-2]
    else:
        output_name = loc_name
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
    gsw_output_shp_file_main_sea_only_clipped_transformed = f'{output_dir}{output_name}_GSW_merged_main_sea_only_clipped_transformed_{epsg_code}.shp'
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
    tile_count = 0
    total_count = 0
    for lon in lon_gsw_range:
        for lat in lat_gsw_range:
            total_count += 1
            if np.logical_or(lat>80,lat<-50):
                continue # GSW doesn't have data for these regions
            tile_count += 1
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
    if tile_count == 0:
        print('No GSW tiles found for this area')
        return None,None
    elif tile_count < total_count:
        print(f'Warning! Some tiles not found for this area.')
        print(f'Doing GSW analysis for {tile_count} tiles instead of {total_count} tiles.')
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
        if size[label_id]/gsw_area < gsw_pocket_threshold:
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
    gdf_gsw_shp_data = gpd.read_file(gsw_output_shp_file_sea_only_clipped_transformed)
    if len(gdf_gsw_shp_data) == 0:
        print('No coast in this region!')
        return None,None
    idx_area = np.asarray(gdf_gsw_shp_data.area)/np.sum(gdf_gsw_shp_data.area) > gsw_crs_transform_threshold
    gdf_gsw_main_sea_only = gdf_gsw_shp_data[idx_area].reset_index(drop=True)
    gdf_gsw_main_sea_only.to_file(gsw_output_shp_file_main_sea_only_clipped_transformed)
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
    return gdf_gsw_main_sea_only,gsw_output_shp_file_main_sea_only_clipped_transformed

def get_strip_shp(strip,tmp_dir):
    '''
    Returns the shapefile of the real outline of a strip,
    i.e. not the raster, but the actual area of valid data.
    '''
    strip_base = os.path.splitext(os.path.basename(strip))[0]
    if subprocess.os.path.exists(f'{tmp_dir}{strip_base}_binary.tif'):
        subprocess.os.remove(f'{tmp_dir}{strip_base}_binary.tif')
    if subprocess.os.path.exists(f'{tmp_dir}{strip_base}_binary.shp'):
        for fi in glob.glob(f'{tmp_dir}{strip_base}_binary.*'):
            subprocess.os.remove(fi)
    calc_command = f'gdal_calc.py -A {strip} --calc=\"A>-9999\" --outfile={tmp_dir}{strip_base}_binary.tif --format=GTiff --co=\"COMPRESS=LZW\" --co=\"BIGTIFF=IF_SAFER\" --quiet'
    subprocess.run(calc_command,shell=True)
    polygonize_command = f'gdal_polygonize.py -q {tmp_dir}{strip_base}_binary.tif -f "ESRI Shapefile" {tmp_dir}{strip_base}_binary.shp'
    subprocess.run(polygonize_command,shell=True)
    wv_strip_shp = gpd.read_file(f'{tmp_dir}{strip_base}_binary.shp')
    if subprocess.os.path.exists(f'{tmp_dir}{strip_base}_binary.tif'):
        subprocess.os.remove(f'{tmp_dir}{strip_base}_binary.tif')
    if subprocess.os.path.exists(f'{tmp_dir}{strip_base}_binary.shp'):
        for fi in glob.glob(f'{tmp_dir}{strip_base}_binary.*'):
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
        generation_count = -1
        ref_list = np.empty([0,1],dtype=int)
        src_list = np.empty([0,1],dtype=int)
        print(' ')
        print(f'Mosaic {i}:')
        print(f'Starting at: {mosaic_ID_start}')
        print(strip_dict[mosaic_ID_start])
        generation_dict = {}
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
            generation_dict[generation_count] = {
                'ref':gen_ref_list,
                'src':gen_src_list
            }
            if np.array_equal(np.sort(done_list),groups_dict[str(i)]['group']):
                path_check = True
                break
            current_gen = next_gen
            next_gen = np.empty([0,1],dtype=int)
        mosaic_dict[i] = generation_dict
    print('')
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

def parallel_coregistration(ref_strip_ID,src_strip_ID,strip_shp_data,gsw,landmask_c_file,mosaic_dir,tmp_dir,horizontal_flag,X_SPACING,Y_SPACING,X_MAX_SEARCH,Y_MAX_SEARCH):
    ref_strip = strip_shp_data.strip[ref_strip_ID]
    src_strip = strip_shp_data.strip[src_strip_ID]
    ref_strip_sensor = ref_strip.split('/')[-1].split('_')[0]
    src_strip_sensor = src_strip.split('/')[-1].split('_')[0]
    src_seg = f'seg{src_strip.split("seg")[1].split("_")[0]}'
    src_base = f'{mosaic_dir}{os.path.splitext(src_strip.split("/")[-1].split(src_seg)[0])[0]}{src_seg}'
    src_ext = os.path.splitext(src_strip)[1]
    src_file = glob.glob(f'{src_base}*{src_ext}')[0]
    print(f'Linking {ref_strip_ID} ({ref_strip_sensor}) to {src_strip_ID} ({src_strip_sensor})...')
    geom_intersection = strip_shp_data.geometry[src_strip_ID].intersection(strip_shp_data.geometry[ref_strip_ID])
    x_masked_total,y_masked_total = populate_intersection(geom_intersection,gsw,landmask_c_file,X_SPACING,Y_SPACING)
    strip_sampled_file = f'{mosaic_dir}Mosaic_sampled_{src_strip_ID}_for_coregistering_{ref_strip_ID}.txt'
    np.savetxt(strip_sampled_file,np.c_[x_masked_total,y_masked_total],fmt='%.3f',delimiter=' ')
    df_sampled = sample_two_rasters(src_file,ref_strip,strip_sampled_file,src_strip_ID,ref_strip_ID)
    if horizontal_flag == True:
        x_res = gdal.Open(src_strip).GetGeoTransform()[1]
        y_res = -gdal.Open(src_strip).GetGeoTransform()[5]
        x_shift,y_shift = evaluate_horizontal_shift(df_sampled,ref_strip,tmp_dir,x_res=x_res,y_res=y_res,x_offset_max=X_MAX_SEARCH,y_offset_max=Y_MAX_SEARCH,primary_ID=src_strip_ID,secondary_ID=ref_strip_ID)
        if ~np.logical_and(x_shift==0,y_shift==0):
            x_min,x_max,y_min,y_max = get_raster_extents(ref_strip,'local')
            horizontal_shift_str = f'x_{x_shift:.2f}m_y_{y_shift:.2f}m'.replace('.','p').replace('-','neg')
            if 'Shifted' in ref_strip:
                new_ref_strip = f'{tmp_dir}{os.path.basename(ref_strip).replace("Shifted",f"Shifted_{horizontal_shift_str}")}'
            else:
                new_ref_strip = f'{tmp_dir}{os.path.splitext(os.path.basename(ref_strip))[0]}_Shifted_{horizontal_shift_str}.tif'
            translate_command = f'gdal_translate -q -a_ullr {x_min + x_shift} {y_max + y_shift} {x_max + x_shift} {y_min + y_shift} -co "COMPRESS=LZW" -co "BIGTIFF=YES" {ref_strip} {new_ref_strip}'
            subprocess.run(translate_command,shell=True)
            df_sampled = sample_two_rasters(src_file,new_ref_strip,strip_sampled_file)
            ref_strip_shifted,vertical_shift,rmse,ratio_pts,df_sampled_new = vertical_shift_raster(new_ref_strip,df_sampled,mosaic_dir)
            subprocess.run(f'rm {new_ref_strip}',shell=True)
        else:
            ref_strip_shifted,vertical_shift,rmse,ratio_pts,df_sampled_new = vertical_shift_raster(ref_strip,df_sampled,mosaic_dir)
    else:
        x_shift = 0
        y_shift = 0
        ref_strip_shifted,vertical_shift,rmse,ratio_pts,df_sampled_new = vertical_shift_raster(ref_strip,df_sampled,mosaic_dir)
    print(f'Results for {ref_strip_ID} ({ref_strip_sensor}) to {src_strip_ID} ({src_strip_sensor}):')
    print(f'Retained {ratio_pts*100:.1f}% of points.')
    print(f'Vertical shift: {vertical_shift:.2f} m')
    print(f'RMSE: {rmse:.2f} m')
    coreg_stats_file = f'{mosaic_dir}/Mosaic_Coreg_{ref_strip_ID}_to_{src_strip_ID}.txt'
    coreg_stats = open(coreg_stats_file,'w')
    coreg_stats.write(f'{ref_strip_ID},{src_strip_ID},{x_shift:.2f},{y_shift:.2f},{vertical_shift:.2f},{rmse:.2f}\n')
    coreg_stats.close()


def build_mosaic(strip_shp_data,gsw_main_sea_only_buffered,landmask_c_file,mosaic_dict,mosaic_dir,tmp_dir,output_name,mosaic_number,epsg_code,horizontal_flag,X_SPACING,Y_SPACING,X_MAX_SEARCH,Y_MAX_SEARCH,MOSAIC_TILE_SIZE,N_cpus):
    '''
    Build the mosaic, given list of indices of strips to co-register to each other
    Co-registering ref to src, by sampling src and treating that as truth
    '''
    mosaic_stats_file = mosaic_dir + output_name + f'_Mosaic_{mosaic_number}_{epsg_code}_Statistics.txt'
    mosaic_stats = open(mosaic_stats_file,'w')
    mosaic_stats.write('ref_strip_ID,src_strip_ID,x_shift,y_shift,z_shift,RMSE\n')
    mosaic_stats.close()
    strip_list_coregistered = np.empty([0,1],dtype=str)
    ir = itertools.repeat
    for i in range(len(mosaic_dict)):
        if i == 0:
            orig_id = mosaic_dict[i]['src'][0]
            orig_strip = strip_shp_data.strip[orig_id]
            orig_strip_new = f'{mosaic_dir}{os.path.basename(orig_strip)}'
            subprocess.run(f'cp {orig_strip} {orig_strip_new}',shell=True)
            strip_list_coregistered = np.append(strip_list_coregistered,orig_strip_new)
        gen_ref_list = mosaic_dict[i]['ref']
        gen_src_list = mosaic_dict[i]['src']

        p = multiprocessing.Pool(np.min((len(gen_ref_list),N_cpus)))
        p.starmap(parallel_coregistration,zip(
            gen_ref_list,gen_src_list,
            ir(strip_shp_data),ir(gsw_main_sea_only_buffered),ir(landmask_c_file),
            ir(mosaic_dir),ir(tmp_dir),ir(horizontal_flag),
            ir(X_SPACING),ir(Y_SPACING),ir(X_MAX_SEARCH),ir(Y_MAX_SEARCH)
            ))
        p.close()
        for ref,src in zip(gen_ref_list,gen_src_list):
            coreg_file = f'{mosaic_dir}/Mosaic_Coreg_{ref}_to_{src}.txt'
            subprocess.run(f'cat {coreg_file} >> {mosaic_stats_file}',shell=True)
            subprocess.run(f'rm {coreg_file}',shell=True)
            ref_strip = strip_shp_data.strip[ref]
            ref_seg = f'seg{ref_strip.split("seg")[1].split("_")[0]}'
            ref_base = f'{mosaic_dir}{os.path.splitext(ref_strip.split("/")[-1].split(ref_seg)[0])[0]}{ref_seg}'
            ref_ext = os.path.splitext(ref_strip)[1]
            ref_file = glob.glob(f'{ref_base}*{ref_ext}')[0]
            strip_list_coregistered = np.append(strip_list_coregistered,ref_file)
            strip_sampled_file = f'{mosaic_dir}Mosaic_sampled_{src}_for_coregistering_{ref}.txt'
            full_strip_sampled_file = f'{mosaic_dir}{output_name}_Mosaic_{mosaic_number}_{epsg_code}_sampled_{src}_for_coregistering_{ref}.txt'
            subprocess.run(f'mv {strip_sampled_file} {full_strip_sampled_file}',shell=True)
    print('')
    print('Mosaicing...')
    strip_list_coregistered_date = np.asarray([int(s.split('/')[-1][5:13]) for s in strip_list_coregistered])
    idx_date_coregistered_strip = np.argsort(-strip_list_coregistered_date)
    strip_list_coregistered_sorted = np.array(strip_list_coregistered)[idx_date_coregistered_strip.astype(int)]
    strip_list_coregistered_sorted_file = f'{mosaic_dir}{output_name}_Mosaic_{mosaic_number}_{epsg_code}_Coregistered_Strips_Sorted_Date.txt'
    np.savetxt(strip_list_coregistered_sorted_file,np.c_[strip_list_coregistered_sorted],fmt='%s')
    mosaic_results_file = f'{mosaic_dir}{output_name}_Mosaic_{mosaic_number}_{epsg_code}_results.txt'
    mosaic_command = f'dem_mosaic --tap -l {strip_list_coregistered_sorted_file} --first --georef-tile-size {MOSAIC_TILE_SIZE} -o {mosaic_dir+output_name}_Mosaic_{mosaic_number}_{epsg_code} > {mosaic_results_file}'
    subprocess.run(mosaic_command,shell=True)
    merge_mosaic_output_file = f'{mosaic_dir}{output_name}_Full_Mosaic_{mosaic_number}_{epsg_code}.tif'
    merge_mosaic_output_file_vrt = merge_mosaic_output_file.replace('.tif','.vrt')
    merge_mosaic_vrt_command = f'gdalbuildvrt -q {merge_mosaic_output_file_vrt} {mosaic_dir+output_name}_Mosaic_{mosaic_number}_{epsg_code}-tile-*-first.tif'
    translate_mosaic_command = f'gdal_translate -co COMPRESS=LZW -co BIGTIFF=YES -q {merge_mosaic_output_file_vrt} {merge_mosaic_output_file}'
    subprocess.run(merge_mosaic_vrt_command,shell=True)
    subprocess.run(translate_mosaic_command,shell=True)
    subprocess.run(f'rm {mosaic_dir}{output_name}_Mosaic_{mosaic_number}_{epsg_code}-tile-*-first.tif',shell=True)
    print('')
    return merge_mosaic_output_file

def evaluate_horizontal_shift(df_sampled,raster_secondary,tmp_dir,x_res=2.0,y_res=2.0,x_offset_max=8.0,y_offset_max=8.0,primary_ID='',secondary_ID=''):
    x_offset = np.arange(-1*x_offset_max,x_offset_max+x_res,x_res)
    y_offset = np.arange(-1*y_offset_max,y_offset_max+y_res,y_res)
    rmse_min = np.inf
    x_opt = x_offset[0]
    y_opt = y_offset[0]
    for x in x_offset:
        for y in y_offset:
            df_offset = df_sampled.copy()
            df_offset = df_offset.drop(columns='h_secondary')
            output_file = f'{tmp_dir}offset_x_{x}_y_{y}_{secondary_ID}_to_{primary_ID}.txt'.replace('-','neg').replace('.','p').replace('ptxt','.txt')
            df_offset['x'] = df_offset['x'] - x
            df_offset['y'] = df_offset['y'] - y
            df_offset.to_csv(output_file,columns=['x','y'],float_format='%.1f',sep=' ',index=False,header=False)
            cat_command = f'cat {output_file} | gdallocationinfo -valonly -geoloc {raster_secondary} > {tmp_dir}tmp_sampled_{secondary_ID}_to_{primary_ID}.txt'
            subprocess.run(cat_command,shell=True)
            subprocess.run(f"sed -i 's/^$/-9999/g' {tmp_dir}tmp_sampled_{secondary_ID}_to_{primary_ID}.txt",shell=True)
            df_offset['h_secondary'] = np.loadtxt(f'{tmp_dir}tmp_sampled_{secondary_ID}_to_{primary_ID}.txt')
            df_offset = df_offset.dropna()
            idx_9999 = np.logical_or(df_offset['h_primary'] == -9999,df_offset['h_secondary'] == -9999)
            df_offset = df_offset[~idx_9999].reset_index(drop=True)
            vertical_shift,df_offset_filtered = calculate_shift(df_offset)
            dh = df_offset_filtered['h_primary'] - df_offset_filtered['h_secondary']
            rmse = np.sqrt(np.sum(dh**2)/len(dh))
            subprocess.run(f'rm {output_file} {tmp_dir}tmp_sampled_{secondary_ID}_to_{primary_ID}.txt',shell=True)
            if x == 0 and y == 0:
                rmse_zero = rmse
            if rmse < rmse_min:
                rmse_min = rmse
                x_opt = x
                y_opt = y
    print(f'Optimal horizontal shift: x = {x_opt} m, y = {y_opt} m')
    print(f'RMSE (zero shift): {rmse_zero:.2f} m, RMSE (optimal shift): {rmse_min:.2f} m')
    print(f'Relative RMSE: {rmse_min/rmse_zero:.2f}')
    return x_opt,y_opt

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

def sample_two_rasters(raster_primary,raster_secondary,csv_path,primary_ID='',secondary_ID=''):
    output_file = f'tmp_output_{secondary_ID}_to_{primary_ID}.txt'
    cat_primary_command = f"cat {csv_path} | gdallocationinfo -valonly -geoloc {raster_primary} > tmp_primary_{secondary_ID}_to_{primary_ID}.txt"
    cat_secondary_command = f"cat {csv_path} | gdallocationinfo -valonly -geoloc {raster_secondary} > tmp_secondary_{secondary_ID}_to_{primary_ID}.txt"
    subprocess.run(cat_primary_command,shell=True)
    subprocess.run(cat_secondary_command,shell=True)
    fill_nan_primary_command = f"awk '!NF{{$0=\"NaN\"}}1' tmp_primary_{secondary_ID}_to_{primary_ID}.txt > tmp2_primary_{secondary_ID}_to_{primary_ID}.txt"
    fill_nan_secondary_command = f"awk '!NF{{$0=\"NaN\"}}1' tmp_secondary_{secondary_ID}_to_{primary_ID}.txt > tmp2_secondary_{secondary_ID}_to_{primary_ID}.txt"
    subprocess.run(fill_nan_primary_command,shell=True)
    subprocess.run(fill_nan_secondary_command,shell=True)
    paste_command = f"paste -d , {csv_path} tmp2_primary_{secondary_ID}_to_{primary_ID}.txt tmp2_secondary_{secondary_ID}_to_{primary_ID}.txt > {output_file}"
    subprocess.run(paste_command,shell=True)
    subprocess.run(f"sed -i 's/ /,/g' {output_file}",shell=True)
    subprocess.run(f"sed -i '/-9999/d' {output_file}",shell=True)
    subprocess.run(f"sed -i '/NaN/d' {output_file}",shell=True)
    subprocess.run(f"sed -i '/nan/d' {output_file}",shell=True)
    df = pd.read_csv(output_file,header=None,names=['x','y','h_primary','h_secondary'],dtype={'x':'float','y':'float','h_primary':'float','h_secondary':'float'})
    subprocess.run(f"rm tmp_primary_{secondary_ID}_to_{primary_ID}.txt tmp2_primary_{secondary_ID}_to_{primary_ID}.txt tmp_secondary_{secondary_ID}_to_{primary_ID}.txt tmp2_secondary_{secondary_ID}_to_{primary_ID}.txt {output_file}",shell=True)
    return df

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

def calculate_shift(df_sampled,mean_median_mode='mean',n_sigma_filter=2,vertical_shift_iterative_threshold=0.02,printing=False,primary='h_primary',secondary='h_secondary'):
    df_sampled = df_sampled.rename(columns={primary:'h_primary',secondary:'h_secondary'})
    count = 0
    cumulative_shift = 0
    original_len = len(df_sampled)
    h_primary_original = np.asarray(df_sampled.h_primary)
    h_secondary_original = np.asarray(df_sampled.h_secondary)
    dh_original = h_primary_original - h_secondary_original
    rmse_original = np.sqrt(np.sum(dh_original**2)/len(dh_original))
    while True:
        count = count + 1
        h_primary = np.asarray(df_sampled.h_primary)
        h_secondary = np.asarray(df_sampled.h_secondary)
        dh = h_primary - h_secondary
        dh_filter = filter_outliers(dh,mean_median_mode,n_sigma_filter)
        if mean_median_mode == 'mean':
            incremental_shift = np.mean(dh[dh_filter])
        elif mean_median_mode == 'median':
            incremental_shift = np.median(dh[dh_filter])
        df_sampled = df_sampled[dh_filter].reset_index(drop=True)
        df_sampled.h_secondary = df_sampled.h_secondary + incremental_shift
        cumulative_shift = cumulative_shift + incremental_shift
        if printing == True:
            print(f'Iteration        : {count}')
            print(f'Incremental shift: {incremental_shift:.2f} m\n')
        if np.abs(incremental_shift) <= vertical_shift_iterative_threshold:
            break
        if count == 15:
            break
    h_primary_filtered = np.asarray(df_sampled.h_primary)
    h_secondary_filtered = np.asarray(df_sampled.h_secondary)
    dh_filtered = h_primary_filtered - h_secondary_filtered
    rmse_filtered = np.sqrt(np.sum(dh_filtered**2)/len(dh_filtered))
    if printing == True:
        print(f'Number of iterations: {count}')
        print(f'Number of points before filtering: {original_len}')
        print(f'Number of points after filtering: {len(df_sampled)}')
        print(f'Retained {len(df_sampled)/original_len*100:.1f}% of points.')
        print(f'Cumulative shift: {cumulative_shift:.2f} m')
        print(f'RMSE before filtering: {rmse_original:.2f} m')
        print(f'RMSE after filtering: {rmse_filtered:.2f} m')
    return cumulative_shift,df_sampled


def vertical_shift_raster(raster_path,df_sampled,output_dir,mean_median_mode='mean',n_sigma_filter=2,vertical_shift_iterative_threshold=0.02,primary='h_primary',secondary='h_secondary',return_df=False,write_df=False,sampled_file='tmp.txt'):
    src = gdal.Open(raster_path,gdalconst.GA_ReadOnly)
    raster_nodata = src.GetRasterBand(1).GetNoDataValue()
    vertical_shift,df_new = calculate_shift(df_sampled,mean_median_mode,n_sigma_filter,vertical_shift_iterative_threshold,primary=primary,secondary=secondary)
    raster_base,raster_ext = os.path.splitext(raster_path.split('/')[-1])
    if 'Shifted' in raster_base:
        if 'Shifted_x' in raster_base:
            if '_z_' in raster_base:
                #case: input is Shifted_x_0.00m_y_0.00m_z_0.00m*.tif
                original_shift = float(raster_base.split('Shifted')[1].split('_z_')[1].split('_')[0].replace('p','.').replace('neg','-').replace('m',''))
                original_shift_str = f'{original_shift}'.replace(".","p").replace("-","neg")
                new_shift = original_shift + vertical_shift
                new_shift_str = f'{new_shift:.2f}'.replace('.','p').replace('-','neg')
                raster_shifted = f'{output_dir}{raster_base}{raster_ext}'.replace(original_shift_str,new_shift_str)
            else:
                #case: input is Shifted_x_0.00m_y_0.00m*.tif
                vertical_shift_str = f'{vertical_shift:.2f}'.replace('.','p').replace('-','neg')
                post_string_fill = "_".join(raster_base.split("_y_")[1].split("_")[1:])
                if len(post_string_fill) == 0:
                    raster_shifted = f'{output_dir}{raster_base}{raster_ext}'.replace(raster_ext,f'_z_{vertical_shift_str}m{raster_ext}')
                else:
                    raster_shifted = f'{output_dir}{raster_base.split(post_string_fill)[0]}z_{vertical_shift_str}m_{post_string_fill}{raster_ext}'
        elif 'Shifted_z' in raster_base:
            #case: input is Shifted_z_0.00m*.tif
            original_shift = float(raster_base.split('Shifted')[1].split('_z_')[1].split('_')[0].replace('p','.').replace('neg','-').replace('m',''))
            new_shift = original_shift + vertical_shift
            raster_shifted = f'{output_dir}{raster_base.split("Shifted")[0]}Shifted_z_{"{:.2f}".format(new_shift).replace(".","p").replace("-","neg")}m{raster_ext}'
    else:
        #case: input is *.tif
        raster_shifted = f'{output_dir}{raster_base}_Shifted_z_{"{:.2f}".format(vertical_shift).replace(".","p").replace("-","neg")}m{raster_ext}'
    shift_command = f'gdal_calc.py --quiet -A {raster_path} --outfile={raster_shifted} --calc="A+{vertical_shift:.2f}" --NoDataValue={raster_nodata} --co "COMPRESS=LZW" --co "BIGTIFF=IF_SAFER" --co "TILED=YES"'
    subprocess.run(shift_command,shell=True)
    rmse = np.sqrt(np.sum((df_new.h_primary-df_new.h_secondary)**2)/len(df_new))
    ratio_pts = len(df_new)/len(df_sampled)
    # print(f'Retained {len(df_new)/len(df_sampled)*100:.1f}% of points.')
    # print(f'Vertical shift: {vertical_shift:.2f} m')
    # print(f'RMSE: {rmse:.2f} m')
    df_new.rename(columns={'h_primary':primary,'h_secondary':secondary},inplace=True)
    if write_df == True:
        if sampled_file == 'tmp.txt':
            print('Writing to tmp.txt file, please specify sampled_file to write to a different file.')
        df_new.to_csv(sampled_file,index=False,float_format='%.6f')
    if return_df == True:
        return raster_shifted,vertical_shift,rmse,ratio_pts,df_new
    else:
        return raster_shifted,vertical_shift,rmse,ratio_pts,None

def find_cloud_water(gdf_strips,df_cloud_water):
    '''
    Given a GeoDataFrame with strip outlines and a DataFrame with their cloud/water content, link them
    Caveat: all vertical exceedance is attributed to clouds, but other factors may be the cause too
    Starting them at -1 will ensure that they won't get thrown out if they're not found.
    '''
    cloud_array = -1*np.ones(len(gdf_strips))
    water_array = -1*np.ones(len(gdf_strips))
    strip_array_gdf = np.asarray([s.split('/')[-1] for s in gdf_strips.strip])
    strip_array_df = np.asarray([s.split('/')[-1] for s in df_cloud_water.Strip])
    for i in range(len(gdf_strips)):
        idx = np.atleast_1d(np.argwhere(strip_array_gdf[i] == strip_array_df).squeeze())
        if len(idx) == 0:
            continue
        idx = idx[0]
        cloud_array[i] = df_cloud_water['Percent Exceedance'][idx]
        water_array[i] = df_cloud_water['Percent Water'][idx]
    gdf_strips['Percent Exceedance'] = cloud_array
    gdf_strips['Percent Water'] = water_array
    return gdf_strips

