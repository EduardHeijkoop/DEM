import numpy as np
import subprocess
import os
from osgeo import gdal,gdalconst,osr

'''
Given a longitude and latitude extent, download global DEM tiles from:
    -SRTM (30 m)
    -ASTER (30 m)
    -Copernicus (30 m)
Optional correction to WGS 84 ellipsoidal heights if EGM96 (SRTM & ASTER) or EGM2008 (Copernicus) is provided
Requirements:
    -GDAL CLI installed (gdal_merge, gdal_calc and gdal_translate)
    -For SRTM and ASTER a NASA EarthData account is required
    -For Copernicus the AWS command line tool is required
'''

def copy_nan(old_raster,new_raster,old_nan_value,new_nan_value):
    copy_nan_command = f'gdal_calc.py -A {old_raster} -B {new_raster} --outfile=tmp.tif --calc="numpy.where(numpy.equal(A,{old_nan_value}),{new_nan_value},B)" --NoDataValue={old_nan_value} --format=GTiff --co=\"COMPRESS=LZW\" --co=\"BIGTIFF=IF_SAFER\" --quiet'
    mv_command = f'mv tmp.tif {new_raster}'
    set_nodata_command = f'gdal_edit.py -a_nodata {new_nan_value} {new_raster}'
    subprocess.run(copy_nan_command,shell=True)
    subprocess.run(mv_command,shell=True)
    subprocess.run(set_nodata_command,shell=True)

    
def get_srtm_tiles(lon_min,lon_max,lat_min,lat_max):
    SRTM_list = []
    lon_range = range(int(np.floor(lon_min)),int(np.floor(lon_max))+1)
    lat_range = range(int(np.floor(lat_min)),int(np.floor(lat_max))+1)
    for i in range(len(lon_range)):
        for j in range(len(lat_range)):
            if lon_range[i] >= 0:
                lonLetter = 'E'
            else:
                lonLetter = 'W'
            if lat_range[j] >= 0:
                latLetter = 'N'
            else:
                latLetter = 'S'
            lonCode = f"{int(np.abs(np.floor(lon_range[i]))):03d}"
            latCode = f"{int(np.abs(np.floor(lat_range[j]))):02d}"
            SRTM_id = f'{latLetter}{latCode}{lonLetter}{lonCode}.hgt'
            SRTM_list.append(SRTM_id)
    return sorted(SRTM_list)

def download_srtm(lon_min,lon_max,lat_min,lat_max,username,password,egm96_file,tmp_dir,output_file,copy_nan_flag=True):
    tile_array = get_srtm_tiles(lon_min,lon_max,lat_min,lat_max)
    srtm_usgs_base = 'https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/'
    srtm_suffix = 'SRTMGL1'
    merge_command = f'gdal_merge.py -q -o tmp_merged.tif '
    for tile in tile_array:
        dl_command = f'wget --user {username} --password={password} {srtm_usgs_base}{os.path.splitext(tile)[0]}.{srtm_suffix}{os.path.splitext(tile)[1]}.zip --no-check-certificate --quiet'
        unzip_command = f'unzip -qq {os.path.splitext(tile)[0]}.{srtm_suffix}{os.path.splitext(tile)[1]}.zip'
        subprocess.run(dl_command,shell=True,cwd=tmp_dir)
        subprocess.run(unzip_command,shell=True,cwd=tmp_dir)
        if os.path.isfile(f'{tmp_dir}{tile}'):
            merge_command = f'{merge_command} {tmp_dir}{tile} ' 
            subprocess.run(f'rm {os.path.splitext(tile)[0]}.{srtm_suffix}{os.path.splitext(tile)[1]}.zip',shell=True,cwd=tmp_dir)
    subprocess.run(merge_command,shell=True,cwd=tmp_dir)
    [subprocess.run(f'rm {tile}',shell=True,cwd=tmp_dir) for tile in tile_array if os.path.isfile(f'{tmp_dir}{tile}')]
    warp_command = f'gdalwarp -q -te {lon_min} {lat_min} {lon_max} {lat_max} tmp_merged.tif tmp_merged_clipped.tif'
    subprocess.run(warp_command,shell=True,cwd=tmp_dir)
    subprocess.run(f'rm tmp_merged.tif',shell=True,cwd=tmp_dir)
    if egm96_file is not None:
        resample_raster(egm96_file,f'{tmp_dir}tmp_merged_clipped.tif',f'{tmp_dir}EGM96_resampled.tif',quiet_flag=True)
        calc_command = f'gdal_calc.py -A tmp_merged_clipped.tif -B EGM96_resampled.tif --outfile={output_file} --calc=\"A+B\" --format=GTiff --co=\"COMPRESS=LZW\" --co=\"BIGTIFF=IF_SAFER\" --quiet'
        subprocess.run(calc_command,shell=True,cwd=tmp_dir)
        if copy_nan_flag == True:
            copy_nan(f'{tmp_dir}tmp_merged_clipped.tif',output_file,0,-9999)
        subprocess.run(f'rm tmp_merged_clipped.tif EGM96_resampled.tif',shell=True,cwd=tmp_dir)
    else:
        subprocess.run(f'mv tmp_merged_clipped.tif {output_file}',shell=True,cwd=tmp_dir)

def get_aster_tiles(lon_min,lon_max,lat_min,lat_max):
    ASTER_list = []
    lon_range = range(int(np.floor(lon_min)),int(np.floor(lon_max))+1)
    lat_range = range(int(np.floor(lat_min)),int(np.floor(lat_max))+1)
    for i in range(len(lon_range)):
        for j in range(len(lat_range)):
            if lon_range[i] >= 0:
                lonLetter = 'E'
            else:
                lonLetter = 'W'
            if lat_range[j] >= 0:
                latLetter = 'N'
            else:
                latLetter = 'S'
            lonCode = f"{int(np.abs(np.floor(lon_range[i]))):03d}"
            latCode = f"{int(np.abs(np.floor(lat_range[j]))):02d}"
            ASTER_id = f'ASTGTMV003_{latLetter}{latCode}{lonLetter}{lonCode}_dem.tif'
            ASTER_list.append(ASTER_id)
    return sorted(ASTER_list)

def download_aster(lon_min,lon_max,lat_min,lat_max,username,password,egm96_file,tmp_dir,output_file,copy_nan_flag=True):
    tile_array = get_aster_tiles(lon_min,lon_max,lat_min,lat_max)
    aster_earthdata_base = 'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/ASTGTM.003/'
    merge_command = f'gdal_merge.py -q -o tmp_merged.tif '
    for tile in tile_array:
        dl_command = f'wget --user={username} --password={password} {aster_earthdata_base}{tile} --quiet --no-check-certificate'
        subprocess.run(dl_command,shell=True,cwd=tmp_dir)
        if os.path.isfile(f'{tmp_dir}{tile}'):
            merge_command = f'{merge_command} {tmp_dir}{tile} ' 
    subprocess.run(merge_command,shell=True,cwd=tmp_dir)
    [subprocess.run(f'rm {tile}',shell=True,cwd=tmp_dir) for tile in tile_array if os.path.isfile(f'{tmp_dir}{tile}')]
    warp_command = f'gdalwarp -q -te {lon_min} {lat_min} {lon_max} {lat_max} tmp_merged.tif tmp_merged_clipped.tif'
    subprocess.run(warp_command,shell=True,cwd=tmp_dir)
    subprocess.run(f'rm tmp_merged.tif',shell=True,cwd=tmp_dir)
    if egm96_file is not None:
        resample_raster(egm96_file,f'{tmp_dir}tmp_merged_clipped.tif',f'{tmp_dir}EGM96_resampled.tif',quiet_flag=True)
        calc_command = f'gdal_calc.py -A tmp_merged_clipped.tif -B EGM96_resampled.tif --outfile={output_file} --calc=\"A+B\" --format=GTiff --co=\"COMPRESS=LZW\" --co=\"BIGTIFF=IF_SAFER\" --quiet'
        subprocess.run(calc_command,shell=True,cwd=tmp_dir)
        if copy_nan_flag == True:
            copy_nan(f'{tmp_dir}tmp_merged_clipped.tif',output_file,0,-9999)
        subprocess.run(f'rm tmp_merged_clipped.tif EGM96_resampled.tif',shell=True,cwd=tmp_dir)
    else:
        subprocess.run(f'mv tmp_merged_clipped.tif {output_file}',shell=True,cwd=tmp_dir)

def get_copernicus_tiles(lon_min,lon_max,lat_min,lat_max):
    COPERNICUS_list = []
    lon_range = range(int(np.floor(lon_min)),int(np.floor(lon_max))+1)
    lat_range = range(int(np.floor(lat_min)),int(np.floor(lat_max))+1)
    for i in range(len(lon_range)):
        for j in range(len(lat_range)):
            if lon_range[i] >= 0:
                lonLetter = 'E'
            else:
                lonLetter = 'W'
            if lat_range[j] >= 0:
                latLetter = 'N'
            else:
                latLetter = 'S'
            lonCode = f"{int(np.abs(np.floor(lon_range[i]))):03d}"
            latCode = f"{int(np.abs(np.floor(lat_range[j]))):02d}"
            COPERNICUS_id = f'Copernicus_DSM_COG_10_{latLetter}{latCode}_00_{lonLetter}{lonCode}_00_DEM/Copernicus_DSM_COG_10_{latLetter}{latCode}_00_{lonLetter}{lonCode}_00_DEM.tif'
            COPERNICUS_list.append(COPERNICUS_id)
    return sorted(COPERNICUS_list)

def download_copernicus(lon_min,lon_max,lat_min,lat_max,egm2008_file,tmp_dir,output_file,copy_nan_flag=True):
    tile_array = get_copernicus_tiles(lon_min,lon_max,lat_min,lat_max)
    copernicus_aws_base = 's3://copernicus-dem-30m/'
    merge_command = f'gdal_merge.py -q -o tmp_merged.tif '
    for tile in tile_array:
        dl_command = f'aws s3 cp --quiet --no-sign-request {copernicus_aws_base}{tile} .'
        subprocess.run(dl_command,shell=True,cwd=tmp_dir)
        if os.path.isfile(f'{tmp_dir}{tile.split("/")[-1]}'):
            merge_command = f'{merge_command} {tmp_dir}{tile.split("/")[-1]} ' 
    subprocess.run(merge_command,shell=True,cwd=tmp_dir)
    [subprocess.run(f'rm {tile.split("/")[-1]}',shell=True,cwd=tmp_dir) for tile in tile_array if os.path.isfile(f'{tmp_dir}{tile.split("/")[-1]}')]
    warp_command = f'gdalwarp -q -te {lon_min} {lat_min} {lon_max} {lat_max} tmp_merged.tif tmp_merged_clipped.tif'
    subprocess.run(warp_command,shell=True,cwd=tmp_dir)
    subprocess.run(f'rm tmp_merged.tif',shell=True,cwd=tmp_dir)
    if egm2008_file is not None:
        resample_raster(egm2008_file,f'{tmp_dir}tmp_merged_clipped.tif',f'{tmp_dir}EGM2008_resampled.tif',quiet_flag=True)
        calc_command = f'gdal_calc.py -A tmp_merged_clipped.tif -B EGM2008_resampled.tif --outfile={output_file} --calc=\"A+B\" --format=GTiff --co=\"COMPRESS=LZW\" --co=\"BIGTIFF=IF_SAFER\" --quiet'
        subprocess.run(calc_command,shell=True,cwd=tmp_dir)
        if copy_nan_flag == True:
            copy_nan(f'{tmp_dir}tmp_merged_clipped.tif',output_file,0,-9999)
        subprocess.run(f'rm tmp_merged_clipped.tif EGM2008_resampled.tif',shell=True,cwd=tmp_dir)
    else:
        subprocess.run(f'mv tmp_merged_clipped.tif {output_file}',shell=True,cwd=tmp_dir)

def resample_raster(src_filename,match_filename,dst_filename,nodata=-9999,resample_method='bilinear',compress=True,quiet_flag=False):
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