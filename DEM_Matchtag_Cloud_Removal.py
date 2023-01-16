import numpy as np
from osgeo import gdal,gdalconst,osr
import scipy.ndimage
from skimage.measure import label
import argparse
import subprocess
import sys
import os

def raster_to_geotiff(x,y,arr,epsg_code,output_file,dtype='float'):
    '''
    given numpy array and x and y coordinates, produces a geotiff in the right epsg code
    '''
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    xres = x[1] - x[0]
    yres = y[1] - y[0]
    geotransform = (xmin-xres/2,xres,0,ymax+yres/2,0,-yres)
    driver = gdal.GetDriverByName('GTiff')
    if dtype == 'float':
        dataset = driver.Create(output_file,arr.shape[1],arr.shape[0],1,gdal.GDT_Float32)
    elif dtype == 'int':
        dataset = driver.Create(output_file,arr.shape[1],arr.shape[0],1,gdal.GDT_Int32)
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(f'EPSG:{epsg_code}')
    dataset.GetRasterBand(1).WriteArray(arr)
    dataset.FlushCache()
    dataset = None
    return None

def erode_matchtag(matchtag_file,threshold):
    '''
    Given a matchtag file, will erode the matchtag file
    and return a new matchtag file with only the largest components.
    '''
    src_matchtag = gdal.Open(matchtag_file,gdalconst.GA_ReadOnly)
    matchtag_array = np.asarray(src_matchtag.GetRasterBand(1).ReadAsArray(),dtype=int)
    matchtag_eroded = scipy.ndimage.binary_erosion(matchtag_array,structure=np.ones((3,3))).astype(matchtag_array.dtype)
    matchtag_dilated = scipy.ndimage.binary_dilation(matchtag_eroded,structure=np.ones((3,3))).astype(matchtag_array.dtype)
    matchtag_labels = label(matchtag_dilated)
    matchtag_new = np.zeros(matchtag_array.shape,dtype=np.uint8)
    matchtag_labels_threshold = np.argwhere(np.bincount(matchtag_labels.flat)[1:] > threshold) + 1
    for lbl in matchtag_labels_threshold:
        matchtag_new[matchtag_labels == lbl] = 1
    return matchtag_new

def get_xy_raster(raster_file):
    '''
    Given a raster file, will return the x and y coordinates
    '''
    src_raster = gdal.Open(raster_file,gdalconst.GA_ReadOnly)
    gt = src_raster.GetGeoTransform()
    x = np.arange(gt[0]+gt[1]*0.5,gt[0]+(src_raster.RasterXSize*gt[1])-gt[1]*0.5+gt[1],gt[1])
    y = np.arange(gt[3]+gt[5]*0.5,gt[3]+(src_raster.RasterYSize*gt[5])-gt[5]*0.5+gt[5],gt[5])
    y = np.flip(y)
    return x,y

def compress_raster(filename,nodata=-9999):
    '''
    Compress a raster using gdal_translate
    '''
    file_ext = os.path.splitext(filename)[-1]
    tmp_filename = filename.replace(file_ext,f'_LZW{file_ext}')
    if nodata is not None:
        compress_command = f'gdal_translate -co "COMPRESS=LZW" -co "BIGTIFF=IF_SAFER" -a_nodata {nodata} {filename} {tmp_filename}'
    else:
        compress_command = f'gdal_translate -co "COMPRESS=LZW" -co "BIGTIFF=IF_SAFER" {filename} {tmp_filename}'
    move_command = f'mv {tmp_filename} {filename}'
    subprocess.run(compress_command,shell=True)
    subprocess.run(move_command,shell=True)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dem',help='Path to DEM file to filter.')
    parser.add_argument('--threshold',help='Pixel threshold for matchtag filtering.',default=1000)
    args = parser.parse_args()
    dem_file = args.dem
    threshold = int(args.threshold)

    src_dem = gdal.Open(dem_file,gdalconst.GA_ReadOnly)
    epsg_code = osr.SpatialReference(wkt=src_dem.GetProjection()).GetAttrValue('AUTHORITY',1)
    matchtag_file = f'{dem_file.split("_dem")[0]}_matchtag.tif'
    matchtag_filtered_file = matchtag_file.replace('.tif','_Cloud_Filtered.tif')
    dem_filtered_file = dem_file.replace('.tif','_Cloud_Filtered.tif')
    matchtag_new = erode_matchtag(matchtag_file,threshold)
    x_dem,y_dem = get_xy_raster(dem_file)
    write_code = raster_to_geotiff(x_dem,y_dem,matchtag_new,epsg_code,matchtag_filtered_file,dtype='int')
    if write_code is None:
        matchtag_filtered_file_compressed = matchtag_filtered_file.replace('.tif','.lzw.tif')
        compress_command = f'gdal_translate -ot Byte -co "COMPRESS=LZW" {matchtag_filtered_file} {matchtag_filtered_file_compressed}'
        move_command = f'mv {matchtag_filtered_file_compressed} {matchtag_filtered_file}'
        subprocess.run(compress_command,shell=True)
        subprocess.run(move_command,shell=True)
    else:
        print('Error writing matchtag file.')
        sys.exit()
    
    dem_array = np.asarray(src_dem.GetRasterBand(1).ReadAsArray(),dtype=float)
    dem_array_filtered = dem_array * matchtag_new
    dem_array_filtered[dem_array_filtered==0] = -9999
    write_code = raster_to_geotiff(x_dem,y_dem,dem_array_filtered,epsg_code,dem_filtered_file,dtype='float')
    compress_code = compress_raster(dem_filtered_file,nodata=-9999)

if __name__ == '__main__':
    main()