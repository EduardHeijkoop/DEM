import numpy as np
import subprocess
import warnings
import argparse
import configparser
from osgeo import gdal,gdalconst,osr
import os
from scipy import ndimage

def main():
    warnings.simplefilter(action='ignore')
    config_file = 'dem_config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)

    parser = argparse.ArgumentParser()
    parser.add_argument('--vlm',help='Path to VLM file to interpolate.')
    parser.add_argument('--n_pixels',help='Number of pixels to interpolate.',default=100)
    parser.add_argument('--n_dilations',help='Number of dilation iterations.',default=1)
    args = parser.parse_args()
    vlm_file = args.vlm
    n_pixels = int(args.n_pixels)
    n_dilations = int(args.n_dilations)
    
    vlm_file_extension = os.path.splitext(vlm_file)[-1]
    vlm_file_interpolated = vlm_file.replace(vlm_file_extension,f'_interpolated_{n_pixels}pix.tif')
    vlm_file_interpolated_clipped = vlm_file_interpolated.replace('.tif','_clipped.tif')

    #create mask
    src_vlm = gdal.Open(vlm_file,gdalconst.GA_ReadOnly)
    src_vlm_proj = src_vlm.GetProjection()
    src_vlm_geotrans = src_vlm.GetGeoTransform()
    wide = src_vlm.RasterXSize
    high = src_vlm.RasterYSize

    vlm_array = np.array(src_vlm.GetRasterBand(1).ReadAsArray())
    vlm_binary = (vlm_array !=0).astype(int) #nodata = 0
    dilation_structure = ndimage.generate_binary_structure(2,2)
    vlm_binary_dilated = ndimage.binary_dilation(vlm_binary,structure=dilation_structure,iterations=n_dilations)
    vlm_binary_dilated_filled = ndimage.binary_fill_holes(vlm_binary_dilated).astype(int)
    vlm_mask_file = vlm_file.replace(vlm_file_extension,'_valid_mask.tif')
    dst = gdal.GetDriverByName('GTiff').Create(vlm_mask_file, wide, high, 1 , gdalconst.GDT_UInt16)
    outBand = dst.GetRasterBand(1)
    outBand.WriteArray(vlm_binary_dilated_filled,0,0)
    outBand.FlushCache()
    outBand.SetNoDataValue(0)
    dst.SetProjection(src_vlm_proj)
    dst.SetGeoTransform(src_vlm_geotrans)
    del dst

    interpolation_command = f'gdal_fillnodata.py -q -md {n_pixels} {vlm_file} {vlm_file_interpolated}'
    gdal_calc_command = f'gdal_calc.py --quiet -A {vlm_file_interpolated} -B {vlm_mask_file} --calc="A*B" --outfile={vlm_file_interpolated_clipped} --NoDataValue=0 --co="COMPRESS=LZW"'
    subprocess.run(interpolation_command,shell=True)
    subprocess.run(gdal_calc_command,shell=True)
    subprocess.run(f'rm {vlm_mask_file}',shell=True)
    subprocess.run(f'mv {vlm_file_interpolated_clipped} {vlm_file_interpolated}',shell=True)

if __name__ == '__main__':
    main()