import numpy as np
import geopandas as gpd
import pandas as pd
import subprocess
import warnings
from shapely.geometry import Polygon
import argparse
import configparser
from osgeo import gdal,gdalconst,osr
import os,sys

from dem_utils import lonlat2epsg,get_lonlat_gdf_center,buffer_gdf

def main():
    warnings.simplefilter(action='ignore')
    config_file = 'dem_config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)

    parser = argparse.ArgumentParser()
    parser.add_argument('--vlm',help='Path to VLM file to interpolate.')
    parser.add_argument('--n_pixels',help='Number of pixels to interpolate.',default=100)
    args = parser.parse_args()
    vlm_file = args.vlm
    n_pixels = int(args.n_pixels)
    
    vlm_file_extension = os.path.splitext(vlm_file)[-1]
    vlm_file_interpolated = vlm_file.replace(vlm_file_extension,f'_interpolated_{n_pixels}pix.tif')
    vlm_file_gt_neg9999 = vlm_file.replace(vlm_file_extension,'_gt_neg9999.tif')
    vlm_shp_file_gt_neg9999 = vlm_file_gt_neg9999.replace('.tif','.shp')
    vlm_shp_file_dissolved = vlm_file.replace(vlm_file_extension,'_dissolved.shp')
    vlm_file_interpolated_clipped = vlm_file_interpolated.replace('.tif','_clipped.tif')

    interpolation_command = f'gdal_fillnodata.py -q {vlm_file} {vlm_file_interpolated}'
    gdal_calc_command = f'gdal_calc.py --quiet -A {vlm_file} --calc="A>-9999" --outfile={vlm_file_gt_neg9999} --NoDataValue=-9999'
    gdal_polygonize_command = f'gdal_polygonize.py -q {vlm_file_gt_neg9999} -f "ESRI Shapefile" {vlm_shp_file_gt_neg9999}'
    subprocess.run(interpolation_command,shell=True)
    subprocess.run(gdal_calc_command,shell=True)
    subprocess.run(gdal_polygonize_command,shell=True)

    src_vlm = gdal.Open(vlm_file,gdalconst.GA_ReadOnly)
    epsg_code_vlm = osr.SpatialReference(wkt=src_vlm.GetProjection()).GetAttrValue('AUTHORITY',1)

    gdf = gpd.read_file(vlm_shp_file_gt_neg9999)
    lon_center,lat_center = get_lonlat_gdf_center(gdf)
    epsg_code = lonlat2epsg(lon_center,lat_center)
    gdf_dissolved = gpd.GeoDataFrame()
    for geom in gdf.geometry:
        gdf_dissolved = pd.concat([gdf_dissolved,gpd.GeoDataFrame(geometry=[Polygon(geom.exterior)],crs=f'EPSG:{epsg_code_vlm}')])
    # gdf_dissolved.to_file(vlm_shp_file_dissolved)
    gdf_dissolved_buffered = buffer_gdf(gdf_dissolved,100,1e3)
    gdf_dissolved_buffered_dissolved = gpd.GeoDataFrame()
    for geom in gdf_dissolved_buffered.geometry[0].geoms:
        gdf_dissolved_buffered_dissolved = pd.concat([gdf_dissolved_buffered_dissolved,gpd.GeoDataFrame(geometry=[Polygon(geom.exterior)],crs=f'EPSG:{epsg_code_vlm}')])
    gdf_dissolved_buffered_dissolved.to_file(vlm_shp_file_dissolved)

    crop_command = f'gdalwarp -q -s_srs EPSG:{epsg_code_vlm} -t_srs EPSG:{epsg_code_vlm} -cutline {vlm_shp_file_dissolved} -crop_to_cutline {vlm_file_interpolated} {vlm_file_interpolated_clipped} -co "COMPRESS=LZW"'
    subprocess.run(crop_command,shell=True)
    subprocess.run(f'rm {vlm_shp_file_gt_neg9999.replace(".shp",".*")}',shell=True)
    subprocess.run(f'rm {vlm_shp_file_dissolved.replace(".shp",".*")}',shell=True)
    subprocess.run(f'rm {vlm_file_interpolated}',shell=True)


if __name__ == '__main__':
    main()