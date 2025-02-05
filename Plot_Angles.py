import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import glob
import argparse
import shapely.geometry
from mpl_toolkits.mplot3d import Axes3D

from dem_utils import get_lonlat_gdf

def get_angles(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    azimuth = float(root.find('IMD').find('IMAGE').find('MEANSATAZ').text)
    elevation = float(root.find('IMD').find('IMAGE').find('MEANSATEL').text)
    return azimuth,elevation

def get_outline(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    ullon = float(root.find('IMD').find('BAND_P').find('ULLON').text)
    ullat = float(root.find('IMD').find('BAND_P').find('ULLAT').text)
    urlon = float(root.find('IMD').find('BAND_P').find('URLON').text)
    urlat = float(root.find('IMD').find('BAND_P').find('URLAT').text)
    lrlon = float(root.find('IMD').find('BAND_P').find('LRLON').text)
    lrlat = float(root.find('IMD').find('BAND_P').find('LRLAT').text)
    lllon = float(root.find('IMD').find('BAND_P').find('LLLON').text)
    lllat = float(root.find('IMD').find('BAND_P').find('LLLAT').text)
    outline = shapely.geometry.Polygon([(ullon,ullat),(urlon,urlat),(lrlon,lrlat),(lllon,lllat),(ullon,ullat)])
    return outline

def az_el_to_xyz(az,el,r=1):
    '''
    Convert azimuth and elevation to xyz coordinates.
    Uses inclination angle (inc) instead of elevation angle.
    Azimuth is defined here starting from x axis in xy plane,
    but from the s/c azimuth starts at y axis in xy plane and goes opposite.
    '''
    inc = 0.5*np.pi - np.radians(el)
    az_conv = np.mod(-1*np.radians(az)+0.5*np.pi,2*np.pi)
    x = r*np.sin(inc)*np.cos(az_conv)
    y = r*np.sin(inc)*np.sin(az_conv)
    z = r*np.cos(inc)
    return x,y,z

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',help='Input file with paths to .NTF files to build DSM.',default=None)
    parser.add_argument('--coast',help='Path to coast file to plot.',default=None)
    args = parser.parse_args()

    input_file = args.input_file
    coast = args.coast

    if coast is not None:
        gdf_coast = gpd.read_file(coast)
        lon_coast,lat_coast = get_lonlat_gdf(gdf_coast)
    
    #We want the xml files, but if NTFs are given, we need to convert them
    input_file = input_file.replace('.NTF','.xml').replace('.ntf','.xml')

    if not os.path.isfile(input_file):
        file_list = np.asarray(sorted(glob.glob(input_file)))
    else:
        try:
            tree = ET.parse(input_file)
            file_list = np.atleast_1d(input_file)
        except:
            df_input = pd.read_csv(input_file,header=None,names=['filename'])
            file_list = np.asarray(df_input['filename'])
            file_list = np.asarray([f.replace('.NTF','.xml').replace('.ntf','.xml') for f in file_list])

    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    linestyles = ['-','--','-.',':']

    if len(file_list) > len(colors) * len(linestyles):
        print(f'Too many files ({len(file_list)}) to plot. Maximum number of files is {len(colors) * len(linestyles)}.')
        print(f'Only doing first {len(colors) * len(linestyles)} files.')
        file_list = file_list[:len(colors) * len(linestyles)]

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    if coast is not None:
        ax.plot(lon_coast,lat_coast,color='k',linewidth=0.5)

    for i,f in enumerate(file_list):
        short_name = '_'.join(f.split('/')[-1].split('_')[:2])
        color_coord = i % len(colors)
        linestyle_coord = int(np.floor(i/len(colors)))
        azimuth,elevation = get_angles(f)
        outline = get_outline(f)
        dx = outline.bounds[2] - outline.bounds[0]
        dy = outline.bounds[3] - outline.bounds[1]
        x_vec,y_vec,z_vec = az_el_to_xyz(azimuth,elevation,r=np.max((dx,dy)))
        x,y = outline.exterior.xy
        x_center = 0.5*(outline.bounds[0] + outline.bounds[2])
        y_center = 0.5*(outline.bounds[1] + outline.bounds[3])
        ax.plot(x,y,label=short_name,color=colors[color_coord],linestyle=linestyles[linestyle_coord])
        ax.plot([x_center,x_center+x_vec],[y_center,y_center+y_vec],[0,z_vec],color=colors[color_coord],linestyle=linestyles[linestyle_coord])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.show()


if __name__ == '__main__':
    main()