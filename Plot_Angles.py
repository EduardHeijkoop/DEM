import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os
import glob
import argparse
import shapely.geometry
from mpl_toolkits.mplot3d import Axes3D


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
    inc = 0.5*np.math.pi - np.radians(el)
    az_conv = np.mod(-1*np.radians(az)+0.5*np.math.pi,2*np.math.pi)
    x = r*np.sin(inc)*np.cos(az_conv)
    y = r*np.sin(inc)*np.sin(az_conv)
    z = r*np.cos(inc)
    return x,y,z

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',help='Input file with paths to .NTF files to build DSM.',default=None)
    args = parser.parse_args()

    input_file = args.input_file

    if not os.path.isfile(input_file):
        file_list = np.asarray(sorted(glob.glob(input_file)))
    else:
        try:
            tree = ET.parse(input_file)
            file_list = np.atleast_1d(input_file)
        except:
            df_input = pd.read_csv(input_file,header=None,names=['filename'])
            file_list = np.asarray(df_input['filename'])

    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    linestyles = ['-','--','-.',':']

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    for i,f in enumerate(file_list):
        short_name = '_'.join(f.split('_')[:2])
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