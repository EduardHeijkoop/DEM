import numpy as np
import pandas as pd
import subprocess
import argparse
import os
import sys
from osgeo import gdal,gdalconst,osr



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

def calculate_shift(df_sampled,mean_median_mode='mean',n_sigma_filter=2,vertical_shift_iterative_threshold=0.02,printing=False,write_file=None,primary='h_primary',secondary='h_secondary',N_iterations=15,sigma_flag=False):
    df_sampled = df_sampled.rename(columns={primary:'h_primary',secondary:'h_secondary'})
    count = 0
    cumulative_shift = 0
    original_len = len(df_sampled)
    h_primary_original = np.asarray(df_sampled.h_primary)
    h_secondary_original = np.asarray(df_sampled.h_secondary)
    dh_original = h_primary_original - h_secondary_original
    rmse_original = np.sqrt(np.sum(dh_original**2)/len(dh_original))
    if write_file is not None:
        f = open(write_file,'w')
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
        if write_file is not None:
            f.writelines(f'Iteration        : {count}\n')
            f.writelines(f'Incremental shift: {incremental_shift:.2f} m\n')
        if np.abs(incremental_shift) <= vertical_shift_iterative_threshold:
            break
        if count == N_iterations:
            if write_file is not None:
                f.writelines('Co-registration did not converge!\n')
            if printing == True:
                print('Co-registration did not converge!')
            break
    h_primary_filtered = np.asarray(df_sampled.h_primary)
    h_secondary_filtered = np.asarray(df_sampled.h_secondary)
    dh_filtered = h_primary_filtered - h_secondary_filtered
    rmse_filtered = np.sqrt(np.sum(dh_filtered**2)/len(dh_filtered))
    if sigma_flag == True:
        sigma_i2 = np.sqrt(np.sum(df_sampled.sigma**2)/len(df_sampled))
        sigma_dsm = np.sqrt(np.var(dh_filtered) - sigma_i2**2)
    if printing == True:
        print(f'Number of iterations: {count}')
        print(f'Number of points before filtering: {original_len}')
        print(f'Number of points after filtering: {len(df_sampled)}')
        print(f'Retained {len(df_sampled)/original_len*100:.1f}% of points.')
        print(f'Cumulative shift: {cumulative_shift:.2f} m')
        print(f'RMSE before filtering: {rmse_original:.2f} m')
        print(f'RMSE after filtering: {rmse_filtered:.2f} m')
        if sigma_flag == True:
            print(f'DSM sigma: {sigma_dsm:.2f} m')
    if write_file is not None:
        f.writelines(f'Number of iterations: {count}\n')
        f.writelines(f'Number of points before filtering: {original_len}\n')
        f.writelines(f'Number of points after filtering: {len(df_sampled)}\n')
        f.writelines(f'Retained {len(df_sampled)/original_len*100:.1f}% of points.\n')
        f.writelines(f'Cumulative shift: {cumulative_shift:.2f} m\n')
        f.writelines(f'RMSE before filtering: {rmse_original:.2f} m\n')
        f.writelines(f'RMSE after filtering: {rmse_filtered:.2f} m\n')
        if sigma_flag == True:
            f.writelines(f'DSM sigma: {sigma_dsm:.2f} m\n')
        f.close()
    return cumulative_shift,df_sampled

def vertical_shift_raster(raster_path,df_sampled,output_dir,mean_median_mode='mean',n_sigma_filter=2,vertical_shift_iterative_threshold=0.02,primary='h_primary',secondary='h_secondary',return_df=False,printing=False,write_file=None,N_iterations=15,sigma_flag=False):
    src = gdal.Open(raster_path,gdalconst.GA_ReadOnly)
    raster_nodata = src.GetRasterBand(1).GetNoDataValue()
    vertical_shift,df_new = calculate_shift(df_sampled,mean_median_mode,n_sigma_filter,vertical_shift_iterative_threshold,primary=primary,secondary=secondary,printing=printing,write_file=write_file,N_iterations=N_iterations,sigma_flag=sigma_flag)
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
    if return_df == True:
        df_new.rename(columns={'h_primary':primary,'h_secondary':secondary},inplace=True)
        return raster_shifted,vertical_shift,rmse,ratio_pts,df_new
    else:
        return raster_shifted,vertical_shift,rmse,ratio_pts,None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raster', help="Path to DEM file")
    parser.add_argument('--csv', help="Path to txt/csv file")
    parser.add_argument('--mean',default=False,action='store_true')
    parser.add_argument('--median',default=False,action='store_true')
    parser.add_argument('--sigma', nargs='?', type=int, default=2)
    parser.add_argument('--threshold', nargs='?', type=float, default=0.05)
    parser.add_argument('--resample',default=False,action='store_true')
    parser.add_argument('--keep_original_sample',default=False,action='store_true')
    parser.add_argument('--no_writing',default=False,action='store_true')
    parser.add_argument('--nodata', nargs='?', type=str,default='-9999')
    parser.add_argument('--print',default=False,action='store_true')
    parser.add_argument('--write_file',default=None)
    parser.add_argument('--output_dir',default=None,help='Directory for output files.')
    parser.add_argument('--N_iterations',default=15,type=int,help='Number of iterations before breaking loop.')

    args = parser.parse_args()
    raster_path = args.raster
    csv_path = args.csv
    mean_mode = args.mean
    median_mode = args.median
    n_sigma_filter = args.sigma
    vertical_shift_iterative_threshold = args.threshold
    resample_flag = args.resample
    keep_original_sample_flag = args.keep_original_sample
    no_writing_flag = args.no_writing
    nodata_value = args.nodata
    print_flag = args.print
    write_file = args.write_file
    output_dir = args.output_dir
    N_iterations = args.N_iterations
    if np.logical_xor(mean_mode,median_mode) == True:
        if mean_mode == True:
            mean_median_mode = 'mean'
        elif median_mode == True:
            mean_median_mode = 'median'
    else:
        print('Please choose exactly one mode: mean or median.')
        sys.exit()

    if output_dir is None:
        output_dir = f'{os.path.dirname(raster_path)}/'
    elif output_dir[-1] != '/':
        output_dir = f'{output_dir}/'

    if write_file is not None and len(os.path.dirname(write_file)) == 0:
        write_file = f'{output_dir}{write_file}'

    '''
    Read header line of csv, if it has "sigma" in it, read 
    '''
    csv_header_line = subprocess.check_output(f'head -n 1 {csv_path}',shell=True).decode('utf-8').strip()
    if 'sigma' in csv_header_line.lower():
        sigma_flag = True
        df_csv = pd.read_csv(csv_path)
        idx_sigma = df_csv.sigma != 0
        if np.sum(idx_sigma) < len(df_csv):
            df_csv = df_csv[idx_sigma].reset_index(drop=True)
            new_csv_path = csv_path.replace(os.path.splitext(csv_path)[1],f'_nonzero_sigma{os.path.splitext(csv_path)[1]}')
            df_csv.to_csv(new_csv_path,index=False,float_format='%.6f')
            csv_path = new_csv_path
    else:
        sigma_flag = False



    sampled_file = f'{output_dir}{os.path.basename(os.path.splitext(csv_path)[0])}_Sampled_{os.path.basename(os.path.splitext(raster_path)[0])}{os.path.splitext(csv_path)[1]}'
    sample_code = sample_raster(raster_path, csv_path, sampled_file,nodata=nodata_value,header='height_dsm')
    if sample_code is not None:
        print('Error in sampling raster.')
    df_sampled_original = pd.read_csv(sampled_file)
    raster_shifted,vertical_shift,rmse,ratio_pts,df_sampled_filtered = vertical_shift_raster(raster_path,df_sampled_original,output_dir,mean_median_mode,n_sigma_filter,vertical_shift_iterative_threshold,primary='height_icesat2',secondary='height_dsm',return_df=True,printing=print_flag,write_file=write_file,N_iterations=N_iterations,sigma_flag=sigma_flag)
    if no_writing_flag == False:
        output_csv = f'{output_dir}{os.path.splitext(os.path.basename(csv_path))[0]}_Filtered_{os.path.basename(os.path.splitext(raster_path)[0])}_{mean_median_mode}_{n_sigma_filter}sigma_Threshold_{str(vertical_shift_iterative_threshold).replace(".","p")}m{os.path.splitext(csv_path)[1]}'
        df_sampled_filtered.to_csv(output_csv,index=False,float_format='%.6f')

    if resample_flag == True:
        resampled_file = f'{output_dir}{os.path.splitext(os.path.basename(csv_path))[0]}_Sampled_Coregistered_{os.path.splitext(os.path.basename(raster_path))[0]}{os.path.splitext(csv_path)[1]}'
        resample_code = sample_raster(raster_shifted, csv_path, resampled_file,nodata=nodata_value,header='height_dsm')
        if resample_code is not None:
            print('Error in sampling co-registered raster.')
    if keep_original_sample_flag == False:
        os.remove(sampled_file)

if __name__ == '__main__':
    main()