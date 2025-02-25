import numpy as np
import argparse
import netCDF4 as nc
import warnings
import configparser
import os

from dem_utils import great_circle_distance

def get_ar6_data(ar6_dir,confidence,scenario,years,quantiles,tg_id=None,coords=[None,None],print_info=True,baseline=None,surge=0.0):
    AR6_file = os.path.join(*[ar6_dir,f'Regional/{confidence}_confidence/{scenario}/total_{scenario}_{confidence}_confidence_values.nc'])
    if not os.path.isfile(AR6_file):
        raise FileNotFoundError(f'File {AR6_file} does not exist!')
    AR6_data = nc.Dataset(AR6_file)
    AR6_years = np.asarray(AR6_data['years'])
    AR6_quantiles = np.asarray(AR6_data['quantiles'])
    AR6_sea_level_change = np.asarray(AR6_data['sea_level_change'])
    if tg_id is not None:
        if tg_id > 2358:
            raise ValueError('Tide gauge ID must be valid number between 1 and 2358')
        idx_loc = np.atleast_1d(np.argwhere(np.asarray(AR6_data['locations']) == tg_id).squeeze())[0]
    elif coords[0] is not None:
        lon_AR6 = np.asarray(AR6_data['lon'])
        lat_AR6 = np.asarray(AR6_data['lat'])
        idx_no_tg = np.arange(1030,len(lon_AR6))
        lon_AR6 = lon_AR6[idx_no_tg]
        lat_AR6 = lat_AR6[idx_no_tg]
        lon_input = coords[0]
        lat_input = coords[1]
        distance = great_circle_distance(lon_AR6,lat_AR6,lon_input,lat_input)
        distance = distance[~np.isnan(distance)]
        idx_loc = np.argmin(distance)
        AR6_sea_level_change = AR6_sea_level_change[:,:,idx_no_tg]
    AR6_sea_level_change = AR6_sea_level_change[:,:,idx_loc].squeeze()
    AR6_sea_level_change = AR6_sea_level_change/1000
    sea_level_change_list = []
    for y in years:
        idx_year = np.atleast_1d(np.argwhere(AR6_years == y).squeeze())[0]
        for q in quantiles:
            idx_quantile = np.atleast_1d(np.argwhere(AR6_quantiles == q).squeeze())[0]
            AR6_sea_level_change_select = AR6_sea_level_change[idx_quantile,idx_year]
            sea_level_change_list.append(AR6_sea_level_change_select)
            if print_info:
                if q == 0.5:
                    q_str = 'median'
                else:
                    q_str = f'{int(np.round(q*100))}%ile'
                scenario_str = f'{scenario.upper()[:4]}-{scenario.upper()[4]}.{scenario.upper()[5]}'
                if surge > 0:
                    print(f'Inundation Risk in {y} ({q_str}) with {scenario_str} (= {AR6_sea_level_change_select:.2f} m)\\nwith {surge:.2f} m surge above {baseline}')
                else:
                    print(f'Inundation Risk in {y} ({q_str}) with {scenario_str} (= {AR6_sea_level_change_select:.2f} m)\\nabove {baseline}')
    print(sea_level_change_list)
    return None

def main():
    warnings.simplefilter(action='ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',help='Config file for DEM',default='dem_config.ini')
    parser.add_argument('--tg_id',help='Tide gauge ID?',default=None,type=int)
    parser.add_argument('--coords',help='Lon/lat coordinates?',default=[None,None],nargs=2,type=float)
    parser.add_argument('--scenario',help='SSP to use?',default='ssp585',choices=['ssp119','ssp126','ssp245','ssp370','ssp585'])
    parser.add_argument('--quantiles',help='Quantiles to use?',default=[0.5],type=float,nargs='*')
    parser.add_argument('--years',help='Years to use?',default=[2020],type=int,nargs='*')
    parser.add_argument('--surge',help='Surge to use?',default=None,type=float)
    parser.add_argument('--baseline',help='Baseline to use?',default='MHHW',type=str,choices=['MHHW','MSL'])
    parser.add_argument('--confidence',help='Confidence level?',default='medium',choices=['low','medium'])
    parser.add_argument('--print',help='Print information?',default=False,action='store_true')
    args = parser.parse_args()

    config_file = args.config_file
    tg_id = args.tg_id
    coords = args.coords
    scenario = args.scenario
    quantiles = np.atleast_1d(args.quantiles)
    years = np.atleast_1d(args.years)
    surge = args.surge
    confidence = args.confidence
    print_info = args.print
    baseline = args.baseline


    config = configparser.ConfigParser()
    config.read(config_file)
    ar6_dir = config.get('INUNDATION_PATHS','AR6_dir')

    if coords[0] > 180:
        coords[0] -= 360

    get_ar6_data(ar6_dir,confidence,scenario,years,quantiles,tg_id,coords,print_info=print_info,baseline=baseline,surge=surge)

if __name__ == "__main__":
    main()