import os
import time
from typing import Dict, List, Any
import datetime as dt

import pandas as pd
import xarray as xr
import cdsapi


class WindByPressureLevels:

    """
    Home Page: https://cds-beta.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download
    """

    def __init__(self, pressure_level: int, destination_dir: str):
        self.client = cdsapi.Client()
        self.pressure_level: int = pressure_level
        self.destination_dir: str = destination_dir
        os.makedirs(name=f'{self.destination_dir}/{self.pressure_level}/src', exist_ok=True)

    def retrieve(self, years: List[int]):
        years: List[str] = [str(y) for y in years]
        
        request: Dict[str, Any] = {
            'product_type': ['reanalysis'],
            'variable': ['u_component_of_wind', 'v_component_of_wind'],
            'year': years,
            'month': [
                '01', '02', '03', '04', '05', '06', 
                '07', '08', '09', '10', '11', '12'
            ],
            'day': [
                '01', '02', '03', '04', '05', '06', 
                '07', '08', '09', '10', '11', '12', 
                '13', '14', '15', '16', '17', '18', 
                '19', '20', '21', '22', '23', '24', 
                '25', '26', '27', '28', '29', '30', 
                '31',
            ],
            'time': [
                '00:00', '01:00', '02:00', '03:00', '04:00', '05:00', 
                '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', 
                '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', 
                '18:00', '19:00', '20:00', '21:00', '22:00', '23:00',
            ],
            'pressure_level': [self.pressure_level],
            'data_format': 'grib',
            'download_format': 'unarchived'
        }

        filename: str = f"{'_'.join(years)}.grib"
        self.client.retrieve(
            name="reanalysis-era5-pressure-levels", request=request
        ).download(f'{self.destination_dir}/{self.pressure_level}/src/{filename}')
        return f'{self.destination_dir}/{self.pressure_level}/src/{filename}'

    def split(self, fileroot: str):
        ds = xr.open_dataset(fileroot, engine='cfgrib')
        times = pd.to_datetime(ds.time.values)
        unique_dates = pd.Series(times).dt.normalize().unique()

        for date in unique_dates:
            ds_date = ds.sel(time=slice(date, date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)))
            output_file = f"{date.strftime('%Y%m%d')}.grib"
            ds.to_netcdf(output_file, engine='cfgrib')

        ds.close()
                   


if __name__ == '__main__':

    for pressure in [1000]:     # [1000, 700, 400, 200, 50]:
        self = WindByPressureLevels(pressure_level=pressure, destination_dir='data/2d/era5/wind')
        self.retrieve(years=[2013, 2012, 2011])

