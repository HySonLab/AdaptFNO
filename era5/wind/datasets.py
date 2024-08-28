import os
from typing import Tuple, List, Dict, Literal
import datetime as dt
import itertools

import xarray as xr

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F


class Wind2dERA5(Dataset):

    def __init__(
        self,
        dataroot: str,
        pressure_level: int,
        fromdate: str,
        todate: str,
        global_latitude: Tuple[float, float] | None,
        global_longitude: Tuple[float, float] | None,
        global_resolution: Tuple[int, int] | None,
        local_latitude: Tuple[float, float] | None,
        local_longitude: Tuple[float, float] | None,
        local_resolution: Tuple[int, int] | None,
        time_resolution: int,
        bundle_size: int,
        input_size: int,
    ):
        
        """
        latitude: (a, b) in the range [90.0, 89.75, 89.5, ..., -89.5, -89.75, -90.0]
        longitude: (a, b) in the range [0.0, 0.25, 0.5, ..., 359.25, 359.5, 359.75]
        """
        super().__init__()
        self.dataroot: str = dataroot
        self.pressure_level: int = pressure_level
        self.fromdate: dt.datetime = dt.datetime.strptime(fromdate, '%Y%m%d')
        self.todate: dt.datetime = dt.datetime.strptime(todate, '%Y%m%d')
        self.local_latitude: Tuple[int, int] | None = local_latitude
        self.local_longitude: Tuple[int, int] | None = local_longitude
        self.local_resolution: Tuple[int, int] | None = local_resolution
        self.global_latitude: Tuple[int, int] | None = global_latitude
        self.global_longitude: Tuple[int, int] | None = global_longitude
        self.global_resolution: Tuple[int, int] | None = global_resolution
        self.bundle_size: int = bundle_size
        self.time_resolution: int = time_resolution
        self.input_size: int = input_size

        assert 24 % self.time_resolution == 0 and self.time_resolution <= 24, (
            f'time_resolution must be a divisor of 24, got {self.time_resolution}'
        )

        self.datafolder: str = f'{dataroot}/{pressure_level}'
        self.filenames: List[str] = sorted([
            name for name in os.listdir(self.datafolder)
            if name.endswith('.grib')
            and self.fromdate <= dt.datetime.strptime(name.replace('.grib',''), '%Y%m%d') <= self.todate
        ])
        self.n_days: int = len(self.filenames)
        self.timesteps_per_day: int = 24 // self.time_resolution
        self.n_timesteps: int = self.timesteps_per_day * self.bundle_size
        self.in_timesteps: int = self.timesteps_per_day * self.bundle_size * self.input_size
        self.out_timesteps: int = self.timesteps_per_day * self.bundle_size
        self.total_timesteps: int = self.timesteps_per_day * self.n_days
        
        self.has_global: bool = all([global_latitude, global_longitude, global_resolution])
        self.has_local: bool = all([local_latitude, local_longitude, local_resolution])
        assert self.has_global or self.has_local, 'either global or local must be specified'

        if self.has_global:
            self.global_tensors: List[torch.Tensor] = []
            self.global_timestamps: List[List[dt.datetime]] = []
            for fname in self.filenames:
                data: torch.Tensor; timestamps: List[dt.datetime]
                data, timestamps = self._retrieve_data(
                    filename=fname, 
                    latitude=self.global_latitude, longitude=self.global_longitude, 
                    spatial_resolution=self.global_resolution,
                )
                self.global_tensors.append(data); self.global_timestamps.append(timestamps)

        if self.has_local:
            self.local_tensors = List[torch.Tensor] = []
            self.local_timestamps = List[List[dt.datetime]] = []
            for fname in self.filenames:
                data: torch.Tensor; timestamps: List[dt.datetime]
                data, timestamps = self._retrieve_data(
                    filename=fname, 
                    latitude=self.local_latitude, longitude=self.local_longitude, 
                    spatial_resolution=self.local_resolution,
                )
                self.local_tensors.append(data); self.local_timestamps.append(timestamps)

    def __getitem__(self, bundle_idx: int) -> Tuple[torch.Tensor, ...]:
        if bundle_idx >= len(self):
            raise IndexError

        input_slice, output_slice = self._compute_slices(bundle_idx=bundle_idx)

        sample: Tuple[torch.Tensor, ...] = tuple()  # (global_input, global_output, local_input, local_output) 
        if self.has_global:
            global_input: torch.Tensor = torch.cat(tensors=self.global_tensors[input_slice], dim=0)
            global_output: torch.Tensor = torch.cat(tensors=self.global_tensors[output_slice], dim=0)
            sample += (global_input, global_output)

        if self.has_local:
            local_input: torch.Tensor = torch.cat(tensors=self.local_tensors[input_slice], dim=0)
            local_output: torch.Tensor = torch.cat(tensors=self.local_tensors[output_slice], dim=0)
            sample += (local_input, local_output)
        
        return sample

    def __len__(self) -> int:
        return (self.n_days - (self.input_size + 1) * self.bundle_size) // self.bundle_size + 1

    def compute_timestamp(self, bundle_idx: int) -> Tuple[List[dt.datetime], List[dt.datetime]]:
        input_slice, output_slice = self._compute_slices(bundle_idx=bundle_idx)
        in_timestamps: List[dt.datetime] = list(itertools.chain(*self.global_timestamps[input_slice]))
        out_timestamps: List[dt.datetime] = list(itertools.chain(*self.global_timestamps[output_slice]))
        return in_timestamps, out_timestamps

    def _compute_slices(self, bundle_idx: int) -> Tuple[slice, slice]:
        left_index: int = bundle_idx * self.bundle_size
        mid_index: int = left_index + self.input_size * self.bundle_size
        right_index: int = mid_index + self.bundle_size
        return slice(left_index, mid_index, 1), slice(mid_index, right_index, 1)

    def _retrieve_data(
        self, 
        filename: str, 
        latitude: Tuple[float, float], 
        longitude: Tuple[float, float],
        spatial_resolution: Tuple[int, int],
    ) -> Tuple[torch.Tensor, List[dt.datetime]]:
        dataset: xr.Dataset = xr.open_dataset(f'{self.datafolder}/{filename}', engine='cfgrib')
        dataset = dataset.sel(
            latitude=slice(*latitude), longitude=slice(*longitude),
        )
        dataset = dataset.isel(time=slice(0, 24, self.time_resolution))
        data: torch.Tensor = torch.tensor(data=dataset.to_dataarray().values)
        # Convert to shape (timesteps, 2, *resolution)
        data = data.permute(1, 0, 2, 3)
        # Transform resolution
        data = F.interpolate(
            input=data, size=spatial_resolution, mode='bicubic',
        )
        # Retrieve timestamps
        timestamps: List[dt.datetime] = [t.astype('M8[s]').astype(dt.datetime) for t in dataset.valid_time.values]
        assert data.shape[0] == len(timestamps)
        return data, timestamps



if __name__ == '__main__':

    dataset = Wind2dERA5(
        dataroot='data/2d/era5/wind',
        pressure_level=1000,
        fromdate='20170101',
        todate='20171231',
        global_latitude=(45, -45),
        global_longitude=(0, 180),
        global_resolution=(128, 128),
        local_latitude=None,
        local_longitude=None,
        local_resolution=None,
        time_resolution=3,
        bundle_size=1,
        input_size=1,
    )

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
    for batch, (input, output) in enumerate(dataloader, start=0):
        print(f'Load batch {batch + 1}/{len(dataloader)}')
        print(input.shape)
        print(output.shape)
        print('--------')

    print(dataset.compute_timestamp(bundle_idx=0)[0])
    print(dataset.compute_timestamp(bundle_idx=0)[1])

    print('Len of dataset:', len(dataset))




