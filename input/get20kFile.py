import numpy as np
import xarray as xr

inname = '/Volumes/jklymak/ForcedChannel1000/results/Channel1000_01/input/'

with xr.open_dataset(inname+'spinup.nc') as ds:
    ds = ds.isel(record=-1)
    print(ds.time)
    ds.to_netcdf('Channel1000Spinup.nc', 'w')

with xr.open_dataset(inname+'spinup2d.nc') as ds:
    ds = ds.isel(record=-1)
    print(ds.time)
    ds.to_netcdf('Channel1000Spinup2d.nc', 'w')
