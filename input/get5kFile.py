import numpy as np
import xarray as xr

"""
Read the 5k data from 5 y on...
"""

inname = '../results/Channel5k1000_01/input/'

with xr.open_dataset(inname+'spinup.nc') as ds:
    ds = ds.isel(record=5)
    print(ds.time)
    ds.to_netcdf('Channel5k_5y_Spinup.nc', 'w')

with xr.open_dataset(inname+'spinup2d.nc') as ds:
    ds = ds.isel(record=5)
    print(ds.time)
    ds.to_netcdf('Channel5k_5y_Spinup2d.nc', 'w')
