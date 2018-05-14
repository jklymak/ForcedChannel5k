import xarray as xr

for td in ['Channel5k01']:
    todo = '../results/{}/input/spinup.nc'.format(td)

    with xr.open_dataset(todo) as ds:
        print(ds)
        ds = ds.isel(j=50, j_g=50)
        ds.to_netcdf('../reduceddata/Slice50{}.nc'.format(td))
