import xarray as xr

for td in ['ChannelCoarse01']:
    todo = '../results/{}/input/spinup.nc'.format(td)

    with xr.open_dataset(todo) as ds:
        print(ds)
        ds = ds.isel(j=30, j_g=30)
        ds.to_netcdf('../reduceddata/Slice30{}.nc'.format(td))
