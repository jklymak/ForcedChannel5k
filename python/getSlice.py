import xarray as xr
# 'Channel5k1000_01',
for td in [ 'Channel5k1000_02']:
    todo = '../results/{}/input/spinup.nc'.format(td)

    with xr.open_dataset(todo) as ds:
        print(ds)
        ds = ds.isel(j=50, j_g=50)
        ds.to_netcdf('../reduceddata/SliceY50{}.nc'.format(td))
