import xarray as xr
# 'Channel5k1000_01',
for td in [ 'Channel5k1000_rough_01', 'Channel5k1000_01']:
    todo = '../results/{}/input/spinup.nc'.format(td)

    with xr.open_dataset(todo) as ds:
        print(ds)
        ds = ds.isel(i=180, i_g=180)
        ds.to_netcdf('../reduceddata/SliceX180{}.nc'.format(td))
