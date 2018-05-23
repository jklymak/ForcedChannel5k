import xarray as xr
# 'Channel5k1000_01',
for td in [ 'Channel5k1000_02']:
    todo = '../results/{}/input/spinup.nc'.format(td)

    with xr.open_dataset(todo) as ds:
        print(ds)
        ds = ds.isel(i=250, i_g=250)
        ds.to_netcdf('../reduceddata/SliceX250{}.nc'.format(td))
