import xarray as xr
# 'Channel5k1000_01',
for td in [ 'Channel5k1000_vrough_01']:
    todo = '../results/{}/input/fastlevels.nc'.format(td)

    with xr.open_dataset(todo) as ds:
        print(ds)
        ds = ds.drop(['UVEL', 'VVEL', 'THETA'])
        ds = ds.isel(k_level=2)
        ds.to_netcdf('../reduceddata/Wonlyfast{}.nc'.format(td))
