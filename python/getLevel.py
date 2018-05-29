import xarray as xr
# 'Channel5k1000_01',
for td in [ 'Channel5k1000_01', 'Channel5k1000_rough_01']:
    todo = '../results/{}/input/spinup.nc'.format(td)

    kk = 70
    with xr.open_dataset(todo) as ds:
        print(ds)
        ds = ds.isel(k=kk, k_l=kk, k_u=kk, k_p1=kk)
        ds.to_netcdf('../reduceddata/SliceLevel{}{}.nc'.format(kk, td))
