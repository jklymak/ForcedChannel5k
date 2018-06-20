import xarray as xr
# 'Channel5k1000_01',
for td in [ 'Channel5k1000_lindrag_01']:
    todo = '../results/{}/input/levels.nc'.format(td)

    with xr.open_dataset(todo) as ds:
        print(ds)
        ds = ds.isel(record=-1)
        ds.to_netcdf('../reduceddata/LevelLast{}.nc'.format(td))
