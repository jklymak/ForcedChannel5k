import xarray as xr

for f in ['038', '073', '100', '126', '141']:
    td = 'LW1kmlowU10Amp305f' + f
    todo = '../results/{}/input/final.nc'.format(td)

    with xr.open_dataset(todo) as ds:
        print(ds)
        ds = ds.isel(j=10, j_g=10, record=-1)
        ds.to_netcdf('../reduceddata/AllSlice{}.nc'.format(td))
