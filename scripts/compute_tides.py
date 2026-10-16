import xarray as xr
import numpy as np
import pyTMD
from pathlib import Path


root = Path(__file__).parent.parent

lon, lat = 114.5, -67.25   # x = longitude, y = latitude (EPSG:4326)

t = np.arange(np.datetime64('2018-12-01T00:00'),
              np.datetime64('2018-12-01T00:00') + np.timedelta64(100, 'D'),
              np.timedelta64(10, 'm'))

tide = pyTMD.compute.tide_elevations(
    lon, lat, t,
    directory='/Users/jaredmagyar/Documents',   # the folder containing the CATS2008 directory
    model='CATS2008',
    type='time series',
    standard='datetime',
    method='linear',
    extrapolate=True,
    cutoff=50.0,
)

print(tide)


ds = xr.Dataset(
    {'tide': ('time', tide.values.squeeze())},
    coords={'time': t},
)

ds['tide'].attrs = {'units': 'm', 'long_name': 'Tidal elevation (CATS2008)'}
ds.to_netcdf(root / 'totten_tides.nc')
