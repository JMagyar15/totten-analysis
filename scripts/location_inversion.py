import numpy as np
import xarray as xr
from cryoquake import spatial_analysis as sa
import pandas as pd
import os
from pathlib import Path
from obspy.core.inventory import inventory


directory = '/Users/jmagyar/Documents/MappingProducts/' #! replace with path to bed topography netcdf file.
bedmap3_file = directory + 'bedmap3.nc' #! replace with name of bed topography file
bedmap3 = xr.open_dataset(bedmap3_file, engine="netcdf4")


xmin = 2.2615e6
xmax = 2.2735e6
ymin = -1.0060e6
ymax = -0.9940e6

resolution = 25 #m

xlin = np.mgrid[xmin:xmax:resolution]
ylin = np.mgrid[ymin:ymax:resolution]

V = 3870 #m/s est
dV = 100 #m/s std

Delta = 10000 #m

root = Path(__file__).parent.parent
m_path = root / "misfit"
p_path = root / "stacked_waveforms"
s_path = root / "stations"


bed_grid = xr.Dataset(coords=dict(x=("x", xlin),y=("y", ylin)))
bedmap_interp = bedmap3.interp_like(bed_grid)
bed_grid = bed_grid.assign(dict(bed_topography = bedmap_interp.bed_topography,bed_uncertainty=bedmap_interp.bed_uncertainty))

inv_files = os.listdir(s_path)

inv = inventory.Inventory()

for file in inv_files:
    temp_path = os.path.join(s_path,file)
    inv += inventory.read_inventory(temp_path,level='response',format='STATIONXML')


for i, cluster_num in enumerate([31,43,34,44,33,32,39,54]):
    
    p_picks = pd.read_csv(os.path.join(m_path,'manual_picks_' + str(cluster_num) + '.csv'),index_col=0)
    p_picks['mu'] = p_picks['P1'] + (p_picks['P2'] - p_picks['P1'])/2
    p_picks['sigma'] = (p_picks['P2'] - p_picks['P1'])/4

    gamma_xr = sa.BedScan(inv,p_picks,bed_grid,V=3870,sigma_V=50,Delta=10000,bed_error=True,no_model_error=False)

    #save the xarray dataset
    gamma_xr.to_netcdf(os.path.join(m_path,'misfit_surface_'+str(cluster_num)+'.nc'))