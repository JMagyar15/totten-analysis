"""
SCRIPT FOR ESTIMATING SOURCE LOCATIONS. A LOCAL VERSION OF BEDMAP3 (OR ANOTHER BED TOPOGRAPHY PRODUCT) AND A .CSV OF ARRIVAL TIMES IS NEEDED. 
THESE ARRIVAL TIMES CAN BE FOUND IN THE GITHUB REPOSITORY, AND THE FULL PRE-COMPUTED MISFIT SURFACES IN THE MISFIT DIRECTORY.
"""

import numpy as np
import xarray as xr
from cryoquake import spatial_analysis as sa
from cryoquake import data_objects as do
import pandas as pd
import os
from pathlib import Path
from obspy.core.inventory import inventory
from obspy.core import UTCDateTime


directory = '/Users/jaredmagyar/Documents/MappingProducts/' #! replace with path to bed topography netcdf file.
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
sigma_V = 100 #m/s std
sigma_floor = 0.2
Delta = 10000 #m

root = Path(__file__).parent.parent
m_path = root / "misfit"
p_path = root / "stacked_waveforms"
s_path = root / "stations"
a_path = root / "arrivals"
c_path = root / "catalogues" / "classic_high_threshold"

for path in [m_path]:
    if not os.path.exists(path):
        os.mkdir(path)


bed_grid = xr.Dataset(coords=dict(x=("x", xlin),y=("y", ylin)))
bedmap_interp = bedmap3.interp_like(bed_grid)
bed_grid = bed_grid.assign(dict(bed_topography = bedmap_interp.bed_topography,bed_uncertainty = bedmap_interp.bed_uncertainty))

inv_files = os.listdir(s_path)

inv = inventory.Inventory()

for file in inv_files:
    temp_path = os.path.join(s_path,file)
    inv += inventory.read_inventory(temp_path,level='response',format='STATIONXML')

inv = inv.select(station='TI?A',channel='CH?')


bedscan = sa.BedScanner(inv,bed_grid)
bedscan.assign_parameters(V,sigma_V,Delta)


temp_cat = do.EventCatalogue(UTCDateTime(2018,12,24),UTCDateTime(2019,1,30),c_path,templates=True)
groups = temp_cat.group_split()


for cluster_num in groups.keys():

    try:
        p_picks = pd.read_csv(os.path.join(a_path,'manual_picks_' + str(cluster_num) + '.csv'),index_col=0)
        p_picks.loc['TI7A','P1'] = np.nan
        p_picks.loc['TI7A','P2'] = np.nan
        p_picks['mu'] = p_picks['P1'] + (p_picks['P2'] - p_picks['P1'])/2
        p_picks['sigma'] = (p_picks['P2'] - p_picks['P1'])/4

        gamma_xr = bedscan.fit(p_picks,sigma_floor=sigma_floor)
        gamma_xr.attrs['N'] = groups[cluster_num].N

        gamma_xr.to_netcdf(os.path.join(m_path,'misfit_surface_'+str(cluster_num)+'.nc'))
    
    except FileNotFoundError:
        print('No arrivals for ' +  str(cluster_num) + ', moving to next')