#do the backmigration, save coalescence surfaces to file, compute uncertainties and save these with the locations, might as well remake the masks when plotting uncertainty contours...
import numpy as np
import os
from cryoquake import spatial_analysis as sa
from cryoquake import data_objects as do
from obspy.core import read, UTCDateTime, inventory
from obspy.geodetics import degrees2kilometers, kilometers2degrees
import pandas as pd
import xarray as xr


decimation = 10

backmigrate = False
migrate_P_wave = True
compute_results = False
compute_P_results = False

t1 = UTCDateTime(2018,12,24)
t2 = UTCDateTime(2019,1,30)

chunk = do.SeismicChunk(t1,t2)

path = '/Users/jmagyar/Documents/TottenData'
w_path = os.path.join(path,'stacked_waveforms')
s_path = os.path.join(path,'stations')
c_path = os.path.join(path,'catalogues')
coal_path = os.path.join(path,'coalescence')

temp_cat = do.EventCatalogue(t1,t2,os.path.join(c_path),templates=True)

moment_mags = chunk.load_csv(os.path.join(c_path,'network'),'moment_magnitude')
temp_cat.attributes['Mw'] = moment_mags['Mw']
groups = temp_cat.group_split()

inv_files = os.listdir(s_path)

inv = inventory.Inventory()

for file in inv_files:
    temp_path = os.path.join(s_path,file)
    inv += inventory.read_inventory(temp_path,level='response',format='STATIONXML')

inv = inv.select(station='TI?A',channel='CH?')


xmin = 2.26e6
xmax = 2.28e6
ymin = -1.01e6
ymax = -9.9e5

resolution = 200

xlin = np.mgrid[xmin:xmax:resolution]
ylin = np.mgrid[ymin:ymax:resolution]

directory = '/Users/jmagyar/Documents/MappingProducts/'
bedmap3_file = directory + 'bedmap3.nc'
bedmap3 = xr.open_dataset(bedmap3_file, engine="netcdf4")

bed_grid = xr.Dataset(coords=dict(x=("x", xlin),y=("y", ylin)))
bedmap_interp = bedmap3.interp_like(bed_grid)
bed_grid = bed_grid.assign(dict(bed_topography = bedmap_interp.bed_topography))


if backmigrate:

    stacked_streams = {}
    for cluster_num in groups.keys():
        stacked_streams[cluster_num] = read(os.path.join(w_path,'stacked_waveforms_cluster_' + str(cluster_num) + '.mseed'))


    for cluster_num, stream in stacked_streams.items():
        stream = stream.filter('bandpass',freqmin=1,freqmax=10) 

        east_grid = np.linspace(-10,5,50)
        north_grid = np.linspace(-5,10,50)
        depth_grid = np.linspace(0,3,20)
        
        east_grid, north_grid, depth_grid, t_grid, coal_surf, sta_xy, centre = sa.CoalescenceSurface(east_grid,north_grid,depth_grid,stream,inv,sta=1,lta=5,normalise=True,modulate=True,g=8,mod_win=0.2,smooth=30,decimation=decimation,mod_overlap=0.95)
        np.savez(os.path.join(coal_path,'coalescence_function_'+str(cluster_num)),x=east_grid,y=north_grid,z=depth_grid,t=t_grid,coal=coal_surf)


if migrate_P_wave:
    stacked_streams = {}
    for cluster_num in groups.keys():
        stacked_streams[cluster_num] = read(os.path.join(w_path,'stacked_waveforms_cluster_' + str(cluster_num) + '.mseed'))


    for cluster_num, stream in stacked_streams.items():
        stream = stream.filter('bandpass',freqmin=1,freqmax=10)

        #east_grid = np.linspace(-8,2,50)
        #north_grid = np.linspace(0,10,50)
        #depth_grid = np.linspace(0,3,30)

        #want to make an xarray dataset with the grid of x,y,z values for the bed.

        coal_grid = sa.CoalescenceBedSearch(bed_grid,stream,inv,sta=0.5,lta=5,normalise=True,modulate=True,decimation=decimation,g=8,smooth=50,mod_win=1,mod_overlap=0.95,p_detect='radial_self',s_detect=None)
        coal_grid.to_netcdf(os.path.join(coal_path,'P_coalescence_function_'+str(cluster_num))+'.nc')
        #np.savez(os.path.join(coal_path,'P_coalescence_function_'+str(cluster_num)),x=east_grid,y=north_grid,z=depth_grid,t=t_grid,coal=coal_surf)

if compute_results:
    #make a dataframe and save with the locations (x,y) and (lon,lat) and error estimates. Also provide an interval and error estimate on r for each station.
    #also take the central location along with the maximum as this could be a good alternate estimate of location...
    rows = []
    clusters = []
    for cluster_num in groups.keys():
        clusters.append(cluster_num)

        coal_file = np.load(os.path.join(coal_path,'coalescence_function_' + str(cluster_num) + '.npz'))
        x, y, z, t, coal = coal_file['x'], coal_file['y'], coal_file['z'], coal_file['t'], coal_file['coal']

        fit, uncertainties, centres = sa.UncertaintyQuantBackM(x,y,z,t,inv.select(station='TI?A'),coal,time_slice=True)
        sta_xy, centre = sa.CartesianStations(inv.select(station='TI?A'))

        row = {}

        x_fit, y_fit, z_fit, t_fit, r_fit = fit

        max_coal = np.max(coal)

        row['max_coal'] = max_coal

        row['med_Mw'] = groups[cluster_num].attributes['Mw'].median()

        row['X'] = x_fit
        row['Y'] = y_fit
        row['Z'] = z_fit
        row['t'] = t_fit

        #now convert to lon/lat

        lon = kilometers2degrees(x_fit,radius=6371.0*np.cos(np.deg2rad(centre[0]))) + centre[1]
        lat = kilometers2degrees(y_fit) + centre[0]

        row['lon'] = lon
        row['lat'] = lat

        #now do similar but with the centres of the connected region #!might want to try centroid means with 0.75 as location estimate...
        means = centres[0.75]
        sigmas = uncertainties[0.75]

        row['cX'] = means[0]
        row['cY'] = means[1]
        row['cZ'] = means[2]

        lon = kilometers2degrees(means[0],radius=6371.0*np.cos(np.deg2rad(centre[0]))) + centre[1]
        lat = kilometers2degrees(means[1]) + centre[0]

        row['clon'] = lon
        row['clat'] = lat

        row['sX'] = sigmas[0]
        row['sY'] = sigmas[1]
        row['sZ'] = sigmas[2]

        rows.append(row)

    df = pd.DataFrame(data=rows,index=clusters)

    df.to_csv(os.path.join(coal_path,'backmigration_results' + '.csv'))


if compute_P_results:
    #make a dataframe and save with the locations (x,y) and (lon,lat) and error estimates. Also provide an interval and error estimate on r for each station.
    #also take the central location along with the maximum as this could be a good alternate estimate of location...
    rows = []
    clusters = []
    for cluster_num in groups.keys():
        clusters.append(cluster_num)

        coal_file = np.load(os.path.join(coal_path,'P_coalescence_function_' + str(cluster_num) + '.npz'))
        x, y, z, t, coal = coal_file['x'], coal_file['y'], coal_file['z'], coal_file['t'], coal_file['coal']

        fit, uncertainties, centres = sa.UncertaintyQuantBackM(x,y,z,t,inv.select(station='TI?A'),coal,time_slice=True)
        sta_xy, centre = sa.CartesianStations(inv.select(station='TI?A'))

        row = {}

        x_fit, y_fit, z_fit, t_fit, r_fit = fit

        max_coal = np.max(coal)

        row['med_Mw'] = groups[cluster_num].attributes['Mw'].median()

        row['max_coal'] = max_coal
        row['X'] = x_fit
        row['Y'] = y_fit
        row['Z'] = z_fit
        row['t'] = t_fit

        #now convert to lon/lat

        lon = kilometers2degrees(x_fit,radius=6371.0*np.cos(np.deg2rad(centre[0]))) + centre[1]
        lat = kilometers2degrees(y_fit) + centre[0]

        row['lon'] = lon
        row['lat'] = lat

        #now do similar but with the centres of the connected region #!might want to try centroid means with 0.75 as location estimate...
        means = centres[0.75]
        sigmas = uncertainties[0.75]

        row['cX'] = means[0]
        row['cY'] = means[1]
        row['cZ'] = means[2]

        lon = kilometers2degrees(means[0],radius=6371.0*np.cos(np.deg2rad(centre[0]))) + centre[1]
        lat = kilometers2degrees(means[1]) + centre[0]

        row['clon'] = lon
        row['clat'] = lat

        row['sX'] = sigmas[0]
        row['sY'] = sigmas[1]
        row['sZ'] = sigmas[2]

        rows.append(row)

    df = pd.DataFrame(data=rows,index=clusters)

    df.to_csv(os.path.join(coal_path,'P_backmigration_results' + '.csv'))