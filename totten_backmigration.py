#do the backmigration, save coalescence surfaces to file, compute uncertainties and save these with the locations, might as well remake the masks when plotting uncertainty contours...
import numpy as np
import os
from iqvis import spatial_analysis as sa
from iqvis import data_objects as do
from obspy.core import read, UTCDateTime, inventory
from obspy.geodetics import degrees2kilometers, kilometers2degrees
import pandas as pd


decimation = 25

backmigrate = True
migrate_P_wave = False
compute_results = True
compute_P_results = True

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

if backmigrate:

    stacked_streams = {}
    for cluster_num in groups.keys():
        stacked_streams[cluster_num] = read(os.path.join(w_path,'stacked_waveforms_cluster_' + str(cluster_num) + '.mseed'))


    for cluster_num, stream in stacked_streams.items():
        stream = stream.filter('bandpass',freqmin=1,freqmax=5)

        east_grid = np.linspace(-10,5,100)
        north_grid = np.linspace(-5,10,100)
        depth_grid = np.linspace(0,4,50)

        east_grid, north_grid, depth_grid, t_grid, coal_surf, sta_xy, centre = sa.CoalescenceSurface(east_grid,north_grid,depth_grid,stream,inv,sta=1,lta=5,normalise=True,modulate=True,g=8,mod_win=0.2,smooth=30,decimation=decimation,mod_overlap=0.95)

        np.savez(os.path.join(coal_path,'coalescence_function_'+str(cluster_num)),x=east_grid,y=north_grid,z=depth_grid,t=t_grid,coal=coal_surf)


if migrate_P_wave:
    stacked_streams = {}
    for cluster_num in groups.keys():
        stacked_streams[cluster_num] = read(os.path.join(w_path,'stacked_waveforms_cluster_' + str(cluster_num) + '.mseed'))


    for cluster_num, stream in stacked_streams.items():
        stream = stream.filter('bandpass',freqmin=1,freqmax=5)

        east_grid = np.linspace(-10,5,100)
        north_grid = np.linspace(-5,10,100)
        depth_grid = np.linspace(0,4,50)

        east_grid, north_grid, depth_grid, t_grid, coal_surf, sta_xy, centre = sa.CoalescenceSurface(east_grid,north_grid,depth_grid,stream,inv,sta=1,lta=5,p_detect='P',normalise=True,modulate=True,mod_win=0.2,smooth=30,decimation=decimation,g=4,polZ=True)

        np.savez(os.path.join(coal_path,'P_coalescence_function_'+str(cluster_num)),x=east_grid,y=north_grid,z=depth_grid,t=t_grid,coal=coal_surf)

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