
import os
#os.chdir('..') #change cwd so local functions can be imported

from cryoquake import stream_handling as sh
from cryoquake import data_objects as do
from cryoquake import moment_magnitude as mm
from obspy.core.inventory import inventory
import tqdm
import numpy as np
from obspy.core import read, UTCDateTime
import xarray as xr
from pathlib import Path



t1 = sh.UTCDateTime(2018,12,24)
t2 = sh.UTCDateTime(2019,1,30)

chunk = do.SeismicChunk(t1,t2)

root = Path(__file__).parent.parent
w_path = root / "stacked_waveforms"
raw_path = root / "waveforms"
s_path = root / "stations"
c_path = root / "catalogues"

# path = '/Users/jmagyar/Documents/TottenData'
# w_path = os.path.join(path,'stacked_waveforms')
# raw_path = os.path.join(path,'waveforms')
# s_path = os.path.join(path,'stations')
# c_path = os.path.join(path,'catalogues')

inv_files = os.listdir(s_path)

inv = inventory.Inventory()

for file in inv_files:
    temp_path = os.path.join(s_path,file)
    inv += inventory.read_inventory(temp_path,level='response',format='STATIONXML')

temp_cat = do.EventCatalogue(t1,t2,os.path.join(c_path,'network','filtered'),templates=True)

groups = temp_cat.group_split()

group_mags = {}
group_refs = {}
group_arrivals = {}
group_stations = {}

for cluster_num, group in groups.items():
    #firstly get the moment information for the stacked waveforms...

    try:
        gamma_xr = xr.load_dataset(os.path.join(path,'misfit','misfit_surface_'+str(cluster_num)+'.nc'))
        arrivals = mm.Misfit2Arrivals(gamma_xr)
        group_stations[cluster_num] = gamma_xr.stations

        stacked_stream = read(os.path.join(w_path,'filtered_stacked_waveforms_cluster_' + str(cluster_num) + '.mseed'))
        stacked_stream.filter('bandpass',freqmin=1,freqmax=10)

        ref_energy = {}

        for tr in stacked_stream:
            if tr.stats.station not in gamma_xr.stations:
                stacked_stream.remove(tr)
        
        for tr in stacked_stream:
            tr.stats.starttime = UTCDateTime(2019,1,10)
            tr.remove_response(inv,output='DISP')

            if tr.id[-1] == 'Z':
                P_win = tr.slice(tr.stats.starttime + arrivals['P_start'][tr.stats.station],tr.stats.starttime + arrivals['P_end'][tr.stats.station])
                P_energy = np.sqrt(np.sum((tr.data**2 * tr.stats.sampling_rate)))

                ref_energy[tr.stats.station] = P_energy

        station_mags = mm.StationMomentMagnitude(stacked_stream,arrivals,freqmin=1,freqmax=10)

        print(cluster_num)
        print(station_mags)

        group_mags[cluster_num] = station_mags
        group_refs[cluster_num] = ref_energy
        group_arrivals[cluster_num] = arrivals
        
    except FileNotFoundError:
        group_mags[cluster_num] = None
        group_refs[cluster_num] = None
        group_arrivals[cluster_num] = None
        group_stations[cluster_num] = []
    

    #keep station mags and make copy of dataframe for each matched event and just mulitply the M0 and errors by amplitude ratio in P window

for daychunk in chunk:
    temp_cat = do.EventCatalogue(daychunk.starttime,daychunk.endtime,os.path.join(c_path,'network','filtered'),templates=True)
    att_cat = temp_cat.attributes
    filename = os.path.join(c_path,'network','filtered','moment_magnitude__' + daychunk.str_name + '.csv')

    for event in tqdm.tqdm(temp_cat,total=temp_cat.N):
        event.attach_waveforms(inv.select(station='TI?A',channel='CHZ'),raw_path,buffer=10,length=25)
        event.filter('bandpass',freqmin=1,freqmax=10)

        cluster_num = event.group

        station_mags = group_mags[cluster_num]
        ref_energy = group_refs[cluster_num]
        arrivals = group_arrivals[cluster_num]

        stream = event.get_data_window()

        if arrivals is not None:
            event_station_mags = station_mags.drop(columns=['logomega0','fc','t_star','dlogomega0','dfc','dt_star','Mw','dMw']).copy()

            for tr in stream:
                if tr.stats.station not in group_stations[cluster_num]:
                    stream.remove(tr)
            
            for tr in stream:
                tr.stats.starttime = UTCDateTime(2019,1,10)
                tr.remove_response(inv,output='DISP')

                P_win = tr.slice(tr.stats.starttime + arrivals['P_start'][tr.stats.station],tr.stats.starttime + arrivals['P_end'][tr.stats.station])
                P_energy = np.sqrt(np.sum((tr.data**2 * tr.stats.sampling_rate)))

                ratio = P_energy / ref_energy[tr.stats.station]

                #print(tr.stats.station,ratio)

                event_station_mags.at[tr.stats.station,'M0'] *= (ratio)
                event_station_mags.at[tr.stats.station,'dM0'] *= (ratio)

            M0_out, Mw_out = mm.CombinedMomentMagnitude(event_station_mags)
            Mw = Mw_out.nominal_value
            M0 = M0_out.nominal_value
            dMw = Mw_out.std_dev
            dM0 = M0_out.std_dev

            att_cat.at[event.event_id,'Mw'] = Mw
            att_cat.at[event.event_id,'M0'] = M0

            att_cat.at[event.event_id,'dMw'] = dMw
            att_cat.at[event.event_id,'dM0'] = dM0
        else:
            att_cat.at[event.event_id,'Mw'] = np.nan
            att_cat.at[event.event_id,'M0'] = np.nan

            att_cat.at[event.event_id,'dMw'] = np.nan
            att_cat.at[event.event_id,'dM0'] = np.nan
    
    att_cat.to_csv(filename)
    