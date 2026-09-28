import os
from cryoquake import stream_handling as sh
from cryoquake import data_objects as do
from obspy.core.inventory import inventory
import tqdm
import numpy as np
from pathlib import Path


single_all_stations = True
network = False
triplet = False
single = False

t1 = sh.UTCDateTime(2018,12,24)
t2 = sh.UTCDateTime(2019,1,30)

chunk = do.SeismicChunk(t1,t2)

root = Path(__file__).parent.parent
w_path = root / "waveforms"
s_path = root / "stations"
c_path = root / "catalogues"
m_path = root / "misfit"


inv_files = os.listdir(s_path)

inv = inventory.Inventory()

for file in inv_files:
    temp_path = os.path.join(s_path,file)
    inv += inventory.read_inventory(temp_path,level='response',format='STATIONXML')


if single_all_stations:
    
    for daychunk in chunk:
    
        event_cat = do.EventCatalogue(daychunk.starttime,daychunk.endtime,os.path.join(c_path,'all_stations'))
        att_cat = event_cat.attributes.drop(labels='group',axis=1)
        filename = os.path.join(c_path,'all_stations','waveform_attributes__' + daychunk.str_name + '.csv')
            
        for event in tqdm.tqdm(event_cat,total=event_cat.N):

            event.attach_waveforms(inv.select(station='TI?A',channel='CH?'),w_path,buffer=10,length=25)
            event.filter('bandpass',freqmin=1,freqmax=30)
            #event.remove_sensitivity()
            event.context('spectral')
            event.get_power_spectrum()
            window = event.get_data_window()
            dt = window[0].stats.delta

            amps = []
            freqs = []
            mean_freqs = []
            energies = []
            for net in event.inv:
                for sta in net:
                    sta_stream = window.select(network=net.code,station=sta.code)
                    flattened = np.stack([tr.data.astype(np.float64) for tr in sta_stream],axis=0)
                    amp = np.linalg.norm(flattened,axis=1) #otherwise compute energy here as squared norm along axis 0.
                    max_amp = np.max(amp)
                    amps.append(max_amp)

                    energy = np.sum(amp**2) * dt
                    energies.append(energy)

                    psd = event.psds[sta.code].Z
                    cumulative_psd = np.cumsum(psd) 
                    cumulative_psd /= cumulative_psd[-1] #normalise to final value to get CDF def

                    i50 = np.abs(cumulative_psd - 0.50).argmin()

                    central_freq = event.psds[sta.code].f[i50]
                    freqs.append(central_freq)

                    psd /= np.sum(psd)
                    weighted_freq = np.sum(psd * event.psds[sta.code].f)
                    mean_freqs.append(weighted_freq)
            
            ind = np.nanargmax(amps)
            max_amp = amps[ind]
            freq = freqs[ind]
            energy = energies[ind]
            mean_freq = mean_freqs[ind]

            att_cat.at[event.event_id,'amplitude'] = max_amp
            att_cat.at[event.event_id,'med_freq'] = freq
            att_cat.at[event.event_id,'duration'] = event.duration
            att_cat.at[event.event_id,'energy'] = energy
            att_cat.at[event.event_id,'mean_freq'] = mean_freq

        att_cat.to_csv(filename)

if network:
    for daychunk in chunk:

        event_cat = do.EventCatalogue(daychunk.starttime,daychunk.endtime,os.path.join(c_path,'network'))
        att_cat = event_cat.attributes.drop(labels='group',axis=1)
        filename = os.path.join(c_path,'network','waveform_attributes__' + daychunk.str_name + '.csv')
            
        for event in tqdm.tqdm(event_cat,total=event_cat.N):

            event.attach_waveforms(inv.select(station='TI?A',channel='CH?'),w_path,buffer=10,length=25)
            event.filter('bandpass',freqmin=1,freqmax=100)
            #event.remove_sensitivity()
            event.context('spectral')
            event.get_power_spectrum()
            window = event.get_data_window()
            dt = window[0].stats.delta

            amps = []
            freqs = []
            mean_freqs = []
            energies = []
            for net in event.inv:
                for sta in net:
                    sta_stream = window.select(network=net.code,station=sta.code)
                    flattened = np.stack([tr.data.astype(np.float64) for tr in sta_stream],axis=0)
                    amp = np.linalg.norm(flattened,axis=1) #otherwise compute energy here as squared norm along axis 0.
                    max_amp = np.max(amp)
                    amps.append(max_amp)

                    energy = np.sum(amp**2) * dt
                    energies.append(energy)

                    psd = event.psds[sta.code].Z
                    cumulative_psd = np.cumsum(psd) 
                    cumulative_psd /= cumulative_psd[-1] #normalise to final value to get CDF def

                    i50 = np.abs(cumulative_psd - 0.50).argmin()

                    central_freq = event.psds[sta.code].f[i50]
                    freqs.append(central_freq)

                    psd /= np.sum(psd)
                    weighted_freq = np.sum(psd * event.psds[sta.code].f)
                    mean_freqs.append(weighted_freq)
            
            ind = np.nanargmax(amps)
            max_amp = amps[ind]
            freq = freqs[ind]
            energy = energies[ind]
            mean_freq = mean_freqs[ind]

            att_cat.at[event.event_id,'amplitude'] = max_amp
            att_cat.at[event.event_id,'med_freq'] = freq
            att_cat.at[event.event_id,'duration'] = event.duration
            att_cat.at[event.event_id,'energy'] = energy
            att_cat.at[event.event_id,'mean_freq'] = mean_freq

        att_cat.to_csv(filename)



if triplet:
    for trip_code in ['TI3','TI6','TI8']:
        for daychunk in chunk:

            event_cat = do.EventCatalogue(daychunk.starttime,daychunk.endtime,os.path.join(c_path,trip_code))
            att_cat = event_cat.attributes.drop(labels='group',axis=1)
            filename = os.path.join(c_path,trip_code,'waveform_attributes__' + daychunk.str_name + '.csv')
                
            for event in tqdm.tqdm(event_cat,total=event_cat.N):

                event.attach_waveforms(inv.select(station=trip_code+'?',channel='CH?'),w_path,buffer=5)
                event.filter('bandpass',freqmin=1,freqmax=100)
                #event.remove_sensitivity()
                event.context('spectral')
                event.get_power_spectrum()
                window = event.get_data_window()
                dt = window[0].stats.delta

                amps = []
                freqs = []
                mean_freqs = []
                energies = []
                for net in event.inv:
                    for sta in net:
                        sta_stream = window.select(network=net.code,station=sta.code)
                        flattened = np.stack([tr.data.astype(np.float64) for tr in sta_stream],axis=0)
                        amp = np.linalg.norm(flattened,axis=1) #otherwise compute energy here as squared norm along axis 0.
                        max_amp = np.max(amp)
                        amps.append(max_amp)

                        energy = np.sum(amp**2) * dt
                        energies.append(energy)

                        psd = event.psds[sta.code].Z
                        cumulative_psd = np.cumsum(psd) 
                        cumulative_psd /= cumulative_psd[-1] #normalise to final value to get CDF def

                        i50 = np.abs(cumulative_psd - 0.50).argmin()

                        central_freq = event.psds[sta.code].f[i50]
                        freqs.append(central_freq)

                        psd /= np.sum(psd)
                        weighted_freq = np.sum(psd * event.psds[sta.code].f)
                        mean_freqs.append(weighted_freq)
                
                ind = np.nanargmax(amps)
                max_amp = amps[ind]
                freq = freqs[ind]
                energy = energies[ind]
                mean_freq = mean_freqs[ind]

                att_cat.at[event.event_id,'amplitude'] = max_amp
                att_cat.at[event.event_id,'med_freq'] = freq
                att_cat.at[event.event_id,'duration'] = event.duration
                att_cat.at[event.event_id,'energy'] = energy
                att_cat.at[event.event_id,'mean_freq'] = mean_freq

            att_cat.to_csv(filename)

if single:
    for net in inv:
        for station in net:
            sub_path = os.path.join(c_path,station.code + '_single')
            print(sub_path)

            for daychunk in chunk:

                event_cat = do.EventCatalogue(daychunk.starttime,daychunk.endtime,sub_path)
                att_cat = event_cat.attributes.drop(labels='group',axis=1)
                filename = os.path.join(sub_path,'waveform_attributes__' + daychunk.str_name + '.csv')
                    
                for event in tqdm.tqdm(event_cat,total=event_cat.N):

                    event.attach_waveforms(inv.select(station=station.code,channel='CH?'),w_path,buffer=5)
                    event.filter('bandpass',freqmin=1,freqmax=100)
                    #event.remove_sensitivity()
                    event.context('spectral')
                    event.get_power_spectrum()
                    window = event.get_data_window()
                    dt = window[0].stats.delta

                    flattened = np.stack([tr.data.astype(np.float64) for tr in window],axis=0)
                    amp = np.linalg.norm(flattened,axis=1) #otherwise compute energy here as squared norm along axis 0.
                    max_amp = np.max(amp)

                    energy = np.sum(amp**2) * dt

                    psd = event.psds[station.code].Z
                    cumulative_psd = np.cumsum(psd) 
                    cumulative_psd /= cumulative_psd[-1] #normalise to final value to get CDF def

                    i50 = np.abs(cumulative_psd - 0.50).argmin()

                    central_freq = event.psds[station.code].f[i50]

                    psd /= np.sum(psd)
                    weighted_freq = np.sum(psd * event.psds[station.code].f)
                    
                    
                    att_cat.at[event.event_id,'amplitude'] = max_amp
                    att_cat.at[event.event_id,'med_freq'] = central_freq
                    att_cat.at[event.event_id,'duration'] = event.duration
                    att_cat.at[event.event_id,'energy'] = energy
                    att_cat.at[event.event_id,'mean_freq'] = weighted_freq

                att_cat.to_csv(filename)