# some of the basic analysis and visualisation for the Totten data

import os
#os.chdir('..') #change cwd so local functions can be imported

from iqvis import stream_handling as sh
from iqvis import data_objects as do
from iqvis import spatial_analysis as sa
from iqvis import moment_magnitude as mm
from obspy.core.inventory import inventory
import matplotlib.pyplot as plt
from iqvis import dayplot_backend as db
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import tqdm
import numpy as np
from obspy.core import read, UTCDateTime

network = False
triplet = False
single = False
templates = False
rel_amps = False
moment_mags = True

t1 = sh.UTCDateTime(2018,12,24)
t2 = sh.UTCDateTime(2019,1,30)

psd_window = 5
psd_overlap = 0.5

chunk = do.SeismicChunk(t1,t2)

path = '/Users/jmagyar/Documents/TottenData'
w_path = os.path.join(path,'waveforms')
s_path = os.path.join(path,'stations')
c_path = os.path.join(path,'catalogues')
p_path = os.path.join(path,'event_plots')
spec_path = os.path.join(path,'spectrograms')
coal_path = os.path.join(path,'coalescence')



inv_files = os.listdir(s_path)

inv = inventory.Inventory()

for file in inv_files:
    temp_path = os.path.join(s_path,file)
    inv += inventory.read_inventory(temp_path,level='response',format='STATIONXML')



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


if templates:
    temp_cat = do.EventCatalogue(chunk.starttime,chunk.endtime,os.path.join(c_path,'network'),templates=True)
    groups = temp_cat.group_split()

    cluster_amps = {}
    cluster_ind = {}
    
    for cluster_num in groups.keys():
        stream = read(os.path.join(path,'stacked_waveforms','high_stacked_waveforms_cluster_'+str(cluster_num) + '.mseed'))
        cluster_amps[cluster_num] = {}
        cluster_ind[cluster_num] = {}

        for net in inv.select(station='TI?A',channel='CH?'):
            for sta in net:
                substream = stream.select(network=net.code,station=sta.code,channel='CH?')
                amp_trace = np.linalg.norm(np.stack([tr.data for tr in substream],axis=1),axis=1)
                rel_amp = np.max(amp_trace)
                rel_loc = np.argmax(amp_trace)
                cluster_amps[cluster_num][sta.code] = rel_amp
                cluster_ind[cluster_num][sta.code] = rel_loc

    #now go through all the template matched events, calculate the attributes and also the amplitude ratio with the group stack.
    for daychunk in chunk:
        event_cat = do.EventCatalogue(daychunk.starttime,daychunk.endtime,os.path.join(c_path,'network'),templates=True)
        att_cat = event_cat.attributes
        filename = os.path.join(c_path,'network','template_waveform_attributes__' + daychunk.str_name + '.csv')

        for event in tqdm.tqdm(event_cat,total=event_cat.N):
            event.attach_waveforms(inv.select(station='TI?A',channel='CH?'),w_path,buffer=10,length=25)
            event.filter('highpass',freq=1)
            #event.remove_sensitivity()
            event.context('spectral')
            event.get_power_spectrum()
            window = event.get_data_window()
            dt = window[0].stats.delta

            amps = []
            ref_amps = []
            ratios = []
            freqs = []
            mean_freqs = []
            stations = []
            for net in event.inv:
                for sta in net:
                    substream = window.select(network=net.code,station=sta.code,channel='CH?')
                    amp_trace = np.linalg.norm(np.stack([tr.data for tr in substream],axis=1),axis=1)
                    
                    max_amp = amp_trace[cluster_ind[event.group][sta.code]]
                    ref_amp = cluster_amps[event.group][sta.code]
                    amp_ratio = max_amp / ref_amp

                    amps.append(max_amp)
                    ref_amps.append(ref_amp)
                    ratios.append(amp_ratio)

                    psd = event.psds[sta.code].Z
                    cumulative_psd = np.cumsum(psd) 
                    cumulative_psd /= cumulative_psd[-1] #normalise to final value to get CDF def

                    i50 = np.abs(cumulative_psd - 0.50).argmin()

                    central_freq = event.psds[sta.code].f[i50]
                    freqs.append(central_freq)

                    psd /= np.sum(psd)
                    weighted_freq = np.sum(psd * event.psds[sta.code].f)
                    mean_freqs.append(weighted_freq)

                    stations.append(sta.code)
            
            ind = np.nanargmax(amps)
            max_amp = amps[ind]
            ref_amp = ref_amps[ind]
            ratio = ratios[ind]
            freq = freqs[ind]
            mean_freq = mean_freqs[ind]
            ref_sta = stations[ind]

            att_cat.at[event.event_id,'amplitude'] = max_amp
            att_cat.at[event.event_id,'ref_amp'] = ref_amp
            att_cat.at[event.event_id,'amp_ratio'] = ratio
            att_cat.at[event.event_id,'med_freq'] = freq
            att_cat.at[event.event_id,'duration'] = event.duration
            att_cat.at[event.event_id,'mean_freq'] = mean_freq
            att_cat.at[event.event_id,'ref_station'] = ref_sta

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



if rel_amps:
    temp_cat = do.EventCatalogue(t1,t2,os.path.join(c_path),templates=True)
    groups = temp_cat.group_split()

    stacked_streams = {}
    for cluster_num in groups.keys():
        stacked_streams[cluster_num] = read(os.path.join(path,'stacked_waveforms','high_stacked_waveforms_cluster_'+str(cluster_num) + '.mseed'))
    
    rel_amp_dict = {}

    for cluster_num, stream in stacked_streams.items():
        #compute the a measure of ampltude for each of the stacked streams to divide each event by, do this with the raw filtered highpass stream, no need to convert to displacement at this point?
        rel_amps = {}
        rel_locs = {}
        for net in inv.select(station='TI?A',channel='CH?'):
            for sta in net:
                substream = stream.select(network=net.code,station=sta.code,channel='CH?')
                amp_trace = np.linalg.norm(np.stack([tr.data for tr in substream],axis=1),axis=1)
                rel_amp = np.max(amp_trace)
                rel_loc = np.argmax(amp_trace)
                rel_amps[sta.code] = rel_amp
                rel_locs[sta.code] = rel_loc

        #now loop through the event catalogue for this stacked stream and compute the amplitude ratios, makes these an attribute for the event...
        group_cat = groups[cluster_num]

        rows = []
        labels = []
        for event in group_cat:
            labels.append(event.event_id)
            row = {}
            event.attach_waveforms(inv.select(station='TI?A',channel='CH?'),w_path,buffer=10,length=25)
            event.filter('highpass',freq=1)

            for net in inv.select(station='TI?A',channel='CH?'):
                for sta in net:
                    substream = event.get_data_window().select(network=net.code,station=sta.code,channel='CH?')
                    amp_trace = np.linalg.norm(np.stack([tr.data for tr in substream],axis=1),axis=1)
                    #rel_amp = np.max(amp_trace)
                    amp_ratio = amp_trace[rel_locs[sta.code]] / rel_amps[sta.code]
                    row[sta.code] = amp_ratio
            rows.append(row)
                    

        rel_amp_df = pd.DataFrame(data=rows,index=labels)
        rel_amp_df['Mean'] = rel_amp_df.mean(axis=1,skipna=True)
        rel_amp_df['Median'] = rel_amp_df.median(axis=1,skipna=True)
        rel_amp_dict[cluster_num] = rel_amp_df

    for cluster_num, rel_amp in rel_amp_dict.items():
        rel_amp.to_csv()


if moment_mags:

    sta_xy, centre = sa.CartesianStationsDF(inv.select(station='TI?A'))
    icequake_loc = pd.read_csv(os.path.join(coal_path,'backmigration_results.csv'),index_col=0)

    for daychunk in chunk:
        #look through all matched events and estimate the moment magnitude by fitting the Brune model to the spectra
        temp_cat = do.EventCatalogue(daychunk.starttime,daychunk.endtime,os.path.join(c_path,'network'),templates=True)
        att_cat = temp_cat.attributes.drop(labels='group',axis=1)
        filename = os.path.join(c_path,'network','moment_magnitude__' + daychunk.str_name + '.csv')
        

        for event in tqdm.tqdm(temp_cat,total=temp_cat.N):
            cluster_num = int(event.group)
            event.attach_waveforms(inv.select(station='TI?A',channel='CH?'),w_path,buffer=10,length=25)
            event.filter('highpass',freq=1)

            stream = event.get_data_window()

            disp_stream = stream.copy()
            vel_stream = stream.copy()

            for tr in disp_stream.select(station='TI?A'):
                tr.stats.starttime = UTCDateTime(2019,1,10)
                tr.remove_response(inv,output='DISP')


            coal_file = np.load(os.path.join(coal_path,'coalescence_function_' + str(cluster_num) + '.npz'))
            x, y, z, t, coal = coal_file['x'], coal_file['y'], coal_file['z'], coal_file['t'], coal_file['coal']

            fit, uncertainties, centres = sa.UncertaintyQuantBackM(x,y,z,t,inv.select(station='TI?A'),coal,contours=[0.75])

            arrivals = sa.ArrivalTimes(inv.select(station='TI?A'),fit,uncertainties[0.75])

            try:
                sta_mag = mm.StationMomentMagnitude(disp_stream,arrivals,freqmin=1,freqmax=10)
                M0_out, Mw_out, fc_out = mm.CombinedMomentMagnitude(sta_mag)
                Mw = Mw_out.nominal_value
                M0 = M0_out.nominal_value
                fc = fc_out.nominal_value
                dMw = Mw_out.std_dev
                dM0 = M0_out.std_dev
                dfc = fc_out.std_dev
            except RuntimeError: 
                Mw = np.nan
                M0 = np.nan
                fc = np.nan
                dMw = np.nan
                dM0 = np.nan
                dfc = np.nan
                

            att_cat.at[event.event_id,'Mw'] = Mw
            att_cat.at[event.event_id,'M0'] = M0
            att_cat.at[event.event_id,'fc'] = fc

            att_cat.at[event.event_id,'dMw'] = dMw
            att_cat.at[event.event_id,'dM0'] = dM0
            att_cat.at[event.event_id,'dfc'] = dfc
        att_cat.to_csv(filename)
