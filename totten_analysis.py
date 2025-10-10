# some of the basic analysis and visualisation for the Totten data

import os
#os.chdir('..') #change cwd so local functions can be imported

from iqvis import stream_handling as sh
from iqvis import data_objects as do
from obspy.core.inventory import inventory
import matplotlib.pyplot as plt
from iqvis import dayplot_backend as db
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

spectra = False
detect_network = False
detect_subarray = False
detect_single = True

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


inv_files = os.listdir(s_path)

inv = inventory.Inventory()

for file in inv_files:
    temp_path = os.path.join(s_path,file)
    inv += inventory.read_inventory(temp_path,level='response',format='STATIONXML')


#compute median spectrograms to get feel for background seismicity, any diurnal/seasonal variations, data quality

if not os.path.exists(spec_path):
    os.mkdir(spec_path)


if spectra:

    for daychunk in chunk:
        print('Computing PSDs for',daychunk.str_name)

        daychunk.context('spectral')
        daychunk.attach_waveforms(inv,w_path)
        daychunk.stream = daychunk.stream.split().merge(fill_value=None)
        daychunk.stream.trim(daychunk.starttime,daychunk.endtime,pad=True,fill_value=None)
        daychunk.decimate(5)
        print(daychunk.stream)
        daychunk.make_periodograms(psd_window,psd_overlap,spec_path)

#event detection on filtered and decimated data


"""
STATION SELECTION
"""

detect_inv = inv.copy()
detect_inv = inv.select(station='TI?A',channel='CH?')

avail_rows = []

if detect_network:
    """
    EVENT DETECTION
    """

    sta = 0.2
    lta = 2.0
    delta_sta = 50
    delta_lta = 50
    epsilon = 2
    thr_on = 6
    thr_off = 4
    thr_coincidence_sum = 3
    avg_wave_speed = 1.5
    thr_event_join = 5.0

    for daychunk in chunk(24*60*60):

        net_path = os.path.join(c_path,'network')
        if not os.path.exists(net_path):
            os.mkdir(net_path)
        
        #firstly get the data availability
        print('Recording data availability for',daychunk.str_name)
        daychunk.attach_waveforms(detect_inv,w_path)
        split_stream = daychunk.stream.split()

        for tr in split_stream:
            network, station, location, channel = tr.id.split('.')
            row = {'Network':network,'Station':station,'Location':location,'Channel':channel,'Start':tr.stats.starttime,'End':tr.stats.endtime}
            row = pd.DataFrame(data=row,index=[tr.id])
            avail_rows.append(row)


        #now make the broadband catalogue
        daychunk.attach_waveforms(detect_inv,w_path,buffer=60*60) #hour long buffer for filtering and STA/LTA
        #daychunk.decimate(5)
        #daychunk.remove_response(pre_filt=[1,2,20,25],taper=False)   
        daychunk.filter('bandpass',freqmin=1,freqmax=100)

        daychunk.context('detect')

        print('Detecting events for',daychunk.str_name)
        daychunk.detect_events(net_path,trigger_type='multistalta',sta=sta,lta=lta,delta_sta=delta_sta,delta_lta=delta_lta,epsilon=epsilon,thr_on=thr_on,thr_off=thr_off,thr_coincidence_sum=thr_coincidence_sum,avg_wave_speed=avg_wave_speed,thr_event_join=thr_event_join) 
        
    
    avail = pd.concat(avail_rows,ignore_index=True)
    avail.to_csv('totten_stream_availability.csv')


if detect_subarray:
    """
    EVENT DETECTION
    """

    sta = 0.2
    lta = 2.0
    delta_sta = 50
    delta_lta = 50
    epsilon = 2
    thr_on = 4
    thr_off = 3
    thr_coincidence_sum = 2
    avg_wave_speed = 1.5
    thr_event_join = 5.0

    for daychunk in chunk(24*60*60):
        
        for station in ['TI3?','TI6?','TI8?']:
            
            sub_path = os.path.join(c_path,station[:-1])
            if not os.path.exists(sub_path):
                os.mkdir(sub_path)

            subarray = inv.select(station=station,channel='CH?')

            #now make the broadband catalogue
            daychunk.attach_waveforms(subarray,w_path,buffer=60*60) #hour long buffer for filtering and STA/LTA
            #daychunk.decimate(5)
            #daychunk.remove_response(pre_filt=[1,2,20,25],taper=False)   
            daychunk.filter('bandpass',freqmin=1,freqmax=100)

            print(daychunk.stream)

            daychunk.context('detect')

            print('Detecting events for',daychunk.str_name)
            daychunk.detect_events(sub_path,trigger_type='multistalta',sta=sta,lta=lta,delta_sta=delta_sta,delta_lta=delta_lta,epsilon=epsilon,thr_on=thr_on,thr_off=thr_off,thr_coincidence_sum=thr_coincidence_sum,avg_wave_speed=avg_wave_speed,thr_event_join=thr_event_join) 
    
if detect_single:
    """
    EVENT DETECTION
    """

    sta = 0.2
    lta = 2.0
    delta_sta = 50
    delta_lta = 50
    epsilon = 2
    thr_on = 4
    thr_off = 3
    thr_coincidence_sum = 1
    avg_wave_speed = 1.5
    thr_event_join = 5.0

    for daychunk in chunk(24*60*60):
        
        for net in inv:
            for station in net:

                sub_path = os.path.join(c_path,station.code+'_single')
                if not os.path.exists(sub_path):
                    os.mkdir(sub_path)

                subarray = inv.select(station=station.code,channel='CH?')

                #now make the broadband catalogue
                daychunk.attach_waveforms(subarray,w_path,buffer=60*60) #hour long buffer for filtering and STA/LTA
                #daychunk.decimate(5)
                #daychunk.remove_response(pre_filt=[1,2,20,25],taper=False)   
                daychunk.filter('bandpass',freqmin=1,freqmax=100)

                print(daychunk.stream)

                daychunk.context('detect')

                print('Detecting events for',daychunk.str_name)
                daychunk.detect_events(sub_path,trigger_type='multistalta',sta=sta,lta=lta,delta_sta=delta_sta,delta_lta=delta_lta,epsilon=epsilon,thr_on=thr_on,thr_off=thr_off,thr_coincidence_sum=thr_coincidence_sum,avg_wave_speed=avg_wave_speed,thr_event_join=thr_event_join) 
        
