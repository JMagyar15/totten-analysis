"""
SCRIPT FOR COMPUTING EVENT CATALOGUES FOR (A) NETWORK, (B) STATION TRIPLETS, AND (C) SINGLE STATIONS. ONLY THE NETWORK CATALOGUES ARE USED IN THE MANUSCRIPT.
"""

import os
from cryoquake import stream_handling as sh
from cryoquake import data_objects as do
from obspy.core.inventory import inventory
import pandas as pd
from pathlib import Path

#switches for different sections of code 
detect_network = True
detect_subarray = False
detect_single = False
detect_separate = False

t1 = sh.UTCDateTime(2018,12,24)
t2 = sh.UTCDateTime(2019,1,30)

chunk = do.SeismicChunk(t1,t2)

root = Path(__file__).parent.parent
w_path = root / "waveforms"
s_path = root / "stations"
c_path = root / "catalogues"

for path in [c_path]:
    if not os.path.exists(path):
        os.mkdir(path)


inv_files = os.listdir(s_path)

inv = inventory.Inventory()

for file in inv_files:
    temp_path = os.path.join(s_path,file)
    inv += inventory.read_inventory(temp_path,level='response',format='STATIONXML')


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
        


if detect_separate:
    """
    EVENT DETECTION
    """

    sta = 1.0
    lta = 10.0
    delta_sta = 10
    delta_lta = 10
    epsilon = 1.2
    thr_on = 3
    thr_off = 2
    thr_coincidence_sum = 1
    avg_wave_speed = 1.5
    thr_event_join = 5.0

    for daychunk in chunk(24*60*60):

        net_path = os.path.join(c_path,'all_stations')
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
        daychunk.filter('bandpass',freqmin=1,freqmax=30)

        daychunk.context('detect')

        print('Detecting events for',daychunk.str_name)
        print('Single-station catalogues')
        daychunk.detect_events(net_path,trigger_type='multistalta',sta=sta,lta=lta,delta_sta=delta_sta,delta_lta=delta_lta,epsilon=epsilon,thr_on=thr_on,thr_off=thr_off,thr_coincidence_sum=thr_coincidence_sum,avg_wave_speed=avg_wave_speed,thr_event_join=thr_event_join) 
    
    avail = pd.concat(avail_rows,ignore_index=True)
    avail.to_csv('totten_stream_availability.csv')
