import os
from cryoquake import stream_handling as sh
from cryoquake import data_objects as do
from cryoquake import spatial_analysis as sa
from obspy.core.inventory import inventory
from obspy.core import Stream, Trace, read, UTCDateTime
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial import distance
from pathlib import Path
from tqdm import tqdm
import fastcluster as fc

buffer = 5
window_length = 15

root = Path(__file__).parent.parent
w_path = root / "waveforms"
stack_path = root / "stacked_waveforms"
s_path = root / "stations"
c_path = root / "catalogues" / "all_stations"

inv_files = os.listdir(s_path)

inv = inventory.Inventory()

for file in inv_files:
    temp_path = os.path.join(s_path,file)
    inv += inventory.read_inventory(temp_path,level='response',format='STATIONXML')

inv = inv.select(station='TI?A',channel='CH?')

t1 = sh.UTCDateTime(2018,12,24)
t2 = sh.UTCDateTime(2019,1,30)


temp_cat = do.EventCatalogue(t1,t2,os.path.join(c_path),templates=True)


groups = temp_cat.group_split()


for key, group_cat in groups.items(): #loop over the event cluster catalogues
    print('Loading cluster ' + str(key) + '...')
    channels = inv.get_contents()['channels']

    streams = []

    for event in tqdm(group_cat,total=group_cat.N):
        event.attach_waveforms(inv,w_path,buffer=buffer,length=window_length,extra=10)
        event.decimate(5)
        event.filter('bandpass',freqmin=1,freqmax=30)  

        event_stream = event.get_data_window()

        if all([len(trace) == int(100*(buffer+window_length)+1) for trace in event_stream]):
            streams.append(event.get_data_window())

    
    stacked_stream = Stream()
    for code in channels:
        net,sta,loc,cha = code.split('.')

        stacked = np.stack([stream.select(network=net,station=sta,channel=cha)[0].data for stream in streams],axis=1)
        data = np.nanmean(stacked,axis=1)
        tr = Trace(data=data,header={'sampling_rate':100,'network':net,'station':sta,'location':loc,'channel':cha})
        stacked_stream += tr

    stacked_stream = stacked_stream.split()
    stacked_stream.write(os.path.join(stack_path,'stacked_waveforms_cluster_' + str(key) + '.mseed'))