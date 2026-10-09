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


def collect_events(chunk, c_path, buffer, window_length):
    catalogue = do.EventCatalogue(chunk.starttime,chunk.endtime,c_path,templates=True)
    groups = catalogue.group_split()

    X_groups = {}
    X_sim = {}
    X_groups_len = {}

    for cluster_num, group_cat in groups.items():
        all_traces = []
        all_sim = []

        for event in group_cat:
            event_stream = chunk.stream.slice(event.starttime - buffer, event.starttime + window_length)
            all_traces.append(np.stack([tr.data for tr in event_stream],axis=0))
            all_sim.append(group_cat.attributes.loc[event.event_id]['similarity'])

        
        X = np.stack(all_traces,axis=1)
        X = X - X.mean(axis=2,keepdims=True)
        X_len = np.linalg.norm(X, axis=2, keepdims=True)
        X /= X_len


        if isinstance(X,np.ma.MaskedArray):
            X = X.filled(np.nan)

        if isinstance(X_len,np.ma.MaskedArray):
            X_len = X_len.filled(np.nan)

        X_sim[cluster_num] = np.array(all_sim)
        
        X_groups[cluster_num] = X
        X_groups_len[cluster_num] = X_len

    return X_groups, X_groups_len, X_sim

def collect_streams(chunk, c_path, buffer, window_length):
    catalogue = do.EventCatalogue(chunk.starttime,chunk.endtime,c_path,templates=True)
    groups = catalogue.group_split()

    X_groups = {}

    for cluster_num, group_cat in groups.items():
        all_streams = Stream()

        for event in group_cat:
            event_stream = chunk.stream.slice(event.starttime - buffer, event.starttime + window_length)
            all_streams += event_stream
        
        X_groups[cluster_num] = all_streams

    return X_groups


buffer = 5
window_length = 15

root = Path(__file__).parent.parent
w_path = root / "waveforms"
stack_path = root / "stacked_waveforms"
s_path = root / "stations"
c_path = root / "catalogues" / "classic_high_threshold"

inv_files = os.listdir(s_path)

inv = inventory.Inventory()

for file in inv_files:
    temp_path = os.path.join(s_path,file)
    inv += inventory.read_inventory(temp_path,level='response',format='STATIONXML')

inv = inv.select(station='TI?A',channel='CH?')

t1 = sh.UTCDateTime(2018,12,24)
t2 = sh.UTCDateTime(2019,1,30)


chunk = do.SeismicChunk(t1,t2)
all_days = []
all_lens = []
all_sim = []


for daychunk in chunk:
    print('Slicing data for ' + str(daychunk.starttime.date))
    daychunk.attach_waveforms(inv.select(station='TI?A',channel='CHZ'),w_path,buffer=60*60)
    daychunk.decimate(5)
    daychunk.filter('bandpass',freqmin=3,freqmax=30) #relatively low high corner to help with correlation
    X_groups, X_groups_len, X_sim = collect_events(daychunk,c_path,10,20)
    all_days.append(X_groups)
    all_lens.append(X_groups_len)
    all_sim.append(X_sim)

sta_lst = [daychunk.stream[i].id for i in range(len(daychunk.stream))]

temp_cat = do.EventCatalogue(t1,t2,c_path,templates=True)
group_cats = temp_cat.group_split()

data_groups = {}
lens_groups = {}
sim_groups = {}
weights_groups = {}
stacks = {}

for clust_num in group_cats.keys():
    data = []
    lens = []
    sim = []
    for i, day in enumerate(all_days):
        if clust_num in day:
            data.append(day[clust_num])
            lens.append(all_lens[i][clust_num])
            sim.append(all_sim[i][clust_num])
    data = np.concatenate(data,axis=1)
    lens = np.concatenate(lens,axis=1)
    sim = np.concatenate(sim,axis=0)

    sort_ind = np.argsort(sim)

    data = data[:,sort_ind,:]
    lens = lens[:,sort_ind,0]
    sim = sim[sort_ind]
    weights = (sim - 0.5)*2

    data_groups[clust_num] = data
    lens_groups[clust_num] = lens
    sim_groups[clust_num] = sim
    weights_groups[clust_num] = weights

    stacks[clust_num] = np.nanmean(weights[None,:,None] * data,axis=1)

    stacked_stream = Stream()
    for i, station in enumerate(sta_lst):
        net,sta,loc,cha = station.split('.')
        tr_data = stacks[clust_num][i,:]
        tr = Trace(data=tr_data,header={'sampling_rate':100,'network':net,'station':sta,'location':loc,'channel':cha})
        stacked_stream += tr

    stacked_stream = stacked_stream.split()
    stacked_stream.write(os.path.join(stack_path,'stacked_waveforms_cluster_' + str(clust_num) + '.mseed'))




# for key, group_cat in groups.items(): #loop over the event cluster catalogues
#     print('Loading cluster ' + str(key) + '...')
#     channels = inv.get_contents()['channels']

#     streams = []

#     for event in tqdm(group_cat,total=group_cat.N):
#         event.attach_waveforms(inv,w_path,buffer=buffer,length=window_length,extra=10)
#         event.decimate(5)
#         event.filter('bandpass',freqmin=1,freqmax=30)  

#         event_stream = event.get_data_window()

#         if all([len(trace) == int(100*(buffer+window_length)+1) for trace in event_stream]):
#             streams.append(event.get_data_window())

    
#     stacked_stream = Stream()
#     for code in channels:
#         net,sta,loc,cha = code.split('.')

#         stacked = np.stack([stream.select(network=net,station=sta,channel=cha)[0].data for stream in streams],axis=1)
#         data = np.nanmean(stacked,axis=1)
#         tr = Trace(data=data,header={'sampling_rate':100,'network':net,'station':sta,'location':loc,'channel':cha})
#         stacked_stream += tr

#     stacked_stream = stacked_stream.split()
#     stacked_stream.write(os.path.join(stack_path,'stacked_waveforms_cluster_' + str(key) + '.mseed'))