"""
SCRIPT FOR TAKING THE STA/LTA NETWORK CATALOGUE AND FINDING THE REPEATING EVENTS USING TEMPLTATE MATCHING.
ALSO COMPUTES THE STACKED WAVEFORMS FOR PICKING ARRIVALS AND ESTIMATING LOCATION AND MAGNITUDE.
"""

import os
from cryoquake import stream_handling as sh
from cryoquake import data_objects as do
from cryoquake import spatial_analysis as sa
from obspy.core.inventory import inventory
from obspy.core import Stream, Trace, read, UTCDateTime
import pandas as pd
from scipy.signal import correlate
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial import distance
from pathlib import Path
from tqdm import tqdm


calculate_xcorr = False
clustering = False
template_matching = False
stack_waveforms = True
mod_waveforms = True


t1 = sh.UTCDateTime(2018,12,24)
t2 = sh.UTCDateTime(2019,1,30)

chunk = do.SeismicChunk(t1,t2)

root = Path(__file__).parent.parent
w_path = root / "waveforms"
stack_path = root / "stacked_waveforms"
s_path = root / "stations"
c_path = root / "catalogues" / "network"


inv_files = os.listdir(s_path)

inv = inventory.Inventory()

for file in inv_files:
    temp_path = os.path.join(s_path,file)
    inv += inventory.read_inventory(temp_path,level='response',format='STATIONXML')

inv = inv.select(station='TI?A',channel='CH?')

network_cat = do.EventCatalogue(t1,t2,c_path)


if calculate_xcorr:
    print('Attaching and processing waveforms...')
    all_traces = []

    for event in network_cat:
        event.attach_waveforms(inv.select(station='TI?A',channel='CHZ'),w_path,buffer=5)      
        event.filter('bandpass',freqmin=1,freqmax=100)
        all_traces.append([tr.data for tr in event.get_data_window()])


    print('Computing cross-correlation matrix...')
    N = len(all_traces)
    num_sta = len(inv.select(station='TI?A',channel='CHZ'))
    cc_mat = np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            if i > j:
                pass
            elif i == j:
                cc_mat[i,j] = 1.0
            else:
                cc = np.stack([correlate(all_traces[i][k],all_traces[j][k]) / ((np.sum(all_traces[i][k]**2) * np.sum(all_traces[j][k]**2))**(0.5)) for k in range(num_sta)],axis=-1) #[time shift,stations]
                sort_cc = np.nan_to_num(np.sort(np.nan_to_num(cc,nan=-np.inf),axis=1),neginf=np.nan)    #sort and put nan at start so not included unless less than three stations    
                ave_cc = np.max(np.nanmean(sort_cc[:,-3:],axis=1)) 
                cc_mat[i,j] = ave_cc
                cc_mat[j,i] = ave_cc

    

    cc_mat[cc_mat > 1] = 1.0

    dissimilarity = 1-cc_mat
    dissimilarity = distance.squareform(dissimilarity) #flattened version - same values as dissimilary so 1 - cc_mat

    np.savez(os.path.join(c_path,'cc_matrix_N.npz'),cc_mat=cc_mat,diss=dissimilarity)


if clustering:
    print('Hierarchial clustering...')
    threshold = 0.2 #make this 0.2 to match the correlation detector

    linkage = hierarchy.linkage(dissimilarity, method="average",optimal_ordering=True)
    clusters = hierarchy.fcluster(linkage, threshold, criterion="distance")

    np.savez(os.path.join(c_path,'clusters_N.npz'),clusters=clusters)


if template_matching:
    print('Template matching...')
    clusters = np.load(os.path.join(c_path,'clusters_N.npz'))['clusters']
    cc_mat = np.load(os.path.join(c_path,'cc_matrix_N.npz'))['cc_mat']


    unique = pd.value_counts(clusters)
    clust_ind = unique[unique>3] #if 3 or more events in cluster, get template 

    templates = {}

    for i, clust_N in enumerate(clust_ind):
        clust_id = clust_ind.index[i]

        cc_ind = np.where(clusters==clust_id)[0]
        temp = cc_mat[cc_ind,:]
        clust_cc = temp[:,cc_ind]

        clust_events = network_cat.events.index[cc_ind]

        central_ind = np.argmax(np.mean(clust_cc,axis=1))
        central_id = clust_events[central_ind]

        templates[str(clust_id)] = central_id

    #drop these groups as appear to be coda or non stick-slip
    templates.pop('116')
    templates.pop('117')

    full_templates = {}
    full_thresholds = {}
    for temp_name, temp_id in templates.items():
        template = network_cat.select_event(temp_id)
        template.attach_waveforms(inv.select(station='TI?A',channel='CHZ'),w_path,buffer=5)
        template.filter('bandpass',freqmin=1,freqmax=10)   

        full_templates[temp_name] = template
        full_thresholds[temp_name] = 0.5

    for daychunk in chunk:
        daychunk.attach_waveforms(inv.select(station='TI?A',channel='CHZ'),w_path,buffer=60*60)
        daychunk.filter('bandpass',freqmin=1,freqmax=10)
        daychunk.context('detect')

        daychunk.template_catalogue(c_path,full_templates,full_thresholds,method='n_mean',num=3)

        
if stack_waveforms:
    temp_cat = do.EventCatalogue(t1,t2,os.path.join(c_path),templates=True)
    groups = temp_cat.group_split()

    for key, group_cat in groups.items(): #loop over the event cluster catalogues
        print('Loading cluster ' + str(key) + '...')
        channels = inv.get_contents()['channels']

        streams = []

        for event in tqdm(group_cat,total=group_cat.N):
            event.attach_waveforms(inv,w_path,buffer=10,length=25)
            event.filter('bandpass',freqmin=1,freqmax=10)  

            streams.append(event.get_data_window())

        
        stacked_stream = Stream()
        for code in channels:
            net,sta,loc,cha = code.split('.')

            stacked = np.stack([stream.select(network=net,station=sta,channel=cha)[0].data for stream in streams],axis=1)
            data = np.nanmean(stacked,axis=1)
            tr = Trace(data=data,header={'sampling_rate':500,'network':net,'station':sta,'location':loc,'channel':cha})
            stacked_stream += tr

        stacked_stream = stacked_stream.split()
        stacked_stream.write(os.path.join(stack_path,'stacked_waveforms_cluster_'+str(key) + '.mseed'))


if mod_waveforms:

    df = 500
    pol_win = 0.2 #s
    freq_smooth = 20
    g = 8
    overlap = 0.95

    for cluster_num in groups.keys():
        stream = read(os.path.join(stack_path,'stacked_waveforms_cluster_' + str(cluster_num) + '.mseed'))
        stream = stream.select(station='TI?A')
        stream = stream.filter("bandpass",freqmin=1,freqmax=10)
        for tr in stream:
            tr.stats.starttime = UTCDateTime(2019,1,10)
            tr.remove_response(inv,output='DISP')

        for network in inv:
            for station in network:
                substream = stream.select(network=network.code,station=station.code)

                Z = substream.select(component='Z')[0]
                N = substream.select(component='1')[0]
                E = substream.select(component='2')[0]

                Zm, Nm, Em = sa.RollingStreamModulation(Z.data,N.data,E.data,int(pol_win*df),g=g,smooth=freq_smooth,overlap=overlap)

                Z.data = Zm
                N.data = Nm
                E.data = Em
        
        stream.write(os.path.join(stack_path,'mod_stacked_waveforms_cluster_' + str(cluster_num) + '.mseed'))