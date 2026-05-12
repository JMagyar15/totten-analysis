
import os
#os.chdir('..') #change cwd so local functions can be imported

from cryoquake import stream_handling as sh
from cryoquake import data_objects as do
from obspy.core.inventory import inventory
import matplotlib.pyplot as plt
from cryoquake import dayplot_backend as db
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from scipy.signal import correlate
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial import distance


t1 = sh.UTCDateTime(2018,12,24)
t2 = sh.UTCDateTime(2019,1,30)

chunk = do.SeismicChunk(t1,t2)

path = '/Users/jmagyar/Documents/TottenData'
w_path = os.path.join(path,'waveforms')
s_path = os.path.join(path,'stations')
c_path = os.path.join(path,'catalogues','network')
p_path = os.path.join(path,'event_plots')
spec_path = os.path.join(path,'spectrograms')


inv_files = os.listdir(s_path)

inv = inventory.Inventory()

for file in inv_files:
    temp_path = os.path.join(s_path,file)
    inv += inventory.read_inventory(temp_path,level='response',format='STATIONXML')

network_cat = do.EventCatalogue(t1,t2,c_path)

calculate_xcorr = False
clustering = False
template_matching = True


if calculate_xcorr:
    print('Attaching and processing waveforms...')
    all_traces = []

    for event in network_cat:
        event.attach_waveforms(inv.select(station='TI?A',channel='CHZ'),w_path,buffer=5) #hour long buffer for filtering and STA/LTA       
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

    #! drop these groups as appear to be coda or non stick-slip
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

        daychunk.template_catalogue(os.path.join(c_path,'filtered'),full_templates,full_thresholds,method='n_mean',num=3)

        
