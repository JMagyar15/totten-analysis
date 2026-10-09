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

clust_threshold = 0.5
match_threshold = 0.5
min_cluster_size = 20

buffer = 5
window_length = 10

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
network_cat = do.EventCatalogue(t1,t2,c_path)

filename = c_path / (str('xcorr_matrix_') + str(t1.date) + '_' + str(t2.date) + '.npz')

xcorr_file = np.load(filename, allow_pickle=True)
xcorr = xcorr_file['cc_mat']
arg_mat = xcorr_file['arg_mat']


"""
Select templates by clustering the cross-correlation matrix and selecting the most central event in each cluster.
"""

dissimilarity = 1.0-xcorr
dissimilarity = distance.squareform(dissimilarity) #flattened version - same values as dissimilary so 1 - cc_mat


linkage = fc.linkage(dissimilarity, method="average")
clusters = hierarchy.fcluster(linkage, clust_threshold, criterion="distance")

unique, counts = np.unique(clusters,return_counts=True)
clust_ind = unique[counts >= min_cluster_size]

templates = {}
template_loc = {}
template_sta = {}

for i, clust_N in enumerate(clust_ind):

    cc_ind = np.where(clusters==clust_N)[0]
    temp = xcorr[cc_ind,:]
    clust_cc = temp[:,cc_ind]

    temp = arg_mat[cc_ind,:]
    clust_arg = temp[:,cc_ind]
    np.fill_diagonal(clust_arg,-1)

    values, counts = np.unique(clust_arg.flatten(),return_counts=True)
    sta = values[np.argmax(counts)] #the station which the template is best represented by

    clust_events = network_cat.events.index[cc_ind]

    #clust_cc[clust_arg!=sta] = np.nan

    central_ind = np.argmax(np.nanmean(clust_cc,axis=1))
    central_id = clust_events[central_ind]


    templates[str(clust_N)] = central_id
    template_loc[str(clust_N)] = network_cat.events.index.get_loc(central_id)
    template_sta[str(clust_N)] = values[np.argmax(counts)]


"""
Use the selected templates to produce a full event catalogue for the season of repeating events.
"""

#going to want to just in here and drop the templates not associated with stick-slip


full_templates = {}
full_thresholds = {}

for temp_name, temp_id in templates.items():
    template = network_cat.select_event(temp_id)
    template.attach_waveforms(inv.select(station='TI?A',channel='CHZ'),w_path,buffer=buffer,length=window_length,extra=10)
    template.decimate(5)
    template.filter('bandpass',freqmin=3,freqmax=30)   
    template.stream = template.stream.split()

    full_templates[temp_name] = template
    full_thresholds[temp_name] = match_threshold

for daychunk in chunk:
    daychunk.attach_waveforms(inv.select(station='TI?A',channel='CHZ'),w_path,buffer=60*60)
    daychunk.decimate(5)
    daychunk.filter('bandpass',freqmin=3,freqmax=30)
    daychunk.context('detect')

    daychunk.template_catalogue(c_path,full_templates,full_thresholds,method='n_mean',num=1)