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


root = Path(__file__).parent.parent
w_path = root / "waveforms"
stack_path = root / "stacked_waveforms"
s_path = root / "stations"
c_path = root / "catalogues" / "all_stations"

t1 = sh.UTCDateTime(2018,12,24)
t2 = sh.UTCDateTime(2019,1,30)

inv_files = os.listdir(s_path)

inv = inventory.Inventory()

for file in inv_files:
    temp_path = os.path.join(s_path,file)
    inv += inventory.read_inventory(temp_path,level='response',format='STATIONXML')

inv = inv.select(station='TI?A',channel='CH?')

network_cat = do.EventCatalogue(t1,t2,c_path)

chunk_i = do.SeismicChunk(t1,t2)


buffer = 5 #in case trigger was early or late, add buffer to contain the beginning of the event
window_length = 10
max_lag = 200 #maximum lag of 2 seconds either side
outer_cc_mat = ()


for i, daychunk in enumerate(chunk_i):
    #want to attach and filter the waveforms for this day

    daychunk.attach_waveforms(inv.select(station='TI?A',channel='CHZ'),w_path,buffer=60*60)
    daychunk.decimate(5)
    daychunk.filter('bandpass',freqmin=1,freqmax=30) #relatively low high corner to help with correlation

    day_cat = do.EventCatalogue(daychunk.starttime,daychunk.endtime,c_path)

    all_traces = []

    for event in day_cat:
        event_stream = daychunk.stream.slice(event.starttime-buffer,event.starttime+10+buffer) #add buffer to help with start/end times
        all_traces.append(np.stack([tr.data for tr in event_stream],axis=0)) #stacked array of [stations,time] for each event

    X = np.stack(all_traces,axis=1) #stacked array of [stations,events,time]
    X = X - X.mean(axis=2, keepdims=True)
    X /= np.linalg.norm(X, axis=2, keepdims=True)

    if isinstance(X, np.ma.MaskedArray):
        X = X.filled(np.nan)  # Fill masked values with np.nan

    chunk_j = do.SeismicChunk(t1,t2) #reset the chunk so loop initises correctly

    inner_cc_mat = ()

    for j, daychunk2 in enumerate(chunk_j):
        if i > j:
            print('Skipping ' + str(daychunk2.starttime.date) + ' since it has already been processed...')
            inner_cc_mat += (outer_cc_mat[j][i].T,) #append the already calculated cc_mat for this day pair
            continue
        elif i == j:
            print('Processing ' + str(daychunk2.starttime.date) + ' against itself...')
            #inner loop for same day, so just use the X matrix already created
            Y = X
        else:
            print('Processing ' + str(daychunk2.starttime.date) + ' against ' + str(daychunk.starttime.date) + '...')
            #want to attach and filter the waveforms for this day

            daychunk2.attach_waveforms(inv.select(station='TI?A',channel='CHZ'),w_path,buffer=60*60)
            daychunk2.decimate(5)
            daychunk2.filter('bandpass',freqmin=1,freqmax=30) #relatively low high corner to help with correlation

            day_cat2 = do.EventCatalogue(daychunk2.starttime,daychunk2.endtime,c_path)

            all_traces = []

            for event in day_cat2:
                event_stream = daychunk2.stream.slice(event.starttime-buffer,event.starttime+10+buffer) #add buffer to help with start/end times
                all_traces.append(np.stack([tr.data for tr in event_stream],axis=0)) #stacked array of [stations,time] for each event

            Y = np.stack(all_traces,axis=1) #stacked array of [stations,events,time]
            Y = Y - Y.mean(axis=2, keepdims=True)
            Y /= np.linalg.norm(Y, axis=2, keepdims=True)

            if isinstance(Y, np.ma.MaskedArray):
                Y = Y.filled(np.nan)  # Fill masked values with np.nan

        max_lag = 200

        N_stations, N_events_i, N_samples = X.shape
        N_stations, N_events_j, N_samples = Y.shape

        similarity = np.full((N_stations, N_events_i, N_events_j), 0.0)


        for lag in range(-max_lag, max_lag + 1):

            if lag > 0:
                C = X[:,:, :-lag] @ np.matrix_transpose(Y[:,:, lag:])

            elif lag < 0:
                C = X[:,:, -lag:] @ np.matrix_transpose(Y[:,:, :lag])

            else:
                C = X @ np.matrix_transpose(Y)

            C = np.nan_to_num(C,nan=0.0)

            similarity = np.maximum(similarity, C)

        cc_mat = np.max(similarity, axis=0) #take max over stations to get a single similarity matrix for all events
        inner_cc_mat += (cc_mat,)

    outer_cc_mat += (inner_cc_mat,)

mat_list = [[mat for mat in group] for group in outer_cc_mat]
xcorr = np.block(mat_list)

#tidy the xcorr matrix so it meets requirements exactly
xcorr[xcorr > 1.0] = 1.0
np.fill_diagonal(xcorr, 1.0)


#now save the xcorr matrix to a file for later use
filename = c_path / 'cc_matrix_N.npz'

np.savez(filename,cc_mat=xcorr)