import numpy as np
from obspy.core.inventory import inventory
from iqvis import data_objects as do
from obspy.core import UTCDateTime, read
import os
from obspy.core import Stream, Trace
from obspy.signal.trigger import aic_simple, pk_baer, ar_pick
from tqdm import tqdm
import pandas as pd

raw_stack_waveforms = False
low_stack_waveforms = False
high_stack_waveforms = False

stack_waveforms = False
pick_arrivals_ar = False
pick_P = False
pick_S = False
aic_picks = False
coalescence = False


t1 = UTCDateTime(2018,12,24)
t2 = UTCDateTime(2019,1,30)

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

#inv = inv.select(station='TI?A',channel='CH?')
inv = inv.select(station='TI??',channel='CH?')


from scipy import odr
import math

def fit_func(beta, x):
    # XXX: Eventually this is correct: return beta[0] * x + beta[1]
    return beta[0] * x

#firstly load in the stacked waveforms to do the picking on

temp_cat = do.EventCatalogue(t1,t2,os.path.join(c_path),templates=True)
groups = temp_cat.group_split()


if raw_stack_waveforms:
    for key, group_cat in groups.items(): #loop over the event cluster catalogues
        print('Loading cluster ' + str(key) + '...')
        channels = inv.get_contents()['channels']

        stacked_stream = Stream()

        for code in channels:
            net,sta,loc,cha = code.split('.')

            print('Stacking',code)

            traces = []

            for event in tqdm(group_cat,total=group_cat.N):
                event.attach_waveforms(inv.select(network=net,station=sta,channel=cha),w_path)
                traces.append(event.get_data_window()[0]) #only should be a single trace, add this to the list

            stacked = np.stack(traces,axis=1)
            tr = Trace(data=np.nanmean(stacked,axis=1),header={'sampling_rate':500,'network':net,'station':sta,'location':loc,'channel':cha})
            stacked_stream += tr

        stacked_stream = stacked_stream.split()
        stacked_stream.write(os.path.join(path,'stacked_waveforms','raw_stacked_waveforms_cluster_'+str(key) + '.mseed'))

if stack_waveforms:

    for key, group_cat in groups.items(): #loop over the event cluster catalogues
        print('Loading cluster ' + str(key) + '...')
        channels = inv.get_contents()['channels']

        streams = []

        for event in tqdm(group_cat,total=group_cat.N):
            event.attach_waveforms(inv,w_path,buffer=10,length=25)
            event.filter('bandpass',freqmin=1,freqmax=100)  

            streams.append(event.get_data_window())

        
        stacked_stream = Stream()
        for code in channels:
            net,sta,loc,cha = code.split('.')

            stacked = np.stack([stream.select(network=net,station=sta,channel=cha)[0].data for stream in streams],axis=1)
            data = np.nanmean(stacked,axis=1)
            tr = Trace(data=data,header={'sampling_rate':500,'network':net,'station':sta,'location':loc,'channel':cha})
            stacked_stream += tr

        stacked_stream = stacked_stream.split()
        stacked_stream.write(os.path.join(path,'stacked_waveforms','stacked_waveforms_cluster_'+str(key) + '.mseed'))

if high_stack_waveforms:
    for key, group_cat in groups.items(): #loop over the event cluster catalogues
        print('Loading cluster ' + str(key) + '...')
        channels = inv.get_contents()['channels']

        streams = []

        for event in tqdm(group_cat,total=group_cat.N):
            event.attach_waveforms(inv,w_path,buffer=10,length=25)
            event.filter('highpass',freq=1)
            streams.append(event.get_data_window())

        
        stacked_stream = Stream()
        for code in channels:
            net,sta,loc,cha = code.split('.')

            stacked = np.stack([stream.select(network=net,station=sta,channel=cha)[0].data for stream in streams],axis=1)
            data = np.nanmean(stacked,axis=1)
            tr = Trace(data=data,header={'sampling_rate':500,'network':net,'station':sta,'location':loc,'channel':cha})
            stacked_stream += tr

        stacked_stream = stacked_stream.split()
        stacked_stream.write(os.path.join(path,'stacked_waveforms','high_stacked_waveforms_cluster_'+str(key) + '.mseed'))


if low_stack_waveforms:

    for key, group_cat in groups.items(): #loop over the event cluster catalogues
        print('Loading cluster ' + str(key) + '...')
        channels = inv.get_contents()['channels']

        streams = []

        for event in tqdm(group_cat,total=group_cat.N):
            event.attach_waveforms(inv,w_path,buffer=10,length=25)
            event.filter('bandpass',freqmin=0.1,freqmax=100)  

            streams.append(event.get_data_window())

        
        stacked_stream = Stream()
        for code in channels:
            net,sta,loc,cha = code.split('.')

            stacked = np.stack([stream.select(network=net,station=sta,channel=cha)[0].data for stream in streams],axis=1)
            data = np.nanmean(stacked,axis=1)
            tr = Trace(data=data,header={'sampling_rate':500,'network':net,'station':sta,'location':loc,'channel':cha})
            stacked_stream += tr

        stacked_stream = stacked_stream.split()
        stacked_stream.write(os.path.join(path,'stacked_waveforms','low_stacked_waveforms_cluster_'+str(key) + '.mseed'))

    #     trace_dict[key] = {}


    #     for code in channels:
    #         trace_dict[key][code] = []


    #         for event in group_cat:
    #             net,sta,loc,cha = code.split('.')
    #             event.attach_waveforms(inv.select(network=net,station=sta,channel=cha),w_path,buffer=5,length=25)
    #             event.filter('bandpass',freqmin=1,freqmax=100)  
    #             tr = event.get_data_window().select(network=net,station=sta,channel=cha)[0]
    #             trace_dict[key][tr.id].append(tr.data)
            

    #     trace_stacks[key] = {}
    #     for code, traces in trace_dict[key].items():
    #         stacked = np.stack(traces,axis=1)
    #         trace_stacks[key][code] = stacked



    # for cluster_num, trace_stack in trace_stacks.items():
    #     stream = Stream()
    #     for code, stack in trace_stack.items():
    #         net, sta, loc, cha = code.split('.')

    #         if cha == 'CH1':
    #             cha = 'CHN'
    #         if cha == 'CH2':
    #             cha = 'CHE'

    #         data = np.nanmean(stack,axis=1)

    #         tr = Trace(data=data,header={'sampling_rate':500,'network':net,'station':sta,'location':loc,'channel':cha})
    #         stream += tr

    #     stream_dict[cluster_num] = stream

    
    # for cluster_num, stream in stream_dict.items():
    #     stream = stream.split()
    #     stream.write(os.path.join(path,'stacked_waveforms','stacked_waveforms_cluster_'+str(cluster_num) + '.mseed'))


if pick_arrivals_ar:
    for key in groups.keys():
        print('Cluster',key)
        stream = read(os.path.join(path,'stacked_waveforms','stacked_waveforms_cluster_'+str(key) + '.mseed'))

        rows = []
        stations = []
        for net in inv:
            for sta in net:
                stations.append(sta.code)

                picks = {}
                substream = stream.select(station=sta.code)

                Z = substream.select(component='Z')[0]
                N = substream.select(component='1')[0]
                E = substream.select(component='2')[0]

                #now do an initial pick based on the vertical component
                df = Z.stats.sampling_rate

                try:
                    p_pick, s_pick = ar_pick(Z.data, N.data, E.data, df,
                            1.0, 10.0, 1.0, 0.1, 4.0, 1.0, 2, 8, 0.1, 0.2)
                except:
                    p_pick = np.nan
                    s_pick = np.nan

                picks['P'] = p_pick
                picks['S'] = s_pick

                rows.append(picks)

        event_picks = pd.DataFrame(data=rows,index=stations)
        print(event_picks)
        event_picks.to_csv(os.path.join(path,'stacked_waveforms','AR_arrivals_'+str(key) + '.csv'))




if pick_P:
    from obspy.signal.trigger import aic_simple, recursive_sta_lta

    def pick_phase(data,df,sta=0.5,lta=5.0,pre_window=6,post_window=1):
        cft = recursive_sta_lta(data, int(sta * df), int(lta * df))
        p_idx = np.argmax(cft)
        snr = cft[p_idx]

        pre_window = int(pre_window*df)
        post_window = int(post_window*df)

        window = data[p_idx-pre_window:p_idx+post_window]
        aic_f = aic_simple(window)
        p_idx2 = p_idx + np.argmin(aic_f) - pre_window

        p_t1 = p_idx / df
        p_t2 = p_idx2 / df
        return p_t1, p_t2, snr
    

    from obspy.core import read

    for cluster_num in groups.keys():
        print('Cluster',cluster_num)
        stream = read(os.path.join(path,'stacked_waveforms','stacked_waveforms_cluster_'+str(cluster_num) + '.mseed'))

        rows = []
        stations = []
        for net in inv:
            for sta in net:
                stations.append(sta.code)

                picks = {}
                substream = stream.select(station=sta.code)

                substream = substream.filter('bandpass',freqmin=1,freqmax=10)
                Z = substream.select(component='Z')[0]

                df = Z.stats.sampling_rate

                t_p1, t_p2, snr = pick_phase(Z.data,df,sta=0.5,lta=7)

                picks['P'] = t_p2
                picks['snr'] = snr

                rows.append(picks)

        event_picks = pd.DataFrame(data=rows,index=stations)
        print(event_picks)

        event_picks.to_csv(os.path.join(path,'stacked_waveforms','P_arrvials_'+str(cluster_num) + '.csv'))


if pick_S:
    from obspy.signal.trigger import aic_simple, recursive_sta_lta
    from scipy import odr
    import math
    from obspy.core import read

    def pick_phase(data,df,sta=0.5,lta=5.0,pre_window=6,post_window=1,min_t=0):
        min_ind = int(min_t/df)
        cft = recursive_sta_lta(data, int(sta * df), int(lta * df))
        p_idx = np.argmax(cft[min_ind:])
        snr = cft[p_idx]

        pre_window = int(pre_window*df)
        post_window = int(post_window*df)

        window = data[p_idx-pre_window:p_idx+post_window]
        aic_f = aic_simple(window)
        p_idx2 = p_idx + np.argmin(aic_f) - pre_window

        p_t1 = p_idx / df
        p_t2 = p_idx2 / df
        return p_t1, p_t2, snr



    def fit_func(beta, x):
        # XXX: Eventually this is correct: return beta[0] * x + beta[1]
        return beta[0] * x


    def get_az(stream,p_t,pre_window=0.2,post_window=1.0):

        window = stream.slice(stream[0].stats.starttime + p_t - pre_window, stream[0].stats.starttime + p_t + post_window,keep_empty_traces=True)

        n = window.select(component='1')[0].data
        e = window.select(component='2')[0].data
        z = window.select(component='Z')[0].data

        #first work out the azimuth of the particle motion
        data = odr.Data(e, n)
        model = odr.Model(fit_func)
        odr_obj = odr.ODR(data, model, beta0=[1.])
        out = odr_obj.run()
        az_slope = out.beta[0]
        az = math.atan2(1.0, az_slope)

        return az
    
    def rotate_stream(stream,az):
        n = stream.select(component='1')[0].data
        e = stream.select(component='2')[0].data
        z = stream.select(component='Z')[0].data


        u = n * np.cos(az) + e * np.sin(az)
        v = e * np.cos(az) - n * np.sin(az)

        return z, u, v

    

    for cluster_num in groups.keys():
        print('Cluster',cluster_num)
        stream = read(os.path.join(path,'stacked_waveforms','stacked_waveforms_cluster_'+str(cluster_num) + '.mseed'))
        stream2 = read(os.path.join(path,'stacked_waveforms','stacked_waveforms_cluster_'+str(cluster_num) + '.mseed'))

        rows = []
        stations = []
        for net in inv:
            for sta in net:
                stations.append(sta.code)

                picks = {}
                substream = stream.select(station=sta.code)
                substream2 = stream2.select(station=sta.code)


                substream = substream.filter('bandpass',freqmin=1,freqmax=10)
                substream2 = substream2.filter('bandpass',freqmin=1,freqmax=5)

                Z = substream.select(component='Z')[0]

                df = Z.stats.sampling_rate

                t_p1, t_p2, snr = pick_phase(Z.data,df,sta=0.5,lta=7,pre_window=5,post_window=2)

                picks['P'] = t_p2
                picks['P_snr'] = snr


                az = get_az(substream,t_p2,pre_window=0.5,post_window=1.2)
                
                z, u, v = rotate_stream(substream2,az)
                t_s1, t_s2, snr = pick_phase(v,df,sta=2.0,lta=7,pre_window=5,post_window=2,min_t=t_p2)

                picks['S'] = t_s2
                picks['S_snr'] = snr

                rows.append(picks)

        event_picks = pd.DataFrame(data=rows,index=stations)
        print(event_picks)

        event_picks.to_csv(os.path.join(path,'stacked_waveforms','S_arrivals_'+str(cluster_num) + '.csv'))




if aic_picks:
    from obspy.signal.trigger import aic_simple, recursive_sta_lta

    from obspy.core import read

    for cluster_num in groups.keys():
        print('Cluster',cluster_num)
        stream = read(os.path.join(path,'stacked_waveforms','stacked_waveforms_cluster_'+str(cluster_num) + '.mseed'))
        stream = stream.slice(stream[0].stats.starttime,stream[0].stats.starttime + 25)
        stream.filter('bandpass',freqmin=1,freqmax=10)

        rows = []
        stations = []
        for net in inv:
            for sta in net:
                stations.append(sta.code)

                picks = {}
                substream = stream.select(station=sta.code)

                Z = substream.select(component='Z')[0]
                N = substream.select(component='1')[0]
                E = substream.select(component='2')[0]

                df = Z.stats.sampling_rate
                t = Z.times()

                if np.isnan(Z.data).any():
                    picks['P'] = np.nan
                    picks['AIC'] = np.nan
                    picks['error'] = np.nan
                else:
                    aic = aic_simple(Z.data)
                    picks['P'] = aic.argmin() / df
                    picks['AIC'] = aic.min()

                    aic_range = np.max(aic) - np.min(aic)
                    thresh = np.min(aic) + aic_range / 4

                    aic_bool = aic <= thresh
                    unc = np.sum(aic_bool) / df
                    picks['error'] = unc

                rows.append(picks)

        event_picks = pd.DataFrame(data=rows,index=stations)
        print(event_picks)

        event_picks.to_csv(os.path.join(path,'stacked_waveforms','aic_arrivals_'+str(cluster_num) + '.csv'))



if coalescence:
    #use backmigration to estimate event locations for stacked waveforms...

    from obspy.geodetics import degrees2kilometers, kilometers2degrees


    def Onset_Function(z,r,t,nsta,nlta,mode='radial_transverse'):

        if mode == 'radial_transverse':
            sta = np.cumsum(r**2)
            lta = np.cumsum(t**2)
        elif mode == 'radial':
            sta = np.cumsum(r**2)
            lta = np.cumsum(z**2 + r**2 + t**2)   
        elif mode == 'absolute':
            sta = np.cumsum(z**2 + r**2 + t**2)
            lta = np.cumsum(z**2 + r**2 + t**2)
        elif mode == 'transverse':
            sta = np.cumsum(t**2)
            lta = np.cumsum(z**2 + r**2 + t**2)
        elif mode == 'transverse_radial':
            sta = np.cumsum(t**2)
            lta = np.cumsum(r**2)
        elif mode == 'transverse_self':
            sta = np.cumsum(t**2)
            lta = np.cumsum(t**2)
        else:
            sta = np.cumsum(z**2)
            lta = np.cumsum(z**2)


        ratio = np.zeros_like(sta)

        sta[:-nsta] = sta[nsta:] - sta[:-nsta]
        sta /= nsta
        sta[-nsta:] = 0


        lta[nlta:] = lta[nlta:] - lta[:-nlta]
        lta /= nlta
        lta[:nlta - 1] = 0

        ratio[nlta:] = sta[nlta:] / lta[nlta:]

        ratio = np.nan_to_num(ratio)
        return ratio
    
    def get_sta_xy(inv):
        N = len(inv)
        locations_ll = np.zeros((N,3))

        i = 0
        for net in inv:
            for sta in net:
                locations_ll[i,:] = np.array([sta.latitude,sta.longitude,-sta.elevation/1000])
                i += 1

        centre = np.mean(locations_ll,axis=0)

        d_ll = locations_ll - centre[None,:]

        sta_xy = np.zeros_like(d_ll)
        sta_xy[:,0] = degrees2kilometers(d_ll[:,1],radius=6371.0*np.cos(np.deg2rad(centre[0])))
        sta_xy[:,1] = degrees2kilometers(d_ll[:,0])
        sta_xy[:,2] = locations_ll[:,2]

        return sta_xy, centre

    def cart_to_deg(x,y,centre):
        lon = centre[1] + kilometers2degrees(x,radius=6371.0*np.cos(np.deg2rad(centre[0])))
        lat = centre[0] + kilometers2degrees(y)
        return lon, lat


    stacked_streams = {}
    for cluster_num in groups.keys():
        stacked_streams[cluster_num] = read(os.path.join(path,'stacked_waveforms','stacked_waveforms_cluster_'+str(cluster_num) + '.mseed'))

    df = 500

    inv_files = os.listdir(s_path)

    inv = inventory.Inventory()

    for file in inv_files:
        temp_path = os.path.join(s_path,file)
        inv += inventory.read_inventory(temp_path,level='response',format='STATIONXML')

    inv = inv.select(station='TI?A')

    sta_xy, centre = get_sta_xy(inv)

    from obspy.core import Trace

    east_grid = np.linspace(-15,15,100)
    north_grid = np.linspace(-15,15,100)
    depth_grid = np.linspace(0,4,30)

    cluster_names = []
    rows = []

    for cluster_num, stream in stacked_streams.items():
        stream = stream.select(station='TI?A') #! may want to include additional stations, although this biases spatial recordings.
        stream.filter('bandpass',freqmin=1,freqmax=5)
        
        row = {}
        cluster_names.append(cluster_num)

        P_obj = np.zeros((east_grid.size,north_grid.size,depth_grid.size))
        S_obj = np.zeros((east_grid.size,north_grid.size,depth_grid.size))
        PS_obj = np.zeros((east_grid.size,north_grid.size,depth_grid.size))
        PS_t = np.zeros((east_grid.size,north_grid.size,depth_grid.size))

        Z_lst = []
        N_lst = []
        E_lst = []

        for net in inv:
            for sta in net:
                Z_lst.append(stream.select(station=sta.code,component='Z')[0].data)
                N_lst.append(stream.select(station=sta.code,component='1')[0].data)
                E_lst.append(stream.select(station=sta.code,component='2')[0].data)

        Z = np.stack(Z_lst,axis=1)
        N = np.stack(N_lst,axis=1)
        E = np.stack(E_lst,axis=1)

        for i, e in enumerate(east_grid):
            for j, n in enumerate(north_grid):
                #compute the backazimuth and rotate the streams...
                diff_e = e - sta_xy[:,0]
                diff_n = n - sta_xy[:,1]

                baz = np.arctan2(diff_e,diff_n) #uses x/y rather than y/x due to way backaz is defined from North

                R = N * np.cos(baz[None,:]) + E * np.sin(baz[None,:])
                T = E * np.cos(baz[None,:]) - N * np.sin(baz[None,:])

                #now get the characteristic functions for the full traces...                
                P_cfts = []
                S_cfts = []
                for ii in range(len(inv)):
                    P_cfts.append(Onset_Function(Z[:,ii],R[:,ii],T[:,ii],int(1*df),int(5*df),mode='radial'))
                    S_cfts.append(Onset_Function(Z[:,ii],R[:,ii],T[:,ii],int(1*df),int(5*df),mode='transverse'))

                P_cft = np.stack(P_cfts,axis=1)
                S_cft = np.stack(S_cfts,axis=1)


                for k, z in enumerate(depth_grid):
                    diff_z = z - sta_xy[:,2]

                    r = np.sqrt(diff_e**2 + diff_n**2 + diff_z**2)
                    P_tt = r / 3.87
                    S_tt = r / 1.90
                    P_tt = P_tt
                    S_tt = S_tt
                    P_tt_samp = (df*P_tt).astype(int) #number of samples for travel time
                    S_tt_samp = (df*S_tt).astype(int)  

                    shifted_P_cfts = []
                    shifted_S_cfts = []
                    for ii in range(len(inv)):
                        shifted_P_cfts.append(np.pad(P_cft[P_tt_samp[ii]:,ii],(0,P_tt_samp[ii])))
                        shifted_S_cfts.append(np.pad(S_cft[S_tt_samp[ii]:,ii],(0,S_tt_samp[ii])))

                    shift_P_cfts = np.stack(shifted_P_cfts,axis=1)
                    shift_S_cfts = np.stack(shifted_S_cfts,axis=1)

                    stacked_P_cfts = np.nansum(shift_P_cfts,axis=1)
                    stacked_S_cfts = np.nansum(shift_S_cfts,axis=1)

                    P_obj[i,j,k] = np.nanmax(stacked_P_cfts)
                    S_obj[i,j,k] = np.nanmax(stacked_S_cfts)
                    PS_obj[i,j,k] = np.nanmax(stacked_P_cfts + stacked_S_cfts)
                    PS_t[i,j,k] = (np.argmax(stacked_P_cfts + stacked_S_cfts)) / 500

        

        """Now construct a dataframe to save the results."""
        #row['Label'] = temp_labels[str(cluster_num)] #! will need to do this later when plotting.

        fit_ind = np.unravel_index(np.argmax(P_obj),P_obj.shape)
        x_fit, y_fit, z_fit = east_grid[fit_ind[0]], north_grid[fit_ind[1]], depth_grid[fit_ind[2]]
        t_fit = PS_t[fit_ind[0],fit_ind[1],fit_ind[2]]

        row['P_X'] = x_fit
        row['P_Y'] = y_fit
        row['P_Z'] = z_fit

        lon, lat = cart_to_deg(x_fit,y_fit,centre)
        row['P_lon'] = lon
        row['P_lat'] = lat

        row['P_coal'] = np.max(P_obj)

        fit_ind = np.unravel_index(np.argmax(PS_obj),PS_obj.shape)
        x_fit, y_fit, z_fit = east_grid[fit_ind[0]], north_grid[fit_ind[1]], depth_grid[fit_ind[2]]

        row['PS_X'] = x_fit
        row['PS_Y'] = y_fit
        row['PS_Z'] = z_fit
        row['PS_t'] = t_fit

        lon, lat = cart_to_deg(x_fit,y_fit,centre)
        row['PS_lon'] = lon
        row['PS_lat'] = lat

        row['PS_coal'] = np.max(PS_obj)

        rows.append(row)


    locations = pd.DataFrame(data=rows,index=cluster_names)
    print(locations)
    locations.to_csv(os.path.join(path,'stacked_waveforms','abs_migration_locations.csv'))