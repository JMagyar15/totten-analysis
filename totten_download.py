import os
#os.chdir('..') #change cwd so local functions can be imported

from iqvis import stream_handling as sh

#we firstly want to download a couple of days of data from the three stations of interest.
t1 = sh.UTCDateTime('2018-01-01T00:00:00.0Z')
t2 = sh.UTCDateTime('2019-12-31T00:00:00.0Z')


path = '/Users/jaredmagyar/Documents/TottenData'
w_path = os.path.join(path,'waveforms')
s_path = os.path.join(path,'stations')

stream = sh.get_waveforms('1J','TI??','','???',t1,t2,waveform_name=w_path,station_name=s_path,download=True)