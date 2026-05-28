from cryoquake import stream_handling as sh
from pathlib import Path
import os


t1 = sh.UTCDateTime('2018-01-01T00:00:00.0Z')
t2 = sh.UTCDateTime('2019-12-31T00:00:00.0Z')

root = Path(__file__).parent.parent
w_path = root / "waveforms"
s_path = root / "stations"

#create folder for waveforms and station files if they do not exist
for path in [w_path,s_path]:
    if not os.path.exists(path):
        os.mkdir(path)

#use mass downloader to download waveforms for full deployment
stream = sh.get_waveforms('1J','TI??','','???',t1,t2,waveform_name=w_path,station_name=s_path,download=True)