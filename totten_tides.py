import pyTMD
import numpy as np

epoch = (2018,12,24,0,0,0) #start time
length = 80 * 24 * 60 * 60
model_times = np.mgrid[0:length:600] #10 minute intervals for the season 

#lat = -66.34825
#lon = 114.88496

lat = -67.25
lon = 114.5

tide_dir = '/Users/jmagyar/Documents/tides'

tides = pyTMD.compute_tide_corrections(np.array([lon]),np.array([lat]),model_times,EPOCH=epoch,DIRECTORY=tide_dir,MODEL='CATS2008',TIME='UTC',TYPE='time series',METHOD='spline',EPSG='4326').flatten()

np.savez('totten_tides.npz',dt=model_times,tide=tides)