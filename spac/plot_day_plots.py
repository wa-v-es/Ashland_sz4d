###
import glob
import obspy
from obspy.clients.fdsn import Client
from obspy.geodetics import locations2degrees, gps2dist_azimuth
from obspy import read_inventory,read,Stream
from obspy.signal.cross_correlation import correlate
from obspy import UTCDateTime

import matplotlib.pyplot as plt
import numpy as np
# jv is for Bessel functions.  spwin for windowing.
from scipy.special import jv
import scipy.signal.windows as spwin
# pyproj helps get accurate UTM-LatLon conversions not in obspy.geodetics
import pyproj
import csv
from math import sin, cos, sqrt, atan2, radians
import sys
from obspy.clients.fdsn import Client
###


client = Client("IRISPH5")
inventory = client.get_stations(network="2E", station="*",
                                starttime=UTCDateTime('2024-07-25T02:00:00.000000'),
                                endtime=UTCDateTime('2024-07-29T02:00:00.000000'),level="response")
# Location of the reference station at array center, as lat/lon
ctrlat = 42.07559574
ctrlon = -122.7472788
t_st=UTCDateTime("2024-07-27T00:00:00")
t_end=UTCDateTime("2024-07-27T16:00:00")

stream_fd = client.get_waveforms("2E", "*", "*", "*Z", t_st, t_end)
stream_fdsn=stream_fd.copy()
stream_clean=Stream()
for tr in stream_fdsn:
    time=tr.stats.endtime
    if UTCDateTime(time.timestamp + 0.5).replace(microsecond=0) == t_end:
        stream_clean.append(tr)
###

stream_clean.remove_response(inventory=inventory, pre_filt=[0.25, 0.5, 45, 50], water_level=80, plot=False)
stream_clean.detrend(type='linear')
stream_clean.taper(.05)
stream_clean.filter('bandpass',freqmin=0.5, freqmax=30, corners=4, zerophase=True)
stream_clean.decimate(5)

for tr in stream_clean:
    tr.plot(type='dayplot',outfile='st_{}.png'.format(tr.stats.station))
