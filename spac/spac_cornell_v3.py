#Geoff's script
#SPAC tests on Cornell 2024 seismology Day2 array: Clean-ish version

# OBSPY start
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
from cmcrameri import cm
#######
#  Function definitions for station spacing, etc. used later
# Utility for getting distance between a pair of stations.
#  sta_sn:      names of stations
#  sta_offset:  list of cartesian station locations
#  n1, n2:      index of the two stations being analyzed
#  RETURNS cartesian distance in m
def getstadistxy(sta_sn, sta_offset, n1, n2):
    x1, y1 = sta_offset[n1,:]
    x2, y2 = sta_offset[n2,:]
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

#  Pre-whitening utility, sometimes useful pre-processing step for ambient noise
#  rawdat:  np.array of time series input
#  wlev - hardwired water level (change here)
# RETURNS np.array of pre-whitened time series
def prewhiten(rawdat):
    wlev = 1E-6
    datfft = np.fft.fft(rawdat)
    eps1 = np.max(np.abs(datfft))*wlev
    datfwhit = datfft / np.abs(datfft + eps1) #
    #Divides each Fourier component by its magnitude plus eps1, effectively flattening the frequency spectrum.
    #By dividing by the magnitude, it reduces the influence of dominant frequencies, thereby whitening the spectrum.
    datwhit = np.real(np.fft.ifft(datfwhit))
    return datwhit

# Calculate power-spec of a Trace() object, return as np.array
def pwrspec(tra):
    tra.detrend(type='linear')
    tra.taper(.05)
    fftdat = np.fft.fft(tra[0].data)
    return fftdat*np.conj(fftdat)

# H/V spectral ratio. Inputs are obspy.Trace. Output is np.array.
def calc_hvsr(trz, trn, tre):
    trz1 = pwrspec(trz)
    trn1 = pwrspec(trn)
    tre1 = pwrspec(tre)
    pwrrat = (trn1 + tre1)/trz1
    return np.sqrt(pwrrat)

def calculate_distance(lat1, lon1, lat2, lon2):
    # Radius of the Earth in meters
    R = 6371000.0  # meters
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Differences in coordinates
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine formula
    a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2))**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    # Distance in meters
    distance = R * c
    return round(distance,3)

#######
datapath = '/Users/keyser/Research/sz4d_ft/SZNET_AllNodeData/enclave_1_overnight/'
# file to save preprocessed data
preprocfile = "preprocess_.mseed"

strttime=UTCDateTime('2024-07-27T02:00:00.000000')
endtime=UTCDateTime('2024-07-27T02:10:00.000000')

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
# Array Geometry Recalculations:  BUILD THE ARRAY HERE

sta_lat=[]
sta_long=[]
sta_ele=[]
sta_name=[]
sta_node=[]
for line in open('Enclave_1_station_data.txt','r'):
# for line in open('/Users/keyser/Research/sz4d_ft/nodal_data/Enclave_1_station_data.txt','r'):
    line=line.split()
    if line[0][0] == '#':
        continue
    sta_lat.append(line[2])
    sta_long.append(line[1])
    sta_ele.append(line[3])
    sta_name.append(line[0])
    sta_node.append(line[6])

sta_lat=[float(lat) for lat in sta_lat]
sta_long=[float(long) for long in sta_long]
sta_ele=[float(ele) for ele in sta_ele]

print('total #station: ',len(sta_ele))

### read sac files..
# stream=read('../SZNET_AllNodeData/enclave_1_overnight/*2024.07.27*')

# print(stream)##
## adding lat long info in the traces.
for i in range(len(sta_ele)):
    for tr in stream_clean:
        # if int(sta_node[i])==int(tr.stats.station): # used if saved traces are used..
        if int(sta_name[i])==int(tr.stats.station):
            tr.stats.latitude=sta_lat[i]
            tr.stats.longitude=sta_long[i]
            tr.stats.elevation=sta_ele[i]
            tr.stats.local_name=sta_name[i]
##
#preprocessing
stream_clean.remove_response(inventory=inventory, pre_filt=[0.25, 0.5, 45, 50], water_level=80, plot=False)
stream_clean.detrend(type='linear')
stream_clean.taper(.05)
stream_clean.filter('bandpass',freqmin=0.5, freqmax=30, corners=4, zerophase=True)
stream_clean.decimate(5)
##
st_filt=stream_clean.copy()
##
##### Build the array rings #########

#   winlen:  length of window to subset (s)
#   max_seconds:  max correlation shift to test (s)
winlen = 30.
max_seconds = 10.
###
pair_dist=[]
for i in range(len(sta_ele)):
    for j in range(i+1,len(sta_ele)):
        dist=calculate_distance(sta_lat[i],sta_long[i],sta_lat[j],sta_long[j])
        # print(f'Distance {dist} for {sta_name[i]} and {sta_name[j]}\n')
        pair_dist.append((dist,sta_name[i],sta_name[j]))
        # print(j)
pair_dist=np.array(pair_dist)
pair_dist=pair_dist.astype(float) #converts all elements to floats
pair_dist=pair_dist[pair_dist[:, 0].argsort()]
####
##### Build the array rings #########
#  ORDER in the sta_sn etc lists are NOT same as order of tr in st_all
#    pairlist corresponds to stations in sta_sn
###
pairs=[]
pairs_all=[]
dist= pair_dist[0][0] # first distance to get the loop started
for i in range(len(pair_dist)):
    if pair_dist[i][0] - dist < 1: # if consecutive elements are less than 1 m
        pairs.append(pair_dist[i])
        dist=pair_dist[i][0]
    else:
        # print('breaks at', pair_dist[i])
        if len(pairs) > 1:
            pairs_all.append(pairs)
        pairs=[]
        dist=pair_dist[i][0]
        pairs.append(pair_dist[i])
        # i=i-1
#########
#following calculates the mean and std in each pair in pair_all
dist_avg=[]
dist_std=[]
for pairs in pairs_all:
    dist_temp=[]
    for pair in pairs:
        dist_temp.append(pair[0])
    dist_temp=np.array(dist_temp)
    dist_avg.append(np.mean(dist_temp))
    dist_std.append(np.std(dist_temp))
print(dist_avg)
print('########\n')
print(dist_std)
###


def calc_SPAC_pair(tr1, tr2, winlen, nlag):
    str2 = obspy.Stream(traces=[tr1, tr2])
    stackcc = np.zeros(nlag*2+1)
    stacksq = np.zeros(nlag*2+1)
    nstak = 0
    for windowed_st in str2.slide(window_length=winlen, step=winlen):
        if (len(windowed_st)< 2):
            continue

        # Try1: time-domain correlation (low-freq issues) - normalization is to std-dev of signals NOT ...
        #windowed_st.detrend()

        cc = correlate(windowed_st[0], windowed_st[1], nlag)
        stackcc = stackcc + cc
        stacksq = stacksq + cc*cc
        nstak += 1
    stackcc /= nstak
    stacksd = np.sqrt( (stacksq-stackcc*stackcc*nstak)/(nstak-1))
    # Now, to Fourier domain
    wdow = spwin.tukey(len(stackcc), alpha=0.2)
    stackcc *= wdow
    stacksd *= wdow
    stackcc = stackcc.T
    stacksd = stacksd.T
    unwrapped=np.append(stackcc[nlag:nlag*2+1],stackcc[0:nlag])
    fstak = np.fft.fft(unwrapped)
    fstdak = np.fft.fft( np.append(stacksd[nlag:nlag*2+1],stacksd[0:nlag]) )
    N = len(fstak)
    freqs = np.arange(N)/(N*tr1.stats.delta)

    return freqs, fstak, fstdak, stackcc
##
def mysmoo(y, n):
    # my n-point smoothing function of array y
    smoo1 = np.convolve(y, np.ones(n)/n, 'same')
    return smoo1
# Algorithm 2: direct freq-domain stacking of correlations
#computes the frequency-domain cross-coherence (tr1 and tr2) over a specified window length, with optional smoothing,
#and returns the frequency, coherence, and standard deviation of the coherence
def calc_SPACfreq_pair(tr1, tr2, winlen, nlag, nsmoo):
    # nsmoo = 5 # for smoothing before tapering
    str2 = obspy.Stream(traces=[tr1, tr2])
    dt = tr1.stats.delta
    N = int(winlen/dt)
    stackcc = np.zeros(N)
    stacksq = np.zeros(N)
    nstak = 0
    for windowed_st in str2.slide(window_length=winlen, step=winlen):
        if (len(windowed_st)< 2):
            continue
        tr1w = windowed_st[0].copy()
        tr2w = windowed_st[1].copy()
        tr1w.detrend().taper(0.25)
        tr2w.detrend().taper(0.25)
        ff1w = np.fft.fft(tr1w.data)
        ff2w = np.fft.fft(tr2w.data)
        # coher = ff1w*np.conj(ff2w)/np.abs(np.sqrt(ff1w*np.conj(ff1w)*ff2w*np.conj(ff2w)))
        C12 = ff1w*np.conj(ff2w)
        C11 = np.abs((ff1w*np.conj(ff1w)))
        C22 = np.abs((ff2w*np.conj(ff2w)))
        C12 = mysmoo(C12, nsmoo)
        C11 = mysmoo(C11, nsmoo)
        C22 = mysmoo(C22, nsmoo)
        coher = C12/np.sqrt(C11)/np.sqrt(C22)
        coher = coher[:len(stackcc)]
        stackcc = stackcc + coher
        stacksq = stacksq + coher*np.conj(coher)
        nstak += 1

    fstak = stackcc / nstak # fstak is the averaged coherence over all windows.
    stackr = np.real(fstak) #stackr is the real part of this average coherence.
    fstdak = np.sqrt((np.real(stacksq) - stackr*stackr*nstak)/(nstak-1)) #fstdak calculates the standard deviation of the coherence.
    freqs = np.arange(N)/(N*dt)
    return freqs, fstak, fstdak
##
##########
# sys.exit()
print('------------------------------------------\n')
print('starting SPAC\n')
delta = stream_clean[0].stats.delta
npt = stream_clean[0].stats.npts
nlag = int(max_seconds/delta)
print('Lags N:', nlag*2+1)
rescalefac = 3.0   # this is MULTIPLIED by he freq-domain stack; why?
#   winlen:  length of window to subset (s)
#   max_seconds:  max correlation shift to test (s)
winlen = 30.
max_seconds = 10.
# iring=0
nsmoo = 21
# pairs_all = pairs_all[iring]
N = int(winlen/delta)
allstaks = np.zeros([len(pairs_all), N],  dtype=np.complex128)

for iring, pairlist in enumerate(pairs_all):
    sumstak = np.zeros(nlag*2+1, dtype=np.complex128)
    sumstak2 = np.zeros(N, dtype=np.complex128)
    sumsdstak = np.zeros(nlag*2+1, dtype=np.complex128)
    delt = dist_avg[iring]
    std=dist_std[iring]
    for pair in pairlist:
        print('working on ring {:d} delt= {:.2f} m std={:.2f} m'.format(iring, delt,std))
        for tr in stream_clean.select(channel='DPZ'):
            if str(int(pair[1]))==tr.stats.station:
                tr1=tr.copy()
            if str(int(pair[2]))==tr.stats.station:
                tr2=tr.copy()
        #
        t_st=UTCDateTime("2024-07-27T00:00:00")
        t_end=UTCDateTime("2024-07-27T16:00:00")

        tr1.trim(t_st,t_end).detrend()
        tr2.trim(t_st,t_end).detrend()
        # comment prewhitening
        tr1.data = prewhiten(tr1.data)
        tr1.taper(.05)
        tr2.data = prewhiten(tr2.data)
        tr2.taper(.05)

        freqs2, fstak2, fstdak2 = calc_SPACfreq_pair(tr1, tr2, winlen, nlag, nsmoo)
        sumstak2 += fstak2

    sumstak2 /= len(pairlist)
    allstaks[iring,:] = sumstak2
    # if iring==2:
    #     break

# sys.exit()
# Plotting:  first "record" section
allstaks=np.load('allstack.npy')
freqs2=np.load('freqs2.npy')
allstaks_normalized = allstaks / np.max(np.abs(allstaks), axis=0)

sclmax= 20/(np.max(np.max(abs(allstaks))))
rescalefac = 3


for iring, pairlist in enumerate(pairs_all):
    delt = dist_avg[iring]
    yplt = np.real(allstaks_normalized[iring,:])*sclmax + delt
    plt.plot(freqs2, yplt)


plt.xlim([0, 30])
plt.xlabel('Frequency [Hz]')
plt.ylabel('Pair spacing [m]')
plt.title('SPAC function record section')
plt.grid()
# plt.savefig('SPAC_whit_rr_normalized.jpg',dpi=300,bbox_inches='tight', pad_inches=0.1)
plt.show()
#######
#

#
# allstaks_normalized* freqs[:, np.newaxis] ** 2

dphv = 20.0   # m/s grid spacing
phvmax = 1000.
frqfitmin = 1.0  # Min frequency to fit, in Hz
bessmin = 0.4     # approx value of first minimum of J0
phvels=np.arange(dphv,phvmax, dphv)
phvgrd = np.zeros([len(phvels), len(freqs2)])
idx = np.where(freqs2>frqfitmin)

for iring, delt in enumerate(dist_avg):
    spacdat = np.real(allstaks_normalized[iring,:])
    spacfit = spacdat[idx]
    # Scaling: find first minimum in spacfit and match that to first minimum of J0
    sclfac = abs(np.min(spacfit))/bessmin    # multiply this to bess0
    # if delt>200.:
    #     continue
    for iphv, phv in enumerate(phvels):
        xarg = 2*np.pi*freqs2*delt/phv
        bess0 = jv(0,xarg)*sclfac
        resid = spacdat - bess0
        phvgrd[iphv, :] = resid**2


plt.pcolor( freqs2, phvels, phvgrd, shading='nearest', vmin=0)#, vmax=0.9)
plt.set_cmap('plasma')
plt.xlim([0, 10])
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase velocity [m/s]')
print(np.max(np.max(phvgrd)))
plt.show()
# plt.savefig('dispersion_whit_rr.jpg',dpi=300,bbox_inches='tight', pad_inches=0.1)
