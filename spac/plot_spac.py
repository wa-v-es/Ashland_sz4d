# uses freqs2.npy and allstack.npy calculated by spac_cornell_v3.py.
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
##

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

# Array Geometry Recalculations:  BUILD THE ARRAY HERE

sta_lat=[]
sta_long=[]
sta_ele=[]
sta_name=[]
sta_node=[]
for line in open('/Users/keyser/Research/sz4d_ft/nodal_data/Enclave_1_station_data.txt','r'):
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
##
##### Build the array rings #########
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

##

allstaks=np.load('allstack.npy')
freqs2=np.load('freqs2.npy')
allstaks_normalized = allstaks / np.max(np.abs(allstaks), axis=0)

sclmax= 20/(np.max(np.max(abs(allstaks))))
rescalefac = 3


for iring, pairlist in enumerate(pairs_all):
    delt = dist_avg[iring]
    yplt = np.real(allstaks_normalized[iring,:])*sclmax + delt
    plt.plot(freqs2, yplt)


plt.xlim([0, 6])
plt.xlabel('Frequency [Hz]')
plt.ylabel('Pair spacing [m]')
plt.title('SPAC function record section')
plt.grid()
# plt.savefig('SPAC_whit_rr_normalized.jpg',dpi=300,bbox_inches='tight', pad_inches=0.1)
plt.show()
#######
#
sys.exit()
#
# allstaks_normalized* freqs[:, np.newaxis] ** 2

dphv = 20.0   # m/s grid spacing
phvmax = 1000.
frqfitmin = 2.0  # Min frequency to fit, in Hz
frqfitmax =23.0  # Min frequency to fit, in Hz

bessmin = 0.4     # approx value of first minimum of J0
phvels=np.arange(dphv,phvmax, dphv)
phvgrd = np.zeros([len(phvels), len(freqs2)])
idx = np.where((freqs2 > frqfitmin) & (freqs2 < frqfitmax))

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
plt.xlim([2, 10])
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase velocity [m/s]')
print(np.max(np.max(phvgrd)))
plt.show()
# plt.savefig('dispersion_whit_rr.jpg',dpi=300,bbox_inches='tight', pad_inches=0.1)
