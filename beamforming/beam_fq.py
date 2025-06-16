import numpy as np
from obspy import read, UTCDateTime
from obspy.signal.array_analysis import array_processing
import matplotlib.pyplot as plt
import sys
import glob
import obspy
from obspy.clients.fdsn import Client
from obspy.geodetics import locations2degrees, gps2dist_azimuth
from obspy import read_inventory,read,Stream
from obspy.signal.cross_correlation import correlate
from obspy import UTCDateTime
from obspy.core.util import AttribDict
from obspy.imaging.cm import obspy_sequential
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
##
def plot_baz_slow(out,frqlow, frqhigh):
    cmap = obspy_sequential
    # cmap = 'cividis'
    #
    t, rel_power, abs_power, baz, slow = out.T
    baz[baz < 0.0] += 360

    #  number of fractions in plot
    N = 36
    N2 = 30
    abins = np.arange(N + 1) * 360. / N
    sbins = np.linspace(0, 3, N2 + 1)

    # sum rel power in bins given by abins and sbins
    hist, baz_edges, sl_edges = \
        np.histogram2d(baz, slow, bins=[abins, sbins], weights=rel_power)

    # transform to radian
    baz_edges = np.radians(baz_edges)

    # add polar and colorbar axes
    fig = plt.figure(figsize=(8, 8))
    cax = fig.add_axes([0.85, 0.2, 0.05, 0.5])
    ax = fig.add_axes([0.10, 0.1, 0.70, 0.7], polar=True)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")

    dh = abs(sl_edges[1] - sl_edges[0])
    dw = abs(baz_edges[1] - baz_edges[0])

    # circle through backazimuth
    for i, row in enumerate(hist):
        bars = ax.bar((i * dw) * np.ones(N2),
                      height=dh * np.ones(N2),
                      width=dw, bottom=dh * np.arange(N2),
                      color=cmap(row / hist.max()))

    ax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    ax.set_xticklabels(['N', 'E', 'S', 'W'])

    # set slowness limits
    ax.set_ylim(0, 3)
    [i.set_color('grey') for i in ax.get_yticklabels()]
    ColorbarBase(cax, cmap=cmap,
                 norm=Normalize(vmin=hist.min(), vmax=hist.max()))

    # plt.show()
    plt.savefig('baz_slow_{}_{}.png'.format(frqlow, frqhigh),dpi=350,bbox_inches='tight')
###
#
# client = Client("IRISPH5")
# inventory = client.get_stations(network="2E", station="*",
#                                 starttime=UTCDateTime('2024-07-25T02:00:00.000000'),
#                                 endtime=UTCDateTime('2024-07-29T02:00:00.000000'),level="response")
# # Location of the reference station at array center, as lat/lon
# ctrlat = 42.07559574
# ctrlon = -122.7472788
starttime=UTCDateTime("2024-07-27T00:00:00")
endtime=UTCDateTime("2024-07-27T16:00:00")
#
# stream_fd = client.get_waveforms("2E", "*", "*", "*Z", starttime, endtime)
# stream_fdsn=stream_fd.copy()
# stream_clean=Stream()
# for tr in stream_fdsn:
#     time=tr.stats.endtime
#     if UTCDateTime(time.timestamp + 0.5).replace(microsecond=0) == t_end:
#         stream_clean.append(tr)
###
# sys.exit()
# data_path = "data/*.mseed"     # all 10 stations
# starttime = UTCDateTime("2020-01-01T00:00:00")
# endtime = UTCDateTime("2020-01-02T00:00:00")
win_len = 20.0                 # length of each window in seconds
win_frac = 0.5                 # 50% overlap
prewhiten = 1

slowness_grid = {
    'sll': .01,  # min slowness [s/km]
    'slm': 0.5,   # max slowness [s/km]
    'sls': 0.01,  # slowness step
}

freq_bands = [
    (0.2, 0.3),
    (0.3, 0.4),
    (0.4, 0.5),
    (0.5,0.6),
    (0.6,1),
    (1,2),
    (2,5)
]

# --- Load and preprocess ---
st = read('all_traces_16hr.mseed')
sta_lat=[]; sta_long=[]; sta_ele=[] ;sta_name=[]; sta_node=[]
# for line in open('Enclave_1_station_data.txt','r'):
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

for i in range(len(sta_ele)):
    for tr in st:
        # if int(sta_node[i])==int(tr.stats.station): # used if saved traces are used..
        if int(sta_name[i])==int(tr.stats.station):
            tr.stats.coordinates = AttribDict({
            'latitude': sta_lat[i],
            'elevation': sta_ele[i],
            'longitude': sta_long[i]})

# sys.exit()
# st.merge(fill_value='interpolate')
st.detrend('linear')
st.taper(max_percentage=0.05)


phase_velocity_results = []

# Loop over frequency bands
for frqlow, frqhigh in freq_bands:
    print(f"Processing frequency band: {frqlow}-{frqhigh} Hz")

    out = array_processing(
        st,
        sll_x=-4.0, slm_x=4.0, sll_y=-4.0, slm_y=4.0, sl_s=0.03,
        win_len=win_len,
        win_frac=win_frac,
        frqlow=frqlow,
        frqhigh=frqhigh,
        prewhiten=prewhiten,
        coordsys='lonlat',
        semb_thres=-1e9, vel_thres=-1e9,
        stime=starttime+1,
        etime=endtime-1,
        verbose=False
    )
    plot_baz_slow(out, frqlow, frqhigh)
    out = np.array(out)
    # Columns are  time, backazimuth, slowness, coherence, ...

    # Filter to high-coherence windows (e.g., top 10%)
    coherence = out[:, 3]
    threshold = np.percentile(coherence, 90)
    high_confidence = out[coherence >= threshold]

    # Average slowness of confident picks
    if len(high_confidence) > 0:
        avg_slowness = np.mean(high_confidence[:, 2])
        phase_velocity = 1.0 / avg_slowness if avg_slowness != 0 else np.nan
        phase_velocity_results.append((np.mean([frqlow, frqhigh]), phase_velocity))
    else:
        phase_velocity_results.append((np.mean([frqlow, frqhigh]), np.nan))

# --- Plot dispersion curve ---
# freqs, velocities = zip(*phase_velocity_results)
freqs, velocities = zip(*phase_velocity_results[2:])


plt.figure(figsize=(6, 4))
plt.plot(freqs, velocities, 'o-',color='maroon' ,label='Phase Velocity',alpha=.7)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase Velocity [km/s]')
plt.grid()
plt.title('Dispersion Curve from Ambient Noise Beamforming')
plt.legend()
plt.tight_layout()
# plt.savefig('bf_16hr_20s_high.png',dpi=300,bbox_inches='tight')
plt.show()
