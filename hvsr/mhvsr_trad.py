import pathlib

import numpy as np
import matplotlib.pyplot as plt

import hvsrpy
from hvsrpy.data_wrangler import _read_mseed

plt.style.use(hvsrpy.HVSRPY_MPL_STYLE)

#####
srecords = _read_mseed("100_16hr.mseed",degrees_from_north=0)
# st=read
# fnames = [["2E.100.1.mseed", "2E.100.2.mseed", "2E.100.Z.mseed"]]
#
# fnames = [["100_16hr_1.mseed", "100_16hr_2.mseed", "100_16hr_Z.mseed"]]
# print(f"Number of recordings: {len(fnames)}")
# for fname_set in fnames:
#     for file in fname_set:
#         if not pathlib.Path(file).exists():
#             raise FileNotFoundError(f"file {file} not found; check spelling.")
# print("All files exist.")
###
preprocessing_settings = hvsrpy.settings.HvsrPreProcessingSettings()
preprocessing_settings.detrend = "linear"
preprocessing_settings.window_length_in_seconds = 100
preprocessing_settings.orient_to_degrees_from_north = 0.0
preprocessing_settings.filter_corner_frequencies_in_hz = (None, None)
preprocessing_settings.ignore_dissimilar_time_step_warning = False

print("Preprocessing Summary")
print("-"*60)
preprocessing_settings.psummary()
##
processing_settings = hvsrpy.settings.HvsrTraditionalProcessingSettings()
processing_settings.window_type_and_width = ("tukey", 0.2)
processing_settings.smoothing=dict(operator="konno_and_ohmachi",
                                   bandwidth=40,
                                   center_frequencies_in_hz=np.geomspace(0.2, 50, 200))
processing_settings.method_to_combine_horizontals = "geometric_mean"
processing_settings.handle_dissimilar_time_steps_by = "frequency_domain_resampling"

print("Processing Summary")
print("-"*60)
processing_settings.psummary()
##
# srecords = hvsrpy.read(fnames)
srecords = hvsrpy.preprocess(srecords, preprocessing_settings)
hvsr = hvsrpy.process(srecords, processing_settings)
#
print("\nStatistical Summary:")
print("-"*20)
hvsrpy.summarize_hvsr_statistics(hvsr)
(fig, ax) = hvsrpy.plot_single_panel_hvsr_curves(hvsr,)
ax.get_legend().remove()
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()
