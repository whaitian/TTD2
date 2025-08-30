import os
import numpy as np
from pycbc import frame
base_name = os.path.expanduser('~')

##########################################
label_source = 'GW231123'
te = 1384782888.6
det_names = ["H1", "L1"]
##########################################
# Init CONFIG
f_filter = 20.
sampling_frequency = 1024## *2

slice_duration = 8.
strain_data = {name:{} for name in det_names}
time_se = {'start_time':te-slice_duration-1., 'end_time':te+3.}

acfs1 = {name:{} for name in det_names}
PSDs1 = {name:{} for name in det_names}

from pycbc.psd import interpolate as ppi
from pycbc.psd import inverse_spectrum_truncation as ppist

start_t0 = 1384779777.
end_t0 = start_t0+4094.
##########################################
## Create two files with names: Data, TD_data in your directory
## Download GW data before running this script!
##########################################
for det in det_names:
    strain = frame.read_frame('./Data/{0}-{1}_GWOSC_DiscO4a_4KHZ_R1-1384779776-4096.gwf'.format(det[0], det), f'{det}:GWOSC-4KHZ_R1_STRAIN', start_t0, end_t0)
    strain = strain.resample(1./sampling_frequency).highpass_fir(f_filter, order=512)
    strain_data[det] = strain.time_slice(time_se['start_time'], time_se['end_time'])
    #######################################
    psd1 = ppi(strain.psd(slice_duration), strain.delta_f)
    psd1 = ppist(psd1, int(slice_duration*strain.sample_rate), low_frequency_cutoff=f_filter)
    PSDs1[det] = ppi(psd1, 1./slice_duration)
    acf1 = PSDs1[det].astype(complex).to_timeseries()/2
    acfs1[det] = acf1[:int(slice_duration/strain.delta_t+10)]
    #######################################

Notes = {"acfs1|PSDs1": f"Estimated by the PyCBC package, using {slice_duration} s slice duration."}
kwargs = dict(f_filter=f_filter, sampling_frequency=sampling_frequency, slice_duration=slice_duration)
f_name = './TD_data/PyCBC_psd_acfs_{0}_{1}-{2}Hz_t{3}s-v0.npy'.format(label_source, int(f_filter), int(sampling_frequency), int(slice_duration))
with open(f_name, 'wb') as g:
    np.save(g, {'Notes':Notes, 'kwargs':kwargs, 'time_se':time_se, 'PSDs1':PSDs1, 'acfs1':acfs1, 'strain_data':strain_data
})
