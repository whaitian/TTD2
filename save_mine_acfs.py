import os, json
import ringdown
import numpy as np
import bilby.gw.utils as bgu
base_name = os.path.expanduser('~')

b_file = base_name+'/Project/'
d_file = base_name+'/Allthing/Data/'
##########################################
label_source = 'GW150914'## 'GW190521'## 

with open(b_file+'Data/event_tc_ifos.json', 'r') as f:
    eti = json.load(f)

time_of_event = eti[label_source]['tc']
print('Event time of %s: '%label_source, time_of_event)
det_names = eti[label_source]['ifos']
##########################################
# Init CONFIG
f_filter = 20.
sf_n = 1
sampling_frequency = 2048*sf_n
buffer_time = 0.5
slice_duration = 2.## 4./sf_n
dls = dict(zip([2048,4096,8192,16384], [8.,4.,2.,2.]))
duration_longer = dls[int(sampling_frequency)]
data_len = {'H1': 2046., 'L1': 2046., 'V1': 512.} ## 512 for GW190521 V1, 2046 for others
strain_data = {name:{} for name in det_names}
random_strain = {name:{} for name in det_names}
acfs = {name:{} for name in det_names}
PSDs = {name:{} for name in det_names}
acfs_mix = {name:{} for name in det_names}
## covariance_inverse = {name:{} for name in det_names}
time_se = {'start_time':time_of_event-slice_duration*2., 'end_time':time_of_event+slice_duration+1.}
from pycbc.psd import interpolate as ppi
from pycbc.psd import inverse_spectrum_truncation as ppist
##########################################
c_sf = '4KHZ' if sampling_frequency<=2048. else '16KHZ'
Notes = {'Notes':"The mixed acfs are truncated from slice duration of {0}s. The duration of acfs is {1}s. We down-sampling data of {2} to the sample rate we need. We do not perform the lowpass filter. We perform the inverse spectrum truncation.".format(duration_longer, slice_duration, c_sf)}
## c_sf = '16KHZ'
for det in det_names:
    frame_file = d_file+'GW/%s-%s_GWOSC_%s_R1-%s-4096.gwf'%(det[0], det, c_sf, int(time_of_event-2047))
    strain = bgu.read_frame_file(frame_file, start_time=time_of_event-data_len[det], end_time=time_of_event+data_len[det], buffer_time=buffer_time, channel='%s:GWOSC-%s_R1_STRAIN'%(det, c_sf)).to_pycbc()
    if sampling_frequency<strain.sample_rate:
        strain = strain.resample(1./sampling_frequency).highpass_fir(f_filter, order=512)
    else:
        assert int(sampling_frequency)==int(strain.sample_rate), "Please check the sample rate!"
        strain = strain.highpass_fir(f_filter, order=512)
    strain_data[det] = strain.time_slice(time_se['start_time'], time_se['end_time'])
    data = ringdown.Data(strain.data, index=strain.sample_times, ifo=det)
    acf = data.get_acf()
    acfs[det] = acf.values[:int(slice_duration/strain.delta_t)]
    #######################################
    psd0 = ppi(strain.psd(duration_longer), strain.delta_f)
    psd0 = ppist(psd0, int(duration_longer*strain.sample_rate), low_frequency_cutoff=f_filter)
    PSDs[det] = ppi(psd0, 1./duration_longer)
    #######################################
    acf0 = PSDs[det].astype(complex).to_timeseries()/2
    acfs_mix[det] = acf0[:int(slice_duration/strain.delta_t)]

kwargs = dict(f_filter=f_filter, sampling_frequency=sampling_frequency)
f_name = b_file+'TD_likelihood/TD_data/PyCBC_psd_acfs_{0}_{1}-{2}Hz.npy'.format(label_source, int(f_filter), int(sampling_frequency))
with open(f_name, 'wb') as g:
    np.save(g, {'kwargs':kwargs, 'time_se':time_se, 'PSDs':PSDs, 'acfs':acfs, 'acfs_mix':acfs_mix, 'strain_data':strain_data, 'Notes':Notes
})
