"""
td_FBI_optimized.py

This script runs a time-domain Fast Bayesian Inference (FBI) analysis for a
gravitational wave ringdown signal. It uses a custom likelihood defined in
`td_likelihood.py` to perform parameter estimation on a specified event.

The script is configurable via command-line arguments for parameters like the
ringdown modes to be analyzed and the time delay for the analysis segment.
"""

import os
import argparse
import lal
import bilby
import numpy as np
import scipy.linalg as sl
import bilby.core.utils as bcu
from gwpy.timeseries import TimeSeries as gtts
from bilby.gw.detector import InterferometerList
from pycbc.detector import Detector as pdd

# Import the custom classes from your previously optimized script
from td_likelihood import RDTDFBITransient, TDWaveformGenerator, pycbc_ringdown_lmn

def main(args):
    """
    Main function to run the Bayesian inference analysis.
    """
    # --- 1. Setup: Environment, Directories, and Logging ---

    # Control CPU thread usage for numerical libraries to avoid conflicts
    # with parallel sampling.
    os.environ['OMP_NUM_THREADS'] = '1'

    # Generate a unique label for this analysis run based on the input arguments
    label = f'{args.modes.replace(",", "+")}_{args.time_delay_multiplier}M_0-4s'
    
    # Create the full output directory path and set up logging
    full_outdir = os.path.join(args.outdir, label)
    bcu.setup_logger(outdir=full_outdir, label=label)
    
    # --- 2. Event and Detector Configuration ---

    # Define event parameters. These are fixed for the specific event being analyzed.
    # The time_of_event is fine-tuned to match the peak of the signal polarization.
    time_of_event = 1384782888.6 + 0.0153731
    
    # These sky location values are likely taken from a maximum likelihood estimate
    # from a previous, broader analysis of GW231123.
    ra0, dec0 = 3.24, 0.25

    # Set up Bilby interferometer objects
    det_names = ["H1", "L1"]
    ifos = InterferometerList(det_names)
    
    # --- 3. Time Delay and Data Segment Calculation ---

    # Create PyCBC detector objects to calculate geometric time delays
    dets = {det_name: pdd(det_name) for det_name in det_names}
    
    # Calculate the time delay for each detector from the geocenter
    delays0 = {
        detname: det.time_delay_from_earth_center(ra0, dec0, time_of_event)
        for detname, det in dets.items()
    }

    # Define the characteristic time scale using the approximate total mass
    m0 = 298.0 * lal.MTSUN_SI
    
    # Calculate the analysis start time, which is a specified number of
    # characteristic timescales after the event time.
    t0 = time_of_event + m0 * args.time_delay_multiplier
    
    # Define the start and end times for the data slice for each detector
    slice_duration = 0.4  # seconds
    time_slices = {
        dn: {'start_time': t0 + delays0[dn], 'end_time': t0 + delays0[dn] + slice_duration + 0.1}
        for dn in det_names
    }

    # --- 4. Loading Pre-computed Data (ACFs and Strain) ---

    sampling_frequency = 1024.
    f_filter = 20.
    
    # Construct the filename for the pre-computed data
    data_filename = (
        f'./TD_data/PyCBC_psd_acfs_{args.event_name}_{int(f_filter)}-'
        f'{int(sampling_frequency)}Hz_t8s-v0.npy'
    )
    
    bcu.logger.info(f"Loading pre-computed data from: {data_filename}")
    with open(data_filename, 'rb') as g:
        loaded_data = np.load(g, allow_pickle=True).item()

    acfs = loaded_data['acfs1']
    notes = loaded_data['Notes']
    strain_data = loaded_data['strain_data']

    bcu.logger.info(f"Notes for the loaded ACFs: \n{notes}")

    # --- 5. Data Conditioning and Whitening Matrix Preparation ---

    l_cov = int(slice_duration * sampling_frequency)

    for ifo in ifos:
        # Slice the strain data to the analysis segment for the current detector
        strain_td = strain_data[ifo.name].time_slice(
            time_slices[ifo.name]['start_time'], time_slices[ifo.name]['end_time']
        )[:l_cov]
        
        # Convert to a gwpy TimeSeries object and set it as the detector's strain data
        ifo_data = gtts(
            strain_td.numpy(),
            sample_rate=strain_td.get_sample_rate(),
            times=strain_td.sample_times.numpy()
        )
        ifo.set_strain_data_from_gwpy_timeseries(ifo_data)

        # The core of the time-domain likelihood: preparing the whitening matrix.
        # The ACF defines the noise covariance matrix C, which is a Toeplitz matrix.
        # 1. Perform a Cholesky decomposition: C = L * L^T, where L is lower-triangular.
        # 2. The whitening matrix is then W = L^-1.
        # 3. The whitened residual is W * (d - h), and the chi-squared is sum((W*(d-h))^2).
        toeplitz_matrix = sl.toeplitz(acfs[ifo.name][:l_cov])
        L0 = sl.cholesky(toeplitz_matrix, lower=True)
        acfs[ifo.name] = sl.solve_triangular(L0, np.eye(L0.shape[0]), lower=True)

    # Parse the input modes string into a list of all modes to analyze
    lmns = [item.strip() for item in args.modes.split(',')]
    lmn_all = [f"{lmn[:2]}{n}" for lmn in lmns for n in range(int(lmn[-1]))]
    bcu.logger.info(f'Modes considered in this run are: \n{lmn_all}')

    # --- 6. Setting Priors for Bayesian Inference ---
    
    priors = bilby.core.prior.PriorDict()
    
    # Fixed parameters (extrinsic) based on prior analysis
    priors['geocent_time'] = time_of_event
    priors['ra'] = ra0
    priors['dec'] = dec0
    priors['psi'] = 2.23  # Polarization angle
    
    # Sampled parameters
    priors['inclination'] = bilby.core.prior.Sine(name='inclination')
    
    # Priors on the final black hole properties (intrinsic)
    priors['final_mass'] = bilby.core.prior.Uniform(
        name='final_mass', minimum=100, maximum=500, unit='$M_{\\odot}$'
    )
    priors['final_spin'] = bilby.core.prior.Uniform(
        name='final_spin', minimum=0.0, maximum=0.99
    )
    
    # Priors for the amplitude and phase of each ringdown mode
    for lmn in lmn_all:
        priors[f'amp{lmn}'] = bilby.core.prior.Uniform(
            name=f'amp{lmn}', minimum=0.0, maximum=50.0
        )
        priors[f'phi{lmn}'] = bilby.core.prior.Uniform(
            name=f'phi{lmn}', minimum=0.0, maximum=2 * np.pi
        )

    # --- 7. Waveform Generator and Likelihood Setup ---

    waveform_arguments = {'lmns': lmns, 'delta_t': 1. / sampling_frequency}
    bcu.logger.info(f'Waveform arguments: {waveform_arguments}')

    waveform_generator = TDWaveformGenerator(
        duration=1.0,
        sampling_frequency=sampling_frequency,
        time_domain_source_model=pycbc_ringdown_lmn,
        waveform_arguments=waveform_arguments
    )

    likelihood = RDTDFBITransient(
        interferometers=ifos,
        waveform_generator=waveform_generator,
        acfs=acfs,
        priors=priors
    )

    # --- 8. Running the Sampler ---

    sampler_settings = {
        'nact': 20,
        'sample': 'rwalk',
        'bound': 'live-multi',
        'proposals': ['diff', 'volumetric'],
        'n_check_point': 1000
    }
    bcu.logger.info(f'Sampler settings: {sampler_settings}')

    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler='dynesty',
        npoints=1000,
        nthreads=10,
        outdir=full_outdir,
        label=label,
        resume=True,
        **sampler_settings
    )

    # --- 9. Plotting Results ---
    
    result.plot_corner(
        parameters=['final_mass', 'final_spin', 'inclination'],
        filename=f'{full_outdir}/{label}_corner.png',
        quantiles=[0.05, 0.95]
    )
    bcu.logger.info(f"Analysis complete. Results saved in {full_outdir}")


if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Run a time-domain Bayesian analysis on a GW ringdown signal."
    )
    
    parser.add_argument(
        '--modes',
        type=str,
        default='221',
        help="String specifying the ringdown modes to include. "
             "Example: '221' for the 220 mode, or '221,201' for 220 and 200 modes. "
             "Default: '221'"
    )
    
    parser.add_argument(
        '--time-delay-multiplier',
        type=int,
        default=10,
        help="The time delay after the event peak to start the analysis, "
             "measured in units of the characteristic time M (final mass). "
             "Default: 10"
    )
    
    parser.add_argument(
        '--event-name',
        type=str,
        default='GW231123',
        help="The name of the gravitational wave event. Default: 'GW231123'"
    )
    
    parser.add_argument(
        '--outdir',
        type=str,
        default='FBI_analysis_results',
        help="The top-level directory to save the analysis results. "
             "A subdirectory will be created for each specific run. "
             "Default: 'FBI_analysis_results'"
    )
    
    args = parser.parse_args()
    main(args)
