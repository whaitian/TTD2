import numpy as np
from pycbc import waveform
from bilby.gw.detector import InterferometerList
from bilby.core.likelihood import Likelihood
from bilby.gw.waveform_generator import WaveformGenerator

def toeplitz_slogdet(r):
    """
    Compute the sign and natural logarithm of the determinant of a positive-definite
    symmetric Toeplitz matrix.

    This method is from Marano et al. "Fitting Earthquake Spectra: Colored Noise
    and Incomplete Data", Bulletin of the Seismological Society of America,
    Vol. 107, No. 1, pp. –, February 2017, doi: 10.1785/0120160030.
    The determinant is computed efficiently using a modified Levinson recursion.

    Original code available at:
    http://mercalli.ethz.ch/~marra/publications/2017_fitting_earthquake_spectra_colored_noise_and_incomplete_data/
    All credits go to the original authors.

    Parameters
    ----------
    r : np.ndarray
        The first row of the symmetric Toeplitz matrix.

    Returns
    -------
    tuple:
        (sign, logdet): A tuple containing the sign of the determinant and the
        natural log of its absolute value.
    """
    n = len(r)
    r_0 = r[0]

    # Normalize the system so that the main diagonal of the Toeplitz matrix is all ones.
    # The log determinant of the original matrix is recovered by adding n*log(r_0).
    r = np.concatenate((r, np.array([r_0]))) / r_0
    logdet = n * np.log(np.abs(r_0))
    sign = np.sign(r_0)**n

    if n == 1:
        return sign, logdet

    # The following is a modification of the Levinson recursion algorithm
    # to efficiently compute the determinant.
    y = np.zeros((n,))
    x = np.zeros((n,))
    b = -r[1:n + 1]
    r = r[:n]

    # Initialization for k=0
    y[0] = -r[1]
    x[0] = b[0]
    beta = 1.0
    alpha = -r[1]

    # The determinant is the product of the diagonal elements of the Cholesky
    # decomposition. This loop calculates those elements recursively.
    d = 1 + np.dot(-b[0], x[0])
    sign *= np.sign(d)
    logdet += np.log(np.abs(d))

    # Main recursion loop for k = 0 to n-3
    for k in range(n - 2):
        beta = (1 - alpha * alpha) * beta
        # mu is the reflection coefficient
        mu = (b[k + 1] - np.dot(r[1:k + 2], x[k::-1])) / beta
        x[0:k + 1] += mu * y[k::-1]
        x[k + 1] = mu

        d = 1 + np.dot(-b[0:k + 2], x[0:k + 2])
        sign *= np.sign(d)
        logdet += np.log(np.abs(d))

        if k < n - 2:
            alpha = -(r[k + 2] + np.dot(r[1:k + 2], y[k::-1])) / beta
            y[0:k + 1] += alpha * y[k::-1]
            y[k + 1] = alpha

    return sign, logdet


def pycbc_ringdown_lmn(**kwargs):
    """
    Generate a time-domain gravitational waveform for a black hole ringdown
    signal using the pycbc library.

    This function supports multiple spherical harmonic modes (l, |m|, n) and can
    generate waveforms based on either the final black hole properties ('kerr' model)
    or directly from mode frequencies and damping times ('ftau' model).

    Note: The 'tau' parameters in kwargs are expected to be the inverse damping
    time (1/tau) for numerical stability with priors.

    Parameters
    ----------
    **kwargs : dict
        A dictionary of waveform parameters passed to pycbc. Expected keys include
        'model' ('kerr' or 'ftau'), 'final_mass', 'final_spin', 'lmns',
        amplitudes (e.g., 'amp220'), phases (e.g., 'phi220'), and quality factors
        or taus (e.g., 'tau_220').

    Returns
    -------
    dict:
        A dictionary containing the 'plus' and 'cross' polarizations of the
        generated waveform as pycbc.types.TimeSeries objects.
    """
    # Default waveform parameters
    waveform_params = dict(
        taper=False, final_mass=20., final_spin=None, lmns=['222'],
        amp220=1., phi220=0., inclination=0., delta_t=1./2048, model='kerr'
    )
    waveform_params.update(kwargs)

    model = waveform_params['model']
    # Generate a list of all mode strings, e.g., '220', '221' from '222'
    lmn_all = [f"{lmn[:2]}{n}" for lmn in waveform_params['lmns'] for n in range(int(lmn[-1]))]

    if len(lmn_all) > 1:
        # The dominant 220 mode is treated as the reference.
        # Normalize other mode amplitudes relative to the 220 amplitude.
        if '220' in lmn_all:
            lmn_all.remove('220')
        for lmn in lmn_all:
            waveform_params[f'amp{lmn}'] /= waveform_params['amp220']

    # Scale the primary amplitude to a physically realistic strain value.
    waveform_params['amp220'] *= 1.e-20

    # Generate waveform based on the selected model
    if model == 'kerr':
        hplus, hcross = waveform.ringdown.get_td_from_final_mass_spin(**waveform_params)
    elif model == 'ftau':
        # Convert inverse tau values back to tau for pycbc function call.
        for lmn in lmn_all + ['220']: # Ensure all taus are inverted
             if f'tau_{lmn}' in waveform_params:
                waveform_params[f'tau_{lmn}'] = 1. / waveform_params[f'tau_{lmn}']
        hplus, hcross = waveform.ringdown.get_td_from_freqtau(**waveform_params)
    else:
        raise ValueError(f"Unknown waveform model '{model}'. Use 'kerr' or 'ftau'.")

    return {'plus': hplus, 'cross': hcross}


class TDWaveformGenerator(WaveformGenerator):
    """
    A Bilby WaveformGenerator that adds a caching layer to avoid redundant
    time-domain waveform generations.

    This class overrides the `time_domain_strain` method to check if the
    waveform for the current set of parameters has already been computed.
    If so, it returns the cached result; otherwise, it calls the source model
    to generate a new one.
    """
    def time_domain_strain(self, parameters=None):
        """
        Generates the time-domain strain polarizations for a given set of parameters.

        Implements a cache to store and retrieve the last generated waveform.

        Parameters
        ----------
        parameters : dict, optional
            A dictionary of parameters for the waveform generation.

        Returns
        -------
        dict or None:
            A dictionary containing the 'plus' and 'cross' waveform polarizations.
            Returns None if waveform generation fails.
        """
        # Return cached waveform if parameters are unchanged
        if parameters == self._cache['parameters']:
            return self._cache['waveform']

        # If parameters are new, generate the waveform
        try:
            # Merge sampling parameters with fixed waveform arguments
            all_params = {**parameters, **self.waveform_arguments}
            waveform_polarizations = self.time_domain_source_model(**all_params)
        except RuntimeError:
            # Waveform generation can fail for certain parameter combinations
            return None

        # Update cache with the new waveform and parameters
        self._cache['waveform'] = waveform_polarizations
        self._cache['parameters'] = parameters.copy()

        return waveform_polarizations


class RDTDFBITransient(Likelihood):
    """
    A time-domain likelihood for analyzing transient gravitational wave ringdown
    signals, designed for Fast Bayesian Inference (TDFBI).

    This likelihood assumes the noise is stationary and Gaussian, characterized by
    its auto-correlation function (ACF). The likelihood is computed directly in
    the time domain.

    Attributes
    ----------
    interferometers : bilby.gw.detector.InterferometerList
        A list of interferometer objects containing the strain data.
    waveform_generator : WaveformGenerator
        The waveform generator object to produce the signal template.
    acfs : dict
        A dictionary mapping interferometer names to their pre-computed matrices
        used for whitening the data (related to the inverse noise covariance).
    normalisations : dict
        A dictionary mapping interferometer names to the log-determinant term of
        the noise covariance matrix, used for normalizing the likelihood.
    """
    def __init__(
        self, interferometers, waveform_generator, acfs,
        normalisations={'H1': 0., 'L1': 0., 'V1': 0.},
        priors=None, sky_average=False
    ):
        """
        Parameters
        ----------
        interferometers : list
            A list of `bilby.gw.detector.Interferometer` objects.
        waveform_generator : object
            An instance of a `WaveformGenerator` class.
        acfs : dict
            Dictionary of whitening matrices for each interferometer.
        normalisations : dict
            Dictionary of likelihood normalization constants for each interferometer.
        priors : bilby.core.prior.PriorDict, optional
            The priors for the sampling.
        sky_average : bool, optional
            If True, average the likelihood over sky location and orientation.
        """
        super(RDTDFBITransient, self).__init__(dict())
        self.interferometers = InterferometerList(interferometers)
        self.waveform_generator = waveform_generator
        self.acfs = acfs
        self.normalisations = normalisations
        self.priors = priors
        self.sky_average = sky_average
        self._meta_data = {}

    def noise_log_likelihood(self):
        """
        Calculates the log-likelihood of the data given the noise model only
        (the null or noise-only hypothesis).

        The result is cached to avoid re-computation.

        Returns
        -------
        float:
            The natural logarithm of the noise-only likelihood.
        """
        # Return cached value if available
        if 'nll' in self._meta_data:
            return self._meta_data['nll']

        log_likelihood_total = 0.
        for ifo in self.interferometers:
            strain_ts = ifo.strain_data.to_pycbc_timeseries()
            # Whiten the strain data using the pre-computed ACF matrix
            whitened_strain = self.acfs[ifo.name] @ strain_ts.data
            # Calculate the chi-squared term for the noise-only hypothesis
            # This is equivalent to -0.5 * (d^T * C^-1 * d)
            log_likelihood_total -= 0.5 * np.sum(whitened_strain**2)
            # Add the normalization term related to the determinant of the
            # noise covariance matrix.
            log_likelihood_total -= self.normalisations[ifo.name]

        # Cache and return the result
        self._meta_data['nll'] = log_likelihood_total
        return log_likelihood_total

    def get_pycbc_detector_response_td(self, ifo, waveform_polarizations, start_t):
        """
        Projects the waveform polarizations onto a single detector and aligns
        the resulting signal in time with the detector's data segment.

        Parameters
        ----------
        ifo : bilby.gw.detector.Interferometer
            The interferometer object.
        waveform_polarizations : dict
            A dictionary containing the 'plus' and 'cross' polarizations.
        start_t : float
            The start GPS time of the detector data segment.

        Returns
        -------
        pycbc.types.TimeSeries:
            The time-domain detector response.
        """
        signal = {}
        # Project each polarization onto the detector using antenna response functions
        for mode in waveform_polarizations.keys():
            det_response = ifo.antenna_response(
                self.parameters['ra'], self.parameters['dec'],
                self.parameters['geocent_time'], self.parameters['psi'], mode
            )
            signal[mode] = waveform_polarizations[mode] * det_response

        signal_ifo = sum(signal.values())

        # Calculate the total time shift needed to align the waveform with the data
        time_delay = ifo.time_delay_from_geocenter(
            self.parameters['ra'], self.parameters['dec'], self.parameters['geocent_time']
        )
        # dt is the shift relative to the start of the data segment
        dt = (time_delay + self.parameters['geocent_time'] - float(start_t) +
              float(signal_ifo.end_time))

        # Pad and roll the waveform to place it correctly in the time series
        # representing the analysis segment.
        total_samples_to_pad = int((dt + ifo.strain_data.duration) / signal_ifo.delta_t)
        signal_ifo.prepend_zeros(total_samples_to_pad)
        samples_to_roll = int(round(dt / signal_ifo.delta_t, 0))
        signal_ifo.roll(samples_to_roll)
        signal_ifo.start_time = start_t

        return signal_ifo

    def log_likelihood(self):
        """
        Calculates the log-likelihood of the data given the signal-plus-noise hypothesis.

        Returns
        -------
        float:
            The natural logarithm of the likelihood. Returns -inf for invalid parameters.
        """
        # Generate the signal template (waveform)
        try:
            waveform_polarizations = self.waveform_generator.time_domain_strain(self.parameters)
        except RuntimeError:
            return np.nan_to_num(-np.inf)

        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)

        log_likelihood_total = 0.
        for ifo in self.interferometers:
            strain_ts = ifo.strain_data.to_pycbc_timeseries()
            strain_len = len(strain_ts.data)

            if self.sky_average:
                # Use a sky-averaged template if specified
                waveform_det = (waveform_polarizations['plus'] + waveform_polarizations['cross']) * self.sky_average
                waveform_det.append_zeros(strain_len)
                waveform_det.start_time = float(strain_ts.start_time)
            else:
                # Project the waveform onto the detector
                waveform_det = self.get_pycbc_detector_response_td(
                    ifo, waveform_polarizations, strain_ts.start_time
                )

            # Calculate the residual (data - template)
            residual = strain_ts.data - waveform_det.data[:strain_len]

            # Whiten the residual
            whitened_residual = self.acfs[ifo.name] @ residual

            # Calculate the chi-squared term for the signal-plus-noise hypothesis
            # This is equivalent to -0.5 * ((d-h)^T * C^-1 * (d-h))
            log_likelihood_total -= 0.5 * np.sum(whitened_residual**2)
            # Add the normalization term
            log_likelihood_total -= self.normalisations[ifo.name]

        return log_likelihood_total

    def log_likelihood_ratio(self):
        """
        Calculates the log-likelihood ratio (or log Bayes factor) between the
        signal-plus-noise hypothesis and the noise-only hypothesis.

        Returns
        -------
        float:
            The log likelihood ratio.
        """
        return self.log_likelihood() - self.noise_log_likelihood()
