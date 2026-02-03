"""
Signal preprocessing module for BEAM.

Provides data quality control and preprocessing functions adapted from
AELUMA driver patterns, including:
- Bandpass filtering
- Gap detection and handling
- Spike/discontinuity rejection
- Envelope computation (STA/LTA)
- Data decimation
"""

import numpy as np
import logging
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)


def bandpass_filter(data, sampling_rate, freqmin, freqmax, corners=4):
    """
    Apply zero-phase Butterworth bandpass filter.
    
    Parameters
    ----------
    data : ndarray
        Input data array
    sampling_rate : float
        Sampling rate in Hz
    freqmin : float
        Minimum frequency (Hz)
    freqmax : float
        Maximum frequency (Hz)
    corners : int
        Number of filter corners (order)
        
    Returns
    -------
    filtered : ndarray
        Bandpass filtered data
    """
    nyquist = 0.5 * sampling_rate
    low = freqmin / nyquist
    high = freqmax / nyquist
    
    # Ensure frequencies are valid
    low = max(0.001, min(low, 0.999))
    high = max(low + 0.001, min(high, 0.999))
    
    sos = butter(corners, [low, high], btype='band', output='sos')
    filtered = sosfiltfilt(sos, data)
    
    return filtered


def detect_spikes(data, jump_max_size, jump_max_number):
    """
    Detect spiky/discontinuous data.
    
    Adapted from AELUMA ifdumpspike handling.
    
    Parameters
    ----------
    data : ndarray
        Input data array
    jump_max_size : float
        Maximum allowed amplitude jump between consecutive samples
    jump_max_number : int
        Maximum number of jumps allowed
        
    Returns
    -------
    is_spiky : bool
        True if data exceeds spike thresholds
    n_spikes : int
        Number of detected spikes
    """
    diff_trace = np.diff(data)
    spike_indices = np.where(np.abs(diff_trace) > jump_max_size)[0]
    n_spikes = len(spike_indices)
    
    is_spiky = (n_spikes > jump_max_number) or (np.max(np.abs(diff_trace)) > jump_max_size)
    
    return is_spiky, n_spikes


def reject_spikes(stream, jump_max_size=1e6, jump_max_number=100):
    """
    Remove traces with spiky data from stream.
    
    Parameters
    ----------
    stream : obspy.Stream
        Input stream
    jump_max_size : float
        Maximum allowed amplitude jump
    jump_max_number : int
        Maximum number of jumps allowed
        
    Returns
    -------
    stream : obspy.Stream
        Stream with spiky traces removed
    rejected : list
        List of rejected station codes
    """
    rejected = []
    traces_to_remove = []
    
    for tr in stream:
        is_spiky, n_spikes = detect_spikes(tr.data, jump_max_size, jump_max_number)
        if is_spiky:
            logger.warning(f"{tr.stats.station}: {n_spikes} discontinuities detected, removing")
            traces_to_remove.append(tr)
            rejected.append(tr.stats.station)
    
    for tr in traces_to_remove:
        stream.remove(tr)
    
    return stream, rejected


def handle_gaps(trace, maxgap=None, fill_value=0.0):
    """
    Handle gaps in trace data.
    
    Parameters
    ----------
    trace : obspy.Trace
        Input trace
    maxgap : int, optional
        Maximum gap size in samples. If exceeded, returns None.
    fill_value : float
        Value to fill gaps with
        
    Returns
    -------
    data : ndarray or None
        Gap-handled data, or None if gap exceeds maxgap
    gap_info : dict
        Information about detected gaps
    """
    data = trace.data
    
    gap_info = {
        'has_gaps': False,
        'total_gap_samples': 0,
        'n_gaps': 0
    }
    
    # Check if data is masked (indicates gaps)
    if hasattr(data, 'mask') and np.any(data.mask):
        gap_info['has_gaps'] = True
        gap_info['total_gap_samples'] = np.sum(data.mask)
        
        # Count separate gap regions
        mask_diff = np.diff(data.mask.astype(int))
        gap_info['n_gaps'] = np.sum(mask_diff == 1)
        
        if maxgap is not None and gap_info['total_gap_samples'] > maxgap:
            logger.warning(f"{trace.stats.station}: gap of {gap_info['total_gap_samples']} "
                         f"samples exceeds maxgap {maxgap}")
            return None, gap_info
        
        # Fill gaps
        data = data.filled(fill_value)
    
    return data, gap_info


def compute_envelope(data, sampling_rate, sta_window, lta_window):
    """
    Compute STA/LTA envelope of data.
    
    Adapted from AELUMA envelope computation.
    
    Parameters
    ----------
    data : ndarray
        Input data (usually bandpass filtered)
    sampling_rate : float
        Sampling rate in Hz
    sta_window : float
        Short-term average window length in seconds
    lta_window : float
        Long-term average window length in seconds
        
    Returns
    -------
    ratio : ndarray
        STA/LTA ratio (envelope)
    """
    # Compute analytic signal using Hilbert transform
    analytic = hilbert(data)
    envelope = np.abs(analytic)
    
    # Compute STA (short-term average)
    sta_samples = int(round(sta_window * sampling_rate))
    sta = smooth(envelope, sta_samples)
    
    # Compute LTA (long-term average) 
    lta_samples = int(round(lta_window * sampling_rate))
    lta = smooth(sta, lta_samples)
    
    # Avoid division by zero
    lta = np.maximum(lta, 1e-10)
    
    ratio = sta / lta
    
    return ratio


def smooth(data, window_size):
    """
    Smooth data using moving average.
    
    Parameters
    ----------
    data : ndarray
        Input data
    window_size : int
        Size of smoothing window in samples
        
    Returns
    -------
    smoothed : ndarray
        Smoothed data
    """
    if window_size < 1:
        return data
    
    return uniform_filter1d(data, size=window_size, mode='nearest')


def decimate_data(data, factor, filter_order=8):
    """
    Decimate data by an integer factor.
    
    Parameters
    ----------
    data : ndarray
        Input data
    factor : int
        Decimation factor
    filter_order : int
        Order of anti-aliasing filter
        
    Returns
    -------
    decimated : ndarray
        Decimated data
    """
    if factor <= 1:
        return data
    
    # Apply low-pass anti-aliasing filter
    nyq = 0.5
    cutoff = 0.8 / factor  # 80% of new Nyquist
    sos = butter(filter_order, cutoff, btype='low', output='sos')
    filtered = sosfiltfilt(sos, data)
    
    # Decimate
    decimated = filtered[::factor]
    
    return decimated


def apply_taper_to_trace(trace, max_percentage=0.05):
    """
    Apply a Hann taper to a single obspy.Trace using numpy (safe across SciPy versions).

    Parameters
    ----------
    trace : obspy.Trace
        Input trace (modified in-place)
    max_percentage : float
        Fraction of trace length used for the taper (default 0.05)
    """
    import numpy as _np

    n = len(trace.data)
    if n <= 0:
        return trace

    win_len = int(round(max_percentage * n))
    if win_len <= 0:
        return trace

    # Build taper: using numpy's hanning for robustness
    w = _np.ones(n, dtype=_np.float64)
    taper = _np.hanning(2 * win_len)
    w[:win_len] = taper[:win_len]
    w[-win_len:] = taper[-win_len:]

    # Ensure float dtype
    if trace.data.dtype.kind == 'i':
        trace.data = trace.data.astype(_np.float64)

    trace.data = trace.data * w
    return trace


def apply_taper_to_stream(stream, max_percentage=0.05):
    """
    Apply taper to each trace in an obspy.Stream.
    """
    for tr in stream:
        apply_taper_to_trace(tr, max_percentage=max_percentage)
    return stream


def compute_decimation_factor(sampling_rate, freq_max, envelope_rate=None):
    """
    Compute optimal decimation factor for data.
    
    Adapted from AELUMA ndec computation.
    
    Parameters
    ----------
    sampling_rate : float
        Original sampling rate in Hz
    freq_max : float
        Maximum frequency of interest in Hz
    envelope_rate : float, optional
        Target envelope sampling rate
        
    Returns
    -------
    ndec : int
        Decimation factor
    """
    # Maximum decimation based on Nyquist
    ndec = int(np.floor(sampling_rate / (2 * freq_max)))
    
    # Ensure ndec is at least 1
    ndec = max(1, ndec)
    
    # If envelope rate specified, ensure it's a factor
    if envelope_rate is not None:
        nmult = sampling_rate / envelope_rate
        # Find largest factor of nmult that is <= ndec
        factors = get_factors(int(nmult))
        valid = [f for f in factors if f <= ndec]
        if valid:
            ndec = max(valid)
    
    return ndec


def get_factors(n):
    """
    Get all factors of an integer.
    
    Parameters
    ----------
    n : int
        Integer to factorize
        
    Returns
    -------
    factors : list
        List of factors
    """
    factors = []
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            factors.append(i)
            if i != n // i:
                factors.append(n // i)
    return sorted(factors)


def preprocess_stream(stream, freqmin, freqmax, 
                      maxgap=None, jump_max_size=None, jump_max_number=None,
                      envelope=False, sta_window=1.0, lta_window=10.0,
                      decimate=None):
    """
    Apply full preprocessing pipeline to a stream.
    
    Parameters
    ----------
    stream : obspy.Stream
        Input stream
    freqmin : float
        Minimum frequency for bandpass
    freqmax : float
        Maximum frequency for bandpass
    maxgap : int, optional
        Maximum gap size in samples
    jump_max_size : float, optional
        Maximum spike size (enables spike rejection)
    jump_max_number : int, optional
        Maximum number of spikes allowed
    envelope : bool
        If True, compute STA/LTA envelope
    sta_window : float
        STA window in seconds
    lta_window : float
        LTA window in seconds
    decimate : int, optional
        Decimation factor
        
    Returns
    -------
    processed : dict
        Dictionary mapping station codes to processed data arrays
    rejected : list
        List of rejected station codes
    """
    processed = {}
    rejected = []
    
    for tr in stream:
        station = tr.stats.station
        data = tr.data.copy()
        fs = tr.stats.sampling_rate
        
        # Handle gaps
        if maxgap is not None:
            data, gap_info = handle_gaps(tr, maxgap=maxgap)
            if data is None:
                rejected.append(station)
                continue
        
        # Check for spikes
        if jump_max_size is not None and jump_max_number is not None:
            is_spiky, n_spikes = detect_spikes(data, jump_max_size, jump_max_number)
            if is_spiky:
                logger.warning(f"{station}: {n_spikes} spikes detected, skipping")
                rejected.append(station)
                continue
        
        # Remove mean
        data = data - np.mean(data)
        
        # Bandpass filter
        data = bandpass_filter(data, fs, freqmin, freqmax)
        
        # Decimate
        if decimate and decimate > 1:
            data = decimate_data(data, decimate)
            fs = fs / decimate
        
        # Compute envelope if requested
        if envelope:
            data = compute_envelope(data, fs, sta_window, lta_window)
        
        processed[station] = {
            'data': data,
            'sampling_rate': fs,
            'starttime': tr.stats.starttime
        }
    
    return processed, rejected
