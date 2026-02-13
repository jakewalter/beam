#!/usr/bin/env python
"""
BEAM Driver - Array-Based Seismic Event Detector

Main entry point for the BEAM package implementing two detection methodologies:

1. CORRELATION MODE (Gibbons & Ringdal 2006):
   - Requires a master event template
   - Uses normalized cross-correlation for detection
   - Best for detecting repeating events similar to the master

2. TRADITIONAL MODE (STA/LTA Beamforming):
   - No master event required
   - Grid search over velocity and backazimuth
   - Uses delay-and-sum beamforming with envelope/characteristic functions
   - Best for detecting new/unknown events

Data access patterns are adapted from AELUMA for consistency.

Usage:
    # Correlation mode (requires master event):
    python beam_driver.py --mode correlation --data-dir /path/to/data \\
        --start 20200101 --end 20200131 --master-time "2020-01-15T12:30:45"
    
    # Traditional beamforming mode (no master event needed):
    python beam_driver.py --mode traditional --data-dir /path/to/data \\
        --start 20200101 --end 20200131
"""

import os
import sys
import glob
import logging
import argparse
from datetime import datetime, timedelta
import numpy as np
import multiprocessing as mp
from functools import partial
from scipy.signal import hilbert

from obspy import read, Stream, UTCDateTime, read_inventory
from obspy.signal.trigger import classic_sta_lta, trigger_onset

# Import BEAM modules
from beam.io.waveform_loader import WaveformLoader, load_day_waveforms
from beam.io.inventory import load_inventory, get_station_coordinates, get_station_coords_dict
from beam.core.preprocessing import (
    preprocess_stream,
    bandpass_filter,
    apply_taper_to_stream,
    decimate_data,
    compute_decimation_factor,
    compute_envelope,
)

# Module-level logger
logger = logging.getLogger(__name__)

# GPU helpers
from beam.core import gpu_beam
from beam.visualization import plot_daily_detections


def apply_gpu_safety_factor(value, logger=logger):
    """Set BEAM_GPU_SAFETY_FACTOR in the environment and update the module
    attribute `GPU_MEMORY_SAFETY_FACTOR` in beam.core.gpu_beam so that
    child processes (spawn/fork) import the correct value during module
    initialization.
    """
    if value is None:
        return
    try:
        os.environ['BEAM_GPU_SAFETY_FACTOR'] = str(int(value))
        import beam.core.gpu_beam as _gpu_mod
        _gpu_mod.GPU_MEMORY_SAFETY_FACTOR = int(value)
        logger.info(f"Set GPU memory safety factor to {_gpu_mod.GPU_MEMORY_SAFETY_FACTOR}")
        print(f"Set GPU memory safety factor to {_gpu_mod.GPU_MEMORY_SAFETY_FACTOR}")
    except Exception as e:
        logger.warning(f"Failed to set global GPU safety factor: {e}")
class BeamArrayDetector:
    """Array-based correlation detector (Gibbons & Ringdal style).

    Lightweight detector focused on a master/template-based correlation
    workflow. Many of the methods below (compute_running_correlation,
    detect_events_single_day, _stack_and_detect, save_detections) are
    instance methods on this class.
    """

    def __init__(self, data_dir, network='*', channel='*Z', location='*',
                 freqmin=1.0, freqmax=10.0, fs_min=None, fs_target=None,
                 min_station_spacing=None, max_station_spacing=None,
                 maxgap=None, jump_max_size=None, jump_max_number=None):
        self.data_dir = data_dir
        self.network = network
        self.channel = channel
        self.location = location
        self.freqmin = freqmin
        self.freqmax = freqmax
        self.fs_min = fs_min
        self.fs_target = fs_target
        self.min_station_spacing = min_station_spacing
        self.max_station_spacing = max_station_spacing
        self.maxgap = maxgap
        self.jump_max_size = jump_max_size
        self.jump_max_number = jump_max_number
        self.location = location
        self.freqmin = freqmin
        self.freqmax = freqmax
        self.fs_min = fs_min
        self.fs_target = fs_target
        self.min_station_spacing = min_station_spacing
        self.max_station_spacing = max_station_spacing
        self.maxgap = maxgap
        self.jump_max_size = jump_max_size
        self.jump_max_number = jump_max_number
        
        # Initialize waveform loader
        self.loader = WaveformLoader(
            data_dir=data_dir,
            network=network,
            channel=channel,
            location=location
        )
        
        # Master event storage
        self.master_waveforms = None
        self.master_time = None
        self.master_duration = None
        
        # Station information
        self.inventory = None
        self.station_coords = None
        self.array_geometry = None
    
    def load_inventory_from_folder(self, folder=None, pattern='*.xml', tag=None, name=None):
        """
        Load station inventory from StationXML files.
        
        Parameters
        ----------
        folder : str, optional
            Folder containing XML files. If None, uses data_dir.
        """
        if folder is None:
            folder = self.data_dir
        
        self.inventory = load_inventory(folder, pattern=pattern, tag=tag, name=name)
        self.station_coords = get_station_coords_dict(self.inventory)
        
        # Compute array geometry
        df = get_station_coordinates(self.inventory)
        # Handle empty inventory gracefully
        if df.empty:
            logger.warning("No station coordinates found in inventory; array geometry not computed")
            self.array_geometry = {}
        else:
            self.array_geometry = compute_array_geometry(
                df['latitude'].values,
                df['longitude'].values,
                df['elevation'].values
            )
        
        logger.info(f"Loaded {len(self.station_coords)} stations")
    
    def load_master_event(self, master_time, duration, date_folder=None, decimate=1):
        """
        Load and prepare master event waveforms.
        
        Parameters
        ----------
        master_time : str or UTCDateTime
            Time of master event
        duration : float
            Duration of master event window in seconds
        date_folder : str, optional
            Specific folder to load from. If None, determines from master_time.
            
        Returns
        -------
        master_waveforms : dict
            Dictionary of master waveforms keyed by station.channel
        """
        if isinstance(master_time, str):
            master_time = UTCDateTime(master_time)
        
        self.master_time = master_time
        self.master_duration = duration
        
        logger.info(f"Loading master event at {master_time}")
        
        # Determine date folder
        if date_folder is None:
            date_str = master_time.strftime('%Y%m%d')
            date_folder = os.path.join(self.data_dir, date_str)
        
        # Load waveforms for the day
        stream = self.loader.load_day(
            master_time,
            starttime=master_time - 60,  # Buffer before
            endtime=master_time + duration + 60  # Buffer after
        )
        
        if len(stream) == 0:
            raise ValueError(f"No data found for master event at {master_time}")
        
        # Check sample rates
        if self.fs_min or self.fs_target:
            stream = self.loader.check_sample_rate(
                stream, 
                fs_min=self.fs_min,
                fs_target=self.fs_target
            )
        
        # Preprocess
        stream.detrend('demean')
        # Use numpy-based taper helper to avoid SciPy.hann compatibility issues
        apply_taper_to_stream(stream, max_percentage=0.05)
        stream.filter('bandpass', freqmin=self.freqmin, freqmax=self.freqmax,
                     corners=4, zerophase=True)
        
        # Trim to master event window
        stream.trim(master_time, master_time + duration)

        # Decimate master waveforms if requested (surface-wave mode)
        if decimate and decimate > 1:
            logger.info(f"Decimating master event waveforms by factor {decimate}")
            for tr in stream:
                # ensure float dtype then decimate
                tr.data = tr.data.astype(np.float64)[::decimate]
                tr.stats.sampling_rate = tr.stats.sampling_rate / decimate
        
        # Store master waveforms
        self.master_waveforms = {}
        for tr in stream:
            key = f"{tr.stats.station}.{tr.stats.channel}"
            self.master_waveforms[key] = tr.copy()
        
        logger.info(f"Loaded {len(self.master_waveforms)} master waveforms")
        return self.master_waveforms
    
    def compute_correlation_coefficient(self, data1, data2):
        """
        Compute fully normalized cross-correlation coefficient (Eq. 3 in G&R 2006).
        
        Parameters
        ----------
        data1 : ndarray
            Reference (master) waveform
        data2 : ndarray
            Test waveform (same length)
            
        Returns
        -------
        corr : float
            Correlation coefficient in range [-1, 1]
        """
        if len(data1) != len(data2):
            raise ValueError("Data arrays must have same length")
        
        inner_product = np.dot(data1, data2)
        norm1 = np.sqrt(np.dot(data1, data1))
        norm2 = np.sqrt(np.dot(data2, data2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return inner_product / (norm1 * norm2)
    
    def compute_running_correlation(self, master_trace, data_trace):
        """
        Compute running correlation coefficient for detection.
        
        Parameters
        ----------
        master_trace : obspy.Trace
            Master event waveform
        data_trace : obspy.Trace
            Continuous data waveform
            
        Returns
        -------
        times : ndarray
            Time array (epoch seconds)
        corr_trace : ndarray
            Correlation coefficient vs time
        scaled_corr : ndarray
            Scaled correlation coefficient (Eq. 6 in G&R 2006)
        """
        master_data = master_trace.data
        data = data_trace.data
        N = len(master_data)
        
        if len(data) < N:
            return None, None, None
        
        # Sliding window correlation
        n_windows = len(data) - N + 1
        corr_trace = np.zeros(n_windows)
        
        # Precompute master norm
        master_norm = np.sqrt(np.dot(master_data, master_data))
        if master_norm == 0:
            return None, None, None
        
        for i in range(n_windows):
            window = data[i:i + N]
            inner_product = np.dot(master_data, window)
            window_norm = np.sqrt(np.dot(window, window))
            
            if window_norm > 0:
                corr_trace[i] = inner_product / (master_norm * window_norm)
        
        # Compute scaled correlation (Eq. 6-8 in G&R 2006)
        scaled_corr = self._compute_scaled_correlation(
            corr_trace,
            a=1.0,  # seconds
            b=2.5,  # seconds
            sampling_rate=data_trace.stats.sampling_rate
        )
        
        # Time array
        times = np.arange(n_windows) / data_trace.stats.sampling_rate
        times += data_trace.stats.starttime.timestamp
        
        return times, corr_trace, scaled_corr
    
    def _compute_scaled_correlation(self, corr_trace, a, b, sampling_rate):
        """
        Compute scaled correlation coefficient (Eq. 6-7 in G&R 2006).
        
        The scaling normalizes by the RMS of correlation values in
        flanking time windows, enhancing detection of coherent signals.
        
        Parameters
        ----------
        corr_trace : ndarray
            Raw correlation trace
        a : float
            Inner exclusion zone in seconds
        b : float
            Outer window boundary in seconds
        sampling_rate : float
            Sampling rate in Hz
            
        Returns
        -------
        scaled : ndarray
            Scaled correlation trace
        """
        n_a = int(a * sampling_rate)
        n_b = int(b * sampling_rate)
        
        scaled = np.zeros_like(corr_trace)
        
        for i in range(len(corr_trace)):
            # Define flanking windows (before and after, excluding center)
            idx_before = range(max(0, i - n_b), max(0, i - n_a))
            idx_after = range(min(len(corr_trace), i + n_a),
                            min(len(corr_trace), i + n_b))
            
            indices = list(idx_before) + list(idx_after)
            
            if len(indices) > 0:
                c_rms = np.sqrt(np.mean(corr_trace[indices]**2))
                if c_rms > 0:
                    scaled[i] = corr_trace[i] / c_rms
        
        return scaled
    
    def detect_events_single_day(self, date_str, detection_threshold=6.0, decimate=1):
        """
        Detect events for a single day using array correlation.
        
        Parameters
        ----------
        date_str : str
            Date in YYYYMMDD format
        detection_threshold : float
            Threshold for scaled array correlation coefficient
            
        Returns
        -------
        detections : list
            List of detection dictionaries
        """
        if self.master_waveforms is None:
            raise ValueError("Master waveforms not loaded. Call load_master_event() first.")
        
        logger.info(f"Processing {date_str}")
        
        # Load waveforms for the day
        stream = self.loader.load_day(date_str)
        
        if len(stream) == 0:
            logger.warning(f"No data found for {date_str}")
            return []
        
        # Check sample rates
        if self.fs_min or self.fs_target:
            stream = self.loader.check_sample_rate(
                stream,
                fs_min=self.fs_min,
                fs_target=self.fs_target
            )
        
        # Merge traces (handle gaps)
        stream.merge(fill_value=0)

        # Optional decimation for long-period / surface-wave mode
        if decimate and decimate > 1:
            logger.info(f"Decimating day data by factor {decimate} before detection for {date_str}")
            for tr in stream:
                tr.data = tr.data.astype(np.float64)[::decimate]
                tr.stats.sampling_rate = tr.stats.sampling_rate / decimate
        
        # Preprocess
        stream.detrend('demean')
        # Use numpy-based taper helper to avoid SciPy.hann compatibility issues
        apply_taper_to_stream(stream, max_percentage=0.05)
        stream.filter('bandpass', freqmin=self.freqmin, freqmax=self.freqmax,
                     corners=4, zerophase=True)
        
        # Compute correlation for each master channel
        correlation_traces = {}
        
        for master_key, master_tr in self.master_waveforms.items():
            station, channel = master_key.split('.')
            
            # Find matching trace
            data_tr = stream.select(station=station, channel=channel)
            
            if len(data_tr) == 0:
                continue
            
            data_tr = data_tr[0]
            
            times, corr, scaled = self.compute_running_correlation(master_tr, data_tr)
            
            if times is not None:
                correlation_traces[master_key] = {
                    'times': times,
                    'correlation': corr,
                    'scaled': scaled,
                    'trace': data_tr
                }
        
        if len(correlation_traces) == 0:
            logger.warning(f"No matching channels found for {date_str}")
            return []
        
        # Stack correlations and detect
        detections = self._stack_and_detect(correlation_traces, detection_threshold)
        
        logger.info(f"{date_str}: Found {len(detections)} detections")
        return detections
    
    def _stack_and_detect(self, correlation_traces, threshold):
        """
        Stack correlation traces across array and detect events.
        
        Parameters
        ----------
        correlation_traces : dict
            Dictionary of correlation trace data
        threshold : float
            Detection threshold for scaled correlation
            
        Returns
        -------
        detections : list
            List of detection dictionaries
        """
        from scipy.interpolate import interp1d
        
        # Find common time window
        all_times = [ct['times'] for ct in correlation_traces.values()]
        start_time = max([t[0] for t in all_times])
        end_time = min([t[-1] for t in all_times])
        
        if start_time >= end_time:
            return []
        
        # Create common time base
        first_key = list(correlation_traces.keys())[0]
        sampling_rate = self.master_waveforms[first_key].stats.sampling_rate
        dt = 1.0 / sampling_rate
        common_times = np.arange(start_time, end_time, dt)
        
        # Stack correlations
        stacked_corr = np.zeros(len(common_times))
        stacked_scaled = np.zeros(len(common_times))
        
        for ct in correlation_traces.values():
            f_corr = interp1d(ct['times'], ct['correlation'],
                            kind='linear', fill_value=0, bounds_error=False)
            f_scaled = interp1d(ct['times'], ct['scaled'],
                              kind='linear', fill_value=0, bounds_error=False)
            
            stacked_corr += f_corr(common_times)
            stacked_scaled += f_scaled(common_times)
        
        # Normalize by number of channels
        n_channels = len(correlation_traces)
        array_corr = stacked_corr / n_channels
        array_scaled = stacked_scaled / n_channels
        
        # Find detections above threshold
        detection_indices = np.where(array_scaled > threshold)[0]
        
        if len(detection_indices) == 0:
            return []
        
        # Group consecutive detections
        groups = self._group_consecutive(detection_indices, gap=10)
        
        detections = []
        for group in groups:
            max_idx = group[np.argmax(array_scaled[group])]
            detection_time = UTCDateTime(common_times[max_idx])
            
            detection = {
                'time': detection_time,
                'array_correlation': array_corr[max_idx],
                'scaled_correlation': array_scaled[max_idx],
                'n_channels': n_channels
            }
            
            detections.append(detection)
        
        return detections
    
    def _group_consecutive(self, indices, gap=10):
        """Group consecutive indices allowing for gaps."""
        if len(indices) == 0:
            return []
        
        groups = []
        current_group = [indices[0]]
        
        for i in range(1, len(indices)):
            if indices[i] - indices[i-1] <= gap:
                current_group.append(indices[i])
            else:
                groups.append(np.array(current_group))
                current_group = [indices[i]]
        
        groups.append(np.array(current_group))
        return groups
    
    def process_date_range(self, start_date, end_date,
                          detection_threshold=6.0, n_processes=None,
                          decimate=1):
        """
        Process multiple days, optionally in parallel.
        
        Parameters
        ----------
        start_date : str or UTCDateTime
            Start date (YYYYMMDD format)
        end_date : str or UTCDateTime
            End date (YYYYMMDD format)
        detection_threshold : float
            Detection threshold
        n_processes : int, optional
            Number of parallel processes. If None, uses all cores.
            If 1, runs sequentially (useful for debugging).
            
        Returns
        -------
        all_detections : dict
            Dictionary with dates as keys and detection lists as values
        """
        if isinstance(start_date, str):
            start_date = UTCDateTime(start_date)
        if isinstance(end_date, str):
            end_date = UTCDateTime(end_date)
        
        # Generate list of dates
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current.strftime('%Y%m%d'))
            current += 86400
        
        logger.info(f"Processing {len(dates)} days")
        
        all_detections = {}
        
        if n_processes == 1:
            # Sequential processing
            for date in dates:
                try:
                    detections = self.detect_events_single_day(
                        date, detection_threshold=detection_threshold, decimate=decimate
                    )
                    if len(detections) > 0:
                        all_detections[date] = detections
                except Exception as e:
                    logger.error(f"Error processing {date}: {e}")
        else:
            # Parallel processing
            n_procs = n_processes or mp.cpu_count()
            logger.info(f"Using {n_procs} processes")
            
            with mp.Pool(processes=n_procs) as pool:
                detect_func = partial(
                    self.detect_events_single_day,
                    detection_threshold=detection_threshold,
                    decimate=decimate
                )
                results = pool.map(detect_func, dates)
            
            for date, detections in zip(dates, results):
                if detections and len(detections) > 0:
                    all_detections[date] = detections
        
        total = sum(len(d) for d in all_detections.values())
        logger.info(f"Total detections: {total}")
        
        return all_detections
    
    def save_detections(self, detections, output_file, min_snr: float = None):
        """
        Save detections to file.
        
        Parameters
        ----------
        detections : dict
            Detection dictionary from process_date_range()
        output_file : str
            Output file path
        """
        # Ensure target directory exists
        out_dir = os.path.dirname(output_file)
        if out_dir:
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception:
                logger.warning(f"Could not create output directory {out_dir}; will try writing directly")

        with open(output_file, 'w') as f:
            f.write("# BEAM Array Correlation Detections\n")
            f.write("# Time, Array_Corr, Scaled_Corr, N_Channels\n")
            
            for date in sorted(detections.keys()):
                for det in detections[date]:
                    # allow optional filtering by SNR when saving
                    if min_snr is not None and 'snr' in det and det['snr'] < min_snr:
                        continue
                    f.write(f"{det['time']}, {det['array_correlation']:.4f}, "
                           f"{det['scaled_correlation']:.4f}, {det['n_channels']}\n")
        
        logger.info(f"Saved detections to {output_file}")


class TraditionalBeamformer:
    """
    Traditional STA/LTA beamforming detector for continuous seismic data.
    
    Does NOT require a master event template. Instead, performs:
    - Grid search over velocity and backazimuth
    - Delay-and-sum beamforming
    - STA/LTA detection on beamed traces
    - Optional envelope/characteristic function beamforming
    
    Parameters
    ----------
    data_dir : str
        Base directory containing YYYYMMDD folders with miniseed files
    network : str
        Network code(s) to process
    channel : str
        Channel code(s) to process
    location : str
        Location code(s) to process
    freqmin : float
        Minimum frequency for bandpass filter (Hz)
    freqmax : float
        Maximum frequency for bandpass filter (Hz)
    """
    
    def __init__(self, data_dir, network='*', channel='*Z', location='*',
                 freqmin=1.0, freqmax=10.0, use_gpu=False):
        
        self.data_dir = data_dir
        self.network = network
        self.channel = channel
        self.location = location
        self.freqmin = freqmin
        self.freqmax = freqmax
        
        # Initialize waveform loader
        self.loader = WaveformLoader(
            data_dir=data_dir,
            network=network,
            channel=channel,
            location=location
        )
        
        # Station information
        self.inventory = None
        self.station_coords = None
        self.array_geometry = None
        # optional GPU acceleration
        self.use_gpu = use_gpu
    
    def load_inventory_from_folder(self, folder=None, pattern='*.xml', tag=None, name=None):
        """Load station inventory from StationXML files."""
        if folder is None:
            folder = self.data_dir
        
        self.inventory = load_inventory(folder, pattern=pattern, tag=tag, name=name)
        self.station_coords = get_station_coords_dict(self.inventory)
        
        # Compute array geometry in local coordinates
        self._compute_array_geometry()

        # Partition inventory into subarrays (groups of ~7 stations)
        try:
            codes = list(self.station_coords.keys())
            lats = [c[0] for c in self.station_coords.values()]
            lons = [c[1] for c in self.station_coords.values()]
            self.subarrays = partition_into_subarrays(codes, lats, lons, group_size=7)
            # build mapping index -> stations
            self.subarray_geometries = {}
            for i, group in enumerate(self.subarrays):
                # compute per-subarray geometry dict mapping station->(x,y,elev)
                lat_vals = np.array([self.station_coords[s][0] for s in group])
                lon_vals = np.array([self.station_coords[s][1] for s in group])
                elev_vals = np.array([self.station_coords[s][2] for s in group])

                center_lat = np.mean(lat_vals)
                center_lon = np.mean(lon_vals)

                geom = {}
                for sta, lat, lon, elev in zip(group, lat_vals, lon_vals, elev_vals):
                    x = (lon - center_lon) * 111.32 * np.cos(np.radians(center_lat))
                    y = (lat - center_lat) * 110.54
                    geom[sta] = (x, y, elev)

                self.subarray_geometries[i] = geom
        except Exception:
            # If partitioning fails, fall back to single array
            self.subarrays = [list(self.station_coords.keys())]
            self.subarray_geometries = {0: self.array_geometry}
        
        logger.info(f"Loaded {len(self.station_coords)} stations")
    
    def set_station_coords(self, station_coords):
        """
        Set station coordinates directly.
        
        Parameters
        ----------
        station_coords : dict
            Dictionary with station codes as keys and (lat, lon, elev) tuples
        """
        self.station_coords = station_coords
        self._compute_array_geometry()

    def set_subarrays(self, subarray_groups):
        """
        Set explicit subarray groupings.

        Parameters
        ----------
        subarray_groups : list of list of str
            Each sub-list contains station codes for one subarray.
        """
        if not self.station_coords:
            raise RuntimeError("station_coords not set; load inventory first")

        # Validate and map station names, with a case-insensitive fallback
        codes = set(self.station_coords.keys())
        codes_lower_map = {c.lower(): c for c in codes}
        remapped_groups = []
        for g in subarray_groups:
            remapped = []
            for s in g:
                if s in codes:
                    remapped.append(s)
                    continue
                # Try case-insensitive match
                s_stripped = s.strip()
                sl = s_stripped.lower()
                if sl in codes_lower_map:
                    real = codes_lower_map[sl]
                    remapped.append(real)
                    continue
                # Try stripping spaces and dashes
                s2 = s_stripped.replace('-', '').replace('_', '')
                candidates = [c for c in codes if c.replace('-', '').replace('_', '').lower() == s2.lower()]
                if len(candidates) == 1:
                    remapped.append(candidates[0])
                    continue
                # Not found - raise with helpful suggestion
                suggestions = [c for c in codes if c.lower().startswith(sl[:2])]
                raise ValueError(
                    f"Station '{s}' in subarray group not found in station_coords. "
                    f"Available station count: {len(codes)}. Suggestions: {suggestions[:5]}"
                )
            remapped_groups.append(remapped)
        subarray_groups = remapped_groups

        # Set subarrays and rebuild geometries
        self.subarrays = [list(g) for g in subarray_groups]
        self.subarray_geometries = {}
        for i, group in enumerate(self.subarrays):
            lat_vals = [self.station_coords[s][0] for s in group]
            lon_vals = [self.station_coords[s][1] for s in group]
            elev_vals = [self.station_coords[s][2] for s in group]

            center_lat = float(np.mean(lat_vals))
            center_lon = float(np.mean(lon_vals))

            geom = {}
            for sta, lat, lon, elev in zip(group, lat_vals, lon_vals, elev_vals):
                x = (lon - center_lon) * 111.32 * np.cos(np.radians(center_lat))
                y = (lat - center_lat) * 110.54
                geom[sta] = (x, y, elev)

            self.subarray_geometries[i] = geom
        return self.subarrays

    def force_subarrays(self, n_subarrays: int):
        """Force splitting stations into exactly n_subarrays groups.

        This is a simple even-split that distributes station codes in the
        original order across N groups. Results are stored in `self.subarrays`
        and `self.subarray_geometries` is recomputed.
        """
        if not self.station_coords:
            raise RuntimeError("station_coords not set; load inventory first")

        codes = list(self.station_coords.keys())
        n_stations = len(codes)
        n = max(1, min(int(n_subarrays), n_stations))

        groups = [[] for _ in range(n)]
        for i, code in enumerate(codes):
            groups[i % n].append(code)

        # drop empty groups (if any) and set subarrays
        self.subarrays = [g for g in groups if len(g) > 0]

        # rebuild subarray_geometries
        self.subarray_geometries = {}
        for i, group in enumerate(self.subarrays):
            lat_vals = [self.station_coords[s][0] for s in group]
            lon_vals = [self.station_coords[s][1] for s in group]
            elev_vals = [self.station_coords[s][2] for s in group]

            center_lat = float(np.mean(lat_vals))
            center_lon = float(np.mean(lon_vals))

            geom = {}
            for sta, lat, lon, elev in zip(group, lat_vals, lon_vals, elev_vals):
                x = (lon - center_lon) * 111.32 * np.cos(np.radians(center_lat))
                y = (lat - center_lat) * 110.54
                geom[sta] = (x, y, elev)

            self.subarray_geometries[i] = geom

        return self.subarrays
    
    def _compute_array_geometry(self):
        """Compute array geometry in km relative to center."""
        if not self.station_coords:
            return
        
        lats = np.array([c[0] for c in self.station_coords.values()])
        lons = np.array([c[1] for c in self.station_coords.values()])
        
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)
        
        self.array_geometry = {}
        for sta, (lat, lon, elev) in self.station_coords.items():
            x = (lon - center_lon) * 111.32 * np.cos(np.radians(center_lat))
            y = (lat - center_lat) * 110.54
            self.array_geometry[sta] = (x, y, elev)
        
        self.center_lat = center_lat
        self.center_lon = center_lon
    
    def load_day_data(self, date_str):
        """
        Load and preprocess data for a single day.
        
        Parameters
        ----------
        date_str : str
            Date in YYYYMMDD format
            
        Returns
        -------
        stream : obspy.Stream or None
            Preprocessed stream
        """
        stream = self.loader.load_day(date_str)
        
        if len(stream) == 0:
            return None
        
        # Preprocess
        stream.detrend('demean')
        stream.detrend('linear')
        # Use numpy-based taper helper to avoid SciPy.hann compatibility issues
        apply_taper_to_stream(stream, max_percentage=0.05)
        stream.filter('bandpass', freqmin=self.freqmin, freqmax=self.freqmax,
                     corners=4, zerophase=False)
        stream.merge(fill_value=0)
        
        # Resample to common sampling rate if needed
        if len(stream) > 0:
            target_sr = stream[0].stats.sampling_rate
            for tr in stream:
                if tr.stats.sampling_rate != target_sr:
                    tr.resample(target_sr)
        
        return stream
    
    def compute_delays(self, slowness, backazimuth):
        """
        Compute time delays for beamforming.
        
        Parameters
        ----------
        slowness : float
            Slowness in s/km
        backazimuth : float
            Backazimuth in degrees (direction TO source)
            
        Returns
        -------
        delays : dict
            Station delays in seconds
        """
        az_rad = np.radians(backazimuth)
        
        # Slowness vector components
        sx = slowness * np.sin(az_rad)
        sy = slowness * np.cos(az_rad)
        
        delays = {}
        for sta, (x, y, _) in self.array_geometry.items():
            delays[sta] = sx * x + sy * y
        
        return delays
    
    def compute_envelope(self, data):
        """
        Compute envelope using Hilbert transform.
        
        Parameters
        ----------
        data : ndarray
            Input time series
            
        Returns
        -------
        envelope : ndarray
            Envelope (instantaneous amplitude)
        """
        analytic_signal = hilbert(data)
        return np.abs(analytic_signal)
    
    def compute_characteristic_function(self, trace, method='envelope'):
        """
        Compute characteristic function for detection.
        
        Parameters
        ----------
        trace : obspy.Trace
            Input trace
        method : str
            'envelope' - Hilbert envelope (good for all events)
            'energy' - Instantaneous energy (data^2)
            'kurtosis' - Kurtosis-based CF (good for impulsive events)
            
        Returns
        -------
        cf_trace : obspy.Trace
            Characteristic function trace
        """
        tr = trace.copy()
        
        if method == 'envelope':
            cf_data = self.compute_envelope(tr.data)
        elif method == 'energy':
            cf_data = tr.data ** 2
        elif method == 'kurtosis':
            window = 50
            cf_data = np.zeros(len(tr.data))
            for i in range(window, len(tr.data)):
                segment = tr.data[i-window:i]
                cf_data[i] = np.mean(segment**4) / (np.mean(segment**2)**2 + 1e-10)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        tr.data = cf_data
        tr.stats.channel = f"CF_{method.upper()}"
        return tr
    
    def beamform(self, stream, slowness, backazimuth, normalize=True,
                 use_envelope=True, cf_method='envelope'):
        """
        Perform delay-and-sum beamforming.
        
        Parameters
        ----------
        stream : obspy.Stream
            Input waveform data
        slowness : float
            Slowness in s/km
        backazimuth : float
            Backazimuth in degrees
        normalize : bool
            Whether to normalize by number of traces
        use_envelope : bool
            If True, beamform the envelope/characteristic function
        cf_method : str
            Characteristic function method: 'envelope', 'energy', 'kurtosis'
            
        Returns
        -------
        beam_trace : obspy.Trace
            Beamformed trace
        """
        if len(stream) == 0:
            return None
        
        delays = self.compute_delays(slowness, backazimuth)
        
        # Convert to characteristic function if requested
        if use_envelope:
            stream_cf = Stream()
            for tr in stream:
                cf_tr = self.compute_characteristic_function(tr, method=cf_method)
                stream_cf.append(cf_tr)
            stream_to_beam = stream_cf
        else:
            stream_to_beam = stream
        
        # Find common time window
        start_times = [tr.stats.starttime for tr in stream_to_beam]
        end_times = [tr.stats.endtime for tr in stream_to_beam]
        common_start = max(start_times)
        common_end = min(end_times)
        
        if common_start >= common_end:
            return None
        
        # Initialize beam
        reference_tr = stream_to_beam[0].copy()
        reference_tr.trim(common_start, common_end)
        beam_data = np.zeros(len(reference_tr.data))
        n_traces = 0
        
        # Sum delayed traces
        for tr in stream_to_beam:
            sta = tr.stats.station
            
            if sta not in delays:
                logger.debug(f"Station '{sta}' not in delays mapping; skipping")
                continue
            
            delay = delays[sta]
            # Trim using the unshifted common window adjusted for station delay
            tr_shifted = tr.copy()
            tr_shifted.trim(common_start - delay, common_end - delay)
            
            # Allow small length mismatches by taking the minimum overlapping
            # length after trimming. This handles minor rounding or off-by-one
            # sample differences across traces.
            min_len = min(len(tr_shifted.data), len(beam_data))
            if min_len > 0:
                beam_data[:min_len] += tr_shifted.data[:min_len]
                n_traces += 1
        
        if n_traces == 0:
            logger.debug('No traces included in beam sum after delay alignment')
            return None
        else:
            logger.debug(f"Beam assembled: n_traces={n_traces}, beam_len={len(beam_data)}, slowness={slowness}, backazimuth={backazimuth}")
            return None
        
        if normalize:
            beam_data /= n_traces
        
        beam_trace = reference_tr.copy()
        beam_trace.data = beam_data
        beam_trace.stats.station = 'BEAM'
        beam_trace.stats.channel = f'BEAM_{cf_method.upper()}' if use_envelope else 'BEAM'
        
        return beam_trace
    
    def sta_lta_detect(self, trace, sta_len=1.0, lta_len=30.0,
                       threshold=3.0, threshold_off=1.5):
        """
        STA/LTA detector on a trace.
        
        Parameters
        ----------
        trace : obspy.Trace
            Input trace
        sta_len : float
            Short-term average window (seconds)
        lta_len : float
            Long-term average window (seconds)
        threshold : float
            Detection threshold (STA/LTA ratio)
        threshold_off : float
            Threshold for end of detection
            
        Returns
        -------
        detections : list
            List of (trigger_time, detrigger_time) tuples
        sta_lta : ndarray
            STA/LTA ratio time series
        """
        sr = trace.stats.sampling_rate
        sta_samples = int(sta_len * sr)
        lta_samples = int(lta_len * sr)
        
        sta_lta = classic_sta_lta(trace.data, sta_samples, lta_samples)
        triggers = trigger_onset(sta_lta, threshold, threshold_off)
        
        detections = []
        for on, off in triggers:
            trigger_time = trace.stats.starttime + on / sr
            detrigger_time = trace.stats.starttime + off / sr
            detections.append((trigger_time, detrigger_time))
        
        return detections, sta_lta
    
    def grid_search_detection(self, stream, date_str,
                              velocity_range=(3.0, 8.0),
                              azimuth_range=(0, 360),
                              velocity_step=0.5,
                              azimuth_step=10,
                              sta_len=1.0, lta_len=30.0,
                              threshold=3.0,
                              use_envelope=True,
                              cf_method='envelope',
                              max_beams=None,
                              decimate=1,
                              progress=True):
        """
        Perform grid search over velocity and backazimuth space.
        
        Parameters
        ----------
        stream : obspy.Stream
            Input data stream
        date_str : str
            Date being processed
        velocity_range : tuple
            (min, max) apparent velocity in km/s
        azimuth_range : tuple
            (min, max) backazimuth in degrees
        velocity_step : float
            Grid spacing for velocity
        azimuth_step : float
            Grid spacing for azimuth
        sta_len, lta_len : float
            STA/LTA window lengths
        threshold : float
            Detection threshold
        use_envelope : bool
            If True, beamform envelope
        cf_method : str
            Characteristic function method
            
        Returns
        -------
        detections : list
            List of detection dictionaries
        """
        import time as time_module
        
        # Optional decimation to dramatically speed up heavy full-day runs
        if decimate and decimate > 1:
            logger.info(f"Decimating data by factor {decimate} to speed-up grid search")
            stream = stream.copy()
            for tr in stream:
                tr.data = tr.data[::decimate]
                tr.stats.sampling_rate = tr.stats.sampling_rate / decimate

        velocities = np.arange(velocity_range[0], velocity_range[1], velocity_step)
        azimuths = np.arange(azimuth_range[0], azimuth_range[1], azimuth_step)

        n_beams = len(velocities) * len(azimuths)
        logger.info(f"Grid search: {len(velocities)} velocities × {len(azimuths)} azimuths = {n_beams} beams")
        
        all_detections = []
        beam_count = 0
        start_time = time_module.time()
        
        # Pre-compute characteristic functions ONCE (major optimization)
        if use_envelope:
            logger.info("Pre-computing characteristic functions...")
            stream_cf = Stream()
            for tr in stream:
                cf_tr = self.compute_characteristic_function(tr, method=cf_method)
                stream_cf.append(cf_tr)
            logger.info(f"  Done. {len(stream_cf)} traces pre-processed.")
        else:
            stream_cf = stream
        
        # Track progress so we can show ETA and optionally abort early
        t0 = None
        beam_count = 0

        # When GPU is available, do batched multi-beam generation to avoid
        # repeated FFTs per-beam. We collect delays and beam params then
        # call the GPU batch beamformer.
        use_gpu_batch = getattr(self, 'use_gpu', False) and gpu_beam.CUPY_AVAILABLE
        batch_size = 32
        
        # If GPU doesn't have enough memory, it will fall back to CPU.
        # CPU beamforming needs much less memory per beam but can't handle
        # large batches. Detect low GPU memory and reduce batch size preemptively.
        if use_gpu_batch:
            try:
                import cupy as cp
                free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                # Reduce batch size if GPU memory is limited
                # With full-day data (~8.64M samples), even "plenty" of GPU memory
                # can run out after loading data due to temporary allocations
                if free_mem < 10 * 1024**3:  # Less than 10 GB
                    batch_size = 8  # Much smaller batches
                    logger.info(f"Limited GPU memory ({free_mem / (1024**3):.2f} GB free), using batch_size={batch_size}")
            except Exception:
                pass

        # If using GPU, precompute a common trimmed data array for the stream
        if use_gpu_batch:
            # find common time window across stream_cf
            start_times = [tr.stats.starttime for tr in stream_cf]
            end_times = [tr.stats.endtime for tr in stream_cf]
            common_start = max(start_times)
            common_end = min(end_times)

            if common_start >= common_end:
                logger.debug("Common data window empty for GPU batching; disabling GPU batching")
                use_gpu_batch = False

        # prepare batch containers
        delays_batch = []
        meta_batch = []

        for i_vel, vel in enumerate(velocities):
            slowness = 1.0 / vel
            vel_start = time_module.time()
            vel_detections = 0
            
            for az in azimuths:
                beam_count += 1

                # Respect optional beam-limit for debugging or quick tests
                if max_beams is not None and beam_count > max_beams:
                    logger.info(f"Reached max_beams={max_beams}; stopping grid search early")
                    break

                # Print progress periodically so long runs are visible
                if progress and (beam_count == 1 or beam_count % 50 == 0):
                    if t0 is None:
                        t0 = time_module.time()
                        elapsed = 0.0
                    else:
                        elapsed = time_module.time() - t0
                    avg = elapsed / beam_count if beam_count else 0.0
                    remaining = max(0, n_beams - beam_count)
                    eta = remaining * avg
                    logger.info(f"Beam {beam_count}/{n_beams} avg {avg:.3f}s/beam ETA {eta/60:.1f} min")
                # If we have a GPU and batch-mode enabled, accumulate this
                # beam's delays and only beamform in batches. Otherwise
                # fall back to the per-beam CPU/stream-based path.
                if use_gpu_batch:
                    # build delays array ordered to the stream_cf station order
                    # use the same ordering across batch
                    # get list of stations and trimmed trace arrays
                    reference_tr = stream_cf[0].copy()
                    reference_tr.trim(common_start, common_end)
                    n_samples = len(reference_tr.data)

                    traces_for_beam = []
                    station_order = []
                    for tr in stream_cf:
                        tcopy = tr.copy()
                        tcopy.trim(common_start, common_end)
                        if len(tcopy.data) == n_samples:
                            traces_for_beam.append(tcopy.data)
                            station_order.append(tcopy.stats.station)

                    if len(traces_for_beam) == 0:
                        # nothing to beam on
                        continue

                    # compute station delays for this beam in seconds
                    delays_this = np.array([self.compute_delays(slowness, az).get(sta, 0.0) for sta in station_order], dtype=np.float32)
                    delays_batch.append(delays_this)
                    meta_batch.append((vel, slowness, az, reference_tr, station_order))

                    # when batch is full (or last beam reached), process
                    if len(delays_batch) >= batch_size or (i_vel == len(velocities)-1 and az == azimuths[-1]):
                        # stack traces array
                        traces_arr = np.vstack(traces_for_beam).astype(np.float32)
                        try:
                            # ask for the device-array output so we can keep processing
                            # on the GPU (CF + STA/LTA) and avoid unnecessary host
                            # transfers for every beam.
                            res = gpu_beam.beamform_freq_domain(
                                traces_arr, np.vstack(delays_batch), reference_tr.stats.sampling_rate, use_gpu=True, return_device_array=True
                            )
                            if res is None:
                                raise RuntimeError("GPU beamformer returned no result (None)")
                            beams_dev, freqs = res
                        except Exception as e:
                            # GPU batch failed or couldn't provide device arrays;
                            # process this batch on CPU instead of skipping it.
                            logger.warning(f"GPU batch beamforming failed; running CPU fallback for this batch: {e}")
                            try:
                                beams, freqs = gpu_beam.beamform_freq_domain(
                                    traces_arr, np.vstack(delays_batch), reference_tr.stats.sampling_rate,
                                    use_gpu=False, return_device_array=False
                                )
                            except Exception as e2:
                                logger.error(f"CPU batch processing also failed: {e2}")
                                # clear batch and continue to avoid infinite loop
                                delays_batch = []
                                meta_batch = []
                                continue

                            # process each output beam (CPU arrays)
                            xp = np
                            if use_envelope:
                                env = gpu_beam.compute_envelope(beams, use_gpu=False)
                            else:
                                env = np.abs(beams)

                            sr = reference_tr.stats.sampling_rate
                            sta_samples = max(1, int(sta_len * sr))
                            lta_samples = max(1, int(lta_len * sr))
                            ratio = gpu_beam.classic_sta_lta(env, sta_samples, lta_samples, use_gpu=False)

                            n_batch_beams = int(ratio.shape[0])
                            for bidx in range(n_batch_beams):
                                vel_b, sl_b, az_b, ref_tr_b, station_order_b = meta_batch[bidx]

                                mask = ratio[bidx] > threshold
                                ratio_row = ratio[bidx]

                                if not np.any(mask):
                                    continue

                                padded = np.concatenate([[0], mask.astype(int), [0]])
                                diffs = np.diff(padded)
                                starts = np.where(diffs == 1)[0]
                                ends = np.where(diffs == -1)[0]

                                for st, ed in zip(starts, ends):
                                    idx_on = st
                                    idx_off = ed
                                    max_idx = idx_on + np.argmax(ratio_row[idx_on:idx_off])
                                    max_snr = float(ratio_row[max_idx])
                                    max_time = ref_tr_b.stats.starttime + max_idx / ref_tr_b.stats.sampling_rate

                                    detection = {
                                        'time': max_time,
                                        'date': date_str,
                                        'velocity': vel_b,
                                        'slowness': sl_b,
                                        'backazimuth': az_b,
                                        'snr': max_snr,
                                        'duration': (idx_off - idx_on) / ref_tr_b.stats.sampling_rate,
                                        'freqmin': self.freqmin,
                                        'freqmax': self.freqmax,
                                        'used_envelope': use_envelope,
                                        'cf_method': cf_method if use_envelope else 'none'
                                    }
                                    all_detections.append(detection)
                                    vel_detections += 1

                            # clear batches
                            delays_batch = []
                            meta_batch = []
                            # continue to next beam
                            continue

                        # process each output beam
                        # GPU-path: compute CF and STA/LTA on the device so we
                        # only transfer trigger indices (small) back to host.
                        try:
                            xp = gpu_beam.cp if (gpu_beam.CUPY_AVAILABLE) else np
                        except Exception:
                            xp = np

                        # compute characteristic function (envelope) on device
                        if use_envelope:
                            env_dev = gpu_beam.compute_envelope(beams_dev, use_gpu=True)
                        else:
                            # if not using envelope, operate on absolute signal
                            env_dev = xp.abs(beams_dev)

                        # compute STA/LTA ratio on device, result aligned to n_samples
                        sr = reference_tr.stats.sampling_rate
                        sta_samples = max(1, int(sta_len * sr))
                        lta_samples = max(1, int(lta_len * sr))
                        # use classic STA/LTA implementation to match ObsPy's
                        # behavior for apples-to-apples comparison
                        ratio_dev = gpu_beam.classic_sta_lta(env_dev, sta_samples, lta_samples, use_gpu=True)

                        # process each beam in the batch, finding triggers on device
                        n_batch_beams = int(ratio_dev.shape[0])
                        for bidx in range(n_batch_beams):
                            vel_b, sl_b, az_b, ref_tr_b, station_order_b = meta_batch[bidx]

                            # mask where SNR ratio exceeds threshold
                            mask_dev = ratio_dev[bidx] > threshold

                            # move this small boolean mask to host for simple edge detection
                            if gpu_beam.CUPY_AVAILABLE and hasattr(gpu_beam, 'cp') and isinstance(mask_dev, gpu_beam.cp.ndarray):
                                mask = mask_dev.get()
                                ratio_row = ratio_dev[bidx].get()
                            else:
                                mask = mask_dev
                                ratio_row = ratio_dev[bidx]

                            if not np.any(mask):
                                continue

                            # find start/end sample indices for contiguous True regions
                            padded = np.concatenate([[0], mask.astype(int), [0]])
                            diffs = np.diff(padded)
                            starts = np.where(diffs == 1)[0]
                            ends = np.where(diffs == -1)[0]

                            for s_idx, e_idx in zip(starts, ends):
                                # convert sample indices to times
                                idx_on = int(s_idx)
                                idx_off = int(e_idx)
                                if idx_on >= idx_off:
                                    continue
                                # pick the maximum ratio sample inside the trigger window
                                max_idx = idx_on + int(np.argmax(ratio_row[idx_on:idx_off]))
                                max_snr = float(ratio_row[max_idx])
                                max_time = ref_tr_b.stats.starttime + max_idx / ref_tr_b.stats.sampling_rate
                                trigger_on_time = ref_tr_b.stats.starttime + idx_on / ref_tr_b.stats.sampling_rate
                                trigger_off_time = ref_tr_b.stats.starttime + idx_off / ref_tr_b.stats.sampling_rate
                                detection = {
                                    'time': max_time,
                                    'date': date_str,
                                    'velocity': vel_b,
                                    'slowness': sl_b,
                                    'backazimuth': az_b,
                                    'snr': max_snr,
                                    'duration': trigger_off_time - trigger_on_time,
                                    'freqmin': self.freqmin,
                                    'freqmax': self.freqmax,
                                    'used_envelope': use_envelope,
                                    'cf_method': cf_method if use_envelope else 'none'
                                }
                                all_detections.append(detection)

                        # clear batches
                        delays_batch = []
                        meta_batch = []
                    # continue to next beam
                    continue

                # (fallback) per-beam CPU-based beamforming
                beam = self._beamform_precomputed(stream_cf, slowness, az)
                
                if beam is None:
                    continue
                
                triggers, sta_lta = self.sta_lta_detect(
                    beam, sta_len, lta_len, threshold, threshold * 0.5
                )
                
                for trigger_on, trigger_off in triggers:
                    idx_on = int((trigger_on - beam.stats.starttime) * beam.stats.sampling_rate)
                    idx_off = int((trigger_off - beam.stats.starttime) * beam.stats.sampling_rate)
                    idx_on = max(0, idx_on)
                    idx_off = min(len(sta_lta), idx_off)
                    
                    if idx_on >= idx_off:
                        continue
                    
                    max_idx = idx_on + np.argmax(sta_lta[idx_on:idx_off])
                    max_snr = sta_lta[max_idx]
                    max_time = beam.stats.starttime + max_idx / beam.stats.sampling_rate
                    
                    detection = {
                        'time': max_time,
                        'date': date_str,
                        'velocity': vel,
                        'slowness': slowness,
                        'backazimuth': az,
                        'snr': max_snr,
                        'duration': trigger_off - trigger_on,
                        'freqmin': self.freqmin,
                        'freqmax': self.freqmax,
                        'used_envelope': use_envelope,
                        'cf_method': cf_method if use_envelope else 'none'
                    }
                    all_detections.append(detection)
                    vel_detections += 1
            
            vel_elapsed = time_module.time() - vel_start
            total_elapsed = time_module.time() - start_time
            pct_done = 100.0 * (i_vel + 1) / len(velocities)
            logger.info(f"  Velocity {vel:.1f} km/s: {len(azimuths)} azimuths in {vel_elapsed:.1f}s, "
                       f"{vel_detections} triggers, {pct_done:.0f}% complete")
        
        total_time = time_module.time() - start_time
        logger.info(f"Grid search complete: {n_beams} beams in {total_time:.1f}s ({n_beams/total_time:.1f} beams/s)")
        
        return all_detections
    
    def _beamform_precomputed(self, stream_cf, slowness, backazimuth, normalize=True):
        """
        Beamform pre-computed characteristic function stream.
        
        This is an optimized version of beamform() that skips the CF computation.
        """
        if len(stream_cf) == 0:
            logger.debug('No traces in stream_cf - returning None')
            return None
        
        delays = self.compute_delays(slowness, backazimuth)
        
        # Find common time window
        start_times = [tr.stats.starttime for tr in stream_cf]
        end_times = [tr.stats.endtime for tr in stream_cf]
        common_start = max(start_times)
        common_end = min(end_times)
        
        if common_start >= common_end:
            logger.debug(f'No common window before shifting: common_start={common_start}, common_end={common_end}')
            return None
        
        # Optional GPU accelerated frequency-domain beamforming path.
        if getattr(self, 'use_gpu', False) and gpu_beam.CUPY_AVAILABLE:
            # Build ordered arrays for traces and station codes
            sta_names = [tr.stats.station for tr in stream_cf]
            data_list = [tr.data for tr in stream_cf]
            # Ensure common length/time window
            reference_tr = stream_cf[0].copy()
            reference_tr.trim(common_start, common_end)
            n_samples = len(reference_tr.data)
            # Ensure all traces are trimmed to the same window
            traces = []
            valid_stas = []
            for tr in stream_cf:
                tcopy = tr.copy()
                tcopy.trim(common_start, common_end)
                if len(tcopy.data) == n_samples:
                    traces.append(tcopy.data)
                    valid_stas.append(tr.stats.station)

            if len(traces) == 0:
                logger.debug('GPU path: no traces after trimming to n_samples')
                return None

            traces_arr = np.vstack(traces).astype(np.float32)

            # Build delays array in seconds for this beam, ordered by valid_stas
            delays_dict = delays
            delays_arr = np.array([delays_dict.get(sta, 0.0) for sta in valid_stas], dtype=np.float32)[None, :]

            # Run GPU frequency-domain beamformer (will fall back to CuPy internal)
            res = gpu_beam.beamform_freq_domain(traces_arr, delays_arr, reference_tr.stats.sampling_rate, use_gpu=self.use_gpu)
            if res is None:
                raise RuntimeError("beamform_freq_domain returned no result (None)")
            beams, freqs = res

            beam_data = beams[0]
            n_traces = len(traces)
            if normalize and n_traces > 0:
                beam_data = beam_data / n_traces

            beam_trace = reference_tr.copy()
            beam_trace.data = beam_data
            beam_trace.stats.station = 'BEAM'

            return beam_trace

        # For per-beam CPU path, compute shifted common window based on delays
        # so that after trimming each trace with its per-station delay we get
        # equal-length arrays for beamforming.
        shifted_starts = [tr.stats.starttime - delays.get(tr.stats.station, 0.0) for tr in stream_cf]
        shifted_ends = [tr.stats.endtime - delays.get(tr.stats.station, 0.0) for tr in stream_cf]
        common_shifted_start = max(shifted_starts)
        common_shifted_end = min(shifted_ends)
        if common_shifted_start >= common_shifted_end:
            logger.debug(f'No common shifted window: start={common_shifted_start}, end={common_shifted_end}')
            return None

        # Use reference station to establish sample count after shift
        ref_sta = stream_cf[0].stats.station
        ref_delay = delays.get(ref_sta, 0.0)
        reference_tr = stream_cf[0].copy()
        reference_tr.trim(common_shifted_start + ref_delay, common_shifted_end + ref_delay)
        beam_data = np.zeros(len(reference_tr.data))
        n_traces = 0
        
        # Sum delayed traces
        for tr in stream_cf:
            sta = tr.stats.station
            
            if sta not in delays:
                continue
            
            delay = delays[sta]
            tr_shifted = tr.copy()
            # Trim using the shifted common window so that the delay
            # applied to each trace lines up with the reference window.
            tr_shifted.trim(common_shifted_start + delay, common_shifted_end + delay)

            # Allow small length mismatches by taking the minimum overlap length
            min_len = min(len(tr_shifted.data), len(beam_data))
            if min_len > 0:
                beam_data[:min_len] += tr_shifted.data[:min_len]
                n_traces += 1
        
        if n_traces == 0:
            return None
        
        if normalize:
            beam_data /= n_traces
        
        beam_trace = reference_tr.copy()
        beam_trace.data = beam_data
        beam_trace.stats.station = 'BEAM'
        
        return beam_trace

    
    def fk_analysis(self, stream, time_window, window_length=3.0,
                    slowness_range=0.3, slowness_step=0.01):
        """
        Frequency-Wavenumber (FK) analysis for a time window.
        
        Parameters
        ----------
        stream : obspy.Stream
            Input data
        time_window : UTCDateTime
            Center time for analysis
        window_length : float
            Length of window in seconds
        slowness_range : float
            Maximum slowness magnitude (s/km)
        slowness_step : float
            Grid spacing for slowness
            
        Returns
        -------
        result : dict
            FK analysis results
        """
        st = stream.copy()
        t1 = time_window - window_length / 2
        t2 = time_window + window_length / 2
        st.trim(t1, t2)
        
        if len(st) < 3:
            return None
        
        sx_range = np.arange(-slowness_range, slowness_range, slowness_step)
        sy_range = np.arange(-slowness_range, slowness_range, slowness_step)
        
        max_power = -np.inf
        best_sx, best_sy = 0, 0
        
        for sx in sx_range:
            for sy in sy_range:
                slowness = np.sqrt(sx**2 + sy**2)
                
                if slowness > slowness_range:
                    continue
                
                azimuth = np.degrees(np.arctan2(sx, sy)) % 360
                beam = self.beamform(st, slowness, azimuth, normalize=True,
                                    use_envelope=False)
                
                if beam is None:
                    continue
                
                power = np.sum(beam.data**2)
                
                if power > max_power:
                    max_power = power
                    best_sx = sx
                    best_sy = sy
        
        best_slowness = np.sqrt(best_sx**2 + best_sy**2)
        best_velocity = 1.0 / best_slowness if best_slowness > 0 else np.inf
        best_azimuth = np.degrees(np.arctan2(best_sx, best_sy)) % 360
        
        return {
            'slowness': best_slowness,
            'velocity': best_velocity,
            'backazimuth': best_azimuth,
            'power': max_power,
            'sx': best_sx,
            'sy': best_sy
        }
    
    def cluster_detections(self, detections, time_tolerance=5.0):
        """
        Cluster detections in time to remove duplicates.
        
        Parameters
        ----------
        detections : list
            List of detection dictionaries
        time_tolerance : float
            Time window for grouping (seconds)
            
        Returns
        -------
        clustered : list
            Clustered detections (one per event)
        """
        if len(detections) == 0:
            return []
        
        detections = sorted(detections, key=lambda x: x['time'])
        
        clustered = []
        current_cluster = [detections[0]]
        
        for det in detections[1:]:
            time_diff = det['time'] - current_cluster[-1]['time']
            
            if time_diff <= time_tolerance:
                current_cluster.append(det)
            else:
                best = max(current_cluster, key=lambda x: x['snr'])
                clustered.append(best)
                current_cluster = [det]
        
        if len(current_cluster) > 0:
            best = max(current_cluster, key=lambda x: x['snr'])
            clustered.append(best)
        
        return clustered
    
    def process_single_day(self, date_str,
                           velocity_range=(3.0, 8.0),
                           azimuth_range=(0, 360),
                           velocity_step=0.5,
                           azimuth_step=10,
                           sta_len=1.0, lta_len=30.0,
                           threshold=3.0,
                           use_envelope=True,
                           cf_method='envelope',
                           max_beams=None,
                           fk_max_per_subarray=3,
                           fk_min_snr=0.0,
                           decimate=1,
                           plot=False,
                           plot_dir=None):
        """
        Process a single day with traditional beamforming.
        
        Parameters
        ----------
        date_str : str
            Date in YYYYMMDD format
        use_envelope : bool
            If True, beamform the envelope (RECOMMENDED for subtle events)
        cf_method : str
            'envelope' - Hilbert envelope (best for long-period)
            'energy' - Instantaneous energy
            'kurtosis' - Kurtosis-based (best for impulsive)
            
        Returns
        -------
        detections : list
            List of clustered detections
        """
        logger.info(f"Processing {date_str} with traditional beamforming...")
        
        stream = self.load_day_data(date_str)
        
        if stream is None or len(stream) == 0:
            logger.warning(f"No data available for {date_str}")
            return []
        
        logger.info(f"Loaded {len(stream)} traces")
        
        # Instead of beamforming the whole station set at once, split into
        # subarrays (groups of ~7 stations) and run the grid search on each
        # subarray independently. This makes it easier to detect local array
        # coherent signals and to produce per-subarray diagnostics.
        day_all_detections = []

        # Determine subarrays (fall back to full array if not partitioned)
        subarrays = getattr(self, 'subarrays', None)
        if not subarrays:
            subarrays = [list(self.station_coords.keys())]

        for i, group in enumerate(subarrays):
            # Build a sub-stream containing only traces from this subarray
            stations_set = set(group)
            substream = Stream([tr.copy() for tr in stream if tr.stats.station in stations_set])

            if substream is None or len(substream) == 0:
                logger.debug(f"Subarray {i} ({len(group)} stations) has no data; skipping")
                continue

            logger.info(f"Processing subarray {i} ({len(group)} stations) for {date_str}")

            # Temporarily switch to subarray geometry so delays are relative to
            # the subarray center when beamforming. Save originals to restore.
            orig_station_coords = self.station_coords
            orig_array_geometry = self.array_geometry

            # Build station_coords subset
            subset_coords = {sta: self.station_coords[sta] for sta in group if sta in self.station_coords}
            self.station_coords = subset_coords

            # Build array_geometry mapping station -> (x,y,elev) for this subarray
            if hasattr(self, 'subarray_geometries') and i in self.subarray_geometries:
                self.array_geometry = self.subarray_geometries[i]
            else:
                # fallback: compute from subset coordinates
                lats = np.array([v[0] for v in subset_coords.values()])
                lons = np.array([v[1] for v in subset_coords.values()])
                elevs = np.array([v[2] for v in subset_coords.values()])
                center_lat = np.mean(lats)
                center_lon = np.mean(lons)
                geom = {}
                for sta, lat, lon, elev in zip(subset_coords.keys(), lats, lons, elevs):
                    x = (lon - center_lon) * 111.32 * np.cos(np.radians(center_lat))
                    y = (lat - center_lat) * 110.54
                    geom[sta] = (x, y, elev)
                self.array_geometry = geom

            # Run grid search on this substream and perform FK/refinement and
            # plotting while the beamformer is still configured to this
            # subarray's geometry. Ensure restoration happens no matter what.
            try:
                detections = self.grid_search_detection(
                    substream, date_str,
                    velocity_range, azimuth_range,
                    velocity_step, azimuth_step,
                    sta_len, lta_len, threshold,
                    use_envelope, cf_method,
                    max_beams=max_beams,
                    decimate=decimate,
                    progress=True
                )

                logger.info(f"Subarray {i}: Found {len(detections)} raw detections")

                # Cluster per-subarray
                clustered_sub = self.cluster_detections(detections, time_tolerance=5.0)

                # Attach subarray info. FK refinement is expensive; limit
                # the number of FK calls performed per subarray. We pick the
                # top-N detections sorted by SNR (or those exceeding fk_min_snr)
                # to keep runtime manageable when many detections are present.
                for det in clustered_sub:
                    det['subarray_id'] = i
                    det['stations'] = group

                # choose subset to refine
                refine_candidates = [d for d in clustered_sub if d.get('snr', 0.0) >= fk_min_snr]
                refine_candidates = sorted(refine_candidates, key=lambda x: x.get('snr', 0.0), reverse=True)
                refine_candidates = refine_candidates[:fk_max_per_subarray] if fk_max_per_subarray is not None else refine_candidates

                for det in refine_candidates:
                    try:
                        fk_result = self.fk_analysis(substream, det['time'])
                    except Exception:
                        fk_result = None
                    if fk_result:
                        det['fk_velocity'] = fk_result['velocity']
                        det['fk_backazimuth'] = fk_result['backazimuth']
                        det['fk_power'] = fk_result['power']

                logger.info(f"Subarray {i}: FK refinements performed: {len(refine_candidates)} / {len(clustered_sub)} (cap={fk_max_per_subarray}, min_snr={fk_min_snr})")

                # Save per-subarray detections into day's pool
                day_all_detections.extend(clustered_sub)

                # Optionally write per-subarray JSON (times as both ISO and epoch)
                if plot and plot_dir:
                    try:
                        import json
                        os.makedirs(os.path.join(plot_dir, f"subarray_{i}"), exist_ok=True)
                        subjson_path = os.path.join(plot_dir, f"subarray_{i}", f"detections_{date_str}.json")
                        # Serialize times to ISO and epoch for downstream scripts
                        serial = []
                        # sanitize detection dicts recursively for JSON
                        def _sanitize(val):
                            import numpy as _np
                            from obspy import UTCDateTime as _UTC
                            # UTCDateTime -> ISO
                            if hasattr(val, 'isoformat') and not isinstance(val, (str, bytes)):
                                try:
                                    return val.isoformat()
                                except Exception:
                                    pass
                            # obspy UTCDateTime instances sometimes don't have isoformat attr
                            try:
                                if isinstance(val, _UTC):
                                    return val.isoformat()
                            except Exception:
                                pass
                            # numpy types -> python native
                            if isinstance(val, (_np.integer,)):
                                return int(val)
                            if isinstance(val, (_np.floating,)):
                                return float(val)
                            if isinstance(val, (_np.ndarray,)):
                                return val.tolist()
                            # lists/tuples -> recur
                            if isinstance(val, (list, tuple)):
                                return [_sanitize(v) for v in val]
                            if isinstance(val, dict):
                                return {k: _sanitize(v) for k, v in val.items()}
                            # datetime objects
                            import datetime as _dt
                            if isinstance(val, (_dt.datetime,)):
                                return val.isoformat() + 'Z' if val.tzinfo is None else val.isoformat()
                            # fallback
                            return val
                        for d in clustered_sub:
                            dt = d['time']
                            try:
                                # If dt is UTCDateTime/datetime-like, convert to float epoch
                                if hasattr(dt, 'timestamp'):
                                    ts_attr = getattr(dt, 'timestamp')
                                    try:
                                        epoch = float(ts_attr()) if callable(ts_attr) else float(ts_attr)
                                    except Exception:
                                        epoch = float(ts_attr)
                                else:
                                    epoch = float(dt)
                            except Exception:
                                # Fallback: try parsing string iso with obspy/UTCDateTime or python's datetime
                                try:
                                    from obspy import UTCDateTime as _UTC
                                    epoch = float(_UTC(str(d['time'])).timestamp())
                                except Exception:
                                    import datetime as _dt
                                    try:
                                        epoch = float(_dt.datetime.fromisoformat(str(d['time'])).replace(tzinfo=_dt.timezone.utc).timestamp())
                                    except Exception:
                                        epoch = 0.0
                            dcopy = d.copy()
                            # add ISO string if possible
                            try:
                                if hasattr(dt, 'datetime'):
                                    # obspy UTCDateTime: use .isoformat()
                                        dcopy['time_iso'] = dt.isoformat() if hasattr(dt, 'isoformat') else str(dt)
                                else:
                                    # try converting float/epoch to iso
                                    import datetime
                                    dcopy['time_iso'] = datetime.datetime.utcfromtimestamp(epoch).isoformat() + 'Z'
                            except Exception:
                                dcopy['time_iso'] = str(dt)
                            dcopy['time_epoch'] = float(epoch)
                            # ensure original 'time' field is replaced with iso string for JSON
                            dcopy['time'] = dcopy['time_iso']
                            # sanitize entire dict
                            dcopy = _sanitize(dcopy)
                            serial.append(dcopy)
                        with open(subjson_path, 'w') as fh:
                            json.dump(serial, fh, indent=2)
                        logger.info(f"Saved per-subarray JSON: {subjson_path}")
                    except Exception as e:
                        logger.warning(f"Failed to write per-subarray JSON for subarray {i}: {e}")

                # Produce a per-subarray diagnostic plot for the day
                if plot and plot_dir:
                    try:
                        sub_out = os.path.join(plot_dir, f"subarray_{i}")
                        outfile = plot_daily_detections(f"{date_str}_sub{i}", substream, clustered_sub, self, outdir=sub_out)
                        if outfile:
                            logger.info(f"Saved subarray detection figure: {outfile}")
                    except Exception as e:
                        logger.warning(f"Failed to create plot for {date_str} subarray {i}: {e}")
            finally:
                # restore global geometry regardless of success
                self.station_coords = orig_station_coords
                self.array_geometry = orig_array_geometry

        # Optional: create plots for detections (always create a diagnostic
        # figure when --plot is requested — even if there are no detections).
        # After processing all subarrays, cluster across subarrays to remove
        # duplicates and produce an overall per-day diagnostic plot
        clustered = self.cluster_detections(day_all_detections, time_tolerance=5.0)

        # Optionally write per-day detections JSON file (with times in epoch)
        if plot and plot_dir:
            try:
                import json
                os.makedirs(plot_dir, exist_ok=True)
                outpath = os.path.join(plot_dir, f'detections_{date_str}.json')
                # Build mapping date -> list of dicts (time_epoch, backazimuth, subarray_id, snr, velocity, duration)
                serial = []
                def _sanitize_val(val):
                    import numpy as _np
                    import datetime as _dt
                    try:
                        from obspy import UTCDateTime as _UTC
                        if isinstance(val, _UTC):
                            return val.isoformat()
                    except Exception:
                        pass
                    if isinstance(val, (_np.integer,)):
                        return int(val)
                    if isinstance(val, (_np.floating,)):
                        return float(val)
                    if isinstance(val, (_np.ndarray,)):
                        return val.tolist()
                    if isinstance(val, (_dt.datetime,)):
                        return val.isoformat() + 'Z' if val.tzinfo is None else val.isoformat()
                    if isinstance(val, (list, tuple)):
                        return [_sanitize_val(v) for v in val]
                    if isinstance(val, dict):
                        return {k: _sanitize_val(v) for k, v in val.items()}
                    return val
                for d in clustered:
                    entry = {}
                    # time could be obspy UTCDateTime or datetime – convert robustly
                    try:
                        if hasattr(d['time'], 'timestamp'):
                            ts_attr = getattr(d['time'], 'timestamp')
                            try:
                                epoch = float(ts_attr()) if callable(ts_attr) else float(ts_attr)
                            except Exception:
                                epoch = float(ts_attr)
                        else:
                            epoch = float(d['time'])
                    except Exception:
                        # If conversion fails, attempt to coerce to float
                        epoch = float(d.get('time_epoch', 0.0))
                    entry.update({
                        'time': epoch,
                        'backazimuth': float(d.get('backazimuth', d.get('fk_backazimuth', 0.0))),
                        'subarray_id': d.get('subarray_id', None),
                        'snr': float(d.get('snr', 0.0)),
                        'velocity': float(d.get('velocity', d.get('fk_velocity', 0.0))),
                        'duration': float(d.get('duration', 0.0))
                    })
                    serial.append(_sanitize_val(entry))
                # wrap as date -> list mapping
                wrapped = {date_str: serial}
                with open(outpath, 'w') as fh:
                    json.dump(wrapped, fh, indent=2)
                logger.info(f"Wrote per-day detections JSON: {outpath}")
            except Exception as e:
                logger.warning(f"Failed to write per-day detections JSON: {e}")

        # Attempt simple two-array triangulation across subarrays. We pair
        # subarray detections which are close in time and use backazimuth to
        # compute intersections. Results are written to an optional output
        # file in the plot directory when available.
        try:
            from beam.core.triangulation import triangulate_two_arrays

            # build mapping subarray_id -> detections
            sub_map = {}
            for d in clustered:
                sid = d.get('subarray_id', None)
                if sid is None:
                    continue
                sub_map.setdefault(sid, []).append(d)

            # Pre-compute centers for each subarray from full station_coords
            sub_centers = {}
            for sid, group in enumerate(getattr(self, 'subarrays', [list(self.station_coords.keys())])):
                try:
                    lat_vals = [self.station_coords[s][0] for s in group if s in self.station_coords]
                    lon_vals = [self.station_coords[s][1] for s in group if s in self.station_coords]
                    if len(lat_vals) > 0:
                        sub_centers[sid] = (float(np.mean(lat_vals)), float(np.mean(lon_vals)))
                except Exception:
                    continue

            # Pairwise triangulation
            location_estimates = []
            ids = sorted(list(sub_map.keys()))
            for i in range(len(ids)):
                for j in range(i+1, len(ids)):
                    ida = ids[i]
                    idb = ids[j]
                    if ida not in sub_centers or idb not in sub_centers:
                        continue
                    res = triangulate_two_arrays(sub_map.get(ida, []), sub_map.get(idb, []),
                                                 sub_centers[ida], sub_centers[idb], time_tolerance=5.0)
                    if res:
                        for r in res:
                            r['subarrays'] = (ida, idb)
                        location_estimates.extend(res)

            # Optionally save location estimates if plotting directory given
            if plot and plot_dir and len(location_estimates) > 0:
                import json
                os.makedirs(plot_dir, exist_ok=True)
                jsonfile = os.path.join(plot_dir, f"locations_{date_str}.json")
                with open(jsonfile, 'w') as fh:
                    json.dump(location_estimates, fh, indent=2)
                logger.info(f"Wrote triangulation results: {jsonfile}")
        except Exception as e:
            logger.debug(f"Triangulation step failed: {e}")

        # Multi-array least-squares location (when three or more arrays have
        # roughly coincident detections). We use the more sophisticated
        # locator which combines TOA+backazimuth+slowness into a least-squares
        # fit for (lat, lon, origin_time).
        try:
            from beam.core.locator import locate_multarray_least_squares

            # group raw day_all_detections by time-window so we have all
            # per-array picks that belong to the same event candidate.
            groups = []
            if len(day_all_detections) > 0:
                sorted_all = sorted(day_all_detections, key=lambda x: x['time'])
                cur = [sorted_all[0]]
                for d in sorted_all[1:]:
                    if d['time'] - cur[-1]['time'] <= 5.0:
                        cur.append(d)
                    else:
                        groups.append(cur)
                        cur = [d]
                if len(cur) > 0:
                    groups.append(cur)

            lsq_results = []
            for g in groups:
                # require at least 3 distinct arrays for a robust LSQ location
                arr_ids = set([d.get('subarray_id', None) for d in g if d.get('subarray_id', None) is not None])
                if len(arr_ids) < 3:
                    continue

                # prepare centers mapping for arrays present
                centers = {}
                for aid in arr_ids:
                    # attempt to get subarray centers from subarrays list
                    try:
                        group_stations = getattr(self, 'subarrays', [list(self.station_coords.keys())])[aid]
                        # compute center lat/lon
                        lat_vals = [self.station_coords[s][0] for s in group_stations if s in self.station_coords]
                        lon_vals = [self.station_coords[s][1] for s in group_stations if s in self.station_coords]
                        if len(lat_vals) == 0:
                            continue
                        centers[aid] = (float(np.mean(lat_vals)), float(np.mean(lon_vals)))
                    except Exception:
                        continue

                if len(centers) < 3:
                    continue

                res = locate_multarray_least_squares(g, centers)
                if res and res.get('success'):
                    res['member_count'] = len(g)
                    lsq_results.append(res)

            if plot and plot_dir and len(lsq_results) > 0:
                import json
                os.makedirs(plot_dir, exist_ok=True)
                jsonfile = os.path.join(plot_dir, f"locations_lsq_{date_str}.json")
                with open(jsonfile, 'w') as fh:
                    json.dump(lsq_results, fh, indent=2)
                logger.info(f"Wrote least-squares location results: {jsonfile}")
        except Exception as e:
            logger.debug(f"Least-squares locator failed: {e}")

        if plot and plot_dir:
            try:
                outfile = plot_daily_detections(date_str, stream, clustered, self, outdir=plot_dir)
                if outfile:
                    logger.info(f"Saved detection figure: {outfile}")
                else:
                    logger.info(f"Plot function returned no file for {date_str}")
            except Exception as e:
                logger.warning(f"Failed to create plot for {date_str}: {e}")
        
        return clustered
    
    def process_date_range(self, start_date, end_date,
                           velocity_range=(3.0, 8.0),
                           azimuth_range=(0, 360),
                           velocity_step=0.5,
                           azimuth_step=10,
                           sta_len=1.0, lta_len=30.0,
                           threshold=3.0,
                           use_envelope=True,
                           cf_method='envelope',
                           n_processes=None,
                           max_beams=None,
                           fk_max_per_subarray=3,
                           fk_min_snr=0.0,
                           decimate=1,
                           plot=False,
                           plot_dir=None):
        """
        Process multiple days with traditional beamforming.
        
        Parameters
        ----------
        start_date : str or UTCDateTime
            Start date
        end_date : str or UTCDateTime
            End date
        use_envelope : bool
            If True, beamform envelope (STRONGLY RECOMMENDED)
        cf_method : str
            'envelope', 'energy', or 'kurtosis'
        n_processes : int, optional
            Number of parallel processes
            
        Returns
        -------
        all_detections : dict
            Dictionary with dates as keys
        """
        if isinstance(start_date, str):
            start_date = UTCDateTime(start_date)
        if isinstance(end_date, str):
            end_date = UTCDateTime(end_date)
        
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current.strftime('%Y%m%d'))
            current += 86400
        
        logger.info(f"Processing {len(dates)} days with traditional beamforming")
        logger.info(f"Envelope mode: {use_envelope} ({cf_method})")
        
        all_detections = {}

        if n_processes == 1:
            # Sequential
            for date in dates:
                try:
                    dets = self.process_single_day(
                        date, velocity_range, azimuth_range,
                        velocity_step, azimuth_step,
                        sta_len, lta_len, threshold,
                        use_envelope, cf_method,
                        max_beams=max_beams, fk_max_per_subarray=fk_max_per_subarray, fk_min_snr=fk_min_snr, decimate=decimate,
                        plot=plot, plot_dir=plot_dir
                    )
                    if len(dets) > 0:
                        all_detections[date] = dets
                except Exception as e:
                    logger.error(f"Error processing {date}: {e}")
        else:
            # Parallel
            process_func = partial(
                self.process_single_day,
                velocity_range=velocity_range,
                azimuth_range=azimuth_range,
                velocity_step=velocity_step,
                azimuth_step=azimuth_step,
                sta_len=sta_len,
                lta_len=lta_len,
                threshold=threshold,
                use_envelope=use_envelope,
                cf_method=cf_method,
                max_beams=max_beams, fk_max_per_subarray=fk_max_per_subarray, fk_min_snr=fk_min_snr, decimate=decimate, plot=plot, plot_dir=plot_dir
            )
            
            n_procs = n_processes or mp.cpu_count()
            with mp.Pool(processes=n_procs) as pool:
                results = pool.map(process_func, dates)
            
            for date, dets in zip(dates, results):
                if dets and len(dets) > 0:
                    all_detections[date] = dets
        
        total = sum(len(d) for d in all_detections.values())
        logger.info(f"Total unique events detected: {total}")
        
        return all_detections
    
    def save_detections(self, detections, output_file, min_snr: float = None):
        """
        Save detections to file.
        
        Parameters
        ----------
        detections : dict
            Detection dictionary
        output_file : str
            Output file path
        """
        with open(output_file, 'w') as f:
            f.write("# BEAM Traditional Beamforming Detections\n")
            f.write("# Time, Velocity(km/s), Backazimuth(deg), SNR, Duration(s)\n")
            
            for date in sorted(detections.keys()):
                for det in detections[date]:
                    # optional filter by SNR
                    if min_snr is not None and det.get('snr', 0.0) < min_snr:
                        continue
                    f.write(f"{det['time']}, {det['velocity']:.2f}, "
                           f"{det['backazimuth']:.1f}, {det['snr']:.2f}, "
                           f"{det['duration']:.2f}\n")
        
        logger.info(f"Saved detections to {output_file}")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='BEAM - Array-Based Seismic Event Detector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Traditional beamforming (no master event needed):
  python beam_driver.py --mode traditional -d /path/to/data -s 20200101 -e 20200131

  # Correlation detection (requires master event):
  python beam_driver.py --mode correlation -d /path/to/data -s 20200101 -e 20200131 \\
      --master-time "2020-01-15T12:30:45"
        """
    )
    parser.add_argument('--config', default=None,
                        help='Path to YAML config file containing options (overridden by CLI args)')
    
    # Mode selection
    parser.add_argument('--mode', choices=['traditional', 'correlation'],
                       default='traditional',
                       help='Detection mode: traditional (STA/LTA beamforming) or '
                            'correlation (master event matching). Default: traditional')
    
    # Required arguments
    parser.add_argument('--data-dir', '-d', required=True,
                       help='Base directory containing YYYYMMDD data folders')
    
    # Date range
    parser.add_argument('--start', '-s', required=True,
                       help='Start date (YYYYMMDD format)')
    parser.add_argument('--end', '-e', required=True,
                       help='End date (YYYYMMDD format)')
    
    # Master event (required for correlation mode)
    parser.add_argument('--master-time', '-m', default=None,
                       help='Master event time (ISO format) - required for correlation mode')
    parser.add_argument('--master-duration', type=float, default=60.0,
                       help='Master event duration in seconds (default: 60)')
    
    # Filter parameters
    parser.add_argument('--freqmin', type=float, default=2.0,
                       help='Minimum frequency for bandpass (Hz)')
    parser.add_argument('--freqmax', type=float, default=8.0,
                       help='Maximum frequency for bandpass (Hz)')
    
    # Detection parameters
    parser.add_argument('--threshold', '-t', type=float, default=3.5,
                       help='Detection threshold (default: 3.5 for traditional, 6.0 for correlation)')
    
    # Traditional beamforming parameters
    parser.add_argument('--vel-min', type=float, default=3.0,
                       help='Minimum velocity for grid search (km/s)')
    parser.add_argument('--vel-max', type=float, default=8.0,
                       help='Maximum velocity for grid search (km/s)')
    parser.add_argument('--vel-step', type=float, default=0.5,
                       help='Velocity grid step (km/s)')
    parser.add_argument('--az-step', type=float, default=10.0,
                       help='Azimuth grid step (degrees)')
    parser.add_argument('--sta-len', type=float, default=1.0,
                       help='STA window length (seconds)')
    parser.add_argument('--lta-len', type=float, default=30.0,
                       help='LTA window length (seconds)')
    parser.add_argument('--use-envelope', action='store_true', default=True,
                       help='Use envelope beamforming (recommended, default)')
    parser.add_argument('--no-envelope', action='store_true',
                       help='Use raw waveform beamforming instead of envelope')
    parser.add_argument('--cf-method', choices=['envelope', 'energy', 'kurtosis'],
                       default='envelope',
                       help='Characteristic function method for envelope beamforming')
    
    # Processing options
    parser.add_argument('--network', default='*',
                       help='Network code(s)')
    parser.add_argument('--channel', default='*Z',
                       help='Channel code(s)')
    parser.add_argument('--processes', '-p', type=int, default=None,
                       help='Number of parallel processes (default: all cores)')
    
    # Output
    parser.add_argument('--output', '-o', default='detections.txt',
                       help='Output file for detections')
    parser.add_argument('--inventory', default=None,
                        help='Path to StationXML inventory file or folder (use if you have a single combined inventory file)')
    parser.add_argument('--inventory-pattern', default='*.xml',
                        help='Glob pattern to search for inventory files in provided inventory folder (default: "*.xml")')
    parser.add_argument('--inventory-tag', default=None,
                        help='Optional substring tag to filter candidate inventory files by filename')
    parser.add_argument('--inventory-name', default=None,
                        help='Optional exact inventory filename to load from the inventory folder')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')

    # Surface-wave mode (long periods / decimated data, tailored parameters)
    parser.add_argument('--surface-wave', action='store_true',
                        help='Configure detection parameters for surface waves (long-period bandpass + decimation)')

    # Speed / debug options for the traditional grid search
    parser.add_argument('--decimate', type=int, default=1,
                       help='Decimate factor for traditional beamforming (integer >1 speeds up grid search)')
    parser.add_argument('--max-beams', type=int, default=None,
                       help='Optional maximum number of beams to evaluate (for quick tests)')
    parser.add_argument('--fk-max-per-subarray', type=int, default=3,
                       help='Maximum FK refinements per subarray (top-N by SNR)')
    parser.add_argument('--fk-min-snr', type=float, default=0.0,
                       help='Minimum SNR threshold to consider for FK refinement')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Create and save daily detection plots (one file per day)')
    parser.add_argument('--plot-dir', default='plots',
                        help='Directory to save detection figures (default: ./plots)')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Attempt to use GPU-accelerated beamforming (CuPy).')
    parser.add_argument('--min-snr-output', type=float, default=None,
                        help='Minimum SNR required for events to be written to the output file')
    parser.add_argument('--force-subarrays', type=int, default=None,
                        help='Force splitting stations into N subarrays (evenly distributed).')
    parser.add_argument('--subarrays-file', default=None,
                        help='Path to JSON file containing explicit subarray groups: list of lists of station codes')
    parser.add_argument('--subarrays', default=None,
                        help='Inline subarray specification: "STA1,STA2;STA3,STA4" where semicolons separate groups')
    parser.add_argument('--gpu-safety-factor', type=int, default=None,
                        help='Optional override for BEAM GPU safety factor (smaller => more memory used; default from env BEAM_GPU_SAFETY_FACTOR)')
    
    # First pass parse to detect config file, then set defaults from it so CLI args override config
    import sys
    known_args, _ = parser.parse_known_args()
    if known_args.config is not None:
        try:
            try:
                import yaml
            except Exception:
                raise RuntimeError('YAML config file requested but PyYAML is not installed; pip install pyyaml')
            with open(known_args.config) as fh:
                cfg = yaml.safe_load(fh) or {}
            # Normalize keys: replace hyphens with underscore to match argparse dest names
            cfg_norm = {}
            for k, v in cfg.items():
                cfg_norm[k.replace('-', '_')] = v
            parser.set_defaults(**cfg_norm)
        except Exception as e:
            logger.warning(f'Failed to load config file {known_args.config}: {e}')

    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine envelope usage
    use_envelope = not args.no_envelope

    # If user requested surface-wave mode, set recommended defaults for
    # long-period bandpass and decimation unless the user explicitly set them
    if args.surface_wave:
        logger.info("Surface-wave mode enabled: applying long-period defaults and decimation")
        # Bandpass defaults (if not overridden)
        if args.freqmin == 2.0:
            args.freqmin = 0.1
        if args.freqmax == 8.0:
            args.freqmax = 2.0

        # Decimation: aggressive by default for long-period surface waves
        if args.decimate == 1:
            args.decimate = 10

        # Adjust traditional grid defaults for surface waves if unchanged
        if args.vel_min == 3.0:
            args.vel_min = 1.5
        if args.vel_max == 8.0:
            args.vel_max = 5.0
        if args.sta_len == 1.0:
            args.sta_len = 10.0
        if args.lta_len == 30.0:
            args.lta_len = 300.0

    # apply the incoming CLI/config override if present
    apply_gpu_safety_factor(args.gpu_safety_factor)
    
    if args.mode == 'correlation':
        # Correlation mode requires master event
        if args.master_time is None:
            parser.error("--master-time is required for correlation mode")
        
        # Use higher default threshold for correlation
        threshold = args.threshold if args.threshold != 3.5 else 6.0
        
        logger.info("=" * 60)
        logger.info("BEAM - Correlation Detection Mode (Gibbons & Ringdal)")
        logger.info("=" * 60)
        
        detector = BeamArrayDetector(
            data_dir=args.data_dir,
            network=args.network,
            channel=args.channel,
            freqmin=args.freqmin,
            freqmax=args.freqmax
        )
        
        # allow direct inventory specification for faster startup
        if args.inventory:
            try:
                if os.path.isfile(args.inventory):
                    inv = read_inventory(args.inventory)
                    detector.inventory = inv
                    detector.station_coords = get_station_coords_dict(inv)
                    detector._compute_array_geometry()
                    logger.info(f"Loaded inventory from file: {args.inventory}")
                else:
                    detector.load_inventory_from_folder(args.inventory, pattern=args.inventory_pattern, tag=args.inventory_tag, name=args.inventory_name)
            except Exception as e:
                logger.warning(f"Failed to load inventory from {args.inventory}: {e}; falling back to data_dir discovery")
                detector.load_inventory_from_folder(pattern=args.inventory_pattern, tag=args.inventory_tag)
        else:
            detector.load_inventory_from_folder(pattern=args.inventory_pattern, tag=args.inventory_tag, name=args.inventory_name)
        detector.load_master_event(
            master_time=args.master_time,
            duration=args.master_duration,
            decimate=args.decimate
        )
        
        detections = detector.process_date_range(
            start_date=args.start,
            end_date=args.end,
            detection_threshold=threshold,
            n_processes=args.processes,
            decimate=args.decimate,
            plot=args.plot,
            plot_dir=args.plot_dir
        )

        # optionally filter saved detections by SNR when writing out
        detector.save_detections(detections, args.output, min_snr=args.min_snr_output)
        
    else:  # traditional mode
        logger.info("=" * 60)
        logger.info("BEAM - Traditional Beamforming Mode (STA/LTA)")
        logger.info("=" * 60)
        
        beamformer = TraditionalBeamformer(
            data_dir=args.data_dir,
            network=args.network,
            channel=args.channel,
            freqmin=args.freqmin,
            freqmax=args.freqmax,
            use_gpu=args.gpu
        )
        
        # Load inventory: either from CLI-specified inventory file/folder or
        # the default data dir inventory folder. Accept a single StationXML
        # file, a folder of XMLs, or a folder containing dailyinventory.xml.
        if args.inventory:
            try:
                # if it's a file, try to read a single StationXML file
                if os.path.isfile(args.inventory):
                    inv = read_inventory(args.inventory)
                    beamformer.inventory = inv
                    beamformer.station_coords = get_station_coords_dict(inv)
                    beamformer._compute_array_geometry()
                    logger.info(f"Loaded inventory from file: {args.inventory}")
                else:
                    # treat it as a folder path
                    beamformer.load_inventory_from_folder(args.inventory, pattern=args.inventory_pattern, tag=args.inventory_tag, name=args.inventory_name)
            except Exception as e:
                logger.warning(f"Failed to load inventory from {args.inventory}: {e}; falling back to data_dir discovery")
                beamformer.load_inventory_from_folder(pattern=args.inventory_pattern, tag=args.inventory_tag, name=args.inventory_name)
        else:
            beamformer.load_inventory_from_folder(pattern=args.inventory_pattern, tag=args.inventory_tag, name=args.inventory_name)
        # Optionally force number of subarrays (useful for testing / experiments)
        if args.force_subarrays is not None and args.force_subarrays > 0:
            try:
                beamformer.force_subarrays(args.force_subarrays)
                logger.info(f"Forced partitioning into {args.force_subarrays} subarrays")
            except Exception as e:
                logger.warning(f"Failed to force subarrays={args.force_subarrays}: {e}")

        # Optionally load explicit subarray definitions
        if args.subarrays_file is not None or args.subarrays is not None:
            # parse file first if provided
            subarray_groups = None
            if args.subarrays_file is not None:
                try:
                    import json as _json
                    with open(args.subarrays_file) as fh:
                        subarray_groups = _json.load(fh)
                except Exception as e:
                    logger.warning(f"Failed to load subarrays-file {args.subarrays_file}: {e}")
                    subarray_groups = None

            if subarray_groups is None and args.subarrays is not None:
                # argparse or config input may provide: string or list-of-lists
                if isinstance(args.subarrays, list):
                    # assume it's already a list-of-lists
                    subarray_groups = args.subarrays
                elif isinstance(args.subarrays, str):
                    # parse inline spec: 'STA1,STA2;STA3,STA4' -> [[STA1,STA2],[STA3,STA4]]
                    try:
                        groups = [g.strip() for g in args.subarrays.split(';') if g.strip()]
                        subarray_groups = [[s.strip() for s in g.split(',') if s.strip()] for g in groups]
                    except Exception:
                        logger.warning(f"Failed to parse --subarrays string: {args.subarrays}")
                        subarray_groups = None
                else:
                    logger.warning(f"Unsupported --subarrays type: {type(args.subarrays)}; expected str or list")
                    subarray_groups = None

            if subarray_groups is not None:
                try:
                    beamformer.set_subarrays(subarray_groups)
                    logger.info(f"Applied subarray groups from input. Number of subarrays: {len(beamformer.subarrays)}")
                    # Warn if some stations from the inventory were not explicitly included
                    grouped = set([s for g in beamformer.subarrays for s in g])
                    all_codes = set(beamformer.station_coords.keys())
                    leftovers = sorted(list(all_codes - grouped))
                    if leftovers:
                        logger.warning(
                            f"The provided subarray groups do not include these {len(leftovers)} stations in the inventory: {leftovers}. "
                            f"These stations will not be used in any subarray unless you include them in a group."
                        )
                except Exception as e:
                    logger.warning(f"Failed to set explicit subarrays: {e}")

        
        
        detections = beamformer.process_date_range(
            start_date=args.start,
            end_date=args.end,
            velocity_range=(args.vel_min, args.vel_max),
            azimuth_range=(0, 360),
            velocity_step=args.vel_step,
            azimuth_step=args.az_step,
            sta_len=args.sta_len,
            lta_len=args.lta_len,
            threshold=args.threshold,
            use_envelope=use_envelope,
            cf_method=args.cf_method,
            n_processes=args.processes,
            max_beams=args.max_beams,
            decimate=args.decimate
            ,plot=args.plot, plot_dir=args.plot_dir,
            fk_max_per_subarray=args.fk_max_per_subarray, fk_min_snr=args.fk_min_snr
        )
        
        beamformer.save_detections(detections, args.output, min_snr=args.min_snr_output)
    
    total = sum(len(d) for d in detections.values())
    print(f"\nDone. Found {total} detections.")
    print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
