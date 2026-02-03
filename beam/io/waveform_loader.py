"""
Waveform loading module for BEAM.

Provides functionality to load seismic waveforms from:
- Local miniseed files organized in YYYYMMDD folders
- FDSN web services (IRIS, etc.)

Data access patterns adapted from AELUMA for consistency.
"""

import os
import glob
import logging
import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d

from obspy import read, Stream, UTCDateTime, Inventory
from obspy.clients.fdsn import Client

logger = logging.getLogger(__name__)


class WaveformLoader:
    """
    Unified waveform data loader supporting local files and FDSN services.
    
    Parameters
    ----------
    data_dir : str, optional
        Base directory containing YYYYMMDD folders with miniseed files
    fdsn_client : str, optional
        FDSN client name (e.g., 'IRIS', 'GFZ', etc.)
    network : str
        Network code(s) to load
    channel : str
        Channel code(s) to load (e.g., 'BHZ', 'HHZ', '*Z')
    location : str
        Location code(s) to load
    """
    
    def __init__(self, data_dir=None, fdsn_client=None,
                 network='*', channel='*Z', location='*'):
        self.data_dir = data_dir
        self.network = network
        self.channel = channel
        self.location = location
        
        # Initialize FDSN client if specified
        self.client = None
        if fdsn_client:
            self.client = Client(fdsn_client)
            logger.info(f"Initialized FDSN client: {fdsn_client}")
    
    def load_day(self, date, stations=None, starttime=None, endtime=None):
        """
        Load waveforms for a single day.
        
        Parameters
        ----------
        date : str, datetime, or UTCDateTime
            Date to load (YYYYMMDD format or datetime object)
        stations : list of str, optional
            List of station codes to load. If None, loads all available.
        starttime : UTCDateTime, optional
            Start time for trimming. Defaults to start of day.
        endtime : UTCDateTime, optional
            End time for trimming. Defaults to end of day.
            
        Returns
        -------
        stream : obspy.Stream
            Stream containing loaded waveforms
        """
        # Parse date
        if isinstance(date, str):
            if len(date) == 8:
                year = int(date[:4])
                month = int(date[4:6])
                day = int(date[6:8])
                date_obj = UTCDateTime(year=year, month=month, day=day)
            else:
                date_obj = UTCDateTime(date)
        elif isinstance(date, datetime):
            date_obj = UTCDateTime(date)
        else:
            date_obj = date
        
        # Set default time window
        if starttime is None:
            starttime = UTCDateTime(date_obj.year, date_obj.month, date_obj.day, 0, 0, 0)
        if endtime is None:
            endtime = starttime + 86400  # Full day
        
        # Try local files first, then FDSN
        stream = Stream()
        
        if self.data_dir:
            stream = self._load_from_local(date_obj, stations)
        
        if len(stream) == 0 and self.client:
            stream = self._load_from_fdsn(starttime, endtime, stations)
        
        # Trim to requested window (without padding to avoid dtype issues)
        if len(stream) > 0:
            # Convert integer data to float to avoid padding issues
            for tr in stream:
                if tr.data.dtype.kind == 'i':  # integer type
                    tr.data = tr.data.astype(np.float64)
            stream.trim(starttime, endtime, pad=True, fill_value=0.0)
        
        return stream
    
    def _load_from_local(self, date_obj, stations=None):
        """Load waveforms from local miniseed files."""
        date_str = date_obj.strftime('%Y%m%d')
        date_folder = os.path.join(self.data_dir, date_str)
        
        if not os.path.exists(date_folder):
            logger.warning(f"Date folder not found: {date_folder}")
            return Stream()
        
        stream = Stream()
        
        # Build file pattern
        if stations:
            for sta in stations:
                pattern = os.path.join(date_folder, f'*.{sta}.*{self.channel}*.mseed')
                files = glob.glob(pattern)
                for f in files:
                    try:
                        stream += read(f)
                    except Exception as e:
                        logger.debug(f"Failed to read {f}: {e}")
        else:
            # Load all files matching channel pattern
            pattern = os.path.join(date_folder, f'*{self.channel}*mseed')
            files = glob.glob(pattern)
            if not files:
                # Try without channel filter
                pattern = os.path.join(date_folder, '*.mseed')
                files = glob.glob(pattern)
            
            for f in files:
                try:
                    stream += read(f)
                except Exception as e:
                    logger.debug(f"Failed to read {f}: {e}")
        
        logger.info(f"Loaded {len(stream)} traces from local files")
        return stream
    
    def _load_from_fdsn(self, starttime, endtime, stations=None):
        """Load waveforms from FDSN web service."""
        if not self.client:
            return Stream()
        
        station_str = ','.join(stations) if stations else '*'
        
        try:
            stream = self.client.get_waveforms(
                network=self.network,
                station=station_str,
                location=self.location,
                channel=self.channel,
                starttime=starttime,
                endtime=endtime
            )
            logger.info(f"Downloaded {len(stream)} traces from FDSN")
            return stream
        except Exception as e:
            logger.error(f"FDSN download failed: {e}")
            return Stream()
    
    def check_sample_rate(self, stream, fs_min=None, fs_max=None, fs_target=None):
        """
        Check and resample traces to acceptable sample rates.
        
        Parameters
        ----------
        stream : obspy.Stream
            Input stream
        fs_min : float, optional
            Minimum acceptable sample rate (Hz)
        fs_max : float, optional
            Maximum sample rate before resampling
        fs_target : float, optional
            Target sample rate for resampling
            
        Returns
        -------
        stream : obspy.Stream
            Stream with valid sample rates
        """
        traces_to_remove = []
        
        for tr in stream:
            fs = tr.stats.sampling_rate
            
            # Check minimum sample rate
            if fs_min and fs < fs_min:
                logger.warning(f"Station {tr.stats.station} sample rate {fs} Hz "
                             f"< minimum {fs_min} Hz; removing")
                traces_to_remove.append(tr)
                continue
            
            # Resample if above maximum
            if fs_max and fs_target and fs > fs_max:
                logger.info(f"Resampling {tr.stats.station} from {fs} to {fs_target} Hz")
                tr.resample(fs_target)
        
        for tr in traces_to_remove:
            stream.remove(tr)
        
        return stream


def load_day_waveforms(data_dir, date, network='*', channel='*Z', 
                       location='*', stations=None, fs_min=None,
                       fs_max=None, fs_target=None):
    """
    Convenience function to load a day of waveforms.
    
    Parameters
    ----------
    data_dir : str
        Base directory containing YYYYMMDD folders
    date : str
        Date in YYYYMMDD format
    network : str
        Network code filter
    channel : str
        Channel code filter
    location : str
        Location code filter
    stations : list, optional
        List of station codes to load
    fs_min : float, optional
        Minimum sample rate
    fs_max : float, optional
        Maximum sample rate before resampling
    fs_target : float, optional
        Target sample rate for resampling
        
    Returns
    -------
    stream : obspy.Stream
        Loaded and filtered waveforms
    """
    loader = WaveformLoader(
        data_dir=data_dir,
        network=network,
        channel=channel,
        location=location
    )
    
    stream = loader.load_day(date, stations=stations)
    
    if len(stream) > 0:
        stream = loader.check_sample_rate(
            stream, fs_min=fs_min, fs_max=fs_max, fs_target=fs_target
        )
    
    return stream


def fill_gaps(trace, t0, tend, maxgap=None):
    """
    Fill gaps in trace data using linear interpolation.
    
    Adapted from AELUMA driver1 gap handling.
    
    Parameters
    ----------
    trace : obspy.Trace
        Input trace (may have gaps represented as masked array)
    t0 : float
        Start epoch time
    tend : float
        End epoch time
    maxgap : int, optional
        Maximum number of missing samples allowed. If exceeded,
        returns None to indicate trace should be skipped.
        
    Returns
    -------
    data : ndarray or None
        Gap-filled data array, or None if gap exceeds maxgap
    times : ndarray
        Corresponding time array
    """
    srate = trace.stats.sampling_rate
    nsampleexp = int((tend - t0) * srate + 1)
    
    # Create expected time vector
    alltime = t0 + np.arange(nsampleexp) / srate
    
    # Get data and time from trace
    data = trace.data
    if hasattr(data, 'mask'):
        # Masked array - find valid samples
        valid = ~data.mask
        data = data.data
    else:
        valid = np.ones(len(data), dtype=bool)
    
    # Create time vector for trace
    trace_times = trace.stats.starttime.timestamp + np.arange(len(data)) / srate
    
    # Count missing samples
    nmiss = nsampleexp - np.sum(valid)
    
    if maxgap is not None and nmiss > maxgap:
        logger.warning(f"{trace.stats.station}: {nmiss} pts missing, exceeds maxgap {maxgap}")
        return None, None
    
    if nmiss > 0:
        # Interpolate to fill gaps
        valid_times = trace_times[valid]
        valid_data = data[valid]
        
        if len(valid_times) < 2:
            logger.warning(f"{trace.stats.station}: insufficient valid samples for interpolation")
            return None, None
        
        f = interp1d(valid_times, valid_data, kind='linear', 
                     fill_value='extrapolate', bounds_error=False)
        filled_data = f(alltime)
        
        if nmiss > maxgap / 2 if maxgap else nmiss > 1000:
            logger.info(f"{trace.stats.station}: filled gap of {nmiss} samples")
        
        return filled_data, alltime
    else:
        return data, trace_times
