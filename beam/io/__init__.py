"""
I/O modules for waveform data access.

Provides unified interface for loading seismic waveforms from:
- Local miniseed files
- FDSN web services (IRIS, etc.)
- Station inventory/metadata files
"""

from .waveform_loader import WaveformLoader, load_day_waveforms
from .inventory import load_inventory, get_station_coordinates

__all__ = [
    'WaveformLoader',
    'load_day_waveforms',
    'load_inventory',
    'get_station_coordinates',
]
