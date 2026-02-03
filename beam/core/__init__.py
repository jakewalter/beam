"""
Core algorithms for Gibbons & Ringdal array-based waveform correlation.

This module provides:
- Array geometry and station management
- Signal preprocessing (filtering, gap handling, spike rejection)
- Waveform correlation detection
"""

from .geometry import compute_distance_matrix, cull_stations, compute_array_geometry
from .preprocessing import bandpass_filter, handle_gaps, reject_spikes, compute_envelope

__all__ = [
    'compute_distance_matrix',
    'cull_stations',
    'compute_array_geometry',
    'bandpass_filter',
    'handle_gaps',
    'reject_spikes',
    'compute_envelope',
]
