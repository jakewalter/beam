"""
BEAM - Beamforming for Earthquake Array Monitoring

An independent implementation of the Gibbons & Ringdal (2006) 
array-based waveform correlation methodology for detecting 
low-magnitude seismic events.

This package provides:
- Waveform loading from FDSN web services and local files
- Array geometry management and station culling  
- Signal preprocessing (bandpass, gap handling, spike rejection)
- Master event correlation detection
- Array beam stacking for event detection

Data access patterns are adapted from the AELUMA codebase for
consistency, but the detection methodology follows Gibbons & Ringdal.
"""

__version__ = "0.1.0"
__author__ = "Beam Development Team"

from . import core
from . import io

__all__ = ['core', 'io']
