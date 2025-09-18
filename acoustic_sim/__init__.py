"""
Acoustic simulation module for bat echolocation research.

This module provides approximate binaural acoustic simulation tools including:
- ApproxAcousticSimulator: Fast image-source based binaural echo simulation
- Route analysis tools for computing acoustic features along flight paths
- Raster processing tools for converting GeoTIFF data to obstacle segments

The simulator integrates with the LidarBat environment to provide optional
binaural echo simulation while maintaining compatibility with existing code.
"""

from .approx_acoustic_simulator import ApproxAcousticSimulator, EchoResult

__all__ = ['ApproxAcousticSimulator', 'EchoResult']