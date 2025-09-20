# Bat Agent

## This is a simulation that the bat-like system travel by echolocation.

### New: Acoustic Simulation Module

This repository now includes an optional acoustic simulation module (`acoustic_sim/`) that provides:

- **ApproxAcousticSimulator**: Fast first-order image-source binaural acoustic simulation
- **Route Analysis Tools**: Analyze acoustic features along flight paths  
- **Raster Processing**: Convert GeoTIFF data to obstacle segments

The acoustic simulator can be optionally integrated with the existing `LidarBat` environment for enhanced echolocation simulation while maintaining full backward compatibility.

#### Quick Start with Acoustic Simulation

```python
from environments.lidar_bat import LidarBat, Segment, Point
from acoustic_sim import ApproxAcousticSimulator

# Enable binaural echo simulation (optional)
simulator = ApproxAcousticSimulator()
bat = LidarBat(0, 1, 1, 5, 0.01, echo_simulator=simulator)

# Use normally - will automatically use binaural simulation when available
obstacles = [Segment(Point(5, 0), Point(5, 10))]
observation = bat.emit_pulse(0.0, obstacles)
```

See `acoustic_sim/README.md` for detailed documentation and usage examples.
