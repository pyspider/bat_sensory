# Acoustic Simulation Toolkit

This module provides optional acoustic simulation capabilities for binaural echo analysis in bat echolocation scenarios. It offers fast approximate binaural echo simulations and route analysis tools for assessing information complexity along migration routes.

## Features

- **ApproxAcousticSimulator**: 1st-order image-source binaural simulator
- **Raster-to-obstacles conversion**: Extract obstacle boundaries from GeoTIFF files
- **Route analysis**: Analyze acoustic complexity along flight paths
- **Minimal integration**: Optional add-on that doesn't affect existing LidarBat functionality

## Components

### 1. ApproxAcousticSimulator (`approx_acoustic_simulator.py`)

Provides binaural acoustic simulation using the image source method.

**Key features:**
- Synthesizes left/right ear waveforms
- Computes first-arrival indices, ITD (Interaural Time Difference), and ILD (Interaural Level Difference)
- Configurable parameters: sound speed, sample rate, pulse characteristics, reflection coefficient
- 1st-order reflections for fast approximate simulation

**Example usage:**
```python
from acoustic_sim.approx_acoustic_simulator import ApproxAcousticSimulator

# Initialize simulator
simulator = ApproxAcousticSimulator(
    c=343.0,                    # Speed of sound (m/s)
    sample_rate=44100,          # Audio sample rate (Hz)
    pulse_center_freq=50000.0,  # Echolocation frequency (Hz)
    pulse_duration_ms=2.0,      # Pulse duration (ms)
    reflection_coeff=0.7,       # Wall reflection coefficient
    ear_offset=0.02             # Distance between ears (m)
)

# Simulate binaural response
source_pos = (0.0, 0.0)
ear_positions = [(-0.01, 0.0), (0.01, 0.0)]  # Left, right ears
obstacle_segments = [((1.0, -0.5), (1.0, 0.5))]  # Wall segment

result = simulator.simulate(
    source_pos=source_pos,
    source_angle=0.0,
    ear_positions=ear_positions,
    scene_segments=obstacle_segments
)

print(f"ITD: {result.itd_seconds:.6f} seconds")
print(f"ILD: {result.ild_db:.2f} dB")
```

### 2. Raster Processing (`raster_to_obstacles.py`)

Converts raster data (GeoTIFF) to obstacle boundary segments.

**Features:**
- Reads single-band raster files
- Applies threshold for binary classification
- Vectorizes and merges polygons
- Extracts boundary segments as LineStrings

**Example usage:**
```python
from acoustic_sim.raster_to_obstacles import raster_to_obstacle_segments

# Extract obstacle segments from GeoTIFF
segments = raster_to_obstacle_segments(
    raster_path="obstacles.tif",
    threshold=0.5,
    simplify_tolerance=0.1
)

print(f"Extracted {len(segments)} obstacle segments")
```

### 3. Route Analysis (`analyze_routes.py`)

CLI script for analyzing acoustic complexity along migration routes.

**Features:**
- Samples points along route LineStrings
- Runs acoustic simulation for multiple beam angles at each point
- Computes acoustic complexity metrics:
  - Angle energy entropy (Shannon entropy over beam energies)
  - ITD and ILD statistics
  - Spectral entropy
- Outputs CSV results and summary statistics

**Example usage:**
```bash
# Analyze routes from GeoJSON file
python -m acoustic_sim.analyze_routes routes.geojson output_results \
    --spacing 10.0 \
    --beam-angles 16 \
    --obstacle-raster obstacles.tif
```

**Output files:**
- `output_results.csv`: Per-point analysis results
- `output_results_summary.txt`: Overall statistics

## Integration with LidarBat

The acoustic simulator can be optionally integrated with the existing LidarBat environment:

```python
from environments.lidar_bat import LidarBat
from acoustic_sim.approx_acoustic_simulator import ApproxAcousticSimulator

# Create LidarBat with acoustic simulation
simulator = ApproxAcousticSimulator()
bat = LidarBat(
    init_angle=0.0, init_x=1.0, init_y=1.0, 
    init_speed=5.0, dt=0.005, 
    echo_simulator=simulator  # Optional parameter
)

# Normal operation - acoustic simulation runs automatically when available
observation = bat.emit_pulse(0.0, obstacle_segments)
```

When `echo_simulator` is provided, the `emit_pulse` method will:
1. Compute ear positions based on bat position and orientation
2. Run acoustic simulation
3. Estimate distance from echo arrival times
4. Provide acoustic-based observation as fallback or supplement

## Dependencies

### Core dependencies (always required):
- `numpy`

### Optional dependencies (only required for specific features):
- `rasterio` and `shapely`: For raster processing (`raster_to_obstacles.py`)
- `scipy`: For spectral analysis (`analyze_routes.py`)
- `fiona`: For reading geographic vector data (`analyze_routes.py`)

### Installation:

For basic acoustic simulation:
```bash
pip install numpy
```

For full functionality:
```bash
pip install numpy rasterio shapely scipy fiona
```

## Quick Start Example

```python
# Basic acoustic simulation
from acoustic_sim.approx_acoustic_simulator import ApproxAcousticSimulator

# Create simulator
simulator = ApproxAcousticSimulator()

# Define simple scene
source_pos = (0.0, 0.0)
ear_positions = [(-0.01, 0.0), (0.01, 0.0)]  # Left, right ears
wall_segments = [((1.0, -0.5), (1.0, 0.5))]  # Vertical wall

# Run simulation
result = simulator.simulate(
    source_pos=source_pos,
    source_angle=0.0,
    ear_positions=ear_positions,
    scene_segments=wall_segments
)

print(f"ITD: {result.itd_seconds*1e6:.1f} μs")
print(f"ILD: {result.ild_db:.2f} dB")

# Integrate with LidarBat
from environments.lidar_bat import LidarBat

bat = LidarBat(0.0, 1.0, 1.0, 5.0, 0.005, echo_simulator=simulator)
observation = bat.emit_pulse(0.0, [])  # Acoustic simulation runs automatically
```

See `acoustic_sim/demo.py` for a complete demonstration.

## Design Principles

1. **Optional**: Does not affect existing functionality when not used
2. **Modular**: Each component can be used independently
3. **Local imports**: Heavy dependencies are imported only where needed
4. **Backward compatible**: Existing LidarBat code works unchanged
5. **Fast approximation**: Prioritizes speed over acoustic accuracy for large-scale analysis

## Performance Considerations

- The simulator uses 1st-order reflections only for speed
- Default parameters are optimized for bat echolocation scenarios
- For large-scale route analysis, consider:
  - Adjusting sample spacing based on analysis requirements
  - Reducing number of beam angles for faster processing
  - Using simplified obstacle representations

## Limitations

- Simplified acoustic model (geometric acoustics, no diffraction)
- 1st-order reflections only
- Fixed reflection coefficient per simulation
- No frequency-dependent attenuation
- No detailed head-related transfer functions (HRTFs)

These limitations are intentional trade-offs for computational efficiency in large-scale route analysis scenarios.

## Example Workflow

1. **Prepare obstacle data**: Convert environmental raster to obstacle segments
2. **Define routes**: Create route LineStrings (GeoJSON/Shapefile)
3. **Run analysis**: Use `analyze_routes.py` to compute acoustic complexity
4. **Analyze results**: Process CSV output to identify high/low complexity areas
5. **Integrate with simulation**: Use ApproxAcousticSimulator in LidarBat environment

This toolkit enables researchers to assess how environmental acoustic complexity varies along migration routes and incorporate these effects into agent-based bat simulations.