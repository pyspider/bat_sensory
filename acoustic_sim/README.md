# Acoustic Simulation Module

This module provides approximate binaural acoustic simulation and route analysis tools for bat echolocation research. It includes fast image-source based echo simulation and tools for analyzing acoustic features along flight routes.

## Components

### ApproxAcousticSimulator
A fast first-order image-source binaural acoustic simulator that:
- Synthesizes left/right ear waveforms from obstacle reflections
- Computes first-arrival times, ITD (Interaural Time Difference), and ILD (Interaural Level Difference)
- Provides configurable parameters for sample rate, pulse frequency, reflection coefficient, and ear offset
- Uses efficient reflection geometry for real-time applications

### Route Analysis Tools
Tools for analyzing acoustic characteristics along flight paths:
- Sample points along route LineStrings
- Compute binaural echoes at each sample point using multiple beam angles
- Extract features: angle energy entropy, ITD/ILD statistics, spectral entropy
- Output results to CSV and summary formats

### Raster Processing
Convert GeoTIFF raster data to obstacle line segments:
- Read single-band raster files using rasterio
- Apply binary thresholding to create obstacle masks
- Vectorize to polygons and extract boundary segments
- Merge and simplify geometry for efficient simulation

## Dependencies

Core simulator (always required):
- numpy

Route analysis (optional):
- shapely
- scipy

Raster processing (optional):
- rasterio
- fiona 
- shapely

Install optional dependencies with:
```bash
pip install rasterio fiona shapely scipy
```

## Usage Examples

### Basic Acoustic Simulation

```python
from acoustic_sim.approx_acoustic_simulator import ApproxAcousticSimulator
from environments.lidar_bat import Segment, Point

# Create simulator
simulator = ApproxAcousticSimulator(
    sample_rate=44100,
    pulse_frequency=40000,
    reflection_coeff=0.7,
    ear_offset=0.02
)

# Create some obstacle segments
obstacles = [
    Segment(Point(5, 0), Point(5, 10)),  # Vertical wall
    Segment(Point(0, 5), Point(10, 5))   # Horizontal wall
]

# Simulate binaural echo
source_pos = (2, 2)
source_angle = 0.0  # Facing right
left_ear = (1.99, 2)   # Left ear position
right_ear = (2.01, 2)  # Right ear position

echo_result = simulator.simulate(
    source_pos=source_pos,
    source_angle=source_angle,
    receivers=[left_ear, right_ear],
    scene_segments=obstacles
)

if echo_result:
    print(f"ITD: {echo_result.itd:.6f} seconds")
    print(f"ILD: {echo_result.ild:.2f} dB")
    print(f"Total energy: {echo_result.energy:.3f}")
```

### Integration with LidarBat

```python
from environments.lidar_bat import LidarBat, Segment, Point
from acoustic_sim.approx_acoustic_simulator import ApproxAcousticSimulator

# Create acoustic simulator
echo_simulator = ApproxAcousticSimulator()

# Create LidarBat with acoustic simulation
bat = LidarBat(
    init_angle=0.0, 
    init_x=1.0, 
    init_y=1.0, 
    init_speed=5.0, 
    dt=0.01,
    echo_simulator=echo_simulator  # Enable binaural simulation
)

# Create some obstacles
obstacles = [Segment(Point(5, 0), Point(5, 10))]

# When emit_pulse is called, it will use binaural simulation if available
observation = bat.emit_pulse(0.0, obstacles)
print(f"Binaural observation: {observation}")

# For comparison, create a bat without echo simulator (original behavior)
bat_original = LidarBat(0.0, 1.0, 1.0, 5.0, 0.01)
observation_orig = bat_original.emit_pulse(0.0, obstacles)
print(f"Original observation: {observation_orig}")
```

### Route Analysis

```python
from acoustic_sim.analyze_routes import analyze_routes
from shapely.geometry import LineString

# Define flight routes
routes = [
    LineString([(0, 0), (10, 5), (20, 0)]),  # Route 1
    LineString([(0, 10), (10, 5), (20, 10)]) # Route 2
]

# Define obstacles (convert from your obstacle format)
obstacles = [...]  # List of obstacle segments

# Analyze acoustic features along routes
features = analyze_routes(
    routes=routes,
    obstacle_segments=obstacles,
    sample_spacing=1.0,  # Sample every 1 meter
    output_csv="route_analysis.csv",
    output_summary="route_summary.txt"
)

print(f"Analyzed {len(features)} sample points")
```

### Raster to Obstacles

```python
from acoustic_sim.raster_to_obstacles import raster_to_obstacle_segments

# Convert GeoTIFF to obstacle line segments
segments = raster_to_obstacle_segments(
    raster_path="environment_map.tif",
    threshold=0.5,           # Binary threshold
    simplify_tolerance=0.1,  # Polygon simplification
    merge_buffer=0.05        # Merge nearby polygons
)

print(f"Extracted {len(segments)} obstacle segments")

# Convert to format compatible with LidarBat
simple_segments = convert_segments_to_simple_format(segments)
```

## Features Computed

The route analysis tool computes these acoustic features at each sample point:

- **angle_energy_entropy**: Shannon entropy of energy distribution across beam angles (higher = more uniform)
- **mean_itd**: Mean interaural time difference across beam angles
- **std_itd**: Standard deviation of ITD values
- **mean_ild**: Mean interaural level difference
- **mean_spectral_entropy**: Mean spectral entropy of echo waveforms
- **total_energy**: Sum of echo energy across all beam angles
- **max_energy**: Maximum echo energy from any beam angle

## Performance Notes

- The simulator uses first-order reflections only for speed
- Typical performance: ~1000 simulations per second on modern hardware
- Memory usage scales with audio sample rate and simulation duration
- For real-time applications, consider lower sample rates (22 kHz) and shorter pulses

## Limitations

- First-order reflections only (no multiple bounces)
- Simplified acoustic propagation model
- No frequency-dependent absorption or scattering
- Assumes flat obstacles (no surface roughness effects)

These limitations make the simulator suitable for rapid prototyping and analysis but not for high-fidelity acoustic modeling.