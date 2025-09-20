# WE-FDTD Integration Guide

This document provides integration instructions, prerequisites, usage guidelines, and coordinate mapping notes for using WE-FDTD CUDA simulator with the bat_sensory project.

## Overview

The WE-FDTD integration provides a lightweight layer that enables bat_sensory to invoke the WE-FDTD CUDA simulator as a subprocess, prepare input files, and collect receiver waveforms. This integration enables sampling-point FDTD simulations from Python without embedding CUDA code into the repository.

## Prerequisites

### System Requirements

- **CUDA Toolkit**: Version 10.0 or higher
  - NVCC compiler
  - CUDA runtime libraries
  - Compatible NVIDIA GPU with compute capability 3.5+

- **C++ Compiler**: GCC 7.0 or higher
  - Required for building WE-FDTD

- **Python**: Version 3.6 or higher
  - NumPy for array operations
  - Standard library modules (subprocess, pathlib, etc.)

- **MPI (Optional)**: For parallel execution
  - OpenMPI or MPICH
  - Required only for multi-GPU or distributed simulations

### Installation

1. **Install CUDA Toolkit**:
   ```bash
   # Ubuntu/Debian
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
   sudo dpkg -i cuda-keyring_1.0-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda
   
   # Set environment variables
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

2. **Install Build Dependencies**:
   ```bash
   sudo apt-get install build-essential
   sudo apt-get install libopenmpi-dev  # For MPI support
   ```

3. **Verify Installation**:
   ```bash
   nvcc --version
   nvidia-smi
   ```

## Building WE-FDTD

### Using the Build Script

```bash
cd tools/
./build.sh check    # Check dependencies
./build.sh build    # Build WE-FDTD
./build.sh test     # Run integration tests
```

### Using Makefile

```bash
cd tools/
make check-deps     # Verify dependencies
make all           # Build executable
make test          # Run tests
make install       # Install to system (optional)
```

### Manual Build

If you have the actual WE-FDTD source code:

```bash
cd tools/
nvcc -O3 -std=c++11 -gencode arch=compute_70,code=sm_70 \
     -I/usr/local/cuda/include \
     -L/usr/local/cuda/lib64 -lcudart -lcufft \
     -o bin/we-fdtd src/*.cpp src/*.cu
```

## Usage

### Basic Python Integration

```python
from tools.fdtd_runner import FDTDRunner

# Create runner instance
runner = FDTDRunner(fdtd_binary_path="./tools/bin/we-fdtd")

# Configure simulation
config = {
    'domain_size_x': 2.0,    # Domain size in meters
    'domain_size_y': 1.0,
    'domain_size_z': 1.0,
    'grid_points_x': 200,    # Grid resolution
    'grid_points_y': 100,
    'grid_points_z': 100,
    'time_steps': 2000,      # Simulation time steps
    'dt': 5e-7,              # Time step size (seconds)
    'source_freq': 50000.0,  # Source frequency (Hz)
    'source_x': 0.1,         # Source position
    'source_y': 0.5,
    'source_z': 0.5,
    'receivers': [           # Receiver positions
        (1.9, 0.5, 0.5),    # (x, y, z) in meters
        (0.5, 0.5, 0.5),
    ]
}

# Write input file and run simulation
input_file = runner.write_input_dat(config)
results = runner.run_fdtd_simulation(input_file)

if results['success']:
    print(f"Simulation completed: {results['output_dir']}")
    
    # Parse waveform data
    for wave_file in results['wave_files']:
        wave_data = runner.parse_wave_csv(wave_file)
        print(f"Receiver data: {len(wave_data['time'])} samples")
```

### Batch Processing

```python
from tools.run_fdtd_batch import FDTDBatchRunner

# Create batch runner
batch_runner = FDTDBatchRunner(max_workers=4)

# Add multiple cases
for freq in [40000, 45000, 50000, 55000]:
    config = {
        'source_freq': freq,
        'source_x': 0.1,
        'receivers': [(1.5, 0.5, 0.5)]
    }
    batch_runner.add_case(f"freq_{freq}", config)

# Run all cases in parallel
results = batch_runner.run_batch("outputs/frequency_sweep")
```

### Command Line Usage

```bash
# Run demo simulation
python3 tools/fdtd_runner.py --demo

# Batch processing from file
python3 tools/run_fdtd_batch.py --cases-file examples/fdtd_demo/cases_list.txt

# With MPI
python3 tools/run_fdtd_batch.py --cases-file cases.txt --use-mpi --mpi-ranks 4
```

## Coordinate System and Mapping

### FDTD Coordinate System

WE-FDTD uses a Cartesian coordinate system:
- **X-axis**: Horizontal (left-right)
- **Y-axis**: Horizontal (front-back) 
- **Z-axis**: Vertical (up-down)
- **Origin**: Bottom-left-front corner of domain
- **Units**: Meters

### Bat Sensory Coordinate Mapping

When integrating with bat_sensory environments:

```python
def map_bat_to_fdtd(bat_position, bat_angle, domain_size):
    """
    Map bat_sensory coordinates to FDTD coordinates.
    
    Args:
        bat_position: (x, y) from bat_sensory environment
        bat_angle: Bat heading angle in radians
        domain_size: FDTD domain size (x, y, z)
    
    Returns:
        fdtd_coords: (x, y, z) for FDTD
    """
    # Map 2D bat position to 3D FDTD
    fdtd_x = bat_position[0]
    fdtd_y = bat_position[1]
    fdtd_z = domain_size[2] / 2  # Middle height
    
    return (fdtd_x, fdtd_y, fdtd_z)

def calculate_receiver_positions(bat_pos, bat_angle, distances):
    """
    Calculate receiver positions for echolocation simulation.
    
    Args:
        bat_pos: Bat position (x, y, z)
        bat_angle: Bat heading angle
        distances: List of distances for receivers
    
    Returns:
        List of receiver positions
    """
    import math
    
    receivers = []
    for dist in distances:
        rx_x = bat_pos[0] + dist * math.cos(bat_angle)
        rx_y = bat_pos[1] + dist * math.sin(bat_angle)
        rx_z = bat_pos[2]
        receivers.append((rx_x, rx_y, rx_z))
    
    return receivers
```

## Configuration Parameters

### Domain Configuration

- **domain_size_x/y/z**: Physical domain size in meters
- **grid_points_x/y/z**: Number of grid points per dimension
- **boundary_conditions**: 'absorbing', 'reflecting', or 'periodic'

### Time Configuration

- **time_steps**: Number of simulation time steps
- **dt**: Time step size in seconds (typically 1e-6 to 1e-7)

### Source Configuration

- **source_freq**: Source frequency in Hz (typical bat calls: 20-200 kHz)
- **source_x/y/z**: Source position in meters
- **source_type**: 'gaussian_pulse', 'sinusoid', or 'chirp'

### Receiver Configuration

- **receivers**: List of (x, y, z) positions in meters
- **receiver_type**: 'pressure' or 'velocity'

## Performance Considerations

### GPU Memory

Estimate GPU memory requirements:
```
Memory (GB) ≈ (Nx × Ny × Nz × 24 bytes) / 1e9
```

For example:
- 200×100×100 grid: ~4.8 MB
- 500×500×500 grid: ~3.0 GB

### Time Step Stability

Ensure numerical stability with CFL condition:
```
dt ≤ 1 / (c × √(1/dx² + 1/dy² + 1/dz²))
```

Where c is the speed of sound (~343 m/s) and dx, dy, dz are grid spacings.

### MPI Scaling

For large simulations, use MPI:
```bash
mpirun -np 4 python3 tools/run_fdtd_batch.py --cases-file cases.txt --use-mpi
```

## Troubleshooting

### Common Issues

1. **CUDA Errors**:
   - Check GPU memory availability
   - Verify CUDA driver compatibility
   - Reduce grid size if memory issues occur

2. **Build Failures**:
   - Ensure CUDA toolkit is properly installed
   - Check compiler compatibility
   - Verify nvcc is in PATH

3. **Runtime Errors**:
   - Check input file format
   - Verify file permissions
   - Monitor GPU temperature and utilization

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

runner = FDTDRunner(fdtd_binary_path="./bin/we-fdtd")
# ... simulation code
```

### Performance Monitoring

Monitor GPU usage during simulation:
```bash
nvidia-smi -l 1  # Update every second
```

## Integration with Bat Sensory

### Environment Integration

```python
from environments.bat_flying_env import BatFlyingEnv
from tools.fdtd_runner import FDTDRunner

class FDTDBatEnv(BatFlyingEnv):
    """Bat environment with FDTD echolocation simulation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fdtd_runner = FDTDRunner()
    
    def emit_pulse(self, bat_position, bat_angle):
        """Emit echolocation pulse using FDTD."""
        
        # Map bat coordinates to FDTD
        source_pos = map_bat_to_fdtd(bat_position, bat_angle, (2.0, 1.0, 1.0))
        
        # Configure simulation
        config = {
            'source_freq': 50000.0,
            'source_x': source_pos[0],
            'source_y': source_pos[1],
            'source_z': source_pos[2],
            'receivers': calculate_receiver_positions(
                source_pos, bat_angle, [0.5, 1.0, 1.5]
            )
        }
        
        # Run FDTD simulation
        input_file = self.fdtd_runner.write_input_dat(config)
        results = self.fdtd_runner.run_fdtd_simulation(input_file)
        
        if results['success']:
            # Process waveform data for environment
            return self._process_fdtd_results(results)
        else:
            return None
```

## Limitations and Caveats

### Current Limitations

1. **GPU Binding**: Single GPU per process
2. **Memory**: Large simulations may require GPU memory management
3. **Real-time**: Not suitable for real-time applications
4. **Accuracy**: Depends on grid resolution and time step size

### Recommended Usage Patterns

1. **Offline Training**: Pre-compute FDTD responses for training data
2. **Batch Processing**: Process multiple scenarios in parallel
3. **Parameter Studies**: Sweep through different configurations
4. **Validation**: Compare with analytical solutions where possible

### Best Practices

1. **Start Small**: Begin with coarse grids for prototyping
2. **Profile**: Monitor GPU memory and utilization
3. **Validate**: Compare results with known solutions
4. **Document**: Keep track of simulation parameters
5. **Version Control**: Exclude large output files from git

## Support and Resources

### Documentation
- WE-FDTD User Manual: [URL if available]
- CUDA Programming Guide: https://docs.nvidia.com/cuda/
- MPI Tutorial: https://mpitutorial.com/

### Community
- Submit issues to the bat_sensory repository
- FDTD community forums
- CUDA developer forums

### Performance Tips
- Use profiling tools (nvprof, Nsight)
- Optimize grid layouts for memory access
- Consider mixed precision for speed
- Use tensor cores on modern GPUs

---

For questions or issues, please refer to the bat_sensory repository or contact the development team.