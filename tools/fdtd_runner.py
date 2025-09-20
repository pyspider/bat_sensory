"""
FDTD (Finite-Difference Time-Domain) Runner for WE-FDTD Integration

This module provides integration with WE-FDTD for acoustic wave simulation
in bat echolocation analysis.
"""

import logging
import numpy as np
import subprocess
import tempfile
import os
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)


class FDTDResult:
    """Container for FDTD simulation results"""
    def __init__(self, left_waveform=None, right_waveform=None,
                 left_energy=0.0, right_energy=0.0,
                 left_arrival_idx=0, right_arrival_idx=0,
                 itd_seconds=0.0, ild_db=0.0):
        self.left_waveform = left_waveform if left_waveform is not None else np.array([])
        self.right_waveform = right_waveform if right_waveform is not None else np.array([])
        self.left_energy = left_energy
        self.right_energy = right_energy
        self.left_arrival_idx = left_arrival_idx
        self.right_arrival_idx = right_arrival_idx
        self.itd_seconds = itd_seconds
        self.ild_db = ild_db


def run_and_get_wave(point: Tuple[float, float], angle: float,
                     receiver_positions: Dict[str, Tuple[float, float]],
                     obstacle_segments: list,
                     dl: float = 0.01,
                     Nx: int = 256,
                     Ny: int = 256,
                     Ngpu: int = 1,
                     GpuId: int = 0,
                     support_dir: Optional[str] = None,
                     scratch_base: Optional[str] = None,
                     mpirun_np: int = 1,
                     timeout: int = 300) -> Optional[FDTDResult]:
    """
    Run WE-FDTD simulation and return acoustic wave results

    Args:
        point: Source position (x, y)
        angle: Emission angle in radians
        receiver_positions: Dict with 'left' and 'right' receiver positions
        obstacle_segments: List of obstacle segments for the environment,
                           or path to a raster/GeoTIFF that can be converted.
        dl: Grid spacing parameter
        Nx, Ny: Grid dimensions
        Ngpu: Number of GPUs to use
        GpuId: GPU ID for single GPU runs
        support_dir: Directory for support files
        scratch_base: Base directory for scratch files
        mpirun_np: Number of MPI processes
        timeout: Timeout in seconds

    Returns:
        FDTDResult object with simulation results, or None if failed
    """
    try:
        logger.info(f"Starting FDTD simulation at point {point}, angle {angle}")

        # Create temporary directory for simulation files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create simulation configuration
            config_file = os.path.join(temp_dir, "simulation.config")
            _create_config_file(config_file, point, angle, receiver_positions,
                               obstacle_segments, dl, Nx, Ny)

            # Prepare WE-FDTD command
            cmd = _build_wefdtd_command(config_file, temp_dir, Ngpu, GpuId,
                                        support_dir, scratch_base, mpirun_np)

            # Run WE-FDTD simulation
            logger.debug(f"Running WE-FDTD command: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=temp_dir, timeout=timeout,
                                    capture_output=True, text=True)

            if result.returncode != 0:
                # Log both stdout and stderr for debugging
                logger.warning(
                    f"WE-FDTD simulation failed (rc={result.returncode}). stdout: {result.stdout} stderr: {result.stderr}"
                )
                return None

            # Parse results
            return _parse_fdtd_results(temp_dir, receiver_positions)

    except subprocess.TimeoutExpired:
        logger.error(f"FDTD simulation timed out after {timeout} seconds")
        return None
    except Exception as e:
        logger.error(f"FDTD simulation failed with error: {e}")
        return None


def _create_config_file(config_file: str, point: Tuple[float, float], angle: float,
                        receiver_positions: Dict[str, Tuple[float, float]],
                        obstacle_segments: list, dl: float, Nx: int, Ny: int):
    """Create WE-FDTD configuration file"""
    # If a raster path is provided (string), try to convert it to segments.
    segments = obstacle_segments
    if isinstance(obstacle_segments, str):
        raster_path = obstacle_segments
        if os.path.exists(raster_path):
            logger.info(
                f"Obstacle input is a raster path. Attempting to convert raster -> obstacle segments: {raster_path}"
            )
            # Try to use known helper(s). Prefer acoustic_sim.raster_to_obstacles if available.
            converted = None
            try:
                from acoustic_sim.raster_to_obstacles import raster_to_obstacle_segments  # type: ignore
                converted = raster_to_obstacle_segments(raster_path)
                logger.info(
                    "Converted raster to obstacle segments using acoustic_sim.raster_to_obstacles.raster_to_obstacle_segments"
                )
            except Exception:
                # try alternate location
                try:
                    from tools.raster_to_obstacles import raster_to_obstacle_segments  # type: ignore
                    converted = raster_to_obstacle_segments(raster_path)
                    logger.info(
                        "Converted raster to obstacle segments using tools.raster_to_obstacles.raster_to_obstacle_segments"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to import raster->obstacle converter: {e}. No obstacles will be used."
                    )
                    converted = []
            segments = converted if converted is not None else []
        else:
            logger.error(
                f"Provided obstacle raster path does not exist: {raster_path}. Proceeding with no obstacles."
            )
            segments = []

    with open(config_file, 'w') as f:
        f.write(f"# WE-FDTD Configuration\n")
        f.write(f"source_x={point[0]}\n")
        f.write(f"source_y={point[1]}\n")
        f.write(f"source_angle={angle}\n")
        f.write(f"left_receiver_x={receiver_positions['left'][0]}\n")
        f.write(f"left_receiver_y={receiver_positions['left'][1]}\n")
        f.write(f"right_receiver_x={receiver_positions['right'][0]}\n")
        f.write(f"right_receiver_y={receiver_positions['right'][1]}\n")
        f.write(f"dl={dl}\n")
        f.write(f"Nx={Nx}\n")
        f.write(f"Ny={Ny}\n")

        # Write obstacle information
        f.write(f"num_obstacles={len(segments)}\n")
        for i, segment in enumerate(segments):
            if hasattr(segment, 'unpack'):
                x0, y0, x1, y1 = segment.unpack()
            else:
                x0, y0, x1, y1 = segment
            f.write(f"obstacle_{i}={x0},{y0},{x1},{y1}\n")


def _build_wefdtd_command(config_file: str, temp_dir: str, Ngpu: int, GpuId: int,
                         support_dir: Optional[str], scratch_base: Optional[str],
                         mpirun_np: int) -> list:
    """Build WE-FDTD command line"""
    cmd = []

    if mpirun_np > 1:
        cmd.extend(["mpirun", "-np", str(mpirun_np)])

    cmd.extend(["we-fdtd", "--config", config_file])

    if Ngpu > 1:
        cmd.extend(["--ngpu", str(Ngpu)])
    else:
        cmd.extend(["--gpu-id", str(GpuId)])

    if support_dir:
        cmd.extend(["--support-dir", support_dir])

    if scratch_base:
        cmd.extend(["--scratch-base", scratch_base])

    cmd.extend(["--output-dir", temp_dir])

    return cmd


def _parse_fdtd_results(temp_dir: str, receiver_positions: Dict[str, Tuple[float, float]]) -> Optional[FDTDResult]:
    """Parse WE-FDTD output files and create result object"""
    try:
        # Look for output files
        left_wave_file = os.path.join(temp_dir, "left_receiver_waveform.dat")
        right_wave_file = os.path.join(temp_dir, "right_receiver_waveform.dat")

        # Default result in case files don't exist
        result = FDTDResult()

        # Load waveforms if files exist
        if os.path.exists(left_wave_file):
            try:
                result.left_waveform = np.loadtxt(left_wave_file)
            except Exception:
                # If file exists but is malformed, fallback to empty array
                logger.warning(f"Failed to load left waveform file: {left_wave_file}")
                result.left_waveform = np.array([])
            result.left_energy = float(np.sum(result.left_waveform ** 2)) if result.left_waveform.size > 0 else 0.0
            # Find first significant arrival (simple threshold detection)
            threshold = 0.1 * np.max(np.abs(result.left_waveform)) if result.left_waveform.size > 0 else 0.0
            arrivals = np.where(np.abs(result.left_waveform) > threshold)[0] if result.left_waveform.size > 0 else []
            result.left_arrival_idx = int(arrivals[0]) if len(arrivals) > 0 else 0

        if os.path.exists(right_wave_file):
            try:
                result.right_waveform = np.loadtxt(right_wave_file)
            except Exception:
                logger.warning(f"Failed to load right waveform file: {right_wave_file}")
                result.right_waveform = np.array([])
            result.right_energy = float(np.sum(result.right_waveform ** 2)) if result.right_waveform.size > 0 else 0.0
            # Find first significant arrival
            threshold = 0.1 * np.max(np.abs(result.right_waveform)) if result.right_waveform.size > 0 else 0.0
            arrivals = np.where(np.abs(result.right_waveform) > threshold)[0] if result.right_waveform.size > 0 else []
            result.right_arrival_idx = int(arrivals[0]) if len(arrivals) > 0 else 0

        # Calculate ITD (Inter-aural Time Difference) and ILD (Inter-aural Level Difference)
        if result.left_waveform.size > 0 and result.right_waveform.size > 0:
            # Assume sampling rate of 44.1 kHz for time conversion (consider parameterizing later)
            dt = 1.0 / 44100.0
            result.itd_seconds = (result.right_arrival_idx - result.left_arrival_idx) * dt

            # ILD in dB
            if result.left_energy > 0 and result.right_energy > 0:
                result.ild_db = 10 * np.log10(result.right_energy / result.left_energy)

        logger.info(f"FDTD simulation completed successfully. ITD: {result.itd_seconds:.6f}s, ILD: {result.ild_db:.2f}dB")
        return result

    except Exception as e:
        logger.error(f"Failed to parse FDTD results: {e}")
        return None
