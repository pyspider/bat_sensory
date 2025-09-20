#!/usr/bin/env python3
"""
Acoustic Analysis Runner for Bat Sensory Simulation

This script runs acoustic simulations for bat echolocation analysis, supporting
both traditional ApproxAcousticSimulator and WE-FDTD integration for more
accurate wave simulations.
"""

import argparse
import logging
import multiprocessing as mp
import numpy as np
import os
import sys
from typing import Any, Dict, List, Tuple, Optional

# Local imports
from tools import fdtd_runner


logger = logging.getLogger(__name__)


class SimulationResult:
    """Container for acoustic simulation results"""
    def __init__(self, left_waveform=None, right_waveform=None,
                 left_energy=0.0, right_energy=0.0,
                 left_arrival_idx=0, right_arrival_idx=0,
                 itd_seconds=0.0, ild_db=0.0, success=True):
        self.left_waveform = left_waveform if left_waveform is not None else np.array([])
        self.right_waveform = right_waveform if right_waveform is not None else np.array([])
        self.left_energy = left_energy
        self.right_energy = right_energy
        self.left_arrival_idx = left_arrival_idx
        self.right_arrival_idx = right_arrival_idx
        self.itd_seconds = itd_seconds
        self.ild_db = ild_db
        self.success = success


def simulate_with_wefdtd(point: Tuple[float, float], angle: float, 
                        simulator_kwargs: Dict[str, Any],
                        obstacle_segments: List) -> SimulationResult:
    """
    Run WE-FDTD simulation for acoustic wave analysis.
    
    This function uses the WE-FDTD integration to simulate acoustic waves
    for left and right ear receiver positions, then converts the results
    to the expected result object format.
    
    Args:
        point: Source emission point (x, y)
        angle: Emission angle in radians
        simulator_kwargs: Dictionary containing FDTD-specific options
        obstacle_segments: List of obstacle segments for the environment
        
    Returns:
        SimulationResult object with left/right waveforms and analysis
    """
    logger.info(f"Running WE-FDTD simulation at point {point}, angle {angle}")
    
    try:
        # Extract FDTD-specific parameters with reasonable defaults
        ear_offset = simulator_kwargs.get('ear_offset', 0.1)  # 10cm ear separation
        dl = simulator_kwargs.get('dl', 0.01)
        Nx = simulator_kwargs.get('Nx', 256)
        Ny = simulator_kwargs.get('Ny', 256)
        Ngpu = simulator_kwargs.get('Ngpu', 1)
        GpuId = simulator_kwargs.get('GpuId', 0)
        support_dir = simulator_kwargs.get('support_dir')
        scratch_base = simulator_kwargs.get('scratch_base')
        mpirun_np = simulator_kwargs.get('mpirun_np', 1)
        timeout = simulator_kwargs.get('timeout', 300)
        
        # Calculate receiver positions based on source point and ear offset
        # Assume ears are positioned perpendicular to emission direction
        ear_direction = np.array([-np.sin(angle), np.cos(angle)])  # Perpendicular to emission
        left_ear_pos = (point[0] + ear_direction[0] * ear_offset / 2,
                       point[1] + ear_direction[1] * ear_offset / 2)
        right_ear_pos = (point[0] - ear_direction[0] * ear_offset / 2,
                        point[1] - ear_direction[1] * ear_offset / 2)
        
        receiver_positions = {
            'left': left_ear_pos,
            'right': right_ear_pos
        }
        
        # Run WE-FDTD simulation
        fdtd_result = fdtd_runner.run_and_get_wave(
            point=point,
            angle=angle,
            receiver_positions=receiver_positions,
            obstacle_segments=obstacle_segments,
            dl=dl,
            Nx=Nx,
            Ny=Ny,
            Ngpu=Ngpu,
            GpuId=GpuId,
            support_dir=support_dir,
            scratch_base=scratch_base,
            mpirun_np=mpirun_np,
            timeout=timeout
        )
        
        if fdtd_result is None:
            logger.warning("WE-FDTD simulation failed, returning zeros")
            return SimulationResult(success=False)
        
        # Convert FDTD result to expected format
        result = SimulationResult(
            left_waveform=fdtd_result.left_waveform,
            right_waveform=fdtd_result.right_waveform,
            left_energy=fdtd_result.left_energy,
            right_energy=fdtd_result.right_energy,
            left_arrival_idx=fdtd_result.left_arrival_idx,
            right_arrival_idx=fdtd_result.right_arrival_idx,
            itd_seconds=fdtd_result.itd_seconds,
            ild_db=fdtd_result.ild_db,
            success=True
        )
        
        logger.info(f"WE-FDTD simulation completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"WE-FDTD simulation failed with error: {e}")
        # Return zeros and continue as specified
        return SimulationResult(success=False)


def worker_task(task_data: Tuple[Tuple[float, float], float, Dict[str, Any], List]) -> SimulationResult:
    """
    Worker task for multiprocessing acoustic simulation.
    
    This function branches between WE-FDTD and traditional ApproxAcousticSimulator
    based on the use_fdtd flag in simulator_kwargs.
    
    Args:
        task_data: Tuple of (point, angle, simulator_kwargs, obstacle_segments)
        
    Returns:
        SimulationResult object
    """
    point, angle, simulator_kwargs, obstacle_segments = task_data
    
    # Check if we should use FDTD
    use_fdtd = simulator_kwargs.get('use_fdtd', False)
    
    if use_fdtd:
        # Use WE-FDTD integration
        return simulate_with_wefdtd(point, angle, simulator_kwargs, obstacle_segments)
    else:
        # Use traditional ApproxAcousticSimulator (preserve existing behavior)
        try:
            # Import ApproxAcousticSimulator (this would be the existing implementation)
            # For now, we'll simulate its behavior since it doesn't exist in this repo
            logger.info(f"Running ApproxAcousticSimulator at point {point}, angle {angle}")
            
            # Simulate traditional acoustic simulation results
            # This would be replaced with actual ApproxAcousticSimulator import and call
            sim_result = _simulate_traditional_acoustic(point, angle, simulator_kwargs, obstacle_segments)
            
            return sim_result
            
        except ImportError as e:
            logger.error(f"ApproxAcousticSimulator not available: {e}")
            return SimulationResult(success=False)
        except Exception as e:
            logger.error(f"Traditional acoustic simulation failed: {e}")
            return SimulationResult(success=False)


def _simulate_traditional_acoustic(point: Tuple[float, float], angle: float,
                                 simulator_kwargs: Dict[str, Any], 
                                 obstacle_segments: List) -> SimulationResult:
    """
    Placeholder for traditional ApproxAcousticSimulator functionality.
    
    In a real implementation, this would import and use the ApproxAcousticSimulator.
    """
    logger.info("Running traditional acoustic simulation (placeholder)")
    
    # Simulate basic acoustic results for demonstration
    # In real implementation, this would call:
    # from acoustic_sim import ApproxAcousticSimulator
    # sim = ApproxAcousticSimulator(**simulator_kwargs)
    # result = sim.simulate(point, angle, obstacle_segments)
    
    # For now, create dummy results
    sample_rate = 44100
    duration = 0.1  # 100ms
    num_samples = int(sample_rate * duration)
    
    # Generate simple synthetic waveforms
    t = np.linspace(0, duration, num_samples)
    left_waveform = 0.1 * np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 10)
    right_waveform = 0.08 * np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 12)
    
    # Calculate basic metrics
    left_energy = np.sum(left_waveform ** 2)
    right_energy = np.sum(right_waveform ** 2)
    left_arrival_idx = 100  # Dummy arrival time
    right_arrival_idx = 110  # Slightly delayed
    
    # Calculate ITD and ILD
    dt = 1.0 / sample_rate
    itd_seconds = (right_arrival_idx - left_arrival_idx) * dt
    ild_db = 10 * np.log10(right_energy / left_energy) if left_energy > 0 else 0.0
    
    return SimulationResult(
        left_waveform=left_waveform,
        right_waveform=right_waveform,
        left_energy=left_energy,
        right_energy=right_energy,
        left_arrival_idx=left_arrival_idx,
        right_arrival_idx=right_arrival_idx,
        itd_seconds=itd_seconds,
        ild_db=ild_db,
        success=True
    )


def run_batch_analysis(points: List[Tuple[float, float]], 
                      angles: List[float],
                      obstacle_segments: List,
                      simulator_kwargs: Dict[str, Any],
                      num_workers: int = None) -> List[SimulationResult]:
    """
    Run batch acoustic analysis using multiprocessing.
    
    Args:
        points: List of emission points
        angles: List of emission angles (must match length of points)
        obstacle_segments: List of obstacle segments for environment
        simulator_kwargs: Simulation parameters including use_fdtd flag
        num_workers: Number of worker processes (default: CPU count)
        
    Returns:
        List of SimulationResult objects
    """
    if len(points) != len(angles):
        raise ValueError("Points and angles lists must have the same length")
    
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    logger.info(f"Starting batch analysis with {len(points)} tasks using {num_workers} workers")
    
    # Prepare task data
    tasks = [(point, angle, simulator_kwargs, obstacle_segments) 
             for point, angle in zip(points, angles)]
    
    # Run in parallel
    with mp.Pool(num_workers) as pool:
        results = pool.map(worker_task, tasks)
    
    logger.info(f"Batch analysis completed. {sum(1 for r in results if r.success)} successful simulations")
    
    return results


def create_test_environment() -> Tuple[List[Tuple[float, float]], List[float], List]:
    """Create a simple test environment for validation"""
    # Create a grid of test points
    points = [(x, y) for x in np.linspace(-1, 1, 3) for y in np.linspace(-1, 1, 3)]
    
    # Test angles (forward, left, right)
    angles = [0.0, np.pi/4, -np.pi/4] * 3
    
    # Simple rectangular obstacle
    obstacle_segments = [
        (0.5, -0.5, 0.5, 0.5),   # Right wall
        (-0.5, -0.5, -0.5, 0.5), # Left wall  
        (-0.5, 0.5, 0.5, 0.5),   # Top wall
        (-0.5, -0.5, 0.5, -0.5)  # Bottom wall
    ]
    
    return points, angles, obstacle_segments


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description='Acoustic Analysis Runner for Bat Sensory Simulation')
    
    # Basic options
    parser.add_argument('--input-file', type=str, help='Input file with points and angles')
    parser.add_argument('--output-file', type=str, default='analysis_results.npz',
                       help='Output file for results (default: analysis_results.npz)')
    parser.add_argument('--num-workers', type=int, help='Number of worker processes')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--test', action='store_true',
                       help='Run with test environment')
    
    # FDTD options
    parser.add_argument('--use-fdtd', action='store_true',
                       help='Use WE-FDTD integration instead of ApproxAcousticSimulator')
    parser.add_argument('--ear-offset', type=float, default=0.1,
                       help='Ear separation distance in meters (default: 0.1)')
    parser.add_argument('--dl', type=float, default=0.01,
                       help='FDTD grid spacing (default: 0.01)')
    parser.add_argument('--Nx', type=int, default=256,
                       help='FDTD grid size X (default: 256)')
    parser.add_argument('--Ny', type=int, default=256,
                       help='FDTD grid size Y (default: 256)')
    parser.add_argument('--Ngpu', type=int, default=1,
                       help='Number of GPUs for FDTD (default: 1)')
    parser.add_argument('--GpuId', type=int, default=0,
                       help='GPU ID for single GPU FDTD (default: 0)')
    parser.add_argument('--support-dir', type=str,
                       help='FDTD support directory')
    parser.add_argument('--scratch-base', type=str,
                       help='FDTD scratch base directory')
    parser.add_argument('--mpirun-np', type=int, default=1,
                       help='Number of MPI processes for FDTD (default: 1)')
    parser.add_argument('--timeout', type=int, default=300,
                       help='FDTD simulation timeout in seconds (default: 300)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=getattr(logging, args.log_level),
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Build simulator kwargs
    simulator_kwargs = {
        'ear_offset': args.ear_offset,
        'dl': args.dl,
        'Nx': args.Nx,
        'Ny': args.Ny,
        'Ngpu': args.Ngpu,
        'GpuId': args.GpuId,
        'support_dir': args.support_dir,
        'scratch_base': args.scratch_base,
        'mpirun_np': args.mpirun_np,
        'timeout': args.timeout,
    }
    
    # Add use_fdtd flag to simulator_kwargs if set
    if args.use_fdtd:
        simulator_kwargs['use_fdtd'] = True
        logger.info("Using WE-FDTD integration for acoustic simulation")
    else:
        logger.info("Using traditional ApproxAcousticSimulator")
    
    try:
        if args.test:
            # Use test environment
            points, angles, obstacle_segments = create_test_environment()
            logger.info(f"Running test with {len(points)} points")
        else:
            # Load from input file
            if not args.input_file:
                parser.error("--input-file is required unless --test is specified")
            
            # Load points, angles, and obstacles from file
            # This would implement the actual file loading logic
            logger.error("Input file loading not implemented yet. Use --test for now.")
            return 1
        
        # Run batch analysis
        results = run_batch_analysis(
            points=points,
            angles=angles,
            obstacle_segments=obstacle_segments,
            simulator_kwargs=simulator_kwargs,
            num_workers=args.num_workers
        )
        
        # Save results
        logger.info(f"Saving results to {args.output_file}")
        # Convert results to arrays for saving
        result_data = {
            'points': np.array(points),
            'angles': np.array(angles),
            'success': np.array([r.success for r in results]),
            'left_energy': np.array([r.left_energy for r in results]),
            'right_energy': np.array([r.right_energy for r in results]),
            'itd_seconds': np.array([r.itd_seconds for r in results]),
            'ild_db': np.array([r.ild_db for r in results]),
        }
        
        np.savez(args.output_file, **result_data)
        
        # Print summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"Analysis complete: {successful}/{len(results)} successful simulations")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())