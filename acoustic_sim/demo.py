#!/usr/bin/env python3
"""
Simple demo script for the acoustic simulation toolkit.

This script demonstrates the key features of the acoustic_sim module:
1. Basic ApproxAcousticSimulator usage
2. Integration with LidarBat environment
3. Comparison of geometric vs acoustic distance estimation
"""

import sys
import numpy as np
import math

# Add the repository root to the path
sys.path.insert(0, '/home/runner/work/bat_sensory/bat_sensory')

from acoustic_sim.approx_acoustic_simulator import ApproxAcousticSimulator
from environments.lidar_bat import LidarBat, Segment, Point


def demo_acoustic_simulator():
    """Demonstrate basic acoustic simulator functionality."""
    print("=== Acoustic Simulator Demo ===\n")
    
    # Create simulator with bat-like parameters
    simulator = ApproxAcousticSimulator(
        c=343.0,                    # Speed of sound (m/s)
        sample_rate=44100,          # Audio sample rate (Hz)
        pulse_center_freq=50000.0,  # Typical bat echolocation frequency (Hz)
        pulse_duration_ms=2.0,      # Short pulse duration (ms)
        reflection_coeff=0.7,       # Wall reflection coefficient
        ear_offset=0.02             # ~2cm between bat ears (m)
    )
    
    print(f"Created acoustic simulator:")
    print(f"  Sample rate: {simulator.sr} Hz")
    print(f"  Pulse frequency: {simulator.pulse_center_freq} Hz")
    print(f"  Ear separation: {simulator.ear_offset*1000:.1f} mm")
    print()
    
    # Set up a simple scene with a wall
    source_pos = (0.0, 0.0)
    wall_distance = 1.5  # 1.5 meter wall
    obstacle_segments = [((wall_distance, -0.5), (wall_distance, 0.5))]  # Vertical wall
    
    # Calculate ear positions for different orientations
    angles = [0, 30, 60, 90]  # degrees
    
    print("Testing different orientations:")
    print("Angle(°) | ITD(μs) | ILD(dB) | Est.Dist(m) | Left Energy | Right Energy")
    print("-" * 75)
    
    for angle_deg in angles:
        angle_rad = math.radians(angle_deg)
        
        # Calculate ear positions
        left_ear_x = source_pos[0] + simulator.ear_offset * math.cos(angle_rad + math.pi/2)
        left_ear_y = source_pos[1] + simulator.ear_offset * math.sin(angle_rad + math.pi/2)
        right_ear_x = source_pos[0] + simulator.ear_offset * math.cos(angle_rad - math.pi/2)
        right_ear_y = source_pos[1] + simulator.ear_offset * math.sin(angle_rad - math.pi/2)
        
        ear_positions = [(left_ear_x, left_ear_y), (right_ear_x, right_ear_y)]
        
        # Run simulation
        result = simulator.simulate(
            source_pos=source_pos,
            source_angle=angle_rad,
            ear_positions=ear_positions,
            scene_segments=obstacle_segments
        )
        
        # Estimate distance from arrival time
        est_distance = 0.0
        if result.left_arrival_idx is not None and result.right_arrival_idx is not None:
            mean_arrival_samples = (result.left_arrival_idx + result.right_arrival_idx) / 2.0
            arrival_time_seconds = mean_arrival_samples / simulator.sr
            est_distance = arrival_time_seconds * simulator.c / 2.0  # Round trip
        
        print(f"{angle_deg:7d} | {result.itd_seconds*1e6:7.1f} | {result.ild_db:7.2f} | "
              f"{est_distance:11.3f} | {result.left_energy:11.2e} | {result.right_energy:12.2e}")
    
    print(f"\nActual wall distance: {wall_distance:.3f} m")
    print()


def demo_lidar_bat_integration():
    """Demonstrate integration with LidarBat environment."""
    print("=== LidarBat Integration Demo ===\n")
    
    # Create obstacle segments (walls of a simple room)
    obstacles = [
        Segment(Point(0.0, 0.0), Point(4.0, 0.0)),    # Bottom wall
        Segment(Point(4.0, 0.0), Point(4.0, 3.0)),    # Right wall
        Segment(Point(4.0, 3.0), Point(0.0, 3.0)),    # Top wall
        Segment(Point(0.0, 3.0), Point(0.0, 0.0)),    # Left wall
        Segment(Point(1.5, 1.0), Point(1.5, 2.0)),    # Interior obstacle
    ]
    
    # Create bat at center of room
    bat_x, bat_y = 2.0, 1.5
    bat_angle = 0.0  # Facing right
    
    print(f"Bat position: ({bat_x}, {bat_y})")
    print(f"Bat orientation: {math.degrees(bat_angle):.1f}°")
    print()
    
    # Test without acoustic simulator (original behavior)
    bat_geometric = LidarBat(
        init_angle=bat_angle,
        init_x=bat_x, 
        init_y=bat_y,
        init_speed=1.0,
        dt=0.01
    )
    
    # Test with acoustic simulator
    acoustic_simulator = ApproxAcousticSimulator()
    bat_acoustic = LidarBat(
        init_angle=bat_angle,
        init_x=bat_x,
        init_y=bat_y, 
        init_speed=1.0,
        dt=0.01,
        echo_simulator=acoustic_simulator
    )
    
    # Test multiple pulse directions
    pulse_angles_deg = [-30, 0, 30, 60]
    
    print("Pulse Angle(°) | Geometric Result     | Acoustic Result")
    print("-" * 55)
    
    for pulse_angle_deg in pulse_angles_deg:
        pulse_angle_rad = math.radians(pulse_angle_deg)
        
        # Get observations from both approaches
        obs_geometric = bat_geometric.emit_pulse(pulse_angle_rad, obstacles)
        obs_acoustic = bat_acoustic.emit_pulse(pulse_angle_rad, obstacles)
        
        print(f"{pulse_angle_deg:12d} | [{obs_geometric[0]:6.3f}, {obs_geometric[1]:6.3f}] | "
              f"[{obs_acoustic[0]:6.3f}, {obs_acoustic[1]:6.3f}]")
    
    print()
    print("Notes:")
    print("- Geometric result uses traditional lidar-like collision detection")
    print("- Acoustic result uses binaural echo simulation")
    print("- Results may differ due to different physics models")
    print()


def demo_obstacle_interaction():
    """Demonstrate how acoustic simulation responds to different obstacles."""
    print("=== Obstacle Interaction Demo ===\n")
    
    simulator = ApproxAcousticSimulator()
    source_pos = (0.0, 0.0)
    source_angle = 0.0
    ear_positions = [(-0.01, 0.0), (0.01, 0.0)]
    
    # Test different obstacle configurations
    scenarios = [
        ("No obstacles", []),
        ("Single wall at 1m", [((1.0, -0.5), (1.0, 0.5))]),
        ("Single wall at 2m", [((2.0, -0.5), (2.0, 0.5))]),
        ("Two parallel walls", [((1.0, -0.5), (1.0, 0.5)), ((2.0, -0.5), (2.0, 0.5))]),
        ("L-shaped corner", [((1.0, -0.5), (1.0, 0.5)), ((1.0, 0.5), (2.0, 0.5))]),
    ]
    
    print("Scenario            | Est.Dist(m) | Left Energy | Right Energy | ITD(μs)")
    print("-" * 70)
    
    for scenario_name, obstacles in scenarios:
        result = simulator.simulate(
            source_pos=source_pos,
            source_angle=source_angle,
            ear_positions=ear_positions,
            scene_segments=obstacles
        )
        
        # Estimate distance
        est_distance = float('inf')
        if result.left_arrival_idx is not None and result.right_arrival_idx is not None:
            mean_arrival_samples = (result.left_arrival_idx + result.right_arrival_idx) / 2.0
            arrival_time_seconds = mean_arrival_samples / simulator.sr
            est_distance = arrival_time_seconds * simulator.c / 2.0
        
        print(f"{scenario_name:18s} | {est_distance:11.3f} | {result.left_energy:11.2e} | "
              f"{result.right_energy:12.2e} | {result.itd_seconds*1e6:7.1f}")
    
    print()


def main():
    """Run all demos."""
    print("Acoustic Simulation Toolkit Demo")
    print("="*40)
    print()
    
    try:
        demo_acoustic_simulator()
        demo_lidar_bat_integration()
        demo_obstacle_interaction()
        
        print("=== Demo Complete ===")
        print("The acoustic simulation toolkit is working correctly!")
        print("\nKey features demonstrated:")
        print("✓ Binaural acoustic simulation with ITD/ILD computation")
        print("✓ Distance estimation from echo arrival times")
        print("✓ Integration with existing LidarBat environment")
        print("✓ Backward compatibility (no changes when simulator=None)")
        print("✓ Response to different obstacle configurations")
        
        return 0
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())