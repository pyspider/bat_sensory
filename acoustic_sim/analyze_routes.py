"""
Analyze acoustic features along flight routes using binaural simulation.

Samples points along route LineStrings, computes binaural echoes at each point
using ApproxAcousticSimulator, and extracts acoustic features for analysis.
"""

import numpy as np
import csv
import argparse
import math
from typing import List, Tuple, Dict, Optional, Any
import warnings

try:
    from shapely.geometry import LineString, Point
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    warnings.warn("Shapely not available. Install with: pip install shapely")

try:
    import scipy.stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Install with: pip install scipy")

from .approx_acoustic_simulator import ApproxAcousticSimulator, EchoResult


def sample_points_along_route(route: LineString, spacing: float = 1.0) -> List[Tuple[float, float]]:
    """
    Sample points at regular intervals along a route LineString.
    
    Args:
        route: Shapely LineString representing the flight route
        spacing: Distance between sample points in meters
        
    Returns:
        List of (x, y) coordinate tuples
    """
    if not SHAPELY_AVAILABLE:
        raise ImportError("Shapely is required for route sampling")
    
    total_length = route.length
    if total_length < spacing:
        # If route is shorter than spacing, just return start and end
        coords = list(route.coords)
        return [(coords[0][0], coords[0][1]), (coords[-1][0], coords[-1][1])]
    
    # Sample points at regular intervals
    distances = np.arange(0, total_length + spacing, spacing)
    points = []
    
    for dist in distances:
        if dist > total_length:
            dist = total_length
        point = route.interpolate(dist)
        points.append((point.x, point.y))
    
    return points


def build_ear_positions(position: Tuple[float, float], heading: float, 
                       ear_offset: float = 0.02) -> List[Tuple[float, float]]:
    """
    Build left and right ear positions given head position and orientation.
    
    Args:
        position: (x, y) head center position
        heading: Head orientation in radians
        ear_offset: Distance from center to each ear
        
    Returns:
        List of [(left_ear_x, left_ear_y), (right_ear_x, right_ear_y)]
    """
    x, y = position
    
    # Left ear is +90 degrees from heading, right ear is -90 degrees
    left_angle = heading + math.pi / 2
    right_angle = heading - math.pi / 2
    
    left_ear = (x + ear_offset * math.cos(left_angle), 
                y + ear_offset * math.sin(left_angle))
    right_ear = (x + ear_offset * math.cos(right_angle), 
                 y + ear_offset * math.sin(right_angle))
    
    return [left_ear, right_ear]


def compute_angle_energy_entropy(beam_energies: List[float]) -> float:
    """
    Compute Shannon entropy of energy distribution across beam angles.
    
    Args:
        beam_energies: List of energy values for different beam directions
        
    Returns:
        Shannon entropy value (higher = more uniform energy distribution)
    """
    if not beam_energies or all(e <= 0 for e in beam_energies):
        return 0.0
    
    # Normalize to probability distribution
    total_energy = sum(beam_energies)
    if total_energy <= 0:
        return 0.0
        
    probabilities = [e / total_energy for e in beam_energies]
    
    # Compute Shannon entropy
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)
            
    return entropy


def compute_spectral_entropy(waveform: np.ndarray, sample_rate: float = 44100) -> float:
    """
    Compute spectral entropy of a waveform.
    
    Args:
        waveform: Input audio waveform
        sample_rate: Sample rate in Hz
        
    Returns:
        Spectral entropy value
    """
    if len(waveform) == 0:
        return 0.0
    
    # Compute power spectral density
    fft = np.fft.fft(waveform)
    power_spectrum = np.abs(fft) ** 2
    
    # Use only positive frequencies
    n = len(power_spectrum) // 2
    power_spectrum = power_spectrum[:n]
    
    # Normalize to probability distribution
    total_power = np.sum(power_spectrum)
    if total_power <= 0:
        return 0.0
        
    prob_spectrum = power_spectrum / total_power
    
    # Compute spectral entropy
    entropy = 0.0
    for p in prob_spectrum:
        if p > 0:
            entropy -= p * math.log2(p)
            
    return entropy


def analyze_point_acoustics(position: Tuple[float, float], heading: float,
                          obstacle_segments: List[Any], simulator: ApproxAcousticSimulator,
                          beam_angles: List[float] = None) -> Dict[str, float]:
    """
    Analyze acoustic features at a single point along the route.
    
    Args:
        position: (x, y) position
        heading: Orientation in radians
        obstacle_segments: List of obstacle segments for simulation
        simulator: Acoustic simulator instance
        beam_angles: List of beam angles to test (radians relative to heading)
        
    Returns:
        Dictionary of acoustic features
    """
    if beam_angles is None:
        # Default beam angles: -60° to +60° in 15° increments
        beam_angles = [math.radians(a) for a in range(-60, 75, 15)]
    
    # Build ear positions
    ear_positions = build_ear_positions(position, heading, simulator.ear_offset)
    
    # Analyze echoes for each beam angle
    beam_energies = []
    itds = []
    ilds = []
    spectral_entropies = []
    
    for beam_angle in beam_angles:
        global_beam_angle = heading + beam_angle
        
        # Simulate binaural echo
        echo_result = simulator.simulate(
            source_pos=position,
            source_angle=global_beam_angle,
            receivers=ear_positions,
            scene_segments=obstacle_segments
        )
        
        if echo_result is not None:
            beam_energies.append(echo_result.energy)
            itds.append(echo_result.itd)
            ilds.append(echo_result.ild)
            
            # Compute spectral entropy of combined waveform
            combined_waveform = echo_result.left_waveform + echo_result.right_waveform
            spectral_entropy = compute_spectral_entropy(combined_waveform, simulator.sample_rate)
            spectral_entropies.append(spectral_entropy)
        else:
            beam_energies.append(0.0)
            itds.append(0.0)
            ilds.append(0.0)
            spectral_entropies.append(0.0)
    
    # Compute aggregate features
    features = {
        'x': position[0],
        'y': position[1],
        'heading': heading,
        'angle_energy_entropy': compute_angle_energy_entropy(beam_energies),
        'mean_itd': np.mean(itds) if itds else 0.0,
        'std_itd': np.std(itds) if itds else 0.0,
        'mean_ild': np.mean(ilds) if ilds else 0.0,
        'mean_spectral_entropy': np.mean(spectral_entropies) if spectral_entropies else 0.0,
        'total_energy': sum(beam_energies),
        'max_energy': max(beam_energies) if beam_energies else 0.0
    }
    
    return features


def analyze_routes(routes: List[LineString], obstacle_segments: List[Any],
                  sample_spacing: float = 1.0, output_csv: Optional[str] = None,
                  output_summary: Optional[str] = None) -> List[Dict[str, float]]:
    """
    Analyze acoustic features along multiple flight routes.
    
    Args:
        routes: List of LineString routes to analyze
        obstacle_segments: List of obstacle segments for acoustic simulation
        sample_spacing: Distance between sample points along routes
        output_csv: Optional path to save detailed CSV results
        output_summary: Optional path to save summary text
        
    Returns:
        List of feature dictionaries for all sample points
    """
    if not SHAPELY_AVAILABLE:
        raise ImportError("Shapely is required for route analysis")
    
    # Initialize acoustic simulator
    simulator = ApproxAcousticSimulator()
    
    all_features = []
    route_summaries = []
    
    for route_idx, route in enumerate(routes):
        print(f"Analyzing route {route_idx + 1}/{len(routes)}...")
        
        # Sample points along route
        sample_points = sample_points_along_route(route, sample_spacing)
        route_features = []
        
        for i, position in enumerate(sample_points):
            # Estimate heading from route direction
            if i < len(sample_points) - 1:
                next_pos = sample_points[i + 1]
                heading = math.atan2(next_pos[1] - position[1], 
                                   next_pos[0] - position[0])
            else:
                # Use previous heading for last point
                if i > 0:
                    prev_pos = sample_points[i - 1]
                    heading = math.atan2(position[1] - prev_pos[1], 
                                       position[0] - prev_pos[0])
                else:
                    heading = 0.0  # Default heading
            
            # Analyze acoustics at this point
            features = analyze_point_acoustics(
                position, heading, obstacle_segments, simulator
            )
            features['route_id'] = route_idx
            features['point_id'] = i
            
            route_features.append(features)
            all_features.append(features)
        
        # Compute route summary statistics
        if route_features:
            summary = {
                'route_id': route_idx,
                'n_points': len(route_features),
                'route_length': route.length,
                'mean_angle_entropy': np.mean([f['angle_energy_entropy'] for f in route_features]),
                'mean_total_energy': np.mean([f['total_energy'] for f in route_features]),
                'mean_spectral_entropy': np.mean([f['mean_spectral_entropy'] for f in route_features])
            }
            route_summaries.append(summary)
    
    # Save results
    if output_csv and all_features:
        save_features_csv(all_features, output_csv)
    
    if output_summary and route_summaries:
        save_summary_text(route_summaries, output_summary)
    
    return all_features


def save_features_csv(features: List[Dict[str, float]], filename: str):
    """Save detailed features to CSV file."""
    if not features:
        return
    
    fieldnames = list(features[0].keys())
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(features)
    
    print(f"Detailed features saved to {filename}")


def save_summary_text(summaries: List[Dict[str, Any]], filename: str):
    """Save route summaries to text file."""
    with open(filename, 'w') as f:
        f.write("Route Analysis Summary\n")
        f.write("=====================\n\n")
        
        for summary in summaries:
            f.write(f"Route {summary['route_id']}:\n")
            f.write(f"  Points: {summary['n_points']}\n")
            f.write(f"  Length: {summary['route_length']:.2f} m\n")
            f.write(f"  Mean Angle Entropy: {summary['mean_angle_entropy']:.3f}\n")
            f.write(f"  Mean Total Energy: {summary['mean_total_energy']:.3f}\n")
            f.write(f"  Mean Spectral Entropy: {summary['mean_spectral_entropy']:.3f}\n")
            f.write("\n")
    
    print(f"Summary saved to {filename}")


def main():
    """Command-line interface for route analysis."""
    parser = argparse.ArgumentParser(description="Analyze acoustic features along flight routes")
    parser.add_argument("routes", help="File containing route LineStrings (format TBD)")
    parser.add_argument("obstacles", help="File containing obstacle segments (format TBD)")
    parser.add_argument("--spacing", type=float, default=1.0, help="Sample spacing along routes (m)")
    parser.add_argument("--csv", help="Output CSV file for detailed results")
    parser.add_argument("--summary", help="Output text file for summary")
    
    args = parser.parse_args()
    
    # TODO: Implement file loading for routes and obstacles
    print("Note: File loading not implemented in this example")
    print("Use the analyze_routes() function directly with LineString and segment objects")
    
    # Example usage:
    if SHAPELY_AVAILABLE:
        # Create example route
        example_route = LineString([(0, 0), (10, 5), (20, 0)])
        
        # Create example obstacles (simple line segments)
        example_obstacles = []  # Would need to be populated with actual segment objects
        
        print("Example route created. Run with actual obstacle data for full analysis.")


if __name__ == "__main__":
    main()