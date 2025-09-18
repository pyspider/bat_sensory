#!/usr/bin/env python3
"""
Route analysis CLI script for acoustic simulation.

This script samples points along route LineStrings (GeoJSON or Shapefile),
runs ApproxAcousticSimulator over a set of beam angles at each sample,
and computes per-sample features including:
- angle_energy_entropy: Shannon entropy over beam energies
- mean/std ITD: Interaural time difference statistics
- mean ILD: Interaural level difference
- mean spectral entropy: Average spectral entropy across beam angles

Outputs results to CSV and a summary text file.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import warnings


def import_optional_dependencies():
    """Import optional dependencies with helpful error messages."""
    try:
        import numpy as np
    except ImportError:
        raise ImportError("numpy is required. Install with: pip install numpy")
    
    try:
        from scipy.signal import welch
    except ImportError:
        raise ImportError("scipy is required for spectral analysis. Install with: pip install scipy")
    
    try:
        import fiona
        from shapely.geometry import shape, Point, LineString
    except ImportError:
        raise ImportError(
            "fiona and shapely are required for geographic data. "
            "Install with: pip install fiona shapely"
        )
    
    return np, welch, fiona, shape, Point, LineString


def load_routes(route_path: str) -> List[LineString]:
    """
    Load route LineStrings from GeoJSON or Shapefile.
    
    Args:
        route_path: Path to route file (GeoJSON or Shapefile)
        
    Returns:
        List of shapely LineString objects
    """
    np, welch, fiona, shape, Point, LineString = import_optional_dependencies()
    
    routes = []
    
    try:
        with fiona.open(route_path) as src:
            for feature in src:
                geom = shape(feature['geometry'])
                if isinstance(geom, LineString):
                    routes.append(geom)
                elif hasattr(geom, 'geoms'):
                    # Handle MultiLineString
                    for line in geom.geoms:
                        if isinstance(line, LineString):
                            routes.append(line)
                else:
                    warnings.warn(f"Skipping non-LineString geometry: {type(geom)}")
    
    except Exception as e:
        # Try as GeoJSON
        try:
            with open(route_path, 'r') as f:
                geojson_data = json.load(f)
            
            if geojson_data.get('type') == 'FeatureCollection':
                for feature in geojson_data.get('features', []):
                    geom = shape(feature['geometry'])
                    if isinstance(geom, LineString):
                        routes.append(geom)
            elif geojson_data.get('type') == 'Feature':
                geom = shape(geojson_data['geometry'])
                if isinstance(geom, LineString):
                    routes.append(geom)
        except Exception as e2:
            raise ValueError(f"Failed to load routes from {route_path}: {e2}")
    
    return routes


def sample_points_along_route(route: LineString, spacing: float) -> List[Tuple[float, float]]:
    """
    Sample points along a route at specified spacing.
    
    Args:
        route: Shapely LineString representing the route
        spacing: Distance between sample points in same units as route coordinates
        
    Returns:
        List of (x, y) coordinate tuples
    """
    points = []
    route_length = route.length
    
    if route_length == 0:
        return points
    
    # Sample points at regular intervals
    distance = 0.0
    while distance <= route_length:
        point = route.interpolate(distance)
        points.append((point.x, point.y))
        distance += spacing
    
    # Always include the end point
    if distance - spacing < route_length:
        end_point = route.interpolate(route_length)
        points.append((end_point.x, end_point.y))
    
    return points


def calculate_spectral_entropy(waveform: 'np.ndarray', sample_rate: int, 
                             nperseg: Optional[int] = None) -> float:
    """
    Calculate spectral entropy of a waveform using Welch's method.
    
    Args:
        waveform: Input waveform
        sample_rate: Sample rate of the waveform
        nperseg: Length of each segment for Welch's method
        
    Returns:
        Spectral entropy value
    """
    np, welch, fiona, shape, Point, LineString = import_optional_dependencies()
    
    if len(waveform) == 0:
        return 0.0
    
    # Use Welch's method to estimate power spectral density
    if nperseg is None:
        nperseg = min(len(waveform), 256)
    
    try:
        frequencies, psd = welch(waveform, fs=sample_rate, nperseg=nperseg)
        
        # Normalize PSD to get probability distribution
        psd_sum = np.sum(psd)
        if psd_sum == 0:
            return 0.0
        
        psd_normalized = psd / psd_sum
        
        # Calculate Shannon entropy
        # Remove zero values to avoid log(0)
        psd_nonzero = psd_normalized[psd_normalized > 0]
        if len(psd_nonzero) == 0:
            return 0.0
        
        entropy = -np.sum(psd_nonzero * np.log2(psd_nonzero))
        return entropy
        
    except Exception as e:
        warnings.warn(f"Failed to calculate spectral entropy: {e}")
        return 0.0


def analyze_sample_point(point: Tuple[float, float], 
                        beam_angles: List[float],
                        simulator: 'ApproxAcousticSimulator',
                        obstacle_segments: List) -> Dict[str, float]:
    """
    Analyze a single sample point with multiple beam angles.
    
    Args:
        point: (x, y) coordinates of sample point
        beam_angles: List of beam angles to test (in radians)
        simulator: ApproxAcousticSimulator instance
        obstacle_segments: List of obstacle segments
        
    Returns:
        Dictionary of computed features
    """
    np, welch, fiona, shape, Point, LineString = import_optional_dependencies()
    
    x, y = point
    ear_offset = getattr(simulator, 'ear_offset', 0.02)
    
    beam_energies = []
    itd_values = []
    ild_values = []
    spectral_entropies = []
    
    for angle in beam_angles:
        try:
            # Calculate ear positions
            left_ear_x = x + ear_offset * math.cos(angle + math.pi/2)
            left_ear_y = y + ear_offset * math.sin(angle + math.pi/2)
            right_ear_x = x + ear_offset * math.cos(angle - math.pi/2)
            right_ear_y = y + ear_offset * math.sin(angle - math.pi/2)
            
            ear_positions = [(left_ear_x, left_ear_y), (right_ear_x, right_ear_y)]
            
            # Run simulation
            result = simulator.simulate(
                source_pos=(x, y),
                source_angle=angle,
                ear_positions=ear_positions,
                scene_segments=obstacle_segments
            )
            
            # Collect features
            total_energy = result.left_energy + result.right_energy
            beam_energies.append(total_energy)
            
            if result.left_arrival_idx is not None and result.right_arrival_idx is not None:
                itd_values.append(result.itd_seconds)
                ild_values.append(result.ild_db)
            
            # Calculate spectral entropy for left ear (could also use right or average)
            if len(result.left_waveform) > 0:
                spec_entropy = calculate_spectral_entropy(result.left_waveform, simulator.sr)
                spectral_entropies.append(spec_entropy)
            
        except Exception as e:
            warnings.warn(f"Failed to simulate beam angle {angle}: {e}")
            continue
    
    # Calculate angle energy entropy (Shannon entropy over beam energies)
    angle_energy_entropy = 0.0
    if beam_energies:
        energy_array = np.array(beam_energies)
        energy_sum = np.sum(energy_array)
        if energy_sum > 0:
            energy_probs = energy_array / energy_sum
            # Remove zero probabilities
            energy_probs_nonzero = energy_probs[energy_probs > 0]
            if len(energy_probs_nonzero) > 0:
                angle_energy_entropy = -np.sum(energy_probs_nonzero * np.log2(energy_probs_nonzero))
    
    # Calculate statistics
    features = {
        'x': x,
        'y': y,
        'angle_energy_entropy': angle_energy_entropy,
        'mean_itd': np.mean(itd_values) if itd_values else 0.0,
        'std_itd': np.std(itd_values) if itd_values else 0.0,
        'mean_ild': np.mean(ild_values) if ild_values else 0.0,
        'mean_spectral_entropy': np.mean(spectral_entropies) if spectral_entropies else 0.0,
        'num_valid_beams': len(beam_energies),
        'total_energy': np.sum(beam_energies) if beam_energies else 0.0
    }
    
    return features


def analyze_routes_main(route_path: str, 
                       output_prefix: str,
                       spacing: float = 1.0,
                       num_beam_angles: int = 8,
                       obstacle_segments: Optional[List] = None) -> None:
    """
    Main route analysis function.
    
    Args:
        route_path: Path to route file (GeoJSON or Shapefile)
        output_prefix: Prefix for output files
        spacing: Distance between sample points
        num_beam_angles: Number of beam angles to test
        obstacle_segments: Optional list of obstacle segments
    """
    np, welch, fiona, shape, Point, LineString = import_optional_dependencies()
    
    # Import acoustic simulator
    try:
        from .approx_acoustic_simulator import ApproxAcousticSimulator
    except ImportError:
        from approx_acoustic_simulator import ApproxAcousticSimulator
    
    print(f"Loading routes from: {route_path}")
    routes = load_routes(route_path)
    print(f"Loaded {len(routes)} routes")
    
    if not routes:
        print("No routes found in input file")
        return
    
    # Initialize simulator
    simulator = ApproxAcousticSimulator()
    
    if obstacle_segments is None:
        obstacle_segments = []
    
    # Generate beam angles
    beam_angles = [2 * math.pi * i / num_beam_angles for i in range(num_beam_angles)]
    
    # Analyze all routes
    all_features = []
    
    for route_idx, route in enumerate(routes):
        print(f"Analyzing route {route_idx + 1}/{len(routes)}")
        
        # Sample points along route
        sample_points = sample_points_along_route(route, spacing)
        print(f"  Sampled {len(sample_points)} points")
        
        # Analyze each sample point
        for point_idx, point in enumerate(sample_points):
            features = analyze_sample_point(point, beam_angles, simulator, obstacle_segments)
            features['route_id'] = route_idx
            features['point_id'] = point_idx
            all_features.append(features)
            
            if (point_idx + 1) % 10 == 0:
                print(f"    Processed {point_idx + 1}/{len(sample_points)} points")
    
    # Save results to CSV
    csv_path = f"{output_prefix}.csv"
    print(f"Saving results to: {csv_path}")
    
    if all_features:
        fieldnames = list(all_features[0].keys())
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_features)
    
    # Generate summary
    summary_path = f"{output_prefix}_summary.txt"
    print(f"Saving summary to: {summary_path}")
    
    with open(summary_path, 'w') as f:
        f.write("Route Analysis Summary\n")
        f.write("=====================\n\n")
        f.write(f"Input file: {route_path}\n")
        f.write(f"Number of routes: {len(routes)}\n")
        f.write(f"Total sample points: {len(all_features)}\n")
        f.write(f"Sample spacing: {spacing}\n")
        f.write(f"Number of beam angles: {num_beam_angles}\n")
        f.write(f"Number of obstacle segments: {len(obstacle_segments)}\n\n")
        
        if all_features:
            # Calculate overall statistics
            angle_entropies = [f['angle_energy_entropy'] for f in all_features]
            mean_itds = [f['mean_itd'] for f in all_features]
            mean_ilds = [f['mean_ild'] for f in all_features]
            spectral_entropies = [f['mean_spectral_entropy'] for f in all_features]
            
            f.write("Overall Statistics:\n")
            f.write(f"  Angle Energy Entropy: mean={np.mean(angle_entropies):.4f}, std={np.std(angle_entropies):.4f}\n")
            f.write(f"  Mean ITD: mean={np.mean(mean_itds):.6f}s, std={np.std(mean_itds):.6f}s\n")
            f.write(f"  Mean ILD: mean={np.mean(mean_ilds):.4f}dB, std={np.std(mean_ilds):.4f}dB\n")
            f.write(f"  Spectral Entropy: mean={np.mean(spectral_entropies):.4f}, std={np.std(spectral_entropies):.4f}\n")
    
    print("Analysis complete!")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze acoustic complexity along migration routes"
    )
    parser.add_argument(
        "route_path",
        help="Path to route file (GeoJSON or Shapefile)"
    )
    parser.add_argument(
        "output_prefix",
        help="Prefix for output files (CSV and summary)"
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=1.0,
        help="Distance between sample points (default: 1.0)"
    )
    parser.add_argument(
        "--beam-angles",
        type=int,
        default=8,
        help="Number of beam angles to test (default: 8)"
    )
    parser.add_argument(
        "--obstacle-raster",
        help="Optional path to obstacle raster file (GeoTIFF)"
    )
    
    args = parser.parse_args()
    
    # Load obstacles if provided
    obstacle_segments = []
    if args.obstacle_raster:
        try:
            from .raster_to_obstacles import raster_to_obstacle_segments
        except ImportError:
            from raster_to_obstacles import raster_to_obstacle_segments
        
        print(f"Loading obstacles from: {args.obstacle_raster}")
        obstacle_segments = raster_to_obstacle_segments(args.obstacle_raster)
        print(f"Loaded {len(obstacle_segments)} obstacle segments")
    
    # Run analysis
    analyze_routes_main(
        route_path=args.route_path,
        output_prefix=args.output_prefix,
        spacing=args.spacing,
        num_beam_angles=args.beam_angles,
        obstacle_segments=obstacle_segments
    )


if __name__ == "__main__":
    main()