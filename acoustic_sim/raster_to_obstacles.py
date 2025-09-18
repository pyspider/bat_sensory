"""
Raster to obstacles conversion utilities.

This module provides functionality to read single-band raster data (GeoTIFF),
threshold to binary mask, vectorize using rasterio.features.shapes, merge polygons,
and extract obstacle boundaries as shapely LineString segments.

Dependencies are kept local to this module so main package users don't need
rasterio unless they use acoustic_sim functionality.
"""

from typing import List, Tuple, Optional
import warnings


def extract_obstacles_from_raster(raster_path: str, 
                                threshold: float = 0.5,
                                simplify_tolerance: float = 0.0) -> List:
    """
    Extract obstacle boundary segments from a single-band raster file.
    
    This function reads a GeoTIFF file, applies a threshold to create a binary
    mask, vectorizes the mask to polygons, and extracts the exterior boundaries
    as line segments representing obstacles.
    
    Args:
        raster_path: Path to the single-band raster file (GeoTIFF)
        threshold: Threshold value for binary classification (default: 0.5)
        simplify_tolerance: Tolerance for polygon simplification (default: 0.0, no simplification)
        
    Returns:
        List of shapely LineString objects representing obstacle boundaries
        
    Raises:
        ImportError: If required dependencies (rasterio, shapely) are not available
        FileNotFoundError: If raster file cannot be found
        ValueError: If raster processing fails
    """
    try:
        import rasterio
        import rasterio.features
        from shapely.geometry import shape, LineString
        from shapely.ops import unary_union
    except ImportError as e:
        raise ImportError(
            "rasterio and shapely are required for raster processing. "
            "Install with: pip install rasterio shapely"
        ) from e
    
    try:
        # Read the raster data
        with rasterio.open(raster_path) as src:
            # Read the first band
            raster_data = src.read(1)
            transform = src.transform
            
            # Apply threshold to create binary mask
            # Assumes obstacles have values >= threshold
            binary_mask = (raster_data >= threshold).astype('uint8')
            
            # Vectorize the binary mask to polygons
            polygon_generator = rasterio.features.shapes(
                binary_mask, 
                mask=binary_mask,  # Only extract shapes where mask is True
                transform=transform
            )
            
            # Convert to shapely polygons
            polygons = []
            for geom_dict, value in polygon_generator:
                if value == 1:  # Only process obstacle pixels
                    poly = shape(geom_dict)
                    if poly.is_valid and poly.area > 0:
                        # Simplify polygon if tolerance specified
                        if simplify_tolerance > 0:
                            poly = poly.simplify(simplify_tolerance, preserve_topology=True)
                        polygons.append(poly)
            
            if not polygons:
                warnings.warn("No obstacle polygons found in raster")
                return []
            
            # Merge overlapping polygons
            try:
                merged_polygons = unary_union(polygons)
                
                # Handle both single polygon and multi-polygon results
                if hasattr(merged_polygons, 'geoms'):
                    # MultiPolygon case
                    polygon_list = list(merged_polygons.geoms)
                else:
                    # Single Polygon case
                    polygon_list = [merged_polygons]
                
            except Exception as e:
                warnings.warn(f"Failed to merge polygons: {e}, using original polygons")
                polygon_list = polygons
            
            # Extract exterior boundaries as LineString segments
            line_segments = []
            for polygon in polygon_list:
                if hasattr(polygon, 'exterior'):
                    # Extract exterior boundary
                    exterior_coords = list(polygon.exterior.coords)
                    if len(exterior_coords) >= 2:
                        # Create LineString segments between consecutive points
                        for i in range(len(exterior_coords) - 1):
                            segment = LineString([exterior_coords[i], exterior_coords[i + 1]])
                            if segment.length > 0:  # Only add non-degenerate segments
                                line_segments.append(segment)
                    
                    # Also extract holes (interior boundaries)
                    if hasattr(polygon, 'interiors'):
                        for interior in polygon.interiors:
                            interior_coords = list(interior.coords)
                            if len(interior_coords) >= 2:
                                for i in range(len(interior_coords) - 1):
                                    segment = LineString([interior_coords[i], interior_coords[i + 1]])
                                    if segment.length > 0:
                                        line_segments.append(segment)
            
            return line_segments
            
    except FileNotFoundError:
        raise FileNotFoundError(f"Raster file not found: {raster_path}")
    except Exception as e:
        raise ValueError(f"Failed to process raster file {raster_path}: {e}")


def convert_linestrings_to_segments(linestrings: List) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Convert shapely LineString objects to simple coordinate tuples.
    
    Args:
        linestrings: List of shapely LineString objects
        
    Returns:
        List of segment tuples in format ((x1, y1), (x2, y2))
    """
    segments = []
    for linestring in linestrings:
        if hasattr(linestring, 'coords'):
            coords = list(linestring.coords)
            if len(coords) >= 2:
                # For multi-segment LineStrings, create segment for each pair
                for i in range(len(coords) - 1):
                    segment = ((coords[i][0], coords[i][1]), (coords[i+1][0], coords[i+1][1]))
                    segments.append(segment)
        else:
            # Handle other formats if needed
            continue
    
    return segments


def raster_to_obstacle_segments(raster_path: str, 
                              threshold: float = 0.5,
                              simplify_tolerance: float = 0.0) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Complete pipeline: raster file to obstacle segments.
    
    Args:
        raster_path: Path to the single-band raster file (GeoTIFF)
        threshold: Threshold value for binary classification (default: 0.5)
        simplify_tolerance: Tolerance for polygon simplification (default: 0.0)
        
    Returns:
        List of segments as ((x1, y1), (x2, y2)) tuples
    """
    linestrings = extract_obstacles_from_raster(raster_path, threshold, simplify_tolerance)
    return convert_linestrings_to_segments(linestrings)


# Example usage function for testing
def demo_raster_processing(raster_path: str) -> None:
    """
    Demonstrate raster processing functionality.
    
    Args:
        raster_path: Path to test raster file
    """
    try:
        print(f"Processing raster: {raster_path}")
        
        # Extract obstacle segments
        segments = raster_to_obstacle_segments(raster_path)
        
        print(f"Extracted {len(segments)} obstacle segments")
        
        # Display first few segments
        for i, segment in enumerate(segments[:5]):
            (x1, y1), (x2, y2) = segment
            print(f"Segment {i+1}: ({x1:.2f}, {y1:.2f}) -> ({x2:.2f}, {y2:.2f})")
        
        if len(segments) > 5:
            print(f"... and {len(segments) - 5} more segments")
            
    except Exception as e:
        print(f"Error processing raster: {e}")