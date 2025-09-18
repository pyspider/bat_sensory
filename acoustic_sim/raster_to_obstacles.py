"""
Convert raster data (GeoTIFF) to obstacle line segments.

Reads single-band raster files, applies binary thresholding, vectorizes to polygons,
and extracts boundary line segments suitable for acoustic simulation.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
import warnings

try:
    import rasterio
    import rasterio.features
    from shapely.geometry import LineString, Polygon, MultiPolygon
    from shapely.ops import unary_union
    RASTER_DEPS_AVAILABLE = True
except ImportError:
    RASTER_DEPS_AVAILABLE = False
    warnings.warn("Raster processing dependencies not available. "
                 "Install rasterio, fiona, and shapely to use raster_to_obstacles.")


def read_raster_as_binary(raster_path: str, threshold: float = 0.5, 
                         band: int = 1) -> Tuple[np.ndarray, dict]:
    """
    Read a raster file and convert to binary mask.
    
    Args:
        raster_path: Path to GeoTIFF or other raster file
        threshold: Threshold value for binary conversion
        band: Band number to read (1-indexed)
        
    Returns:
        Tuple of (binary_mask, transform_info)
        binary_mask: 2D numpy array of 0s and 1s
        transform_info: Dictionary with georeferencing information
    """
    if not RASTER_DEPS_AVAILABLE:
        raise ImportError("rasterio is required for raster processing. "
                         "Install with: pip install rasterio")
    
    with rasterio.open(raster_path) as src:
        # Read the specified band
        data = src.read(band)
        
        # Create binary mask
        binary_mask = (data > threshold).astype(np.uint8)
        
        # Extract transform information
        transform_info = {
            'transform': src.transform,
            'crs': src.crs,
            'width': src.width,
            'height': src.height,
            'bounds': src.bounds
        }
        
    return binary_mask, transform_info


def vectorize_binary_mask(binary_mask: np.ndarray, transform: object,
                         simplify_tolerance: float = 0.1) -> List[Polygon]:
    """
    Convert binary mask to vector polygons.
    
    Args:
        binary_mask: 2D binary array (0s and 1s)
        transform: Rasterio affine transform object
        simplify_tolerance: Tolerance for polygon simplification
        
    Returns:
        List of Shapely Polygon objects representing obstacles
    """
    if not RASTER_DEPS_AVAILABLE:
        raise ImportError("rasterio and shapely are required for vectorization")
    
    # Extract shapes from the binary mask
    shapes = list(rasterio.features.shapes(binary_mask, transform=transform))
    
    # Filter for obstacle shapes (value = 1) and convert to polygons
    polygons = []
    for geom, value in shapes:
        if value == 1:  # Obstacle pixels
            poly = Polygon(geom['coordinates'][0])
            if simplify_tolerance > 0:
                poly = poly.simplify(simplify_tolerance, preserve_topology=True)
            if poly.is_valid and not poly.is_empty:
                polygons.append(poly)
    
    return polygons


def merge_polygons(polygons: List[Polygon], buffer_distance: float = 0.01) -> List[Polygon]:
    """
    Merge nearby polygons to reduce fragmentation.
    
    Args:
        polygons: List of Shapely polygons
        buffer_distance: Buffer distance for merging nearby polygons
        
    Returns:
        List of merged polygons
    """
    if not polygons:
        return []
    
    if not RASTER_DEPS_AVAILABLE:
        raise ImportError("shapely is required for polygon operations")
    
    # Buffer polygons slightly to merge nearby ones
    buffered = [p.buffer(buffer_distance) for p in polygons]
    
    # Union all polygons
    merged = unary_union(buffered)
    
    # Unbuffer to get back to original size
    if hasattr(merged, 'geoms'):  # MultiPolygon
        result = [geom.buffer(-buffer_distance) for geom in merged.geoms]
    else:  # Single Polygon
        result = [merged.buffer(-buffer_distance)]
    
    # Filter valid polygons
    return [p for p in result if p.is_valid and not p.is_empty]


def extract_boundary_segments(polygons: List[Polygon]) -> List[LineString]:
    """
    Extract boundary line segments from polygons.
    
    Args:
        polygons: List of Shapely polygons
        
    Returns:
        List of LineString objects representing obstacle boundaries
    """
    if not RASTER_DEPS_AVAILABLE:
        raise ImportError("shapely is required for boundary extraction")
    
    segments = []
    
    for poly in polygons:
        # Get exterior boundary
        if hasattr(poly.exterior, 'coords'):
            coords = list(poly.exterior.coords)
            
            # Create line segments between consecutive points
            for i in range(len(coords) - 1):
                segment = LineString([coords[i], coords[i + 1]])
                if segment.length > 1e-6:  # Filter very short segments
                    segments.append(segment)
        
        # Get interior boundaries (holes)
        for interior in poly.interiors:
            coords = list(interior.coords)
            for i in range(len(coords) - 1):
                segment = LineString([coords[i], coords[i + 1]])
                if segment.length > 1e-6:
                    segments.append(segment)
    
    return segments


def raster_to_obstacle_segments(raster_path: str, threshold: float = 0.5,
                               band: int = 1, simplify_tolerance: float = 0.1,
                               merge_buffer: float = 0.01) -> List[LineString]:
    """
    Complete pipeline: raster file to obstacle line segments.
    
    Args:
        raster_path: Path to raster file (GeoTIFF, etc.)
        threshold: Threshold for binary conversion
        band: Band number to read (1-indexed)
        simplify_tolerance: Polygon simplification tolerance
        merge_buffer: Buffer distance for merging nearby polygons
        
    Returns:
        List of LineString objects representing obstacle boundaries
    """
    if not RASTER_DEPS_AVAILABLE:
        raise ImportError("Raster processing requires rasterio, fiona, and shapely. "
                         "Install with: pip install rasterio fiona shapely")
    
    # Step 1: Read raster as binary mask
    binary_mask, transform_info = read_raster_as_binary(raster_path, threshold, band)
    
    # Step 2: Vectorize to polygons
    polygons = vectorize_binary_mask(binary_mask, transform_info['transform'], 
                                   simplify_tolerance)
    
    # Step 3: Merge nearby polygons
    merged_polygons = merge_polygons(polygons, merge_buffer)
    
    # Step 4: Extract boundary segments
    segments = extract_boundary_segments(merged_polygons)
    
    return segments


def convert_segments_to_simple_format(segments: List[LineString]) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Convert LineString segments to simple coordinate tuples.
    
    Args:
        segments: List of Shapely LineString objects
        
    Returns:
        List of tuples: [((x0, y0), (x1, y1)), ...]
    """
    simple_segments = []
    for segment in segments:
        coords = list(segment.coords)
        if len(coords) >= 2:
            start = (coords[0][0], coords[0][1])
            end = (coords[-1][0], coords[-1][1])
            simple_segments.append((start, end))
    
    return simple_segments


# Example usage function
def example_usage():
    """
    Example of how to use the raster to obstacles pipeline.
    """
    if not RASTER_DEPS_AVAILABLE:
        print("Example cannot run: missing dependencies")
        print("Install with: pip install rasterio fiona shapely")
        return
    
    # Example raster processing
    try:
        raster_path = "example_map.tif"
        segments = raster_to_obstacle_segments(
            raster_path=raster_path,
            threshold=0.5,
            simplify_tolerance=0.1,
            merge_buffer=0.05
        )
        
        print(f"Extracted {len(segments)} obstacle segments")
        
        # Convert to simple format for use with other systems
        simple_segments = convert_segments_to_simple_format(segments)
        
        # Print first few segments
        for i, ((x0, y0), (x1, y1)) in enumerate(simple_segments[:5]):
            print(f"Segment {i}: ({x0:.2f}, {y0:.2f}) -> ({x1:.2f}, {y1:.2f})")
            
    except FileNotFoundError:
        print(f"Example raster file not found: {raster_path}")
    except Exception as e:
        print(f"Error processing raster: {e}")


if __name__ == "__main__":
    example_usage()