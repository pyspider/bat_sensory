"""
Approximate acoustic simulator for binaural echo analysis.

This module provides a 1st-order image-source binaural simulator that synthesizes
left/right waveforms, computes first-arrival indices, ITD (seconds), ILD (dB),
and energy for bat echolocation simulation.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union


@dataclass
class EchoResult:
    """
    Result of acoustic echo simulation.
    
    Attributes:
        left_waveform: Left ear waveform samples
        right_waveform: Right ear waveform samples
        left_arrival_idx: Sample index of first arrival in left ear (None if no arrival)
        right_arrival_idx: Sample index of first arrival in right ear (None if no arrival)
        itd_seconds: Interaural time difference in seconds
        ild_db: Interaural level difference in dB (left - right)
        left_energy: Total energy in left ear signal
        right_energy: Total energy in right ear signal
    """
    left_waveform: np.ndarray
    right_waveform: np.ndarray
    left_arrival_idx: Optional[int]
    right_arrival_idx: Optional[int]
    itd_seconds: float
    ild_db: float
    left_energy: float
    right_energy: float


class ApproxAcousticSimulator:
    """
    1st-order image-source binaural acoustic simulator.
    
    This simulator synthesizes binaural echo responses using image source method
    for fast approximate binaural echo simulations in bat echolocation scenarios.
    """
    
    def __init__(self, 
                 c: float = 343.0,
                 sample_rate: int = 44100,
                 pulse_center_freq: float = 50000.0,
                 pulse_duration_ms: float = 2.0,
                 reflection_coeff: float = 0.7,
                 ear_offset: float = 0.02,
                 max_reflections: int = 1):
        """
        Initialize the acoustic simulator.
        
        Args:
            c: Speed of sound in m/s
            sample_rate: Audio sample rate in Hz
            pulse_center_freq: Center frequency of echolocation pulse in Hz
            pulse_duration_ms: Duration of pulse in milliseconds
            reflection_coeff: Reflection coefficient for surfaces (0-1)
            ear_offset: Distance between ears in meters
            max_reflections: Maximum number of reflections to simulate
        """
        self.c = c
        self.sr = sample_rate
        self.pulse_center_freq = pulse_center_freq
        self.pulse_duration_ms = pulse_duration_ms
        self.reflection_coeff = reflection_coeff
        self.ear_offset = ear_offset
        self.max_reflections = max_reflections
        
        # Generate default pulse waveform
        self._generate_default_pulse()
    
    def _generate_default_pulse(self):
        """Generate a default Gaussian-windowed sinusoidal pulse."""
        duration_samples = int(self.pulse_duration_ms * self.sr / 1000.0)
        t = np.linspace(0, self.pulse_duration_ms / 1000.0, duration_samples)
        
        # Gaussian window
        sigma = self.pulse_duration_ms / 1000.0 / 6  # 6-sigma window
        window = np.exp(-0.5 * ((t - t[-1]/2) / sigma) ** 2)
        
        # Sinusoidal carrier
        carrier = np.sin(2 * np.pi * self.pulse_center_freq * t)
        
        self.default_pulse = window * carrier
    
    def reflect_point_across_segment(self, point: Tuple[float, float], 
                                   segment: Tuple[Tuple[float, float], Tuple[float, float]]) -> Tuple[float, float]:
        """
        Reflect a point across a line segment.
        
        Args:
            point: (x, y) coordinates of point to reflect
            segment: ((x1, y1), (x2, y2)) defining the line segment
            
        Returns:
            (x, y) coordinates of reflected point
        """
        px, py = point
        (x1, y1), (x2, y2) = segment
        
        # Vector along the segment
        dx = x2 - x1
        dy = y2 - y1
        
        # Normalize segment vector
        length = math.sqrt(dx*dx + dy*dy)
        if length == 0:
            return point  # Degenerate segment
        
        dx /= length
        dy /= length
        
        # Vector from segment start to point
        dpx = px - x1
        dpy = py - y1
        
        # Project point onto segment line (infinite line)
        projection_length = dpx * dx + dpy * dy
        proj_x = x1 + projection_length * dx
        proj_y = y1 + projection_length * dy
        
        # Reflect point across the line
        reflected_x = 2 * proj_x - px
        reflected_y = 2 * proj_y - py
        
        return (reflected_x, reflected_y)
    
    def find_reflection_point_on_segment(self, source: Tuple[float, float],
                                       receiver: Tuple[float, float],
                                       segment: Tuple[Tuple[float, float], Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """
        Find the reflection point on a finite segment for specular reflection.
        
        Args:
            source: (x, y) coordinates of sound source
            receiver: (x, y) coordinates of receiver
            segment: ((x1, y1), (x2, y2)) defining the reflective segment
            
        Returns:
            (x, y) coordinates of reflection point on segment, or None if no valid reflection
        """
        # Use image source method: reflect source across segment line
        image_source = self.reflect_point_across_segment(source, segment)
        
        # Find intersection of line from image source to receiver with segment
        (x1, y1), (x2, y2) = segment
        sx, sy = image_source
        rx, ry = receiver
        
        # Parametric line equations:
        # Segment: P = (x1, y1) + t * ((x2, y2) - (x1, y1)), t ∈ [0, 1]
        # Ray: Q = (sx, sy) + s * ((rx, ry) - (sx, sy)), s ≥ 0
        
        seg_dx = x2 - x1
        seg_dy = y2 - y1
        ray_dx = rx - sx
        ray_dy = ry - sy
        
        # Solve: (x1, y1) + t * (seg_dx, seg_dy) = (sx, sy) + s * (ray_dx, ray_dy)
        # x1 + t * seg_dx = sx + s * ray_dx
        # y1 + t * seg_dy = sy + s * ray_dy
        
        denominator = seg_dx * ray_dy - seg_dy * ray_dx
        if abs(denominator) < 1e-10:
            return None  # Lines are parallel
        
        t = ((sx - x1) * ray_dy - (sy - y1) * ray_dx) / denominator
        s = ((sx - x1) * seg_dy - (sy - y1) * seg_dx) / denominator
        
        # Check if intersection is within segment and ray is forward
        if 0 <= t <= 1 and s >= 0:
            # Calculate intersection point
            intersection_x = x1 + t * seg_dx
            intersection_y = y1 + t * seg_dy
            return (intersection_x, intersection_y)
        
        return None
    
    def simulate(self, 
                 source_pos: Tuple[float, float],
                 source_angle: float,
                 ear_positions: List[Tuple[float, float]],
                 source_waveform: Optional[np.ndarray] = None,
                 scene_segments: Optional[List] = None) -> EchoResult:
        """
        Simulate binaural echo response for a given scene.
        
        Args:
            source_pos: (x, y) position of sound source
            source_angle: Angle of source orientation in radians
            ear_positions: List of (x, y) positions for left and right ears
            source_waveform: Optional source waveform, uses default pulse if None
            scene_segments: List of obstacle segments in the scene
            
        Returns:
            EchoResult containing simulated binaural response
        """
        if source_waveform is None:
            source_waveform = self.default_pulse
        
        if scene_segments is None:
            scene_segments = []
        
        # Convert scene segments to expected format if needed
        segments = []
        for seg in scene_segments:
            if hasattr(seg, 'p0') and hasattr(seg, 'p1'):
                # Handle Segment objects from lidar_bat.py
                segments.append(((seg.p0.x, seg.p0.y), (seg.p1.x, seg.p1.y)))
            elif isinstance(seg, (list, tuple)) and len(seg) == 2:
                segments.append(seg)
            else:
                continue  # Skip unrecognized formats
        
        # Simulate for each ear
        left_ear_pos = ear_positions[0] if len(ear_positions) > 0 else source_pos
        right_ear_pos = ear_positions[1] if len(ear_positions) > 1 else source_pos
        
        # Calculate direct path and reflections for each ear
        left_response = self._simulate_single_ear(source_pos, left_ear_pos, source_waveform, segments)
        right_response = self._simulate_single_ear(source_pos, right_ear_pos, source_waveform, segments)
        
        # Find first arrivals
        left_arrival_idx = self._find_first_arrival(left_response)
        right_arrival_idx = self._find_first_arrival(right_response)
        
        # Calculate ITD and ILD
        itd_seconds = 0.0
        if left_arrival_idx is not None and right_arrival_idx is not None:
            itd_seconds = (left_arrival_idx - right_arrival_idx) / self.sr
        
        left_energy = np.sum(left_response ** 2)
        right_energy = np.sum(right_response ** 2)
        
        ild_db = 0.0
        if left_energy > 0 and right_energy > 0:
            ild_db = 10 * np.log10(left_energy / right_energy)
        
        return EchoResult(
            left_waveform=left_response,
            right_waveform=right_response,
            left_arrival_idx=left_arrival_idx,
            right_arrival_idx=right_arrival_idx,
            itd_seconds=itd_seconds,
            ild_db=ild_db,
            left_energy=left_energy,
            right_energy=right_energy
        )
    
    def _simulate_single_ear(self, source_pos: Tuple[float, float],
                           ear_pos: Tuple[float, float],
                           source_waveform: np.ndarray,
                           segments: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> np.ndarray:
        """Simulate acoustic response for a single ear."""
        # Calculate maximum simulation time based on scene size
        max_distance = 50.0  # Assume max scene dimension of 50m
        max_delay_samples = int(2 * max_distance * self.sr / self.c)  # Round trip
        
        # Initialize output signal
        output_length = len(source_waveform) + max_delay_samples
        output = np.zeros(output_length)
        
        # Direct path
        direct_distance = math.sqrt((ear_pos[0] - source_pos[0])**2 + (ear_pos[1] - source_pos[1])**2)
        direct_delay_samples = int(direct_distance * self.sr / self.c)
        direct_attenuation = 1.0 / max(direct_distance, 0.01)  # Avoid division by zero
        
        if direct_delay_samples < output_length:
            end_idx = min(direct_delay_samples + len(source_waveform), output_length)
            waveform_length = end_idx - direct_delay_samples
            if waveform_length > 0:
                output[direct_delay_samples:end_idx] += (
                    direct_attenuation * source_waveform[:waveform_length]
                )
        
        # First-order reflections
        for segment in segments:
            reflection_point = self.find_reflection_point_on_segment(source_pos, ear_pos, segment)
            if reflection_point is None:
                continue
            
            # Calculate path length: source -> reflection point -> ear
            source_to_reflection = math.sqrt(
                (reflection_point[0] - source_pos[0])**2 + 
                (reflection_point[1] - source_pos[1])**2
            )
            reflection_to_ear = math.sqrt(
                (ear_pos[0] - reflection_point[0])**2 + 
                (ear_pos[1] - reflection_point[1])**2
            )
            
            total_distance = source_to_reflection + reflection_to_ear
            reflection_delay_samples = int(total_distance * self.sr / self.c)
            
            # Apply reflection coefficient and distance attenuation
            reflection_attenuation = (self.reflection_coeff / max(total_distance, 0.01))
            
            if reflection_delay_samples < output_length:
                end_idx = min(reflection_delay_samples + len(source_waveform), output_length)
                waveform_length = end_idx - reflection_delay_samples
                if waveform_length > 0:
                    output[reflection_delay_samples:end_idx] += (
                        reflection_attenuation * source_waveform[:waveform_length]
                    )
        
        return output
    
    def _find_first_arrival(self, waveform: np.ndarray, threshold: float = 0.01) -> Optional[int]:
        """Find the index of the first significant arrival in a waveform."""
        if len(waveform) == 0:
            return None
        
        # Use energy-based threshold
        max_energy = np.max(waveform ** 2)
        if max_energy == 0:
            return None
        
        energy_threshold = threshold * max_energy
        energy_signal = waveform ** 2
        
        # Find first sample above threshold
        above_threshold = np.where(energy_signal > energy_threshold)[0]
        if len(above_threshold) > 0:
            return int(above_threshold[0])
        
        return None