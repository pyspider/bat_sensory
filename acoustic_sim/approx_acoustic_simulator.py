"""
Approximate binaural acoustic simulator using first-order image-source method.

Provides fast computation of binaural echoes for echolocation simulation
with configurable parameters including sample rate, pulse frequency,
reflection coefficient, and ear offset.
"""

import numpy as np
import math
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class EchoResult:
    """Result of binaural echo simulation containing waveforms and analysis."""
    left_waveform: np.ndarray
    right_waveform: np.ndarray
    left_first_sample: Optional[int]
    right_first_sample: Optional[int]
    itd: float  # Interaural Time Difference in samples
    ild: float  # Interaural Level Difference in dB
    energy: float  # Total energy across both channels


class ApproxAcousticSimulator:
    """
    First-order image-source binaural acoustic simulator.
    
    Simulates binaural echoes by reflecting sound sources across obstacle segments
    and computing arrival times and amplitudes at left and right ear positions.
    """
    
    def __init__(self, sample_rate: float = 44100, pulse_frequency: float = 40000,
                 reflection_coeff: float = 0.7, ear_offset: float = 0.02,
                 speed_of_sound: float = 343.0):
        """
        Initialize the acoustic simulator.
        
        Args:
            sample_rate: Audio sample rate in Hz
            pulse_frequency: Echolocation pulse frequency in Hz
            reflection_coeff: Wall reflection coefficient (0-1)
            ear_offset: Distance between ears in meters
            speed_of_sound: Speed of sound in m/s
        """
        self.sample_rate = sample_rate
        self.pulse_frequency = pulse_frequency
        self.reflection_coeff = reflection_coeff
        self.ear_offset = ear_offset
        self.speed_of_sound = speed_of_sound
        
    def simulate(self, source_pos: Tuple[float, float], source_angle: float,
                 receivers: List[Tuple[float, float]], 
                 source_waveform: Optional[np.ndarray] = None,
                 scene_segments: Optional[List] = None) -> Optional[EchoResult]:
        """
        Simulate binaural echoes for a pulse emitted from source_pos.
        
        Args:
            source_pos: (x, y) position of sound source
            source_angle: Angle of source emission in radians
            receivers: List of (x, y) positions for left and right ears
            source_waveform: Optional source waveform (generates default if None)
            scene_segments: List of obstacle segments for reflections
            
        Returns:
            EchoResult with binaural waveforms and analysis, or None if no echoes
        """
        if not receivers or len(receivers) < 2:
            return None
            
        left_ear, right_ear = receivers[0], receivers[1]
        
        # Generate default source waveform if not provided
        if source_waveform is None:
            source_waveform = self._generate_pulse()
            
        # Find reflection points for each obstacle segment
        reflection_points = []
        if scene_segments:
            for segment in scene_segments:
                refl_point = self._find_reflection_point(source_pos, left_ear, segment)
                if refl_point:
                    reflection_points.append((refl_point, segment))
                    
                refl_point = self._find_reflection_point(source_pos, right_ear, segment)
                if refl_point:
                    reflection_points.append((refl_point, segment))
        
        if not reflection_points:
            return None
            
        # Compute echoes for left and right ears
        left_waveform = self._compute_ear_response(source_pos, left_ear, 
                                                  reflection_points, source_waveform)
        right_waveform = self._compute_ear_response(source_pos, right_ear,
                                                   reflection_points, source_waveform)
        
        # Analyze the results
        left_first = self._find_first_arrival(left_waveform)
        right_first = self._find_first_arrival(right_waveform)
        
        itd = 0.0
        if left_first is not None and right_first is not None:
            itd = (right_first - left_first) / self.sample_rate
            
        ild = self._compute_ild(left_waveform, right_waveform)
        energy = np.sum(left_waveform**2) + np.sum(right_waveform**2)
        
        return EchoResult(
            left_waveform=left_waveform,
            right_waveform=right_waveform,
            left_first_sample=left_first,
            right_first_sample=right_first,
            itd=itd,
            ild=ild,
            energy=energy
        )
    
    def _generate_pulse(self, duration: float = 0.005) -> np.ndarray:
        """Generate a default echolocation pulse."""
        n_samples = int(duration * self.sample_rate)
        t = np.arange(n_samples) / self.sample_rate
        
        # Generate frequency-modulated chirp
        freq_sweep = self.pulse_frequency * (1 + 0.2 * t / duration)
        envelope = np.exp(-t / (duration * 0.3))  # Exponential decay
        
        pulse = envelope * np.sin(2 * np.pi * freq_sweep * t)
        return pulse
    
    def _reflect_source_across_segment(self, source_pos: Tuple[float, float],
                                      segment) -> Tuple[float, float]:
        """
        Reflect a source position across a line segment using mirror method.
        
        Args:
            source_pos: (x, y) position to reflect
            segment: Line segment object with p0, p1 points
            
        Returns:
            (x, y) position of reflected source
        """
        # Extract segment endpoints
        x1, y1 = segment.p0.x, segment.p0.y
        x2, y2 = segment.p1.x, segment.p1.y
        x0, y0 = source_pos
        
        # Line equation: ax + by + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2*y1 - x1*y2
        
        # Normalize
        norm = math.sqrt(a*a + b*b)
        if norm < 1e-10:
            return source_pos  # Degenerate segment
            
        a, b, c = a/norm, b/norm, c/norm
        
        # Reflect point across line
        d = a*x0 + b*y0 + c
        x_refl = x0 - 2*a*d
        y_refl = y0 - 2*b*d
        
        return (x_refl, y_refl)
    
    def _find_reflection_point(self, source_pos: Tuple[float, float],
                              receiver_pos: Tuple[float, float],
                              segment) -> Optional[Tuple[float, float]]:
        """
        Find the reflection point on a segment for source-receiver pair.
        
        Args:
            source_pos: (x, y) source position
            receiver_pos: (x, y) receiver position  
            segment: Line segment object
            
        Returns:
            (x, y) reflection point if valid, None otherwise
        """
        # Reflect source across segment
        refl_source = self._reflect_source_across_segment(source_pos, segment)
        
        # Find intersection of reflected-source-to-receiver line with segment
        x1, y1 = segment.p0.x, segment.p0.y
        x2, y2 = segment.p1.x, segment.p1.y
        x3, y3 = refl_source
        x4, y4 = receiver_pos
        
        # Line intersection formula
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-10:
            return None
            
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        
        # Check if intersection is within segment bounds
        if t < 0 or t > 1:
            return None
            
        # Compute intersection point
        x_int = x1 + t*(x2-x1)
        y_int = y1 + t*(y2-y1)
        
        return (x_int, y_int)
    
    def _compute_ear_response(self, source_pos: Tuple[float, float],
                             ear_pos: Tuple[float, float],
                             reflection_points: List[Tuple],
                             source_waveform: np.ndarray) -> np.ndarray:
        """
        Compute the acoustic response at an ear from all reflections.
        
        Args:
            source_pos: Source position
            ear_pos: Ear position
            reflection_points: List of (reflection_point, segment) tuples
            source_waveform: Source pulse waveform
            
        Returns:
            Summed waveform at the ear position
        """
        max_delay_samples = int(0.1 * self.sample_rate)  # 100ms max
        output_length = len(source_waveform) + max_delay_samples
        ear_response = np.zeros(output_length)
        
        for refl_point, segment in reflection_points:
            # Compute total path distance
            dist_source_to_refl = math.sqrt((refl_point[0] - source_pos[0])**2 + 
                                          (refl_point[1] - source_pos[1])**2)
            dist_refl_to_ear = math.sqrt((ear_pos[0] - refl_point[0])**2 + 
                                       (ear_pos[1] - refl_point[1])**2)
            total_distance = dist_source_to_refl + dist_refl_to_ear
            
            # Compute delay and attenuation
            delay_time = total_distance / self.speed_of_sound
            delay_samples = int(delay_time * self.sample_rate)
            
            if delay_samples >= max_delay_samples:
                continue
                
            # Distance and reflection attenuation
            amplitude = self.reflection_coeff / (total_distance + 1e-6)
            
            # Add delayed and attenuated waveform
            end_idx = min(delay_samples + len(source_waveform), output_length)
            waveform_len = end_idx - delay_samples
            ear_response[delay_samples:end_idx] += amplitude * source_waveform[:waveform_len]
            
        return ear_response
    
    def _find_first_arrival(self, waveform: np.ndarray, 
                           threshold: float = 0.01) -> Optional[int]:
        """
        Find the sample index of the first significant arrival.
        
        Args:
            waveform: Input waveform
            threshold: Detection threshold relative to peak
            
        Returns:
            Sample index of first arrival, or None if no arrival detected
        """
        if len(waveform) == 0:
            return None
            
        peak_amplitude = np.max(np.abs(waveform))
        if peak_amplitude < 1e-10:
            return None
            
        abs_threshold = threshold * peak_amplitude
        above_threshold = np.abs(waveform) > abs_threshold
        
        if not np.any(above_threshold):
            return None
            
        return np.argmax(above_threshold)
    
    def _compute_ild(self, left_waveform: np.ndarray, 
                     right_waveform: np.ndarray) -> float:
        """
        Compute Interaural Level Difference in dB.
        
        Args:
            left_waveform: Left ear waveform
            right_waveform: Right ear waveform
            
        Returns:
            ILD in dB (positive means left is louder)
        """
        left_energy = np.sum(left_waveform**2)
        right_energy = np.sum(right_waveform**2)
        
        if right_energy < 1e-10:
            return 20.0  # Large positive ILD
        if left_energy < 1e-10:
            return -20.0  # Large negative ILD
            
        return 10 * np.log10(left_energy / right_energy)