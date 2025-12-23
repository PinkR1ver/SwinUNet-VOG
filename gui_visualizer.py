import os
import sys
import threading
import time
import argparse
import cv2
import numpy as np
import torch
import customtkinter as ctk
from PIL import Image, ImageTk
import mediapipe as mp
from typing import Optional, Tuple
import matplotlib
# matplotlib.use('Agg') # Disable Agg backend to support interactive GUI
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal
from collections import deque
import plistlib
from datetime import datetime, timedelta
import tempfile
import pickle

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import SwinUNet
from preprocessing import EyeImagePreprocessor

class NystagmusAnalyzer:
    """Nystagmus (眼震) detection and analysis algorithm."""
    
    def __init__(self, fps=30.0):
        self.fps = fps
        
    def butter_highpass_filter(self, data, cutoff, fs, order=5):
        """Zero-phase highpass Butterworth filter."""
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        if normal_cutoff >= 1:
            return data
        b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
        filtered_data = scipy.signal.filtfilt(b, a, data, 
                                              padlen=min(len(data)-1, 3*(max(len(b), len(a))-1)))
        return filtered_data
    
    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        """Zero-phase lowpass Butterworth filter."""
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        if normal_cutoff >= 1:
            return data
        b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = scipy.signal.filtfilt(b, a, data, 
                                              padlen=min(len(data)-1, 3*(max(len(b), len(a))-1)))
        return filtered_data
    
    def moving_average_filter(self, data, window_size):
        """Moving average filter."""
        window_size = int(window_size)
        half_window = window_size // 2
        padded_data = np.pad(data, (half_window, half_window), mode='edge')
        ma = np.cumsum(padded_data, dtype=float)
        ma[window_size:] = ma[window_size:] - ma[:-window_size]
        filtered_data = ma[window_size - 1:] / window_size
        return filtered_data[:len(data)]
    
    def signal_preprocess(self, timestamps, eye_angles, 
                         highpass_cutoff=0.1, lowpass_cutoff=6.0, 
                         interpolate_ratio=10):
        """Preprocess signal with filtering and resampling."""
        original_time = timestamps
        original_signal = eye_angles
        
        min_len = min(len(original_time), len(original_signal))
        original_time = original_time[:min_len]
        original_signal = original_signal[:min_len]
        
        if len(original_signal) == 0 or len(original_time) == 0:
            return np.array([]), np.array([])
        
        # 1. Highpass filter (remove low-frequency drift)
        signal_filtered = self.butter_highpass_filter(
            original_signal, cutoff=highpass_cutoff, fs=self.fps, order=5
        )
        
        # 2. Lowpass filter (remove high-frequency noise)
        signal_filtered = self.butter_lowpass_filter(
            signal_filtered, cutoff=lowpass_cutoff, fs=self.fps, order=5
        )
        
        # 3. Resample (increase time resolution)
        signal_filtered = scipy.signal.resample(
            signal_filtered, int(len(original_signal) * interpolate_ratio)
        )
        
        # 4. Generate new time series
        time = np.linspace(original_time[0], original_time[-1], len(signal_filtered))
        
        return signal_filtered, time
    
    def find_turning_points(self, signal_data, prominence=0.1, distance=150):
        """Detect turning points (local maxima and minima)."""
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(signal_data, prominence=prominence, distance=distance)
        valleys, _ = find_peaks(-signal_data, prominence=prominence, distance=distance)
        turning_points = np.sort(np.concatenate([peaks, valleys]))
        return turning_points
    
    def calculate_slopes(self, time, signal, turning_points):
        """Calculate slopes between adjacent turning points."""
        slopes = []
        slope_times = []
        for i in range(len(turning_points)-1):
            t1, t2 = time[turning_points[i]], time[turning_points[i+1]]
            y1, y2 = signal[turning_points[i]], signal[turning_points[i+1]]
            slope = (y2 - y1) / (t2 - t1)
            slope_time = (t1 + t2) / 2
            slopes.append(slope)
            slope_times.append(slope_time)
        return np.array(slope_times), np.array(slopes)
    
    def identify_nystagmus_patterns(self, signal_data, time_data, 
                                    min_time=0.3, max_time=0.8,
                                    min_ratio=1.4, max_ratio=8.0, 
                                    direction_axis="horizontal"):
        """
        Identify nystagmus patterns (fast and slow phases).
        
        Returns:
            patterns: Valid nystagmus patterns
            filtered_patterns: Patterns filtered out by CV
            direction: Nystagmus direction (left/right/up/down)
            spv: Slow Phase Velocity (deg/s)
            cv: Coefficient of Variation (%)
        """
        # Detect turning points
        turning_points = self.find_turning_points(signal_data, prominence=0.1, distance=150)
        
        if len(turning_points) < 3:
            return [], [], "unknown", 0, float('inf')
        
        # Collect potential nystagmus patterns
        potential_patterns = []
        
        for i in range(1, len(turning_points)-1):
            idx1 = turning_points[i-1]
            idx2 = turning_points[i]
            idx3 = turning_points[i+1]
            
            p1 = np.array([time_data[idx1], signal_data[idx1]])
            p2 = np.array([time_data[idx2], signal_data[idx2]])
            p3 = np.array([time_data[idx3], signal_data[idx3]])
            
            # Check time threshold
            total_time = p3[0] - p1[0]
            if not (min_time <= total_time <= max_time):
                continue
            
            # Calculate slopes
            slope_before = (p2[1] - p1[1]) / (p2[0] - p1[0])
            slope_after = (p3[1] - p2[1]) / (p3[0] - p2[0])
            
            # Determine fast/slow phase
            if abs(slope_before) > abs(slope_after):
                fast_slope = slope_before
                slow_slope = slope_after
                fast_phase_first = True
            else:
                fast_slope = slope_after
                slow_slope = slope_before
                fast_phase_first = False
            
            # Check direction consistency (fast and slow should be opposite)
            if fast_slope * slow_slope > 0:
                continue
            
            ratio = abs(fast_slope) / abs(slow_slope) if slow_slope != 0 else float('inf')
            
            if min_ratio <= ratio <= max_ratio:
                potential_patterns.append({
                    'index': i,
                    'time_point': time_data[idx2],
                    'slow_slope': slow_slope,
                    'fast_slope': fast_slope,
                    'ratio': ratio,
                    'fast_phase_first': fast_phase_first,
                    'total_time': total_time
                })
        
        if not potential_patterns:
            return [], [], "unknown", 0, float('inf')
        
        # CV-based outlier filtering using MAD (Median Absolute Deviation)
        slow_slopes = np.array([p['slow_slope'] for p in potential_patterns])
        original_indices = list(range(len(potential_patterns)))
        
        median_slope = np.median(slow_slopes)
        mad = np.median(np.abs(slow_slopes - median_slope))
        mad_normalized = 1.4826 * mad
        cv = (mad_normalized / abs(median_slope)) * 100 if median_slope != 0 else float('inf')
        
        # Iteratively remove outliers until CV <= 20% or only 3 patterns remain
        filtered_indices = []
        while cv > 20 and len(slow_slopes) > 3:
            modified_z_scores = 0.6745 * np.abs(slow_slopes - median_slope) / (mad + 1e-8)
            max_z_idx = np.argmax(modified_z_scores)
            filtered_indices.append(original_indices[max_z_idx])
            slow_slopes = np.delete(slow_slopes, max_z_idx)
            original_indices.pop(max_z_idx)
            
            median_slope = np.median(slow_slopes)
            mad = np.median(np.abs(slow_slopes - median_slope))
            mad_normalized = 1.4826 * mad
            cv = (mad_normalized / abs(median_slope)) * 100 if median_slope != 0 else float('inf')
        
        # Separate valid and filtered patterns
        patterns = []
        filtered_patterns = []
        for idx, pattern in enumerate(potential_patterns):
            if idx in filtered_indices:
                filtered_patterns.append(pattern)
            else:
                patterns.append(pattern)
        
        # Calculate final direction and SPV
        final_median_slope = np.median(slow_slopes)
        
        if direction_axis == "horizontal":
            direction = "left" if final_median_slope > 0 else "right"
        else:
            direction = "up" if final_median_slope > 0 else "down"
        
        spv = abs(final_median_slope)
        
        return patterns, filtered_patterns, direction, spv, cv
    
    def analyze(self, timestamps, angles, blink_mask, axis="horizontal"):
        """
        Full nystagmus analysis pipeline.
        
        Args:
            timestamps: Time array
            angles: Angle array (Pitch or Yaw)
            blink_mask: Boolean mask where True = blink (to exclude)
            axis: "horizontal" (yaw) or "vertical" (pitch)
        
        Returns:
            dict with analysis results
        """
        # Exclude blink data
        valid_mask = ~blink_mask
        valid_times = timestamps[valid_mask]
        valid_angles = angles[valid_mask]
        
        if len(valid_times) < 30:
            return {
                'success': False,
                'error': 'Not enough valid data (too many blinks)',
                'n_valid_samples': len(valid_times)
            }
        
        # Store original data for visualization
        original_times = valid_times.copy()
        original_angles = valid_angles.copy()
        
        # Step 1: High-pass filter only
        signal_highpass = self.butter_highpass_filter(
            valid_angles, cutoff=0.1, fs=self.fps, order=5
        )
        
        # Step 2: Low-pass filter (on highpass output)
        signal_lowpass = self.butter_lowpass_filter(
            signal_highpass, cutoff=6.0, fs=self.fps, order=5
        )
        
        # Step 3: Full preprocessing (includes resampling)
        filtered_signal, time = self.signal_preprocess(
            valid_times, valid_angles,
            highpass_cutoff=0.1, lowpass_cutoff=6.0, interpolate_ratio=10
        )
        
        if len(filtered_signal) == 0:
            return {
                'success': False,
                'error': 'Signal preprocessing failed'
            }
        
        # Detect turning points
        turning_points = self.find_turning_points(filtered_signal, prominence=0.1, distance=150)
        
        # Calculate slopes
        slope_times, slopes = self.calculate_slopes(time, filtered_signal, turning_points)
        
        # Identify nystagmus patterns
        patterns, filtered_patterns, direction, spv, cv = self.identify_nystagmus_patterns(
            filtered_signal, time,
            min_time=0.3, max_time=0.8,
            min_ratio=1.4, max_ratio=8.0,
            direction_axis=axis
        )
        
        return {
            'success': True,
            'axis': axis,
            'n_valid_samples': len(valid_times),
            'n_blink_samples': np.sum(blink_mask),
            # Original data for plotting
            'original_times': original_times,
            'original_angles': original_angles,
            # Intermediate processing steps
            'signal_highpass': signal_highpass,
            'signal_lowpass': signal_lowpass,
            # Final processed data
            'filtered_signal': filtered_signal,
            'time': time,
            'turning_points': turning_points,
            'slope_times': slope_times,
            'slopes': slopes,
            'patterns': patterns,
            'filtered_patterns': filtered_patterns,
            'n_patterns': len(patterns),
            'n_filtered_patterns': len(filtered_patterns),
            'direction': direction,
            'spv': spv,
            'cv': cv
        }


class SignalProcessor:
    """Applies filtering to smooth gaze data."""
    def __init__(self, fps=30.0, low_pass_cutoff=8.0):
        self.fps = fps
        self.nyquist = fps / 2.0
        self.low_pass_cutoff = low_pass_cutoff
        
    def process(self, data):
        """
        data: np.array of shape (N, 3). May contain NaNs for blink frames.
        Returns: smoothed data of shape (N, 3), NaNs are interpolated.
        """
        if len(data) < 15: # Need some data for filters
            return data
            
        # Handle NaNs (Interpolation)
        # Convert to DataFrame for easy interpolation or use numpy
        # We use simple linear interpolation here
        n_samples, n_features = data.shape
        filled_data = data.copy()
        
        # Linear interpolation for each channel
        for i in range(n_features):
            y = filled_data[:, i]
            nans = np.isnan(y)
            
            # If all are NaNs, return zeros (shouldn't happen usually)
            if np.all(nans):
                filled_data[:, i] = 0
                continue
                
            # Get valid indices
            x = np.arange(n_samples)
            
            # Interpolate
            filled_data[nans, i] = np.interp(x[nans], x[~nans], y[~nans])
            
        data = filled_data
            
        # 1. Median Filter to remove spikes (毛刺)
        kernel_size = 5
        filtered_data = np.zeros_like(data)
        for i in range(3):
            filtered_data[:, i] = scipy.signal.medfilt(data[:, i], kernel_size=kernel_size)
            
        # 2. Low-pass Butterworth filter for smoothing
        # Ensure cutoff is valid
        cutoff = min(self.low_pass_cutoff, self.nyquist - 0.1)
        b, a = scipy.signal.butter(2, cutoff / self.nyquist, btype='low')
        for i in range(3):
            filtered_data[:, i] = scipy.signal.filtfilt(b, a, filtered_data[:, i])
            
        return filtered_data

class MediaPipeEyeNormalizer:
    """Using MediaPipe to extract and normalize eye ROI."""
    
    # MediaPipe FaceMesh indices
    # Left Eye - Main landmarks
    LEFT_EYE_OUTER = 263
    LEFT_EYE_INNER = 362
    LEFT_EYE_UPPER = 386
    LEFT_EYE_BOTTOM = 374
    LEFT_PUPIL = 468  # iris center (from refine_landmarks)
    
    # Left Eye - Additional landmarks for better alignment
    LEFT_EYE_UPPER_LEFT = 387
    LEFT_EYE_UPPER_RIGHT = 385
    LEFT_EYE_LOWER_LEFT = 380
    LEFT_EYE_LOWER_RIGHT = 373
    
    # Right Eye - Main landmarks
    RIGHT_EYE_OUTER = 33
    RIGHT_EYE_INNER = 133
    RIGHT_EYE_UPPER = 159
    RIGHT_EYE_BOTTOM = 145
    RIGHT_PUPIL = 473  # iris center (from refine_landmarks)
    
    # Right Eye - Additional landmarks for better alignment
    RIGHT_EYE_UPPER_LEFT = 160
    RIGHT_EYE_UPPER_RIGHT = 158
    RIGHT_EYE_LOWER_LEFT = 144
    RIGHT_EYE_LOWER_RIGHT = 153
    
    def __init__(self, eye: str = 'left', target_size: Tuple[int, int] = (36, 60),
                 padding: float = 0.15, preprocessing_kwargs: Optional[dict] = None,
                 blink_window_extend_sec: float = 0.3,
                 extraction_method: str = 'perspective',  # 'perspective', 'affine', 'affine_6pt', 'bbox_scale', or 'bbox_fill'
                 # Simplified brightness/contrast enhancement (grayscale only)
                 enhance_enabled: bool = True,  # Enable grayscale enhancement
                 enhance_gamma: float = 1.0,    # Gamma for brightness (lower=brighter, 1.0=no change)
                 enhance_clahe_clip: float = 1.2):  # CLAHE clip limit for contrast (higher=more contrast)
        """
        Eye extraction methods:
        - 'perspective': 4-point perspective transform (RECOMMENDED - closest to MPIIGaze normalization)
        - 'affine': 3-point affine transform (simple, but doesn't handle perspective)
        - 'affine_6pt': 6-point homography (more robust affine)
        - 'bbox_scale': Bounding box with aspect ratio preservation
        - 'bbox_fill': Bounding box stretched to fill (may distort)
        
        Simplified enhancement (grayscale output):
        - enhance_enabled: Enable/disable enhancement
        - enhance_gamma: Brightness control (0.3=very bright, 1.0=no change)
        - enhance_clahe_clip: Contrast control (1.0=minimal, 4.0=strong)
        """
        self.eye = eye.lower()
        self.target_size = target_size  # (H, W)
        self.padding = np.clip(padding, 0.05, 0.4)
        self.blink_window_extend_sec = blink_window_extend_sec  # Time-based blink window extension
        self.extraction_method = extraction_method  # Method for eye extraction
        
        # Simplified enhancement settings
        self.enhance_enabled = enhance_enabled
        self.enhance_gamma = enhance_gamma
        self.enhance_clahe_clip = enhance_clahe_clip
        
        # CLAHE with user-specified clip limit
        self.clahe = cv2.createCLAHE(clipLimit=enhance_clahe_clip, tileGridSize=(4, 4))
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        if preprocessing_kwargs is None:
            preprocessing_kwargs = {}
        preprocessing_kwargs = dict(preprocessing_kwargs)
        # Defaults matching FaceAlignmentEyeNormalizer
        preprocessing_kwargs.setdefault('use_geometric_normalization', False)
        # Disable advanced preprocessing for raw visualization
        preprocessing_kwargs.setdefault('normalize_illumination', False)
        preprocessing_kwargs.setdefault('normalize_contrast', False)
        preprocessing_kwargs.setdefault('normalize_color', False)
        preprocessing_kwargs.setdefault('gamma_correction', False)
        preprocessing_kwargs.setdefault('adaptive_hist_eq', False)
        
        self.preprocessor = EyeImagePreprocessor(
            target_size=target_size,
            **preprocessing_kwargs
        )
        
        h, w = target_size
        outer_x = w * self.padding
        inner_x = w * (1.0 - self.padding)
        center_y = h * 0.55
        upper_y = h * 0.25
        lower_y = h * 0.75
        
        # Target points for 3-point affine transform (used if method='affine')
        self.target_points = np.float32([
            [outer_x, center_y],   # outer corner
            [inner_x, center_y],   # inner corner
            [w * 0.5, upper_y],    # upper eyelid
        ])
        
        # Target points for 4-point PERSPECTIVE transform (method='perspective')
        # Using 4 corners of eye: outer, inner, upper, lower
        # This is closest to MPIIGaze's perspective normalization
        self.target_points_4pt = np.float32([
            [outer_x, center_y],           # outer corner
            [inner_x, center_y],           # inner corner
            [w * 0.5, upper_y],            # upper center
            [w * 0.5, lower_y],            # lower center
        ])
        
        # Target points for 6-point affine transform (method='affine_6pt')
        # More points = more robust alignment, similar to MPIIGaze normalization
        # Format: [outer, inner, upper, lower, upper_left, upper_right]
        self.target_points_6pt = np.float32([
            [outer_x, center_y],           # outer corner
            [inner_x, center_y],           # inner corner  
            [w * 0.5, upper_y],            # upper center
            [w * 0.5, lower_y],            # lower center
            [w * 0.35, upper_y + 2],       # upper left
            [w * 0.65, upper_y + 2],       # upper right
        ])
        
        self.last_roi_tensor = None
        self.last_roi_display = None
        
        # Time-based blink tracking
        self.blink_timestamps = []  # Store timestamps of detected blinks

    def _is_blinking(self, landmarks):
        """Calculate eye opening ratio to detect blinking."""
        if self.eye == 'right':
            upper_idx = self.RIGHT_EYE_UPPER
            lower_idx = self.RIGHT_EYE_BOTTOM
            inner_idx = self.RIGHT_EYE_INNER
            outer_idx = self.RIGHT_EYE_OUTER
        else:
            upper_idx = self.LEFT_EYE_UPPER
            lower_idx = self.LEFT_EYE_BOTTOM
            inner_idx = self.LEFT_EYE_INNER
            outer_idx = self.LEFT_EYE_OUTER
            
        p_up = landmarks[upper_idx]
        p_lo = landmarks[lower_idx]
        p_in = landmarks[inner_idx]
        p_out = landmarks[outer_idx]
        
        # Simple 2D Euclidean distance (ignoring Z depth for speed/robustness in 2D image)
        v_dist = np.sqrt((p_up.x - p_lo.x)**2 + (p_up.y - p_lo.y)**2)
        h_dist = np.sqrt((p_in.x - p_out.x)**2 + (p_in.y - p_out.y)**2)
        
        ratio = v_dist / (h_dist + 1e-6)
        
        # Threshold: typically 0.15 - 0.25 indicates blinking
        return ratio < 0.20
    
    def _enhance_simple(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Simple grayscale enhancement with Gamma + CLAHE.
        
        Uses self.enhance_gamma and self.enhance_clahe_clip parameters.
        - gamma=1.0: no brightness change
        - clahe_clip=1.2: light contrast enhancement
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Gamma correction (only if gamma != 1.0)
        if self.enhance_gamma != 1.0:
            inv_gamma = 1.0 / self.enhance_gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                             for i in range(256)]).astype("uint8")
            gray = cv2.LUT(gray, table)
        
        # Step 2: CLAHE for contrast
        gray = self.clahe.apply(gray)
        
        # Step 3: Contrast stretching to use full range
        p2, p98 = np.percentile(gray, (1, 99))
        if p98 > p2:
            gray = np.clip(gray, p2, p98)
            gray = ((gray - p2) / (p98 - p2) * 255).astype(np.uint8)
        
        # Convert back to BGR for consistent processing
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def extract(self, frame_bgr: np.ndarray, timestamp: float = None) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], bool]:
        """
        Extract eye ROI with time-based blink window extension.
        
        Args:
            frame_bgr: Input frame in BGR format
            timestamp: Current timestamp in seconds (required for time-based blink extension)
            
        Returns:
            (roi_tensor, roi_display, is_in_blink_window)
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None, None, False
        
        # Simple grayscale enhancement (Gamma + CLAHE)
        if self.enhance_enabled:
            frame_for_detection = self._enhance_simple(frame_bgr)
        else:
            frame_for_detection = frame_bgr
            
        frame_rgb = cv2.cvtColor(frame_for_detection, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None, None, False
            
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Check if currently blinking
        is_blinking = self._is_blinking(landmarks)
        
        # Time-based Blink Window Expansion
        if timestamp is not None:
            # Check if current timestamp is within extended window of any pre-detected blink
            # Window is SYMMETRIC: [blink_time - extend_sec, blink_time + extend_sec]
            # This ensures we filter BOTH before and after the blink event
            effective_blink = False
            for blink_time in self.blink_timestamps:
                time_diff = timestamp - blink_time
                # Symmetric window: covers both before (-) and after (+) the blink
                if -self.blink_window_extend_sec <= time_diff <= self.blink_window_extend_sec:
                    effective_blink = True
                    break
        else:
            # Fallback to frame-based if no timestamp provided (backward compatibility)
            if not hasattr(self, '_blink_history'):
                self._blink_history = deque(maxlen=3)
            self._blink_history.append(is_blinking)
            effective_blink = any(self._blink_history)
        
        h_img, w_img, _ = frame_for_detection.shape
        
        # Select points based on eye
        if self.eye == 'right':
            outer_idx = self.RIGHT_EYE_OUTER
            inner_idx = self.RIGHT_EYE_INNER
            upper_idx = self.RIGHT_EYE_UPPER
        else:
            outer_idx = self.LEFT_EYE_OUTER
            inner_idx = self.LEFT_EYE_INNER
            upper_idx = self.LEFT_EYE_UPPER
            
        # Get pixel coordinates
        def get_point(idx):
            lm = landmarks[idx]
            return [lm.x * w_img, lm.y * h_img]
            
        outer_corner = np.array(get_point(outer_idx), dtype=np.float32)
        inner_corner = np.array(get_point(inner_idx), dtype=np.float32)
        upper_center = np.array(get_point(upper_idx), dtype=np.float32)
        
        # Also get bottom point for better bounding box
        if self.eye == 'right':
            bottom_idx = self.RIGHT_EYE_BOTTOM
        else:
            bottom_idx = self.LEFT_EYE_BOTTOM
        bottom_center = np.array(get_point(bottom_idx), dtype=np.float32)
        
        try:
            h, w = self.target_size
            
            if self.extraction_method == 'bbox_fill':
                # Method: Bounding box with FILL - stretches to fill target (allows some distortion)
                # Best for maximizing eye pixels, slight aspect ratio change is acceptable
                
                # Get eye bounding box from landmarks
                all_eye_points = np.array([outer_corner, inner_corner, upper_center, bottom_center])
                x_min, y_min = all_eye_points.min(axis=0)
                x_max, y_max = all_eye_points.max(axis=0)
                
                # Minimal padding - we want the eye to fill the frame
                bbox_w = x_max - x_min
                bbox_h = y_max - y_min
                pad_x = bbox_w * 0.12
                pad_y = bbox_h * 0.35
                
                x_min = max(0, x_min - pad_x)
                x_max = min(w_img, x_max + pad_x)
                y_min = max(0, y_min - pad_y)
                y_max = min(h_img, y_max + pad_y)
                
                # Crop the eye region
                x_min_i, y_min_i = int(x_min), int(y_min)
                x_max_i, y_max_i = int(x_max), int(y_max)
                
                cropped = frame_for_detection[y_min_i:y_max_i, x_min_i:x_max_i]
                
                if cropped.size == 0:
                    return None, None, False
                
                # Resize directly to target size (allows distortion but maximizes eye pixels)
                warped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)
                
            elif self.extraction_method == 'bbox_scale':
                # Method: Bounding box with aspect ratio preservation
                # This avoids the distortion caused by affine transform
                
                # Get eye bounding box from landmarks
                all_eye_points = np.array([outer_corner, inner_corner, upper_center, bottom_center])
                x_min, y_min = all_eye_points.min(axis=0)
                x_max, y_max = all_eye_points.max(axis=0)
                
                # Add minimal padding around the eye (smaller padding = larger eye in final image)
                bbox_w = x_max - x_min
                bbox_h = y_max - y_min
                pad_x = bbox_w * 0.15  # Reduced horizontal padding for larger eye
                pad_y = bbox_h * 0.4   # Reduced vertical padding
                
                x_min = max(0, x_min - pad_x)
                x_max = min(w_img, x_max + pad_x)
                y_min = max(0, y_min - pad_y)
                y_max = min(h_img, y_max + pad_y)
                
                # Crop the eye region
                x_min_i, y_min_i = int(x_min), int(y_min)
                x_max_i, y_max_i = int(x_max), int(y_max)
                
                cropped = frame_for_detection[y_min_i:y_max_i, x_min_i:x_max_i]
                
                if cropped.size == 0:
                    return None, None, False
                
                crop_h, crop_w = cropped.shape[:2]
                
                # Calculate scale to fit within target size while preserving aspect ratio
                scale = min(w / crop_w, h / crop_h)
                new_w = int(crop_w * scale)
                new_h = int(crop_h * scale)
                
                # Resize with preserved aspect ratio
                resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Create padded image (use edge reflection instead of black)
                padded = np.zeros((h, w, 3), dtype=np.uint8)
                
                # Calculate padding offsets to center the eye
                y_offset = (h - new_h) // 2
                x_offset = (w - new_w) // 2
                
                # Fill with edge colors for smooth borders
                if new_h > 0 and new_w > 0:
                    # Top/bottom edge extension
                    if y_offset > 0:
                        padded[:y_offset, x_offset:x_offset+new_w] = resized[0:1, :]
                        padded[y_offset+new_h:, x_offset:x_offset+new_w] = resized[-1:, :]
                    # Left/right edge extension
                    if x_offset > 0:
                        padded[y_offset:y_offset+new_h, :x_offset] = resized[:, 0:1]
                        padded[y_offset:y_offset+new_h, x_offset+new_w:] = resized[:, -1:]
                    # Corners
                    if y_offset > 0 and x_offset > 0:
                        padded[:y_offset, :x_offset] = resized[0, 0]
                        padded[:y_offset, x_offset+new_w:] = resized[0, -1]
                        padded[y_offset+new_h:, :x_offset] = resized[-1, 0]
                        padded[y_offset+new_h:, x_offset+new_w:] = resized[-1, -1]
                    
                    # Place the resized image
                    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                
                warped = padded
            
            elif self.extraction_method == 'perspective':
                # Method: 4-point PERSPECTIVE transform (RECOMMENDED)
                # This is the closest match to MPIIGaze's normalization approach
                # Uses exactly 4 eye corners to compute a perspective transform matrix
                # which can handle 3D perspective distortion (unlike affine transform)
                
                # 4 source points: outer corner, inner corner, upper center, lower center
                source_points_4 = np.float32([
                    outer_corner,
                    inner_corner,
                    upper_center,
                    bottom_center
                ])
                
                # Get perspective transform matrix (3x3)
                # This is what MPIIGaze essentially does - cancel perspective distortion
                M = cv2.getPerspectiveTransform(source_points_4, self.target_points_4pt)
                
                warped = cv2.warpPerspective(
                    frame_for_detection,
                    M,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT101
                )
                
            elif self.extraction_method == 'affine_6pt':
                # Method: 6-point affine transform using homography
                # More similar to MPIIGaze normalization - uses multiple landmarks
                # for more robust geometric alignment
                
                # Get additional landmarks for 6-point alignment
                if self.eye == 'right':
                    upper_left_idx = self.RIGHT_EYE_UPPER_LEFT
                    upper_right_idx = self.RIGHT_EYE_UPPER_RIGHT
                else:
                    upper_left_idx = self.LEFT_EYE_UPPER_LEFT
                    upper_right_idx = self.LEFT_EYE_UPPER_RIGHT
                
                upper_left = np.array(get_point(upper_left_idx), dtype=np.float32)
                upper_right = np.array(get_point(upper_right_idx), dtype=np.float32)
                
                # 6 source points: outer, inner, upper, lower, upper_left, upper_right
                source_points_6 = np.float32([
                    outer_corner,
                    inner_corner,
                    upper_center,
                    bottom_center,
                    upper_left,
                    upper_right
                ])
                
                # Use homography (perspective transform) for better alignment
                # This is closer to what MPIIGaze normalization does
                M, mask = cv2.findHomography(source_points_6, self.target_points_6pt, cv2.RANSAC, 5.0)
                
                if M is not None:
                    warped = cv2.warpPerspective(
                        frame_for_detection,
                        M,
                        (w, h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT101
                    )
                else:
                    # Fallback to 3-point affine if homography fails
                    source_points = np.float32([outer_corner, inner_corner, upper_center])
                    M = cv2.getAffineTransform(source_points, self.target_points)
                    warped = cv2.warpAffine(
                        frame_for_detection,
                        M,
                        (w, h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT101
                    )
                
            else:
                # Original 3-point affine transform method (method='affine')
                source_points = np.float32([
                    outer_corner,
                    inner_corner,
                    upper_center
                ])
                
                M = cv2.getAffineTransform(source_points, self.target_points)
                warped = cv2.warpAffine(
                    frame_for_detection,
                    M,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT101
                )
            
            if self.eye == 'right':
                warped = cv2.flip(warped, 1)
                
            roi_display = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            roi_tensor = self.preprocessor(warped)
            
            self.last_roi_tensor = roi_tensor
            self.last_roi_display = roi_display
            
            return roi_tensor, roi_display, effective_blink
            
        except Exception as e:
            print(f"Error in extraction: {e}")
            return None, None, False
            
    def close(self):
        self.face_mesh.close()

class GazeVisualizerApp(ctk.CTk):
    def flush_buffers_to_disk(self):
        """Write buffered data to disk to free memory."""
        if not self.gaze_buffer:
            return
            
        # Append to existing files
        if os.path.exists(self.gaze_history_file):
            existing_gaze = np.load(self.gaze_history_file)
            new_gaze = np.vstack([existing_gaze, np.array(self.gaze_buffer)])
        else:
            new_gaze = np.array(self.gaze_buffer)
        np.save(self.gaze_history_file, new_gaze)
        
        if os.path.exists(self.time_history_file):
            existing_time = np.load(self.time_history_file)
            new_time = np.concatenate([existing_time, np.array(self.time_buffer)])
        else:
            new_time = np.array(self.time_buffer)
        np.save(self.time_history_file, new_time)
        
        # Eye images (pickle for variable size images)
        if os.path.exists(self.eye_history_file):
            with open(self.eye_history_file, 'rb') as f:
                existing_eye = pickle.load(f)
            existing_eye.extend(self.eye_buffer)
            with open(self.eye_history_file, 'wb') as f:
                pickle.dump(existing_eye, f)
        else:
            with open(self.eye_history_file, 'wb') as f:
                pickle.dump(self.eye_buffer, f)
        
        # Clear buffers
        self.gaze_buffer.clear()
        self.time_buffer.clear()
        self.eye_buffer.clear()
    
    def load_history_from_disk(self):
        """Load all history data from disk."""
        gaze_history = np.load(self.gaze_history_file) if os.path.exists(self.gaze_history_file) else np.array([])
        time_history = np.load(self.time_history_file) if os.path.exists(self.time_history_file) else np.array([])
        
        eye_history = []
        if os.path.exists(self.eye_history_file):
            with open(self.eye_history_file, 'rb') as f:
                eye_history = pickle.load(f)
        
        return gaze_history, time_history, eye_history
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except:
                pass
    
    def __init__(self):
        super().__init__()
        
        self.title("SwinUNet-VOG Gaze Visualization")
        self.geometry("800x600")
        
        # Config
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.normalizer = None
        self.is_processing = False
        self.video_path = None
        
        # Cache for video frames and eye images
        self.frame_cache = []  # List of (frame_bgr, eye_img_rgb)
        self.draggable_marker = None
        
        # Register cleanup on exit
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.setup_ui()
        
        # Try to load default checkpoint
        self.load_default_checkpoint()
    
    def on_closing(self):
        """Handle window close event."""
        # Stop processing if running
        self.is_processing = False
        
        # Close normalizer
        if hasattr(self, 'normalizer') and self.normalizer is not None:
            try:
                self.normalizer.close()
            except:
                pass
        
        # Close video capture
        if hasattr(self, 'review_cap') and self.review_cap is not None:
            try:
                self.review_cap.release()
            except:
                pass
        
        # Close all matplotlib figures
        try:
            plt.close('all')
        except:
            pass
        
        # Clean up temp files
        self.cleanup_temp_files()
        
        self.destroy()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Main container
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        self.label_title = ctk.CTkLabel(self.main_frame, text="Drop Video Here or Select File", font=("Arial", 24))
        self.label_title.grid(row=0, column=0, pady=(50, 20))
        
        # Drop/Select Button
        self.btn_select = ctk.CTkButton(self.main_frame, text="Select Video File", command=self.select_file, height=50, font=("Arial", 16))
        self.btn_select.grid(row=1, column=0, pady=20)
        
        # Status
        self.label_status = ctk.CTkLabel(self.main_frame, text="Ready", text_color="gray")
        self.label_status.grid(row=2, column=0, pady=10)
        
        # Options frame
        self.frame_options = ctk.CTkFrame(self.main_frame)
        self.frame_options.grid(row=3, column=0, pady=10)
        
        # Animation option
        self.animation_var = ctk.BooleanVar(value=False)
        self.chk_animation = ctk.CTkCheckBox(
            self.frame_options, 
            text="🎬 Enable Real-time Animation (slower)", 
            variable=self.animation_var,
            font=("Arial", 13)
        )
        self.chk_animation.pack(pady=5)
        
        # 3D Gaze Vector option
        self.show_3d_var = ctk.BooleanVar(value=True)
        self.chk_3d_gaze = ctk.CTkCheckBox(
            self.frame_options, 
            text="🎯 Show 3D Gaze Vector Window", 
            variable=self.show_3d_var,
            font=("Arial", 13)
        )
        self.chk_3d_gaze.pack(pady=5)
        
        # Controls
        self.frame_controls = ctk.CTkFrame(self.main_frame)
        self.frame_controls.grid(row=4, column=0, pady=20)
        
        self.btn_start = ctk.CTkButton(self.frame_controls, text="Start Processing", command=self.start_processing, state="disabled")
        self.btn_start.pack(side="left", padx=10)
        
        self.btn_stop = ctk.CTkButton(self.frame_controls, text="Stop", command=self.stop_processing, state="disabled", fg_color="red")
        self.btn_stop.pack(side="left", padx=10)
        
        # Checkpoint selection
        self.btn_ckpt = ctk.CTkButton(self.main_frame, text="Load Checkpoint", command=self.select_checkpoint, width=100)
        self.btn_ckpt.grid(row=5, column=0, pady=10)
        self.label_ckpt = ctk.CTkLabel(self.main_frame, text="No checkpoint loaded", font=("Arial", 10))
        self.label_ckpt.grid(row=6, column=0)

    def load_default_checkpoint(self):
        possible_paths = [
            "checkpoints/gaze/checkpoint_best.pth",   # New location
            "checkpoints/gaze/checkpoint_latest.pth",
            "checkpoints/checkpoint_best.pth",        # Fallback to old location
            "checkpoints/checkpoint_latest.pth"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                self.load_model(path)
                break

    def select_checkpoint(self):
        file_path = ctk.filedialog.askopenfilename(filetypes=[("PyTorch Checkpoint", "*.pth")])
        if file_path:
            self.load_model(file_path)

    def load_model(self, path):
        try:
            self.label_status.configure(text=f"Loading model from {os.path.basename(path)}...")
            self.update_idletasks()
            
            checkpoint = torch.load(path, map_location=self.device)
            self.model = SwinUNet(
                img_size=(36, 60),
                in_chans=3,
                embed_dim=96,
                depths=[2, 2, 2],
                num_heads=[3, 6, 12],
                window_size=7,
                drop_rate=0.1
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            self.label_ckpt.configure(text=f"Loaded: {os.path.basename(path)}")
            self.label_status.configure(text="Model loaded. Ready to process video.")
            if self.video_path:
                self.btn_start.configure(state="normal")
                
        except Exception as e:
            self.label_status.configure(text=f"Error loading model: {str(e)}")
            print(f"Error: {e}")

    def select_file(self):
        file_path = ctk.filedialog.askopenfilename(filetypes=[("Video Files", "*.mkv *.mp4 *.avi")])
        if file_path:
            self.video_path = file_path
            self.label_title.configure(text=os.path.basename(file_path))
            if self.model is not None:
                self.btn_start.configure(state="normal")
            self.label_status.configure(text="Video selected.")

    def vector_to_pitch_yaw(self, vector):
        """
        Convert 3D gaze vector to Pitch and Yaw angles in degrees.
        MPIIGaze normalization convention:
        gaze_vector = (x, y, z)
        pitch = arcsin(-y)
        yaw = arctan2(-x, -z)
        """
        if vector.ndim == 1:
            x, y, z = vector
            pitch = np.arcsin(np.clip(-y, -1.0, 1.0))
            yaw = np.arctan2(-x, -z)
            return np.degrees(np.array([pitch, yaw]))
        else:
            x, y, z = vector[:, 0], vector[:, 1], vector[:, 2]
            pitch = np.arcsin(np.clip(-y, -1.0, 1.0))
            yaw = np.arctan2(-x, -z)
            return np.degrees(np.stack([pitch, yaw], axis=1))

    def cleanup_previous_session(self):
        """Clean up resources from previous processing session."""
        # Close previous normalizer
        if hasattr(self, 'normalizer') and self.normalizer is not None:
            try:
                self.normalizer.close()
            except:
                pass
            self.normalizer = None
        
        # Close previous review video capture
        if hasattr(self, 'review_cap') and self.review_cap is not None:
            try:
                self.review_cap.release()
            except:
                pass
            self.review_cap = None
        
        # Close matplotlib figures
        if hasattr(self, 'fig'):
            try:
                plt.close(self.fig)
            except:
                pass
        if hasattr(self, 'fig_3d'):
            try:
                plt.close(self.fig_3d)
            except:
                pass
        
        # Close previous windows
        windows_to_close = ['win_eye', 'win_gaze', 'win_video', 'win_gaze_vector', 'win_progress']
        for win_name in windows_to_close:
            if hasattr(self, win_name):
                win = getattr(self, win_name)
                if win is not None:
                    try:
                        if win.winfo_exists():
                            win.destroy()
                    except:
                        pass
                setattr(self, win_name, None)
        
        # Clean up temp files from previous session
        self.cleanup_temp_files()
        
        # Clear history data
        self.gaze_history = []
        self.time_history = []
        self.eye_history = []
        self.frame_cache = []
        self.frame_index_history = []

    def start_processing(self):
        if not self.video_path or not self.model:
            return
        
        # Clean up previous session first
        self.cleanup_previous_session()
            
        self.is_processing = True
        self.btn_start.configure(state="disabled")
        self.btn_select.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        
        # Initialize history - use temporary file to reduce memory usage
        self.temp_dir = tempfile.mkdtemp(prefix="swinunet_vog_")
        self.gaze_history_file = os.path.join(self.temp_dir, "gaze_history.npy")
        self.time_history_file = os.path.join(self.temp_dir, "time_history.npy")
        self.eye_history_file = os.path.join(self.temp_dir, "eye_history.pkl")
        
        # In-memory buffers (write to disk periodically)
        self.gaze_buffer = []
        self.time_buffer = []
        self.eye_buffer = []
        self.buffer_size = 1000  # Write to disk every N frames
        
        self.start_time = time.time()
        self.frame_index_history = [] # Map time to frame index (keep in memory, small)
        
        # Check if animation mode is enabled
        self.enable_animation = self.animation_var.get()
        
        if self.enable_animation:
            # Create real-time visualization windows
            self.fig = plt.figure(figsize=(5, 4), dpi=100)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_title("Gaze Angles over Time (Real-time)")
            self.ax.set_ylim(-25, 45)
            self.ax.set_xlabel("Video Time (s)")
            self.ax.set_ylabel("Angle (degrees)")
            
            self.lines = []
            colors = ['r', 'b']
            labels = ['Pitch (Vertical)', 'Yaw (Horizontal)']
            for i in range(2):
                line, = self.ax.plot([], [], color=colors[i], label=labels[i])
                self.lines.append(line)
            self.ax.legend(loc='upper right')
            self.ax.grid(True, alpha=0.3)
            
            # Create Popup Windows
            self.win_eye = ctk.CTkToplevel(self)
            self.win_eye.title("Extracted Eye Input (Real-time)")
            self.win_eye.geometry("300x200")
            self.lbl_eye_img = ctk.CTkLabel(self.win_eye, text="")
            self.lbl_eye_img.pack(expand=True, fill="both")
            
            self.win_gaze = ctk.CTkToplevel(self)
            self.win_gaze.title("Gaze Prediction (Real-time)")
            self.win_gaze.geometry("600x550")
            
            # Embed Matplotlib Figure
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.win_gaze)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
            
            # Add Toolbar
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.win_gaze)
            self.toolbar.update()
            self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
            
            self.lbl_gaze_val = ctk.CTkLabel(self.win_gaze, text="Gaze: [0.00, 0.00, 0.00]", font=("Courier", 16))
            self.lbl_gaze_val.pack(pady=5)
        else:
            # Create progress window
            self.win_progress = ctk.CTkToplevel(self)
            self.win_progress.title("Processing Video...")
            self.win_progress.geometry("500x200")
            
            self.lbl_progress_title = ctk.CTkLabel(
                self.win_progress, 
                text="Processing Video", 
                font=("Arial", 20, "bold")
            )
            self.lbl_progress_title.pack(pady=20)
            
            self.progress_bar = ctk.CTkProgressBar(self.win_progress, width=400)
            self.progress_bar.pack(pady=10)
            self.progress_bar.set(0)
            
            self.lbl_progress_status = ctk.CTkLabel(
                self.win_progress, 
                text="Initializing...", 
                font=("Arial", 14)
            )
            self.lbl_progress_status.pack(pady=10)
            
            self.lbl_progress_detail = ctk.CTkLabel(
                self.win_progress, 
                text="", 
                font=("Courier", 12)
            )
            self.lbl_progress_detail.pack(pady=5)
        
        # Start thread
        self.frame_idx = 0
        threading.Thread(target=self.process_video_loop, daemon=True).start()

    def stop_processing(self):
        self.is_processing = False
    
    def update_progress(self, progress_value, status_text):
        """Update progress bar and status (for non-animation mode)."""
        if hasattr(self, 'progress_bar'):
            self.progress_bar.set(progress_value)
        if hasattr(self, 'lbl_progress_status'):
            self.lbl_progress_status.configure(text=status_text)
        if hasattr(self, 'lbl_progress_detail'):
            percent = int(progress_value * 100)
            self.lbl_progress_detail.configure(text=f"Progress: {percent}%")

    def process_video_loop(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError("Could not open video")
                
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0: self.fps = 30.0
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Initialize normalizer with extended blink window (0.3 seconds = ±300ms around blink)
            # This covers both full blinks and partial blinks (微闭) that affect gaze accuracy
            self.normalizer = MediaPipeEyeNormalizer(eye='left', blink_window_extend_sec=0.3)
            
            # Clear cache for new video
            self.frame_cache = []
            
            # ===== PHASE 1: Quick scan to detect all blink timestamps =====
            print("[Phase 1] Scanning video for blink events...")
            if not self.enable_animation:
                self.after(0, lambda: self.lbl_progress_status.configure(text="Phase 1: Detecting blinks..."))
            
            blink_timestamps = []
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                timestamp = frame_count / self.fps
                frame_count += 1
                
                # Update progress
                if not self.enable_animation and frame_count % 30 == 0:
                    progress = frame_count / total_frames * 0.3  # Phase 1 = 30% of total
                    self.after(0, lambda p=progress, fc=frame_count, tf=total_frames: 
                              self.update_progress(p, f"Phase 1: Scanning blinks... {fc}/{tf} frames"))
                
                # Quick blink detection (no ROI extraction)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.normalizer.face_mesh.process(frame_rgb)
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    if self.normalizer._is_blinking(landmarks):
                        # Record blink timestamp (avoid duplicates within 100ms)
                        if not blink_timestamps or (timestamp - blink_timestamps[-1]) > 0.1:
                            blink_timestamps.append(timestamp)
            
            print(f"[Phase 1] Found {len(blink_timestamps)} blink events")
            if not self.enable_animation:
                self.after(0, lambda: self.lbl_progress_status.configure(
                    text=f"Phase 1 Complete: Found {len(blink_timestamps)} blinks"))
            
            # Pre-populate the normalizer's blink timestamps for symmetric window
            self.normalizer.blink_timestamps = blink_timestamps
            
            # ===== PHASE 2: Full processing with blink windows =====
            print("[Phase 2] Processing video with blink windows...")
            if not self.enable_animation:
                self.after(0, lambda: self.lbl_progress_status.configure(text="Phase 2: Processing gaze..."))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            frame_count = 0
            
            while self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate timestamp based on FPS
                timestamp = frame_count / self.fps
                frame_count += 1
                
                # Extract Eye (with pre-detected blink timestamps for symmetric window)
                roi_tensor, roi_display, is_blinking = self.normalizer.extract(frame, timestamp)
                
                # Cache frame and eye image for later scrubbing
                self.frame_cache.append((frame.copy(), roi_display.copy() if roi_display is not None else None))
                
                # Blink handling
                if is_blinking:
                    # If blinking, we skip inference to avoid artifacts
                    # We append NaN to history to indicate missing data
                    # For visualization, we can just skip this frame update or show last valid
                    self.gaze_buffer.append(np.array([np.nan, np.nan, np.nan]))
                    self.time_buffer.append(timestamp)
                    
                    # Store None or placeholder for blink eye image (or the closed eye itself)
                    # Better to store the closed eye for review!
                    if roi_display is not None:
                        self.eye_buffer.append(roi_display)
                    else:
                        # Fallback if detection failed completely
                        self.eye_buffer.append(np.zeros((36, 60, 3), dtype=np.uint8))
                    self.frame_index_history.append(frame_count - 1)
                    
                    # Flush to disk periodically
                    if len(self.gaze_buffer) >= self.buffer_size:
                        self.flush_buffers_to_disk()
                    
                    # Optional: Update visualization with "Blinking" status but no new point
                    # For speed, we might just skip visual update during blink
                    pass
                
                elif roi_tensor is not None:
                    # Inference
                    roi_input = roi_tensor.unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        output = self.model(roi_input)
                        gaze_pred = output.cpu().numpy()[0]
                        
                    # Normalize gaze vector for display
                    gaze_pred = gaze_pred / (np.linalg.norm(gaze_pred) + 1e-8)
                    
                    # Record history (use buffer to reduce memory)
                    self.gaze_buffer.append(gaze_pred)
                    self.time_buffer.append(timestamp)
                    self.eye_buffer.append(roi_display)
                    self.frame_index_history.append(frame_count - 1) # 0-based index
                    
                    # Flush to disk periodically
                    if len(self.gaze_buffer) >= self.buffer_size:
                        self.flush_buffers_to_disk()
                    
                    # Update GUI based on mode
                    if self.enable_animation:
                        # Update visualization less frequently to speed up processing (e.g., every 3rd frame)
                        if frame_count % 3 == 0:
                            self.after(0, self.update_visualization, roi_display, gaze_pred, timestamp, False)
                    else:
                        # Update progress bar
                        if frame_count % 30 == 0:
                            progress = 0.3 + (frame_count / total_frames) * 0.7  # Phase 2 = 70% of total
                            self.after(0, lambda p=progress, fc=frame_count, tf=total_frames: 
                                      self.update_progress(p, f"Phase 2: Processing... {fc}/{tf} frames"))
                
                # Control FPS roughly
                # time.sleep(1.0 / self.fps) # Disabled to process as fast as possible
                pass
                
            cap.release()
            if self.normalizer:
                self.normalizer.close()
            
            # Post-processing
            # Flush remaining buffer to disk
            self.flush_buffers_to_disk()
            
            # Load all data from disk
            gaze_history, time_history, eye_history = self.load_history_from_disk()
            
            if len(gaze_history) > 0:
                print("Applying post-processing filters...")
                if not self.enable_animation:
                    self.after(0, lambda: self.update_progress(0.95, "Applying filters..."))
                
                processor = SignalProcessor(fps=self.fps)
                raw_data = gaze_history
                smoothed_data = processor.process(raw_data)
                
                # Store for later use
                self.gaze_history = gaze_history
                self.time_history = time_history
                self.eye_history = eye_history
                
                if not self.enable_animation:
                    self.after(0, lambda: self.update_progress(1.0, "Complete! Opening results..."))
                    time.sleep(0.5)  # Brief pause to show completion
                
                # Update visualization with smoothed data
                self.after(0, self.show_smoothed_result, smoothed_data)
            else:
                self.after(0, self.reset_ui)
            
        except Exception as e:
            print(f"Processing error: {e}")
            import traceback
            traceback.print_exc()
            self.after(0, lambda: self.label_status.configure(text=f"Error: {str(e)}"))
            self.is_processing = False
            self.after(0, self.reset_ui)

    def update_visualization(self, eye_img, gaze_vector, timestamp, is_blinking=False):
        # Update Eye Image
        if eye_img is not None:
            # Resize for better visibility
            h, w = eye_img.shape[:2]
            scale = 4
            eye_img_big = cv2.resize(eye_img, (w*scale, h*scale), interpolation=cv2.INTER_NEAREST)
            img = Image.fromarray(eye_img_big)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(w*scale, h*scale))
            self.lbl_eye_img.configure(image=ctk_img)
            self.lbl_eye_img.image = ctk_img # keep ref
            
        # Update Gaze Plot
        # Note: History is already updated in the thread to ensure sync
        
        # Convert to numpy for plotting (handle recent history)
        # Load from disk + buffer for real-time display
        recent_len = 300 # Show last N frames in real-time
        
        # Combine disk data and buffer
        if os.path.exists(self.time_history_file):
            disk_times = np.load(self.time_history_file)
            disk_gaze = np.load(self.gaze_history_file)
        else:
            disk_times = np.array([])
            disk_gaze = np.array([]).reshape(0, 3)
        
        all_times = np.concatenate([disk_times, np.array(self.time_buffer)])
        all_gaze = np.vstack([disk_gaze, np.array(self.gaze_buffer)]) if len(self.gaze_buffer) > 0 else disk_gaze
        
        start_idx = max(0, len(all_times) - recent_len)
        times = all_times[start_idx:]
        history_vectors = all_gaze[start_idx:]
        
        if len(times) > 0:
            # Convert to angles
            angles = np.full((len(history_vectors), 2), np.nan)
            valid_mask = ~np.isnan(history_vectors).any(axis=1)
            
            if np.any(valid_mask):
                valid_vectors = history_vectors[valid_mask]
                angles[valid_mask] = self.vector_to_pitch_yaw(valid_vectors)
            
            # Update lines
            for i in range(2):
                self.lines[i].set_data(times, angles[:, i])
                
            # Update x-axis limit to fit all data
            current_time = times[-1]
            min_time = times[0]
            self.ax.set_xlim(min_time, max(min_time + 5, current_time))
            
            # Do NOT auto-scale Y axis in real-time for stability
            # Keep it fixed at [-45, 45] unless data goes way out
            # This makes it easier to see small movements without the axis jumping
            # self.ax.set_ylim(-45, 45)
            
            # Redraw plot
            self.canvas.draw()
        
        # Update gaze value text
        if not is_blinking and not np.isnan(gaze_vector).any():
            angles = self.vector_to_pitch_yaw(gaze_vector)
            self.lbl_gaze_val.configure(text=f"Pitch: {angles[0]:.1f}°, Yaw: {angles[1]:.1f}°")
        else:
             self.lbl_gaze_val.configure(text="Gaze: [Blinking]")

    def show_smoothed_result(self, smoothed_data):
        self.is_processing = False
        self.btn_start.configure(state="normal")
        self.btn_select.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.label_status.configure(text="Review Mode: Drag on plot to see video frame.")
        
        # Close progress window if exists
        if hasattr(self, 'win_progress') and self.win_progress is not None and self.win_progress.winfo_exists():
            self.win_progress.destroy()
        
        # Create or update eye window
        if not hasattr(self, 'win_eye') or self.win_eye is None or not self.win_eye.winfo_exists():
            self.win_eye = ctk.CTkToplevel(self)
            self.win_eye.title("Review: Extracted Eye")
            self.win_eye.geometry("300x200")
            self.lbl_eye_img = ctk.CTkLabel(self.win_eye, text="")
            self.lbl_eye_img.pack(expand=True, fill="both")
        else:
            self.win_eye.title("Review: Extracted Eye")
            
        # Create Video Review Window
        if not hasattr(self, 'win_video') or self.win_video is None or not self.win_video.winfo_exists():
            self.win_video = ctk.CTkToplevel(self)
            self.win_video.title("Review: Original Video")
            self.win_video.geometry("640x480")
            self.lbl_video_img = ctk.CTkLabel(self.win_video, text="Click on plot to load frame")
            self.lbl_video_img.pack(expand=True, fill="both")
            
        # Re-open video for random access
        if hasattr(self, 'review_cap') and self.review_cap.isOpened():
            self.review_cap.release()
        self.review_cap = cv2.VideoCapture(self.video_path)
        
        # Create gaze plot window if not in animation mode
        if not self.enable_animation:
            self.win_gaze = ctk.CTkToplevel(self)
            self.win_gaze.title("Gaze Prediction Results")
            self.win_gaze.geometry("600x600")
            
            self.fig = plt.figure(figsize=(5, 4), dpi=100)
            self.ax = self.fig.add_subplot(111)
            
            # Embed Matplotlib Figure
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.win_gaze)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
            
            # Add Toolbar
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.win_gaze)
            self.toolbar.update()
            self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
            
            self.lbl_gaze_val = ctk.CTkLabel(self.win_gaze, text="Gaze: [0.00, 0.00, 0.00]", font=("Courier", 16))
            self.lbl_gaze_val.pack(pady=5)
            
        # Update plot with smoothed data comparison
        if hasattr(self, 'fig'):
            self.ax.clear()
            self.ax.set_title("Gaze Angles: Raw (with Blink Fill) vs Smoothed")
            self.ax.set_xlabel("Video Time (s)")
            self.ax.set_ylabel("Angle (degrees)")
            
            times = np.array(self.time_history)
            raw_vectors = np.array(self.gaze_history)
            
            # Convert smoothed data to angles
            smoothed_angles = self.vector_to_pitch_yaw(smoothed_data)
            
            # Convert raw data to angles (handle NaNs)
            raw_angles = np.full((len(raw_vectors), 2), np.nan)
            valid_mask = ~np.isnan(raw_vectors).any(axis=1)
            if np.any(valid_mask):
                raw_angles[valid_mask] = self.vector_to_pitch_yaw(raw_vectors[valid_mask])
            
            colors = ['r', 'b']
            labels = ['Pitch', 'Yaw']
            
            # Plot smoothed data with separate solid/dashed segments
            for i in range(2):
                # Plot Raw (Transparent background)
                self.ax.plot(times, raw_angles[:, i], color=colors[i], alpha=0.2, linewidth=1, label=f'{labels[i]} raw')
                
                # Identify Blink Segments (where raw data was NaN)
                blink_mask = np.isnan(raw_angles[:, i])
                
                # Split smoothed data into non-blink (solid) and blink (dashed) segments
                # Non-blink segments: solid line
                non_blink_data = smoothed_angles[:, i].copy()
                non_blink_data[blink_mask] = np.nan
                self.ax.plot(times, non_blink_data, color=colors[i], linewidth=2, linestyle='-', label=f'{labels[i]} smooth')
                
                # Blink segments (interpolated): dashed line
                if np.any(blink_mask):
                    blink_data = smoothed_angles[:, i].copy()
                    blink_data[~blink_mask] = np.nan
                    self.ax.plot(times, blink_data, color=colors[i], linewidth=2, linestyle='--', alpha=0.8, label=f'{labels[i]} interp')
                
            self.ax.legend(loc='upper right', fontsize='small')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_xlim(0, max(10, times[-1]))
            
            # Keep fixed Y-axis for consistent comparison
            self.ax.set_ylim(-25, 45)
            
            # Add vertical cursor line
            self.cursor_line = self.ax.axvline(x=0, color='k', linestyle=':', linewidth=1, alpha=0.8)
            
            self.canvas.draw()
            
            # Connect events
            self.canvas.mpl_connect('button_press_event', self.on_plot_interaction)
            self.canvas.mpl_connect('motion_notify_event', self.on_plot_interaction)
            
            # Create 3D Gaze Vector Visualization Window (if enabled)
            if self.show_3d_var.get():
                self.create_gaze_vector_window(smoothed_data)
            
            # Button frame for actions
            self.frame_buttons = ctk.CTkFrame(self.win_gaze)
            self.frame_buttons.pack(pady=10, fill="x", padx=20)
            
            # Add Save button to export plist data
            self.btn_save_plist = ctk.CTkButton(
                self.frame_buttons, 
                text="[Save] Export Plist", 
                command=lambda: self.save_to_plist(smoothed_data),
                fg_color="green",
                hover_color="darkgreen",
                width=150
            )
            self.btn_save_plist.pack(side="left", padx=5)
            
            # Add Nystagmus Analysis button
            self.btn_nystagmus = ctk.CTkButton(
                self.frame_buttons, 
                text="[Analyze] Nystagmus", 
                command=lambda: self.analyze_nystagmus(smoothed_data),
                fg_color="#8B4513",
                hover_color="#654321",
                width=180
            )
            self.btn_nystagmus.pack(side="left", padx=5)
            
            # Store blink mask for nystagmus analysis
            self.blink_mask = np.isnan(raw_vectors).any(axis=1)
            self.smoothed_angles_cache = smoothed_angles

    def analyze_nystagmus(self, smoothed_data):
        """Perform nystagmus analysis on the gaze data."""
        try:
            times = np.array(self.time_history)
            
            # Create analyzer
            analyzer = NystagmusAnalyzer(fps=self.fps)
            
            # Analyze horizontal (Yaw) - more common for nystagmus
            result_h = analyzer.analyze(
                times, 
                self.smoothed_angles_cache[:, 1],  # Yaw
                self.blink_mask,
                axis="horizontal"
            )
            
            # Analyze vertical (Pitch)
            result_v = analyzer.analyze(
                times, 
                self.smoothed_angles_cache[:, 0],  # Pitch
                self.blink_mask,
                axis="vertical"
            )
            
            # Show results window
            self.show_nystagmus_results(result_h, result_v)
            
        except Exception as e:
            print(f"[Nystagmus Analysis Error] {e}")
            import traceback
            traceback.print_exc()
            if hasattr(self, 'label_status'):
                self.label_status.configure(text=f"❌ Nystagmus analysis error: {str(e)}")
    
    def show_nystagmus_results(self, result_h, result_v):
        """Display nystagmus analysis results in a new window."""
        # Create results window (larger to fit 4 subplots)
        self.win_nystagmus = ctk.CTkToplevel(self)
        self.win_nystagmus.title("Nystagmus Analysis Results")
        self.win_nystagmus.geometry("1100x900")
        
        # Title
        title_label = ctk.CTkLabel(
            self.win_nystagmus, 
            text="Nystagmus Analysis Results", 
            font=("Arial", 20, "bold")
        )
        title_label.pack(pady=10)
        
        # Create tabview for H and V results (larger for 4 subplots)
        tabview = ctk.CTkTabview(self.win_nystagmus, width=1050, height=800)
        tabview.pack(pady=10, padx=20, fill="both", expand=True)
        
        tabview.add("Horizontal (Yaw)")
        tabview.add("Vertical (Pitch)")
        tabview.add("Summary")
        
        # Horizontal results
        self._create_nystagmus_result_tab(tabview.tab("Horizontal (Yaw)"), result_h, "Horizontal")
        
        # Vertical results
        self._create_nystagmus_result_tab(tabview.tab("Vertical (Pitch)"), result_v, "Vertical")
        
        # Summary tab
        self._create_nystagmus_summary_tab(tabview.tab("Summary"), result_h, result_v)
        
        # Info label
        info_label = ctk.CTkLabel(
            self.win_nystagmus, 
            text="Note: Blink segments are automatically excluded from analysis",
            font=("Arial", 11),
            text_color="gray"
        )
        info_label.pack(pady=5)
    
    def _create_nystagmus_result_tab(self, parent, result, axis_name):
        """Create a tab showing nystagmus analysis results for one axis with 4 subplots."""
        if not result['success']:
            error_label = ctk.CTkLabel(
                parent, 
                text=f"❌ Analysis Failed: {result.get('error', 'Unknown error')}",
                font=("Arial", 14),
                text_color="red"
            )
            error_label.pack(pady=20)
            return
        
        # Create scrollable frame for all content
        frame = ctk.CTkScrollableFrame(parent, width=750, height=500)
        frame.pack(pady=5, padx=5, fill="both", expand=True)
        
        # Key metrics row
        metrics_frame = ctk.CTkFrame(frame)
        metrics_frame.pack(pady=5, padx=5, fill="x")
        
        # SPV (Slow Phase Velocity)
        spv_frame = ctk.CTkFrame(metrics_frame)
        spv_frame.pack(side="left", padx=15, pady=5)
        ctk.CTkLabel(spv_frame, text="SPV", font=("Arial", 11)).pack()
        ctk.CTkLabel(
            spv_frame, 
            text=f"{result['spv']:.2f}°/s", 
            font=("Arial", 20, "bold"),
            text_color="cyan"
        ).pack()
        
        # Direction
        dir_frame = ctk.CTkFrame(metrics_frame)
        dir_frame.pack(side="left", padx=15, pady=5)
        ctk.CTkLabel(dir_frame, text="Direction", font=("Arial", 11)).pack()
        dir_symbols = {"left": "<< Left", "right": "Right >>", "up": "Up ^", "down": "Down v", "unknown": "?"}
        dir_colors = {"left": "yellow", "right": "yellow", "up": "orange", "down": "orange", "unknown": "gray"}
        ctk.CTkLabel(
            dir_frame, 
            text=dir_symbols.get(result['direction'], "?"), 
            font=("Arial", 20, "bold"),
            text_color=dir_colors.get(result['direction'], "gray")
        ).pack()
        
        # CV
        cv_frame = ctk.CTkFrame(metrics_frame)
        cv_frame.pack(side="left", padx=15, pady=5)
        ctk.CTkLabel(cv_frame, text="CV", font=("Arial", 11)).pack()
        cv_color = "green" if result['cv'] <= 20 else "yellow" if result['cv'] <= 40 else "red"
        ctk.CTkLabel(
            cv_frame, 
            text=f"{result['cv']:.1f}%", 
            font=("Arial", 20, "bold"),
            text_color=cv_color
        ).pack()
        
        # Pattern count
        count_frame = ctk.CTkFrame(metrics_frame)
        count_frame.pack(side="left", padx=15, pady=5)
        ctk.CTkLabel(count_frame, text="Patterns", font=("Arial", 11)).pack()
        ctk.CTkLabel(
            count_frame, 
            text=f"{result['n_patterns']} ({result['n_filtered_patterns']} filtered)", 
            font=("Arial", 16, "bold"),
            text_color="white"
        ).pack()
        
        # ========== 4-SUBPLOT FIGURE (similar to Streamlit) ==========
        if len(result['filtered_signal']) > 0:
            plot_frame = ctk.CTkFrame(frame)
            plot_frame.pack(pady=5, padx=5, fill="both", expand=True)
            
            fig_nys = plt.Figure(figsize=(10, 12), dpi=100)
            
            # ---------- Subplot 1: Signal Preprocessing Steps ----------
            ax1 = fig_nys.add_subplot(4, 1, 1)
            
            # Plot original data
            if 'original_times' in result and 'original_angles' in result:
                ax1.plot(result['original_times'], result['original_angles'], 
                        label='Original Data', alpha=0.7, color='blue')
                
                # Plot high-pass filtered
                if 'signal_highpass' in result:
                    ax1.plot(result['original_times'], result['signal_highpass'], 
                            label='After High-pass Filter', alpha=0.7, color='orange')
                
                # Plot low-pass filtered
                if 'signal_lowpass' in result:
                    ax1.plot(result['original_times'], result['signal_lowpass'], 
                            label='After Low-pass Filter', alpha=0.9, linewidth=2, color='green')
            
            ax1.set_title(f'1. Signal Preprocessing Steps ({axis_name})', fontsize=10, fontweight='bold')
            ax1.set_ylabel('Position (°)')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper right', fontsize=8)
            
            # ---------- Subplot 2: Turning Point Detection ----------
            ax2 = fig_nys.add_subplot(4, 1, 2)
            
            time = result['time']
            signal = result['filtered_signal']
            tp = result['turning_points']
            
            # Original signal (gray background)
            ax2.plot(time, signal, 'gray', alpha=0.3, label='Filtered Signal')
            
            # Turning points connection
            if len(tp) > 0:
                ax2.plot(time[tp], signal[tp], 'r-', linewidth=2, label='Turning Points Connection')
                ax2.plot(time[tp], signal[tp], 'ro', markersize=5, label='Turning Points')
            
            ax2.set_title(f'2. Eye Movement Signal with Turning Points ({axis_name})', fontsize=10, fontweight='bold')
            ax2.set_ylabel('Position (°)')
            ax2.legend(loc='upper right', fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # ---------- Subplot 3: Slope Calculation ----------
            ax3 = fig_nys.add_subplot(4, 1, 3)
            
            if len(result['slope_times']) > 0:
                ax3.scatter(result['slope_times'], result['slopes'], c='blue', s=30, alpha=0.7)
            ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            
            ax3.set_title(f'3. Calculated Slopes ({axis_name})', fontsize=10, fontweight='bold')
            ax3.set_ylabel('Slope (°/s)')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim([-60, 60])
            
            # ---------- Subplot 4: Nystagmus Pattern Recognition ----------
            ax4 = fig_nys.add_subplot(4, 1, 4)
            
            # Background signal
            ax4.plot(time, signal, 'gray', alpha=0.5, label='Signal')
            
            # Plot filtered patterns (light colors - outliers)
            for pattern_item in result['filtered_patterns']:
                idx = pattern_item['index']
                if idx > 0 and idx + 1 < len(tp):
                    idx1 = tp[idx-1]
                    idx2 = tp[idx]
                    idx3 = tp[idx+1]
                    
                    if pattern_item['fast_phase_first']:
                        fast_segment = time[idx1:idx2+1]
                        slow_segment = time[idx2:idx3+1]
                        fast_signal = signal[idx1:idx2+1]
                        slow_signal = signal[idx2:idx3+1]
                    else:
                        slow_segment = time[idx1:idx2+1]
                        fast_segment = time[idx2:idx3+1]
                        slow_signal = signal[idx1:idx2+1]
                        fast_signal = signal[idx2:idx3+1]
                    
                    ax4.plot(fast_segment, fast_signal, 'lightcoral', linewidth=2, alpha=0.5)
                    ax4.plot(slow_segment, slow_signal, 'lightblue', linewidth=2, alpha=0.5)
            
            # Plot final patterns (bright colors - valid)
            first_fast = True
            first_slow = True
            for pattern_item in result['patterns']:
                idx = pattern_item['index']
                if idx > 0 and idx + 1 < len(tp):
                    idx1 = tp[idx-1]
                    idx2 = tp[idx]
                    idx3 = tp[idx+1]
                    
                    if pattern_item['fast_phase_first']:
                        fast_segment = time[idx1:idx2+1]
                        slow_segment = time[idx2:idx3+1]
                        fast_signal = signal[idx1:idx2+1]
                        slow_signal = signal[idx2:idx3+1]
                    else:
                        slow_segment = time[idx1:idx2+1]
                        fast_segment = time[idx2:idx3+1]
                        slow_signal = signal[idx1:idx2+1]
                        fast_signal = signal[idx2:idx3+1]
                    
                    # Add labels only for the first occurrence
                    fast_label = 'Fast Phase (Red)' if first_fast else None
                    slow_label = 'Slow Phase (Blue)' if first_slow else None
                    
                    ax4.plot(fast_segment, fast_signal, 'red', linewidth=2, alpha=0.8, label=fast_label)
                    ax4.plot(slow_segment, slow_signal, 'blue', linewidth=2, alpha=0.8, label=slow_label)
                    
                    first_fast = False
                    first_slow = False
            
            # Direction label
            english_dir = result['direction'].capitalize()
            if result['axis'] == "horizontal":
                english_dir = "Left" if result['direction'] == "left" else "Right"
            else:
                english_dir = "Down" if result['direction'] in ("up", "left") else "Up"
            
            ax4.set_title(
                f'4. Pattern-Based Nystagmus Analysis ({axis_name} - Direction: {english_dir}, '
                f'SPV: {result["spv"]:.1f}°/s, CV: {result["cv"]:.1f}%)', 
                fontsize=10, fontweight='bold'
            )
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Position (°)')
            ax4.legend(loc='upper right', fontsize=8)
            ax4.grid(True, alpha=0.3)
            
            fig_nys.tight_layout()
            
            canvas_nys = FigureCanvasTkAgg(fig_nys, master=plot_frame)
            canvas_nys.draw()
            canvas_nys.get_tk_widget().pack(fill="both", expand=True)
        
        # Statistics summary at bottom
        stats_frame = ctk.CTkFrame(frame)
        stats_frame.pack(pady=5, padx=5, fill="x")
        
        stats_text = (
            f"Statistics: Valid samples: {result['n_valid_samples']} | "
            f"Blink excluded: {result['n_blink_samples']} | "
            f"Turning points: {len(result['turning_points'])} | "
            f"Valid patterns: {result['n_patterns']} | "
            f"Filtered (outliers): {result['n_filtered_patterns']}"
        )
        
        ctk.CTkLabel(
            stats_frame, 
            text=stats_text, 
            font=("Arial", 11),
            text_color="gray"
        ).pack(pady=5, padx=10)
    
    def _create_nystagmus_summary_tab(self, parent, result_h, result_v):
        """Create summary tab comparing H and V nystagmus."""
        frame = ctk.CTkFrame(parent)
        frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        # Title
        ctk.CTkLabel(
            frame, 
            text="Analysis Summary", 
            font=("Arial", 18, "bold")
        ).pack(pady=10)
        
        # Comparison table
        table_frame = ctk.CTkFrame(frame)
        table_frame.pack(pady=20, padx=20, fill="x")
        
        # Header
        headers = ["Metric", "Horizontal (Yaw)", "Vertical (Pitch)"]
        for i, h in enumerate(headers):
            ctk.CTkLabel(
                table_frame, 
                text=h, 
                font=("Arial", 12, "bold"),
                width=180
            ).grid(row=0, column=i, padx=5, pady=5)
        
        # Data rows
        def get_value(result, key, fmt="{:.2f}"):
            if not result['success']:
                return "N/A"
            val = result.get(key, "N/A")
            if isinstance(val, (int, float)):
                return fmt.format(val)
            return str(val)
        
        rows = [
            ("SPV (°/s)", get_value(result_h, 'spv'), get_value(result_v, 'spv')),
            ("Direction", result_h.get('direction', 'N/A') if result_h['success'] else 'N/A',
                         result_v.get('direction', 'N/A') if result_v['success'] else 'N/A'),
            ("CV (%)", get_value(result_h, 'cv', "{:.1f}"), get_value(result_v, 'cv', "{:.1f}")),
            ("Patterns", get_value(result_h, 'n_patterns', "{}"), get_value(result_v, 'n_patterns', "{}")),
            ("Filtered", get_value(result_h, 'n_filtered_patterns', "{}"), get_value(result_v, 'n_filtered_patterns', "{}")),
        ]
        
        for row_idx, (metric, val_h, val_v) in enumerate(rows, start=1):
            ctk.CTkLabel(table_frame, text=metric, font=("Arial", 12)).grid(row=row_idx, column=0, padx=5, pady=3)
            ctk.CTkLabel(table_frame, text=val_h, font=("Courier", 12)).grid(row=row_idx, column=1, padx=5, pady=3)
            ctk.CTkLabel(table_frame, text=val_v, font=("Courier", 12)).grid(row=row_idx, column=2, padx=5, pady=3)
        
        # Interpretation
        interp_frame = ctk.CTkFrame(frame)
        interp_frame.pack(pady=20, padx=20, fill="x")
        
        ctk.CTkLabel(
            interp_frame, 
            text="📝 Interpretation", 
            font=("Arial", 14, "bold")
        ).pack(pady=5)
        
        # Generate interpretation
        interpretation = []
        
        if result_h['success'] and result_h['n_patterns'] > 0:
            if result_h['spv'] >= 3.0:
                interpretation.append(f"• Significant horizontal nystagmus detected ({result_h['direction']})")
                interpretation.append(f"  SPV = {result_h['spv']:.2f}°/s, {result_h['n_patterns']} valid patterns")
            else:
                interpretation.append(f"• Mild horizontal nystagmus ({result_h['direction']}, SPV = {result_h['spv']:.2f}°/s)")
        else:
            interpretation.append("• No significant horizontal nystagmus detected")
        
        if result_v['success'] and result_v['n_patterns'] > 0:
            if result_v['spv'] >= 3.0:
                interpretation.append(f"• Significant vertical nystagmus detected ({result_v['direction']})")
                interpretation.append(f"  SPV = {result_v['spv']:.2f}°/s, {result_v['n_patterns']} valid patterns")
            else:
                interpretation.append(f"• Mild vertical nystagmus ({result_v['direction']}, SPV = {result_v['spv']:.2f}°/s)")
        else:
            interpretation.append("• No significant vertical nystagmus detected")
        
        # CV quality assessment
        for axis, result in [("Horizontal", result_h), ("Vertical", result_v)]:
            if result['success'] and result['cv'] < float('inf'):
                if result['cv'] <= 20:
                    interpretation.append(f"• {axis} CV = {result['cv']:.1f}% (Good consistency)")
                elif result['cv'] <= 40:
                    interpretation.append(f"• {axis} CV = {result['cv']:.1f}% (Moderate variability)")
                else:
                    interpretation.append(f"• {axis} CV = {result['cv']:.1f}% (High variability - check data quality)")
        
        interp_text = "\n".join(interpretation)
        ctk.CTkLabel(
            interp_frame, 
            text=interp_text, 
            font=("Arial", 12),
            justify="left"
        ).pack(pady=10, padx=10, anchor="w")
        
        # Note about blink exclusion
        ctk.CTkLabel(
            frame, 
            text="ℹ️ Note: Blink segments were automatically excluded from analysis\nto ensure accurate nystagmus detection.",
            font=("Arial", 11),
            text_color="gray",
            justify="center"
        ).pack(pady=10)

    def create_gaze_vector_window(self, smoothed_data):
        """Create a window to visualize 3D gaze vector as arrow."""
        self.win_gaze_vector = ctk.CTkToplevel(self)
        self.win_gaze_vector.title("3D Gaze Vector Visualization")
        self.win_gaze_vector.geometry("500x550")
        
        # Create matplotlib figure for 3D visualization
        self.fig_3d = plt.Figure(figsize=(5, 5), dpi=100)
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        
        # Store smoothed data for vector visualization
        self.smoothed_gaze_vectors = smoothed_data
        
        # Initial visualization (first frame)
        self.update_gaze_vector_display(0)
        
        # Embed figure
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=self.win_gaze_vector)
        self.canvas_3d.draw()
        self.canvas_3d.get_tk_widget().pack(side="top", fill="both", expand=True)
        
        # Add info label
        self.lbl_vector_info = ctk.CTkLabel(
            self.win_gaze_vector, 
            text="Gaze Vector: [0.00, 0.00, 0.00]", 
            font=("Courier", 14)
        )
        self.lbl_vector_info.pack(pady=5)
    
    def update_gaze_vector_display(self, frame_idx):
        """Update 3D gaze vector visualization for a specific frame."""
        if not hasattr(self, 'ax_3d') or frame_idx >= len(self.smoothed_gaze_vectors):
            return
        
        # Get gaze vector for this frame
        gaze_vec = self.smoothed_gaze_vectors[frame_idx]
        x, y, z = gaze_vec
        
        # Convert to Pitch and Yaw for visualization
        pitch, yaw = self.vector_to_pitch_yaw(gaze_vec)
        
        # Clear and redraw
        self.ax_3d.clear()
        
        # Set up 3D space with person and camera
        # X: left-right (Yaw direction)
        # Y: forward-backward (depth from person to camera)
        # Z: up-down (Pitch direction)
        self.ax_3d.set_xlim([-0.5, 0.5])
        self.ax_3d.set_ylim([0, 1.2])
        self.ax_3d.set_zlim([-0.3, 0.5])
        self.ax_3d.set_xlabel('X (Left <-> Right)', fontsize=10)
        self.ax_3d.set_ylabel('Y (Depth: Person -> Camera)', fontsize=10, fontweight='bold')
        self.ax_3d.set_zlabel('Z (Down <-> Up)', fontsize=10)
        self.ax_3d.set_title('Gaze Direction: Eye -> Camera', fontsize=12, fontweight='bold')
        
        # Person position (eye center at origin)
        person_x, person_y, person_z = 0, 0, 0
        
        # Draw person (head as sphere + body as cylinder)
        # Head
        self.ax_3d.scatter([person_x], [person_y], [person_z], 
                          color='yellow', s=400, marker='o', 
                          edgecolors='orange', linewidths=3, alpha=0.95, label='Person (Eye)')
        # Simple body indicator
        self.ax_3d.plot([person_x, person_x], [person_y, person_y], [person_z, person_z-0.2], 
                       'orange', linewidth=6, alpha=0.7)
        
        # Camera position (at y=1.0, typical distance)
        camera_x, camera_y, camera_z = 0, 1.0, 0
        
        # Draw camera (box + lens)
        camera_size = 0.08
        # Camera body (box)
        camera_corners_x = [camera_x-camera_size, camera_x+camera_size, camera_x+camera_size, camera_x-camera_size, camera_x-camera_size]
        camera_corners_z = [camera_z-camera_size, camera_z-camera_size, camera_z+camera_size, camera_z+camera_size, camera_z-camera_size]
        camera_corners_y = [camera_y] * 5
        self.ax_3d.plot(camera_corners_x, camera_corners_y, camera_corners_z, 
                       'k-', linewidth=3, alpha=0.8)
        # Camera lens (circle)
        theta = np.linspace(0, 2*np.pi, 20)
        lens_r = camera_size * 0.5
        lens_x = camera_x + lens_r * np.cos(theta)
        lens_z = camera_z + lens_r * np.sin(theta)
        lens_y = [camera_y] * len(theta)
        self.ax_3d.plot(lens_x, lens_y, lens_z, 'gray', linewidth=4, alpha=0.9)
        self.ax_3d.scatter([camera_x], [camera_y], [camera_z], 
                          color='darkblue', s=300, marker='s', 
                          edgecolors='black', linewidths=3, alpha=0.9, label='Camera')
        
        # Draw ground plane for reference
        xx, yy = np.meshgrid(np.linspace(-0.4, 0.4, 2), np.linspace(0, 1.1, 2))
        zz = np.full_like(xx, -0.25)
        self.ax_3d.plot_surface(xx, yy, zz, alpha=0.15, color='gray')
        
        # Draw reference line (person to camera center)
        self.ax_3d.plot([person_x, camera_x], [person_y, camera_y], [person_z, camera_z], 
                       'g--', linewidth=1.5, alpha=0.4, label='Center line (0°, 0°)')
        
        # Calculate gaze target point in 3D space
        # Convert gaze vector (x, y, z) to 3D position
        # Scale the unit vector to reach approximately camera distance
        gaze_distance = 1.0  # Same as camera distance
        gaze_target_x = person_x + x * gaze_distance
        gaze_target_y = person_y + (-z) * gaze_distance  # -z because z points away from camera
        gaze_target_z = person_z + (-y) * gaze_distance  # -y because y is down in gaze coords
        
        # Draw gaze vector as thick arrow from eye to target
        self.ax_3d.quiver(
            person_x, person_y, person_z,  # Start at eye
            x * gaze_distance, (-z) * gaze_distance, (-y) * gaze_distance,  # Gaze direction
            color='cyan',
            alpha=0.95,
            arrow_length_ratio=0.1,
            linewidth=5,
            label='Gaze Vector'
        )
        
        # Mark the gaze target point
        self.ax_3d.scatter([gaze_target_x], [gaze_target_y], [gaze_target_z], 
                          color='red', s=250, marker='*', 
                          edgecolors='white', linewidths=3, alpha=1.0,
                          label=f'Gaze Target')
        
        # Draw line from gaze target to camera (to show offset)
        self.ax_3d.plot([gaze_target_x, camera_x], [gaze_target_y, camera_y], [gaze_target_z, camera_z], 
                       'yellow', linestyle=':', linewidth=2, alpha=0.6)
        
        # Add text annotation
        annotation_text = f'Pitch: {pitch:.1f}° (Vertical)\n'
        annotation_text += f'Yaw: {yaw:.1f}° (Horizontal)\n'
        annotation_text += f'Vector: ({x:.3f}, {y:.3f}, {z:.3f})'
        
        # Determine gaze direction description
        if abs(pitch) < 5 and abs(yaw) < 5:
            direction = 'Looking straight'
        elif abs(yaw) > abs(pitch):
            direction = f'Looking {"right" if yaw > 0 else "left"}'
        else:
            direction = f'Looking {"up" if pitch > 0 else "down"}'
        annotation_text += f'\n{direction}'
        
        self.ax_3d.text2D(0.05, 0.95, annotation_text, transform=self.ax_3d.transAxes,
                          fontsize=9, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
        
        # Set viewing angle (side view from 90° to see person-camera relationship)
        # elev=10: slight elevation to see ground plane
        # azim=90: viewing from the side (perpendicular to person-camera line)
        self.ax_3d.view_init(elev=10, azim=90)
        
        self.ax_3d.legend(loc='lower left', fontsize=8, framealpha=0.9)
        
        # Update canvas
        if hasattr(self, 'canvas_3d'):
            self.canvas_3d.draw_idle()
        
        # Update info label
        if hasattr(self, 'lbl_vector_info'):
            if abs(pitch) < 5 and abs(yaw) < 5:
                direction = 'Looking straight'
            elif abs(yaw) > abs(pitch):
                direction = f'Looking {"right >>" if yaw > 0 else "<< left"}'
            else:
                direction = f'Looking {"up ^" if pitch > 0 else "down v"}'
            self.lbl_vector_info.configure(
                text=f"Pitch: {pitch:.1f}° | Yaw: {yaw:.1f}° | {direction}"
            )

    def save_to_plist(self, smoothed_data):
        """Save gaze data to plist format for further analysis."""
        try:
            # Ask user for save location (base filename)
            from tkinter import filedialog
            default_filename = f"gaze_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            save_path = filedialog.asksaveasfilename(
                defaultextension=".plist",
                filetypes=[("Plist files", "*.plist"), ("All files", "*.*")],
                initialfile=default_filename + "_filtered.plist"
            )
            
            if not save_path:
                return
            
            # Generate both filenames
            base_path = save_path.rsplit('.', 1)[0]
            if base_path.endswith('_filtered'):
                base_path = base_path[:-9]  # Remove '_filtered'
            
            filtered_path = base_path + "_filtered.plist"
            raw_path = base_path + "_raw.plist"
            
            # Prepare data in plist format (matching your analysis code structure)
            times = np.array(self.time_history)
            raw_vectors = np.array(self.gaze_history)
            
            # Convert to angles (both raw and smoothed)
            raw_angles = self.vector_to_pitch_yaw(raw_vectors)
            smoothed_angles = self.vector_to_pitch_yaw(smoothed_data)
            
            # Generate timestamp strings (similar to your HIT format)
            base_time = datetime.now()
            time_list = []
            for t in times:
                timestamp = base_time + timedelta(seconds=float(t))
                time_list.append(timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            
            # Common metadata
            metadata = {
                'VideoFile': os.path.basename(self.video_path) if self.video_path else 'unknown',
                'FPS': float(self.fps) if hasattr(self, 'fps') else 30.0,
                'TotalFrames': len(times),
                'Duration': float(times[-1]) if len(times) > 0 else 0.0,
                'ProcessingDate': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'BlinkWindowSec': 0.3,
                'CoordinateSystem': 'MPIIGaze',
                'DataFormat': 'Eye angles in degrees, compatible with LeftEyeXDegList/LeftEyeYDegList format',
                'PitchRange': '[-25, 45] degrees (vertical)',
                'YawRange': '[-45, 45] degrees (horizontal)',
            }
            
            # Create FILTERED plist data structure
            plist_filtered = {
                'TimeList': time_list,
                'LeftEyeXDegList': smoothed_angles[:, 1].tolist(),  # Yaw
                'RightEyeXDegList': smoothed_angles[:, 1].tolist(),
                'LeftEyeYDegList': smoothed_angles[:, 0].tolist(),  # Pitch
                'RightEyeYDegList': smoothed_angles[:, 0].tolist(),
                'GazePitchDegList': smoothed_angles[:, 0].tolist(),
                'GazeYawDegList': smoothed_angles[:, 1].tolist(),
                'GazeXList': smoothed_data[:, 0].tolist(),
                'GazeYList': smoothed_data[:, 1].tolist(),
                'GazeZList': smoothed_data[:, 2].tolist(),
                'FilteredData': True,
                'Description': 'Filtered gaze data from SwinUNet-VOG (median + low-pass filtered)',
                **metadata
            }
            
            # Create RAW plist data structure
            plist_raw = {
                'TimeList': time_list,
                'LeftEyeXDegList': raw_angles[:, 1].tolist(),  # Yaw
                'RightEyeXDegList': raw_angles[:, 1].tolist(),
                'LeftEyeYDegList': raw_angles[:, 0].tolist(),  # Pitch
                'RightEyeYDegList': raw_angles[:, 0].tolist(),
                'GazePitchDegList': raw_angles[:, 0].tolist(),
                'GazeYawDegList': raw_angles[:, 1].tolist(),
                'GazeXList': raw_vectors[:, 0].tolist(),
                'GazeYList': raw_vectors[:, 1].tolist(),
                'GazeZList': raw_vectors[:, 2].tolist(),
                'FilteredData': False,
                'Description': 'Raw gaze data from SwinUNet-VOG (no filtering, contains blink interpolation)',
                **metadata
            }
            
            # Save both files
            with open(filtered_path, 'wb') as f:
                plistlib.dump(plist_filtered, f)
            
            with open(raw_path, 'wb') as f:
                plistlib.dump(plist_raw, f)
            
            # Show success message
            if hasattr(self, 'label_status'):
                self.label_status.configure(
                    text=f"✅ Saved: {os.path.basename(raw_path)} & {os.path.basename(filtered_path)}"
                )
            
            print(f"✅ Plist data saved successfully:")
            print(f"   Raw: {os.path.basename(raw_path)}")
            print(f"   Filtered: {os.path.basename(filtered_path)}")
            print(f"   Total frames: {len(times)}, Duration: {times[-1]:.2f}s")
            
        except Exception as e:
            error_msg = f"Error saving plist: {str(e)}"
            print(f"[Save Error] {error_msg}")
            if hasattr(self, 'label_status'):
                self.label_status.configure(text=f"❌ {error_msg}")

    def on_plot_interaction(self, event):
        """Handle mouse clicks and drags on the plot."""
        if event.inaxes != self.ax:
            return
            
        # Only respond to left click drag or click
        if event.button != 1 and event.name == 'button_press_event':
            # Allow other buttons for zoom/pan
            return
        if event.name == 'motion_notify_event' and event.button != 1:
            return
            
        # Get time from x-axis
        target_time = event.xdata
        if target_time is None:
            return
            
        # Update cursor
        self.cursor_line.set_xdata([target_time])
        self.canvas.draw_idle() # Efficient redraw
        
        # Find closest frame
        times = np.array(self.time_history)
        idx = (np.abs(times - target_time)).argmin()
        
        # Update Eye View (Fast)
        if idx < len(self.eye_history):
            eye_img = self.eye_history[idx]
            if eye_img is not None:
                h, w = eye_img.shape[:2]
                scale = 4
                eye_img_big = cv2.resize(eye_img, (w*scale, h*scale), interpolation=cv2.INTER_NEAREST)
                img = Image.fromarray(eye_img_big)
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(w*scale, h*scale))
                if hasattr(self, 'win_eye') and self.win_eye is not None and self.win_eye.winfo_exists():
                    self.lbl_eye_img.configure(image=ctk_img)
                    self.lbl_eye_img.image = ctk_img
        
        # Update 3D Gaze Vector View (if window exists)
        if hasattr(self, 'ax_3d') and hasattr(self, 'win_gaze_vector') and self.win_gaze_vector is not None and self.win_gaze_vector.winfo_exists():
            self.update_gaze_vector_display(idx)
        
        # Update Video View (Slower, maybe skip if dragging too fast?)
        # For now, just do it.
        if hasattr(self, 'review_cap') and self.review_cap.isOpened() and idx < len(self.frame_index_history):
            frame_idx = self.frame_index_history[idx]
            self.review_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.review_cap.read()
            if ret and hasattr(self, 'win_video') and self.win_video is not None and self.win_video.winfo_exists():
                # Resize to fit window
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame_rgb.shape[:2]
                # Scale down if too big
                max_h = 400
                if h > max_h:
                    scale = max_h / h
                    w = int(w * scale)
                    h = int(h * scale)
                    frame_rgb = cv2.resize(frame_rgb, (w, h))
                
                img = Image.fromarray(frame_rgb)
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
                self.lbl_video_img.configure(image=ctk_img, text="")
                self.lbl_video_img.image = ctk_img
            
    def reset_ui(self):
        self.is_processing = False
        self.btn_start.configure(state="normal")
        self.btn_select.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.label_status.configure(text="Processing finished.")
        
        # Clean up windows
        windows_to_close = ['win_eye', 'win_gaze', 'win_video', 'win_gaze_vector', 'win_progress']
        for win_name in windows_to_close:
            if hasattr(self, win_name):
                win = getattr(self, win_name)
                if win is not None:
                    try:
                        if win.winfo_exists():
                            win.destroy()
                    except:
                        pass
            
        # Close figures
        if hasattr(self, 'fig'):
            try:
                plt.close(self.fig)
            except:
                pass
        if hasattr(self, 'fig_3d'):
            try:
                plt.close(self.fig_3d)
            except:
                pass

if __name__ == "__main__":
    app = GazeVisualizerApp()
    app.mainloop()
