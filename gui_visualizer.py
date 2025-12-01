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

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import SwinUNet
from preprocessing import EyeImagePreprocessor

class SignalProcessor:
    """Applies filtering to smooth gaze data."""
    def __init__(self, fps=30.0, low_pass_cutoff=5.0):
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
    # Left Eye
    LEFT_EYE_OUTER = 263
    LEFT_EYE_INNER = 362
    LEFT_EYE_UPPER = 386
    LEFT_EYE_BOTTOM = 374
    
    # Right Eye
    RIGHT_EYE_OUTER = 33
    RIGHT_EYE_INNER = 133
    RIGHT_EYE_UPPER = 159
    RIGHT_EYE_BOTTOM = 145
    
    def __init__(self, eye: str = 'left', target_size: Tuple[int, int] = (36, 60),
                 padding: float = 0.15, preprocessing_kwargs: Optional[dict] = None,
                 blink_window_extend_sec: float = 0.3):
        self.eye = eye.lower()
        self.target_size = target_size  # (H, W)
        self.padding = np.clip(padding, 0.05, 0.4)
        self.blink_window_extend_sec = blink_window_extend_sec  # Time-based blink window extension
        
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
        
        # Target points for affine transform
        self.target_points = np.float32([
            [outer_x, center_y],   # outer corner
            [inner_x, center_y],   # inner corner
            [w * 0.5, upper_y],    # upper eyelid
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
            
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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
        
        h_img, w_img, _ = frame_bgr.shape
        
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
        
        source_points = np.float32([
            outer_corner,
            inner_corner,
            upper_center
        ])
        
        try:
            M = cv2.getAffineTransform(source_points, self.target_points)
            h, w = self.target_size
            warped = cv2.warpAffine(
                frame_bgr,
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
        
        self.setup_ui()
        
        # Try to load default checkpoint
        self.load_default_checkpoint()

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
            "checkpoints/checkpoint_best.pth",
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

    def start_processing(self):
        if not self.video_path or not self.model:
            return
            
        self.is_processing = True
        self.btn_start.configure(state="disabled")
        self.btn_select.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        
        # Initialize history
        self.gaze_history = []
        self.start_time = time.time()
        self.time_history = []
        self.eye_history = [] # Cache processed eye images for review
        self.frame_index_history = [] # Map time to frame index
        
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
                    self.gaze_history.append(np.array([np.nan, np.nan, np.nan]))
                    self.time_history.append(timestamp)
                    
                    # Store None or placeholder for blink eye image (or the closed eye itself)
                    # Better to store the closed eye for review!
                    if roi_display is not None:
                        self.eye_history.append(roi_display)
                    else:
                        # Fallback if detection failed completely
                        self.eye_history.append(np.zeros((36, 60, 3), dtype=np.uint8))
                    self.frame_index_history.append(frame_count - 1)
                    
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
                    
                    # Record history
                    self.gaze_history.append(gaze_pred)
                    self.time_history.append(timestamp)
                    self.eye_history.append(roi_display)
                    self.frame_index_history.append(frame_count - 1) # 0-based index
                    
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
            if self.gaze_history:
                print("Applying post-processing filters...")
                if not self.enable_animation:
                    self.after(0, lambda: self.update_progress(0.95, "Applying filters..."))
                
                processor = SignalProcessor(fps=self.fps)
                raw_data = np.array(self.gaze_history)
                smoothed_data = processor.process(raw_data)
                
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
        recent_len = 300 # Show last N frames in real-time
        start_idx = max(0, len(self.time_history) - recent_len)
        
        times = np.array(self.time_history[start_idx:])
        history_vectors = np.array(self.gaze_history[start_idx:])
        
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
        if hasattr(self, 'win_progress') and self.win_progress.winfo_exists():
            self.win_progress.destroy()
        
        # Create or update eye window
        if not hasattr(self, 'win_eye') or not self.win_eye.winfo_exists():
            self.win_eye = ctk.CTkToplevel(self)
            self.win_eye.title("Review: Extracted Eye")
            self.win_eye.geometry("300x200")
            self.lbl_eye_img = ctk.CTkLabel(self.win_eye, text="")
            self.lbl_eye_img.pack(expand=True, fill="both")
        else:
            self.win_eye.title("Review: Extracted Eye")
            
        # Create Video Review Window
        if not hasattr(self, 'win_video') or not self.win_video.winfo_exists():
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
            
            # Add Save button to export plist data
            self.btn_save_plist = ctk.CTkButton(
                self.win_gaze, 
                text="💾 Save as Plist", 
                command=lambda: self.save_to_plist(smoothed_data),
                fg_color="green",
                hover_color="darkgreen"
            )
            self.btn_save_plist.pack(pady=5)

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
        self.ax_3d.set_xlabel('X (Left ← → Right)', fontsize=10)
        self.ax_3d.set_ylabel('Y (Depth: Person → Camera)', fontsize=10, fontweight='bold')
        self.ax_3d.set_zlabel('Z (Down ↓ ↑ Up)', fontsize=10)
        self.ax_3d.set_title('Gaze Direction: Eye → Camera', fontsize=12, fontweight='bold')
        
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
            direction = '👁️ Looking straight'
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
                direction = '👁️ Looking straight'
            elif abs(yaw) > abs(pitch):
                direction = f'Looking {"right →" if yaw > 0 else "← left"}'
            else:
                direction = f'Looking {"up ↑" if pitch > 0 else "down ↓"}'
            self.lbl_vector_info.configure(
                text=f"Pitch: {pitch:.1f}° | Yaw: {yaw:.1f}° | {direction}"
            )

    def save_to_plist(self, smoothed_data):
        """Save gaze data to plist format for further analysis."""
        try:
            # Ask user for save location
            from tkinter import filedialog
            default_filename = f"gaze_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.plist"
            save_path = filedialog.asksaveasfilename(
                defaultextension=".plist",
                filetypes=[("Plist files", "*.plist"), ("All files", "*.*")],
                initialfile=default_filename
            )
            
            if not save_path:
                return
            
            # Prepare data in plist format (matching your analysis code structure)
            times = np.array(self.time_history)
            raw_vectors = np.array(self.gaze_history)
            
            # Convert to angles
            smoothed_angles = self.vector_to_pitch_yaw(smoothed_data)
            
            # Generate timestamp strings (similar to your HIT format)
            base_time = datetime.now()
            time_list = []
            for t in times:
                timestamp = base_time + timedelta(seconds=float(t))
                time_list.append(timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            
            # Create plist data structure (compatible with your Streamlit analysis code)
            plist_data = {
                # Time data
                'TimeList': time_list,
                
                # Eye movement data in degrees (matching your analysis code's expected format)
                # Horizontal axis: Yaw (left-right eye movement)
                'LeftEyeXDegList': smoothed_angles[:, 1].tolist(),  # Yaw angles
                'RightEyeXDegList': smoothed_angles[:, 1].tolist(), # Same data for both eyes
                
                # Vertical axis: Pitch (up-down eye movement)
                'LeftEyeYDegList': smoothed_angles[:, 0].tolist(),  # Pitch angles
                'RightEyeYDegList': smoothed_angles[:, 0].tolist(), # Same data for both eyes
                
                # Also keep the original naming for reference
                'GazePitchDegList': smoothed_angles[:, 0].tolist(),
                'GazeYawDegList': smoothed_angles[:, 1].tolist(),
                
                # Raw gaze vectors (x, y, z) for advanced analysis
                'GazeXList': smoothed_data[:, 0].tolist(),
                'GazeYList': smoothed_data[:, 1].tolist(),
                'GazeZList': smoothed_data[:, 2].tolist(),
                
                # Metadata
                'VideoFile': os.path.basename(self.video_path) if self.video_path else 'unknown',
                'FPS': float(self.fps) if hasattr(self, 'fps') else 30.0,
                'TotalFrames': len(times),
                'Duration': float(times[-1]) if len(times) > 0 else 0.0,
                'ProcessingDate': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                
                # Processing parameters
                'BlinkWindowSec': 0.3,
                'FilteredData': True,
                'CoordinateSystem': 'MPIIGaze',
                
                # Additional info
                'Description': 'Gaze estimation data from SwinUNet-VOG - Compatible with pVestibular analysis',
                'DataFormat': 'Eye angles in degrees, compatible with LeftEyeXDegList/LeftEyeYDegList format',
                'PitchRange': '[-25, 45] degrees (vertical)',
                'YawRange': '[-45, 45] degrees (horizontal)',
            }
            
            # Save to plist file
            with open(save_path, 'wb') as f:
                plistlib.dump(plist_data, f)
            
            # Show success message
            if hasattr(self, 'label_status'):
                self.label_status.configure(
                    text=f"✅ Data saved to: {os.path.basename(save_path)}"
                )
            
            print(f"[Save] Plist data saved successfully to: {save_path}")
            print(f"[Save] Total frames: {len(times)}, Duration: {times[-1]:.2f}s")
            
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
                if hasattr(self, 'win_eye') and self.win_eye.winfo_exists():
                    self.lbl_eye_img.configure(image=ctk_img)
                    self.lbl_eye_img.image = ctk_img
        
        # Update 3D Gaze Vector View (if window exists)
        if hasattr(self, 'ax_3d') and hasattr(self, 'win_gaze_vector') and self.win_gaze_vector.winfo_exists():
            self.update_gaze_vector_display(idx)
        
        # Update Video View (Slower, maybe skip if dragging too fast?)
        # For now, just do it.
        if hasattr(self, 'review_cap') and self.review_cap.isOpened() and idx < len(self.frame_index_history):
            frame_idx = self.frame_index_history[idx]
            self.review_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.review_cap.read()
            if ret and hasattr(self, 'win_video') and self.win_video.winfo_exists():
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
        if hasattr(self, 'win_eye') and self.win_eye.winfo_exists():
            self.win_eye.destroy()
        if hasattr(self, 'win_gaze') and self.win_gaze.winfo_exists():
            self.win_gaze.destroy()
            
        # Close figure
        if hasattr(self, 'fig'):
            plt.close(self.fig)

if __name__ == "__main__":
    app = GazeVisualizerApp()
    app.mainloop()
