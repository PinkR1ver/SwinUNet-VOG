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
import scipy.signal
from collections import deque

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
                 padding: float = 0.15, preprocessing_kwargs: Optional[dict] = None):
        self.eye = eye.lower()
        self.target_size = target_size  # (H, W)
        self.padding = np.clip(padding, 0.05, 0.4)
        
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

    def extract(self, frame_bgr: np.ndarray) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], bool]:
        if frame_bgr is None or frame_bgr.size == 0:
            return None, None, False
            
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None, None, False
            
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Check blink
        is_blinking = self._is_blinking(landmarks)
        
        # Blink Window Expansion (Dilate Blink Events)
        # If currently blinking, extend the "blink state" slightly to cover
        # the transition phases (closing/opening) which often have bad data.
        
        # Initialize history buffer for blink smoothing if not exists
        if not hasattr(self, '_blink_history'):
            self._blink_history = deque(maxlen=3) # Store last 3 frames blink status
            
        self._blink_history.append(is_blinking)
        
        # If ANY frame in recent history was blinking, consider this frame 'unstable/blinking'
        # This expands the blink window forward. 
        # To expand backward, we would need a delay, but for real-time forward expansion is safer.
        # A simple "if any recent was blink" creates a hold-off period.
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
        
        # Controls
        self.frame_controls = ctk.CTkFrame(self.main_frame)
        self.frame_controls.grid(row=3, column=0, pady=20)
        
        self.btn_start = ctk.CTkButton(self.frame_controls, text="Start Processing", command=self.start_processing, state="disabled")
        self.btn_start.pack(side="left", padx=10)
        
        self.btn_stop = ctk.CTkButton(self.frame_controls, text="Stop", command=self.stop_processing, state="disabled", fg_color="red")
        self.btn_stop.pack(side="left", padx=10)
        
        # Checkpoint selection
        self.btn_ckpt = ctk.CTkButton(self.main_frame, text="Load Checkpoint", command=self.select_checkpoint, width=100)
        self.btn_ckpt.grid(row=4, column=0, pady=10)
        self.label_ckpt = ctk.CTkLabel(self.main_frame, text="No checkpoint loaded", font=("Arial", 10))
        self.label_ckpt.grid(row=5, column=0)

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
        
        # Initialize history and plot
        self.gaze_history = []
        self.start_time = time.time()
        self.time_history = []
        self.eye_history = [] # Cache processed eye images for review
        self.frame_index_history = [] # Map time to frame index
        
        self.fig = plt.figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Gaze Angles over Time")
        self.ax.set_ylim(-25, 45) # Optimized range based on MPIIGaze statistics
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
        self.win_eye.title("Extracted Eye Input")
        self.win_eye.geometry("300x200")
        self.lbl_eye_img = ctk.CTkLabel(self.win_eye, text="")
        self.lbl_eye_img.pack(expand=True, fill="both")
        
        self.win_gaze = ctk.CTkToplevel(self)
        self.win_gaze.title("Gaze Prediction")
        self.win_gaze.geometry("600x550") # Increased height for toolbar
        
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
        
        # Start thread
        self.frame_idx = 0
        threading.Thread(target=self.process_video_loop, daemon=True).start()

    def stop_processing(self):
        self.is_processing = False

    def process_video_loop(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError("Could not open video")
                
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0: self.fps = 30.0
            
            self.normalizer = MediaPipeEyeNormalizer(eye='left')
            
            # Clear cache for new video
            self.frame_cache = []
            
            frame_count = 0
            while self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate timestamp based on FPS
                timestamp = frame_count / self.fps
                frame_count += 1
                
                # Extract Eye
                roi_tensor, roi_display, is_blinking = self.normalizer.extract(frame)
                
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
                    
                    # Update GUI with correct timestamp
                    # Update visualization less frequently to speed up processing (e.g., every 3rd frame)
                    if frame_count % 3 == 0:
                        self.after(0, self.update_visualization, roi_display, gaze_pred, timestamp, False)
                
                # Control FPS roughly
                # time.sleep(1.0 / self.fps) # Disabled to process as fast as possible
                pass
                
            cap.release()
            if self.normalizer:
                self.normalizer.close()
            
            # Post-processing
            if self.gaze_history:
                print("Applying post-processing filters...")
                processor = SignalProcessor(fps=self.fps)
                raw_data = np.array(self.gaze_history)
                smoothed_data = processor.process(raw_data)
                
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
        
        # Keep eye window open for review
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
            
            # Plot smoothed data (Full continuous line)
            for i in range(2):
                # Plot smoothed (Solid)
                self.ax.plot(times, smoothed_angles[:, i], color=colors[i], linewidth=2, label=f'{labels[i]} smooth')
                
                # Plot Raw (Transparent)
                self.ax.plot(times, raw_angles[:, i], color=colors[i], alpha=0.2, linewidth=1, label=f'{labels[i]} raw')
                
                # Identify Blink Segments
                blink_mask = np.isnan(raw_angles[:, i])
                if np.any(blink_mask):
                    # Create a copy of smoothed data, set non-blink parts to NaN
                    blink_segments = smoothed_angles[:, i].copy()
                    blink_segments[~blink_mask] = np.nan
                    
                    # Plot blink segments as Dashed line on top
                    self.ax.plot(times, blink_segments, color=colors[i], linestyle='--', linewidth=2, alpha=0.7)
                
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
