"""
可视化工具：展示模型在测试图像上的预测效果
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
from typing import Optional, Tuple

from face_alignment import FaceAlignment, LandmarksType
import imageio.v2 as imageio

from model import SwinUNet
from data import get_data_loaders, get_person_specific_loaders
from preprocessing import EyeImagePreprocessor


def visualize_predictions(model, test_loader, device, num_samples=16, save_dir='results'):
    """
    可视化模型在测试集上的预测效果。
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 设备
        num_samples: 要可视化的样本数量
        save_dir: 保存目录
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # 收集样本
    images_list = []
    predictions_list = []
    targets_list = []
    errors_list = []
    
    with torch.no_grad():
        for images, gaze, _ in test_loader:
            images = images.to(device)
            gaze = gaze.to(device)
            
            output = model(images)
            
            # 计算角度误差
            pred_norm = output / torch.norm(output, dim=1, keepdim=True)
            target_norm = gaze / torch.norm(gaze, dim=1, keepdim=True)
            dot_product = torch.sum(pred_norm * target_norm, dim=1)
            dot_product = torch.clamp(dot_product, -1.0, 1.0)
            errors = torch.acos(dot_product) * 180.0 / 3.14159
            
            images_list.append(images.cpu())
            predictions_list.append(output.cpu())
            targets_list.append(gaze.cpu())
            errors_list.append(errors.cpu())
            
            if len(images_list) * images.size(0) >= num_samples:
                break
    
    # 合并所有批次
    all_images = torch.cat(images_list, dim=0)[:num_samples]
    all_predictions = torch.cat(predictions_list, dim=0)[:num_samples]
    all_targets = torch.cat(targets_list, dim=0)[:num_samples]
    all_errors = torch.cat(errors_list, dim=0)[:num_samples]
    
    # 创建可视化
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        
        # 显示图像
        img = all_images[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img, cmap='gray' if img.shape[2] == 1 else None)
        
        # 归一化gaze向量
        pred_gaze = all_predictions[i].numpy()
        target_gaze = all_targets[i].numpy()
        pred_gaze = pred_gaze / (np.linalg.norm(pred_gaze) + 1e-8)
        target_gaze = target_gaze / (np.linalg.norm(target_gaze) + 1e-8)
        
        # 显示信息
        error = all_errors[i].item()
        title = f'Error: {error:.2f}°'
        title += f'\nPred: [{pred_gaze[0]:.2f}, {pred_gaze[1]:.2f}, {pred_gaze[2]:.2f}]'
        title += f'\nGT: [{target_gaze[0]:.2f}, {target_gaze[1]:.2f}, {target_gaze[2]:.2f}]'
        
        ax.set_title(title, fontsize=8)
        ax.axis('off')
    
    # 隐藏多余的子图
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_samples.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建更好的可视化：显示最好的和最差的预测
    # 按误差排序
    sorted_indices = torch.argsort(all_errors)
    best_indices = sorted_indices[:8]
    worst_indices = sorted_indices[-8:]
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    # 最好的8个
    for i, idx in enumerate(best_indices):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        img = all_images[idx].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img, cmap='gray' if img.shape[2] == 1 else None)
        
        error = all_errors[idx].item()
        pred_gaze = all_predictions[idx].numpy()
        target_gaze = all_targets[idx].numpy()
        pred_gaze = pred_gaze / (np.linalg.norm(pred_gaze) + 1e-8)
        target_gaze = target_gaze / (np.linalg.norm(target_gaze) + 1e-8)
        
        title = f'Best #{i+1}\nError: {error:.2f}°'
        ax.set_title(title, fontsize=9, color='green')
        ax.axis('off')
    
    # 最差的8个
    for i, idx in enumerate(worst_indices):
        row = (i // 4) + 2
        col = i % 4
        ax = axes[row, col]
        
        img = all_images[idx].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img, cmap='gray' if img.shape[2] == 1 else None)
        
        error = all_errors[idx].item()
        pred_gaze = all_predictions[idx].numpy()
        target_gaze = all_targets[idx].numpy()
        pred_gaze = pred_gaze / (np.linalg.norm(pred_gaze) + 1e-8)
        target_gaze = target_gaze / (np.linalg.norm(target_gaze) + 1e-8)
        
        title = f'Worst #{i+1}\nError: {error:.2f}°'
        ax.set_title(title, fontsize=9, color='red')
        ax.axis('off')
    
    plt.suptitle('Best and Worst Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'best_worst_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Visualization saved to {save_dir}/')


class FaceAlignmentEyeNormalizer:
    """Extract and normalize eye ROI using face-alignment landmarks."""
    
    LEFT_EYE_IDXS = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_IDXS = [42, 43, 44, 45, 46, 47]
    
    def __init__(self, eye: str = 'left', target_size: Tuple[int, int] = (36, 60),
                 device: str = 'cpu', padding: float = 0.15,
                 preprocessing_kwargs: Optional[dict] = None):
        self.eye = eye.lower()
        self.target_size = target_size  # (H, W)
        self.padding = np.clip(padding, 0.05, 0.4)
        
        fa_device = 'cuda' if device.startswith('cuda') and torch.cuda.is_available() else 'cpu'
        landmarks_type = getattr(LandmarksType, '_2D', None)
        if landmarks_type is None:
            landmarks_type = getattr(LandmarksType, 'TWO_D', None)
        if landmarks_type is None:
            raise AttributeError('face-alignment LandmarksType does not expose a 2D mode')
        
        self.face_aligner = FaceAlignment(
            landmarks_type,
            device=fa_device,
            flip_input=False
        )
        
        if preprocessing_kwargs is None:
            preprocessing_kwargs = {}
        preprocessing_kwargs = dict(preprocessing_kwargs)
        preprocessing_kwargs.setdefault('use_geometric_normalization', False)
        preprocessing_kwargs.setdefault('normalize_illumination', True)
        preprocessing_kwargs.setdefault('normalize_contrast', True)
        preprocessing_kwargs.setdefault('normalize_color', False)
        preprocessing_kwargs.setdefault('gamma_correction', True)
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
        self.target_points = np.float32([
            [outer_x, center_y],   # outer corner
            [inner_x, center_y],   # inner corner
            [w * 0.5, upper_y],    # upper eyelid
        ])
        
        self.last_roi_tensor: Optional[torch.Tensor] = None
        self.last_roi_display: Optional[np.ndarray] = None
    
    def _select_eye_points(self, landmarks: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        if self.eye == 'right':
            idxs = self.RIGHT_EYE_IDXS
            outer_idx, inner_idx = 45, 42
        else:
            idxs = self.LEFT_EYE_IDXS
            outer_idx, inner_idx = 36, 39
        
        try:
            eye_points = landmarks[idxs]
            outer_corner = landmarks[outer_idx]
            inner_corner = landmarks[inner_idx]
        except Exception:
            return None
        
        upper_center = eye_points[[1, 2]].mean(axis=0)
        lower_center = eye_points[[4, 5]].mean(axis=0)
        return outer_corner, inner_corner, upper_center, lower_center
    
    def extract(self, frame_bgr: np.ndarray) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
        """Return normalized ROI tensor (C, H, W) and ROI display image (H, W, 3 in RGB)."""
        if frame_bgr is None or frame_bgr.size == 0:
            return None, None
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        preds = self.face_aligner.get_landmarks(frame_rgb)
        if preds is None or len(preds) == 0:
            return None, None
        
        landmarks = preds[0]
        keypoints = self._select_eye_points(landmarks)
        if keypoints is None:
            return None, None
        
        outer_corner, inner_corner, upper_center, lower_center = keypoints
        source_points = np.float32([
            outer_corner,
            inner_corner,
            upper_center
        ])
        
        try:
            M = cv2.getAffineTransform(source_points, self.target_points)
        except cv2.error:
            return None, None
        
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
        return roi_tensor, roi_display
    
    def fallback(self) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
        return self.last_roi_tensor, self.last_roi_display
    
    def close(self):
        self.face_aligner = None


def visualize_video_gaze(model, video_path, device, save_dir='results', resize=(60, 36),
                         roi_normalizer: Optional[FaceAlignmentEyeNormalizer] = None):
    """
    Display video frames alongside real-time gaze predictions.
    
    Args:
        model: Trained model
        video_path: Path to the input video
        device: Torch device
        save_dir: Directory to cache artifacts
        resize: Input resolution (width, height)
        roi_normalizer: Optional FaceAlignmentEyeNormalizer used to extract and normalize ROI
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f'Video file {video_path} not found')
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video file {video_path}')
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0:
        source_fps = 30.0
    
    os.makedirs(save_dir, exist_ok=True)
    plt.ion()
    
    if roi_normalizer is not None:
        fig = plt.figure(figsize=(22, 6))
        frame_ax = fig.add_subplot(1, 4, 1)
        roi_ax = fig.add_subplot(1, 4, 2)
        curve_ax = fig.add_subplot(1, 4, 3)
        scatter_ax = fig.add_subplot(1, 4, 4, projection='3d')
        roi_ax.set_title('Extracted ROI', fontsize=12)
        roi_ax.axis('off')
    else:
        fig = plt.figure(figsize=(18, 6))
        frame_ax = fig.add_subplot(1, 3, 1)
        curve_ax = fig.add_subplot(1, 3, 2)
        scatter_ax = fig.add_subplot(1, 3, 3, projection='3d')
        roi_ax = None
    
    frame_ax.axis('off')
    frame_ax.set_title('Video Frame & Prediction', fontsize=12)
    
    curve_ax.set_title('Gaze Components over Frames', fontsize=12)
    curve_ax.set_xlabel('Frame Index')
    curve_ax.set_ylabel('Gaze Component Value')
    curve_ax.set_ylim(-1.2, 1.2)
    curve_ax.grid(True, alpha=0.3)
    
    colors = ['tab:red', 'tab:green', 'tab:blue']
    labels = ['Gaze X', 'Gaze Y', 'Gaze Z']
    lines = [curve_ax.plot([], [], color=c, label=l)[0] for c, l in zip(colors, labels)]
    curve_ax.legend(loc='upper right')
    
    scatter_ax.set_title('Gaze in 3D Space', fontsize=12)
    scatter_ax.set_xlabel('Gaze X')
    scatter_ax.set_ylabel('Gaze Y')
    scatter_ax.set_zlabel('Gaze Z')
    scatter_ax.set_xlim(-1.2, 1.2)
    scatter_ax.set_ylim(-1.2, 1.2)
    scatter_ax.set_zlim(-1.2, 1.2)
    scatter_ax.grid(True, alpha=0.3)
    line3d, = scatter_ax.plot([], [], [], color='tab:purple', linewidth=1.5)
    point3d = scatter_ax.scatter([0], [0], [0], color='black', s=40)
    
    gaze_history = []
    frame_indices = []
    gif_frames = []

    width, height = resize
    
    try:
        frame_idx = 0
        text_artist = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_display = frame_rgb.copy()
            
            if roi_normalizer is not None:
                roi_tensor, roi_display = roi_normalizer.extract(frame)
                if roi_tensor is None:
                    roi_tensor, roi_display = roi_normalizer.fallback()
                    if roi_tensor is None:
                        frame_idx += 1
                        continue
                frame_tensor = roi_tensor.unsqueeze(0).to(device)
                
                if roi_ax is not None and roi_display is not None:
                    roi_ax.imshow(roi_display)
                    roi_ax.set_title('Extracted ROI', fontsize=12)
                    roi_ax.axis('off')
            else:
                frame_resized = cv2.resize(frame_rgb, (width, height))
                frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                frame_tensor = frame_tensor.to(device)
            
            with torch.no_grad():
                prediction = model(frame_tensor).cpu().numpy()[0]
            
            gaze_history.append(prediction)
            frame_indices.append(frame_idx)
            
            frame_ax.imshow(frame_display)
            if text_artist is not None:
                text_artist.remove()
            text_artist = frame_ax.text(
                0.02, 0.95,
                f'Frame: {frame_idx}\nGaze: [{prediction[0]:.2f}, {prediction[1]:.2f}, {prediction[2]:.2f}]',
                color='white',
                fontsize=10,
                transform=frame_ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.4)
            )
            
            history_array = np.array(gaze_history)
            for i, line in enumerate(lines):
                line.set_data(frame_indices, history_array[:, i])
            
            if frame_idx > 0:
                curve_ax.set_xlim(0, max(frame_indices))
            
            line3d.set_data(history_array[:, 0], history_array[:, 1])
            line3d.set_3d_properties(history_array[:, 2])
            point3d._offsets3d = (
                np.array([history_array[-1, 0]]),
                np.array([history_array[-1, 1]]),
                np.array([history_array[-1, 2]])
            )
            
            fig.canvas.draw()
            canvas_width, canvas_height = fig.canvas.get_width_height()
            canvas_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            canvas_image = canvas_image.reshape(canvas_height, canvas_width, 3)
            gif_frames.append(canvas_image.copy())
            fig.canvas.flush_events()
            plt.pause(0.001)
            
            frame_idx += 1
    
    finally:
        cap.release()
        plt.ioff()
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    if gaze_history:
        history_array = np.array(gaze_history)
        np.save(os.path.join(save_dir, 'video_gaze_predictions.npy'), history_array)
        fig_static = plt.figure(figsize=(18, 6))
        curve_static = fig_static.add_subplot(1, 2, 1)
        scatter_static = fig_static.add_subplot(1, 2, 2, projection='3d')
        
        for i, (c, l) in enumerate(zip(colors, labels)):
            curve_static.plot(frame_indices, history_array[:, i], color=c, label=l)
        curve_static.set_title('Gaze Components over Frames')
        curve_static.set_xlabel('Frame Index')
        curve_static.set_ylabel('Gaze Component Value')
        curve_static.set_ylim(-1.2, 1.2)
        curve_static.grid(True, alpha=0.3)
        curve_static.legend()
        
        scatter_static.plot(history_array[:, 0], history_array[:, 1], history_array[:, 2],
                            color='tab:purple', linewidth=1.5)
        scatter_static.scatter(history_array[-1, 0], history_array[-1, 1], history_array[-1, 2],
                               color='black', s=40)
        scatter_static.set_title('Gaze in 3D Space')
        scatter_static.set_xlabel('Gaze X')
        scatter_static.set_ylabel('Gaze Y')
        scatter_static.set_zlabel('Gaze Z')
        scatter_static.set_xlim(-1.2, 1.2)
        scatter_static.set_ylim(-1.2, 1.2)
        scatter_static.set_zlim(-1.2, 1.2)
        scatter_static.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'video_gaze_curve.png'), dpi=300, bbox_inches='tight')
        plt.close(fig_static)
    
    if gif_frames:
        gif_path = os.path.join(save_dir, 'video_gaze_demo.gif')
        imageio.mimsave(gif_path, gif_frames, fps=min(20, source_fps))
        print(f'GIF saved to {gif_path}')
    
    print('Video gaze demo completed.')


def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint to load')
    parser.add_argument('--data_dir', type=str, default='MPIIGaze/Data/Normalized',
                        help='Path to normalized data')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--num_samples', type=int, default=16,
                        help='Number of samples to visualize')
    parser.add_argument('--person_specific', action='store_true',
                        help='Use person-specific evaluation protocol')
    parser.add_argument('--person_id', type=str, default='p00',
                        help='Person ID for person-specific evaluation')
    parser.add_argument('--video', type=str, default=None,
                        help='Run real-time gaze demo with video input (skip dataset visualization)')
    parser.add_argument('--use_roi', action='store_true',
                        help='Use face landmark detection to extract eye ROI before inference')
    parser.add_argument('--eye', type=str, default='left', choices=['left', 'right'],
                        help='Eye to extract when using ROI mode')
    parser.add_argument('--roi_padding', type=float, default=0.25,
                        help='Padding ratio around detected eye ROI')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Load checkpoint
    print(f'Loading checkpoint from {args.checkpoint}...')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create model
    print('Creating model...')
    model = SwinUNet(
        img_size=(36, 60),
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 2],
        num_heads=[3, 6, 12],
        window_size=7,
        drop_rate=0.1
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    if args.video:
        print('Running video gaze demo...')
        roi_normalizer = None
        if args.use_roi:
            roi_normalizer = FaceAlignmentEyeNormalizer(
                eye=args.eye,
                target_size=(36, 60),
                device=args.device,
                padding=args.roi_padding
            )
        try:
            visualize_video_gaze(
                model,
                args.video,
                device,
                save_dir=args.save_dir,
                roi_normalizer=roi_normalizer
            )
        finally:
            if roi_normalizer is not None:
                roi_normalizer.close()
        print('Visualization completed!')
        return
    
    # Load data only when需要图像样本可视化
    print('Loading data...')
    if args.person_specific:
        train_loader, val_loader, test_loader = get_person_specific_loaders(
            args.data_dir, args.person_id, batch_size=args.batch_size, num_workers=args.num_workers
        )
    else:
        train_loader, val_loader, test_loader = get_data_loaders(
            args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
        )
    
    # Visualize
    print('Visualizing predictions...')
    visualize_predictions(model, test_loader, device, 
                         num_samples=args.num_samples, 
                         save_dir=args.save_dir)
    
    print('Visualization completed!')


if __name__ == '__main__':
    main()
