#!/usr/bin/env python3
"""
Verify training data: visualize what images and targets are fed to the model.
验证训练数据：可视化输入模型的图像和回归目标。
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io

import torch
from data import get_data_loaders, MPIIGazeDataset


def gaze_vector_to_angles(gaze):
    """
    Convert 3D gaze vector to pitch and yaw angles (degrees).
    
    Args:
        gaze: (x, y, z) gaze direction vector
    
    Returns:
        pitch, yaw in degrees
    """
    x, y, z = gaze
    # Normalize
    norm = np.sqrt(x**2 + y**2 + z**2)
    x, y, z = x/norm, y/norm, z/norm
    
    # Convert to angles (MPIIGaze convention)
    pitch = np.arcsin(-y) * 180 / np.pi
    yaw = np.arctan2(-x, -z) * 180 / np.pi
    
    return pitch, yaw


def visualize_samples(dataset, num_samples=16, save_path='verify_samples.png'):
    """
    Visualize random samples from dataset.
    """
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        image, gaze, participant = dataset[idx]
        
        # Convert to numpy for display
        if isinstance(image, torch.Tensor):
            img_np = image.numpy()
        else:
            img_np = image
        
        # Handle different formats
        if img_np.ndim == 3:
            if img_np.shape[0] == 3:  # (C, H, W)
                img_np = img_np.transpose(1, 2, 0)  # -> (H, W, C)
            # If grayscale repeated 3 times, just take one channel
            if img_np.shape[2] == 3 and np.allclose(img_np[:,:,0], img_np[:,:,1]):
                img_np = img_np[:,:,0]
        
        # Convert gaze to numpy
        if isinstance(gaze, torch.Tensor):
            gaze_np = gaze.numpy()
        else:
            gaze_np = gaze
        
        # Calculate angles
        pitch, yaw = gaze_vector_to_angles(gaze_np)
        
        # Display
        ax = axes[i]
        if img_np.ndim == 2:
            ax.imshow(img_np, cmap='gray', vmin=0, vmax=1 if img_np.max() <= 1 else 255)
        else:
            ax.imshow(np.clip(img_np, 0, 1) if img_np.max() <= 1 else img_np)
        
        # Title with gaze info
        title = f'{participant}\n'
        title += f'Gaze: [{gaze_np[0]:.3f}, {gaze_np[1]:.3f}, {gaze_np[2]:.3f}]\n'
        title += f'Pitch: {pitch:.1f}°, Yaw: {yaw:.1f}°'
        ax.set_title(title, fontsize=8)
        ax.axis('off')
    
    # Hide unused axes
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('MPIIGaze Training Samples\n(Image + Gaze Vector Target)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved to: {save_path}')
    plt.show()


def visualize_batch(data_loader, save_path='verify_batch.png'):
    """
    Visualize one batch from data loader (as it would be fed to model).
    """
    images, gazes, participants = next(iter(data_loader))
    
    batch_size = images.shape[0]
    num_show = min(16, batch_size)
    
    grid_size = int(np.ceil(np.sqrt(num_show)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(14, 14))
    axes = axes.flatten()
    
    for i in range(num_show):
        img = images[i].numpy()
        gaze = gazes[i].numpy()
        
        # (C, H, W) -> (H, W, C) or (H, W)
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
            if np.allclose(img[:,:,0], img[:,:,1]):
                img = img[:,:,0]
        
        pitch, yaw = gaze_vector_to_angles(gaze)
        
        ax = axes[i]
        if img.ndim == 2:
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(np.clip(img, 0, 1))
        
        title = f'Gaze: [{gaze[0]:.2f}, {gaze[1]:.2f}, {gaze[2]:.2f}]\n'
        title += f'P:{pitch:.1f}° Y:{yaw:.1f}°'
        ax.set_title(title, fontsize=8)
        ax.axis('off')
    
    for i in range(num_show, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Training Batch (as fed to model)\nShape: {images.shape}, dtype: {images.dtype}', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved to: {save_path}')
    plt.show()


def analyze_gaze_distribution(dataset, save_path='verify_gaze_distribution.png'):
    """
    Analyze the distribution of gaze targets.
    """
    all_gazes = []
    all_pitches = []
    all_yaws = []
    
    print("Analyzing gaze distribution...")
    for i in range(len(dataset)):
        _, gaze, _ = dataset[i]
        if isinstance(gaze, torch.Tensor):
            gaze = gaze.numpy()
        all_gazes.append(gaze)
        pitch, yaw = gaze_vector_to_angles(gaze)
        all_pitches.append(pitch)
        all_yaws.append(yaw)
    
    all_gazes = np.array(all_gazes)
    all_pitches = np.array(all_pitches)
    all_yaws = np.array(all_yaws)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Gaze vector components
    for i, (comp, name) in enumerate(zip(all_gazes.T, ['X', 'Y', 'Z'])):
        ax = axes[0, i]
        ax.hist(comp, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel(f'Gaze {name}')
        ax.set_ylabel('Count')
        ax.set_title(f'Gaze {name} Distribution\nmean={comp.mean():.3f}, std={comp.std():.3f}')
        ax.axvline(comp.mean(), color='r', linestyle='--', label='Mean')
        ax.grid(True, alpha=0.3)
    
    # Pitch distribution
    axes[1, 0].hist(all_pitches, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Pitch (degrees)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title(f'Pitch Distribution\nmean={all_pitches.mean():.1f}°, std={all_pitches.std():.1f}°')
    axes[1, 0].axvline(all_pitches.mean(), color='r', linestyle='--')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Yaw distribution
    axes[1, 1].hist(all_yaws, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('Yaw (degrees)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title(f'Yaw Distribution\nmean={all_yaws.mean():.1f}°, std={all_yaws.std():.1f}°')
    axes[1, 1].axvline(all_yaws.mean(), color='r', linestyle='--')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Pitch vs Yaw scatter
    axes[1, 2].scatter(all_yaws, all_pitches, alpha=0.1, s=1)
    axes[1, 2].set_xlabel('Yaw (degrees)')
    axes[1, 2].set_ylabel('Pitch (degrees)')
    axes[1, 2].set_title('Pitch vs Yaw (Gaze Direction Space)')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axhline(0, color='gray', linestyle='-', alpha=0.5)
    axes[1, 2].axvline(0, color='gray', linestyle='-', alpha=0.5)
    
    plt.suptitle(f'Gaze Target Distribution (N={len(dataset)} samples)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved to: {save_path}')
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("Gaze Target Statistics:")
    print("="*60)
    print(f"Total samples: {len(dataset)}")
    print(f"\nGaze Vector (X, Y, Z):")
    print(f"  X: mean={all_gazes[:,0].mean():.4f}, std={all_gazes[:,0].std():.4f}, range=[{all_gazes[:,0].min():.4f}, {all_gazes[:,0].max():.4f}]")
    print(f"  Y: mean={all_gazes[:,1].mean():.4f}, std={all_gazes[:,1].std():.4f}, range=[{all_gazes[:,1].min():.4f}, {all_gazes[:,1].max():.4f}]")
    print(f"  Z: mean={all_gazes[:,2].mean():.4f}, std={all_gazes[:,2].std():.4f}, range=[{all_gazes[:,2].min():.4f}, {all_gazes[:,2].max():.4f}]")
    print(f"\nAngles:")
    print(f"  Pitch: mean={all_pitches.mean():.2f}°, std={all_pitches.std():.2f}°, range=[{all_pitches.min():.2f}°, {all_pitches.max():.2f}°]")
    print(f"  Yaw:   mean={all_yaws.mean():.2f}°, std={all_yaws.std():.2f}°, range=[{all_yaws.min():.2f}°, {all_yaws.max():.2f}°]")
    print("="*60)


def visualize_raw_mat_file(mat_path, num_samples=8, save_path='verify_raw_mat.png'):
    """
    Visualize raw data from a .mat file to see exactly what's in MPIIGaze.
    """
    print(f"\nLoading: {mat_path}")
    data = scipy.io.loadmat(mat_path)
    
    # Extract data
    left_data = data['data'][0, 0]['left'][0]
    right_data = data['data'][0, 0]['right'][0]
    
    left_images = left_data['image'][0]
    left_gazes = left_data['gaze'][0]
    right_images = right_data['image'][0]
    right_gazes = right_data['gaze'][0]
    
    print(f"Left eye samples: {len(left_images)}")
    print(f"Right eye samples: {len(right_images)}")
    print(f"Image shape: {left_images[0].shape}")
    print(f"Image dtype: {left_images[0].dtype}")
    print(f"Image value range: [{left_images[0].min()}, {left_images[0].max()}]")
    print(f"Gaze shape: {left_gazes[0].shape}")
    
    # Visualize
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 5))
    
    for i in range(min(num_samples, len(left_images))):
        # Left eye
        axes[0, i].imshow(left_images[i], cmap='gray')
        gaze = left_gazes[i]
        pitch, yaw = gaze_vector_to_angles(gaze)
        axes[0, i].set_title(f'P:{pitch:.1f}° Y:{yaw:.1f}°', fontsize=8)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Left Eye', fontsize=10)
        
        # Right eye
        axes[1, i].imshow(right_images[i], cmap='gray')
        gaze = right_gazes[i]
        pitch, yaw = gaze_vector_to_angles(gaze)
        axes[1, i].set_title(f'P:{pitch:.1f}° Y:{yaw:.1f}°', fontsize=8)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Right Eye', fontsize=10)
    
    plt.suptitle(f'Raw MPIIGaze Data: {Path(mat_path).name}\n'
                 f'Shape: {left_images[0].shape}, dtype: {left_images[0].dtype}, '
                 f'range: [{left_images[0].min()}, {left_images[0].max()}]',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved to: {save_path}')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Verify training data for SwinUNet Gaze Estimation')
    parser.add_argument('--data_dir', type=str, default='MPIIGaze/Data/Normalized',
                        help='Path to normalized data')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_samples', type=int, default=16,
                        help='Number of samples to visualize')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='Directory to save verification plots')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'samples', 'batch', 'distribution', 'raw'],
                        help='What to verify')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Training Data Verification")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.mode in ['all', 'raw']:
        # Visualize raw .mat file
        mat_files = list(Path(args.data_dir).glob('p00/day*.mat'))
        if mat_files:
            visualize_raw_mat_file(
                str(mat_files[0]), 
                num_samples=8,
                save_path=os.path.join(args.save_dir, 'verify_raw_mat.png')
            )
    
    if args.mode in ['all', 'samples', 'batch', 'distribution']:
        # Load dataset
        print("\nLoading dataset...")
        train_loader, val_loader, test_loader = get_data_loaders(
            args.data_dir, batch_size=args.batch_size, num_workers=0, train_augment=False
        )
        
        # Get underlying dataset
        train_dataset = train_loader.dataset
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
    
    if args.mode in ['all', 'samples']:
        print("\n--- Visualizing Random Samples ---")
        visualize_samples(
            train_dataset, 
            num_samples=args.num_samples,
            save_path=os.path.join(args.save_dir, 'verify_samples.png')
        )
    
    if args.mode in ['all', 'batch']:
        print("\n--- Visualizing Training Batch ---")
        visualize_batch(
            train_loader,
            save_path=os.path.join(args.save_dir, 'verify_batch.png')
        )
    
    if args.mode in ['all', 'distribution']:
        print("\n--- Analyzing Gaze Distribution ---")
        analyze_gaze_distribution(
            train_dataset,
            save_path=os.path.join(args.save_dir, 'verify_gaze_distribution.png')
        )
    
    print("\nVerification completed!")


if __name__ == '__main__':
    main()


