"""
Evaluation script for SwinUNet gaze estimation model.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from model import SwinUNet
from data import get_data_loaders, get_person_specific_loaders


def angle_error(pred, target):
    """
    Calculate angular error between predicted and target gaze vectors.
    
    Args:
        pred: Predicted gaze vector (batch_size, 3)
        target: Target gaze vector (batch_size, 3)
    
    Returns:
        Angular error in degrees for each sample
    """
    # Normalize vectors
    pred_norm = pred / torch.norm(pred, dim=1, keepdim=True)
    target_norm = target / torch.norm(target, dim=1, keepdim=True)
    
    # Calculate dot product
    dot_product = torch.sum(pred_norm * target_norm, dim=1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # Calculate angle in radians and convert to degrees
    angle = torch.acos(dot_product) * 180.0 / 3.14159
    return angle


def evaluate(model, test_loader, criterion, device, save_plots=True, save_dir='results'):
    """Evaluate model on test set."""
    model.eval()
    
    all_errors = []
    all_predictions = []
    all_targets = []
    running_loss = 0.0
    
    print('Evaluating on test set...')
    
    with torch.no_grad():
        for batch_idx, (images, gaze, _) in enumerate(test_loader):
            images, gaze = images.to(device), gaze.to(device)
            
            output = model(images)
            loss = criterion(output, gaze)
            errors = angle_error(output, gaze)
            
            all_errors.extend(errors.cpu().numpy())
            all_predictions.append(output.cpu().numpy())
            all_targets.append(gaze.cpu().numpy())
            running_loss += loss.item()
            
            if batch_idx % 200 == 0:
                print(f'Batch: {batch_idx}/{len(test_loader)}, '
                      f'Loss: {loss.item():.6f}, Mean Error: {errors.mean().item():.2f}°')
    
    # Calculate statistics
    all_errors = np.array(all_errors)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    avg_loss = running_loss / len(test_loader)
    mean_error = np.mean(all_errors)
    median_error = np.median(all_errors)
    std_error = np.std(all_errors)
    percentile_95 = np.percentile(all_errors, 95)
    percentile_99 = np.percentile(all_errors, 99)
    
    print('\n' + '=' * 80)
    print('Evaluation Results:')
    print('=' * 80)
    print(f'Average Loss: {avg_loss:.6f}')
    print(f'Mean Angular Error: {mean_error:.2f}°')
    print(f'Median Angular Error: {median_error:.2f}°')
    print(f'Std Angular Error: {std_error:.2f}°')
    print(f'95th Percentile: {percentile_95:.2f}°')
    print(f'99th Percentile: {percentile_99:.2f}°')
    print(f'Min Error: {all_errors.min():.2f}°')
    print(f'Max Error: {all_errors.max():.2f}°')
    print('=' * 80)
    
    # Save plots if requested
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        
        # Error distribution histogram
        plt.figure(figsize=(10, 6))
        plt.hist(all_errors, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(mean_error, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.2f}°')
        plt.axvline(median_error, color='g', linestyle='--', linewidth=2, label=f'Median: {median_error:.2f}°')
        plt.xlabel('Angular Error (degrees)')
        plt.ylabel('Frequency')
        plt.title('Angular Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Cumulative distribution
        sorted_errors = np.sort(all_errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_errors, cumulative)
        plt.xlabel('Angular Error (degrees)')
        plt.ylabel('Cumulative Percentage (%)')
        plt.title('Cumulative Error Distribution')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'cumulative_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Scatter plot of predictions vs targets (for one component)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i in range(3):
            axes[i].scatter(all_targets[:, i], all_predictions[:, i], alpha=0.3, s=1)
            axes[i].plot([all_targets[:, i].min(), all_targets[:, i].max()],
                        [all_targets[:, i].min(), all_targets[:, i].max()],
                        'r--', linewidth=2)
            axes[i].set_xlabel('Ground Truth')
            axes[i].set_ylabel('Prediction')
            axes[i].set_title(f'Gaze Component {i+1}')
            axes[i].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'predictions_vs_targets.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'\nPlots saved to {save_dir}/')
    
    return {
        'mean_error': mean_error,
        'median_error': median_error,
        'std_error': std_error,
        'percentile_95': percentile_95,
        'percentile_99': percentile_99,
        'avg_loss': avg_loss
    }


def main():
    parser = argparse.ArgumentParser(description='Test SwinUNet for Gaze Estimation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint to load')
    parser.add_argument('--data_dir', type=str, default='MPIIGaze/Data/Normalized',
                        help='Path to normalized data')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--person_specific', action='store_true',
                        help='Use person-specific evaluation protocol')
    parser.add_argument('--person_id', type=str, default='p00',
                        help='Person ID for person-specific evaluation')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip saving plots')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Load checkpoint
    print(f'Loading checkpoint from {args.checkpoint}...')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Load data
    print('Loading data...')
    if args.person_specific:
        train_loader, val_loader, test_loader = get_person_specific_loaders(
            args.data_dir, args.person_id, batch_size=args.batch_size, num_workers=args.num_workers
        )
    else:
        train_loader, val_loader, test_loader = get_data_loaders(
            args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
        )
    
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
    
    # Print model info
    if 'epoch' in checkpoint:
        print(f'Model trained for {checkpoint["epoch"]} epochs')
    if 'best_val_angle' in checkpoint:
        print(f'Best validation angle: {checkpoint["best_val_angle"]:.2f}°')
    
    # Loss
    criterion = nn.MSELoss()
    
    # Evaluate on test set
    results = evaluate(model, test_loader, criterion, device,
                      save_plots=not args.no_plots, save_dir=args.save_dir)
    
    # Save results
    import json
    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\nTest evaluation completed!')


if __name__ == '__main__':
    main()

