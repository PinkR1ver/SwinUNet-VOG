"""
可视化工具：展示模型在测试图像上的预测效果
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from model import SwinUNet
from data import get_data_loaders, get_person_specific_loaders


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
    
    args = parser.parse_args()
    
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
    
    # Visualize
    print('Visualizing predictions...')
    visualize_predictions(model, test_loader, device, 
                         num_samples=args.num_samples, 
                         save_dir=args.save_dir)
    
    print('Visualization completed!')


if __name__ == '__main__':
    main()
