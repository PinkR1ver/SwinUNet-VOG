"""
Training script for SwinUNet gaze estimation model.
"""

import os
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from model import SwinUNet
from data import get_data_loaders, get_person_specific_loaders


def angle_error(pred, target):
    """
    Calculate angular error between predicted and target gaze vectors.
    
    Args:
        pred: Predicted gaze vector (batch_size, 3)
        target: Target gaze vector (batch_size, 3)
    
    Returns:
        Angular error in degrees
    """
    # Normalize vectors
    pred_norm = pred / torch.norm(pred, dim=1, keepdim=True)
    target_norm = target / torch.norm(target, dim=1, keepdim=True)
    
    # Calculate dot product
    dot_product = torch.sum(pred_norm * target_norm, dim=1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # Calculate angle in radians and convert to degrees
    angle = torch.acos(dot_product) * 180.0 / 3.14159
    return angle.mean()


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, log_interval=500):
    """Train one epoch."""
    model.train()
    
    running_loss = 0.0
    running_angle_error = 0.0
    num_samples = 0
    
    for batch_idx, (images, gaze, _) in enumerate(train_loader):
        images, gaze = images.to(device), gaze.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(images)
        
        # Calculate loss
        loss = criterion(output, gaze)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        angle_err = angle_error(output, gaze)
        running_angle_error += angle_err.item()
        num_samples += images.size(0)
        
        if batch_idx % log_interval == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.6f}, Angle: {angle_err.item():.2f}°')
    
    avg_loss = running_loss / len(train_loader)
    avg_angle = running_angle_error / len(train_loader)
    
    return avg_loss, avg_angle


def validate(model, val_loader, criterion, device):
    """Validate on validation set."""
    model.eval()
    
    running_loss = 0.0
    running_angle_error = 0.0
    
    with torch.no_grad():
        for images, gaze, _ in val_loader:
            images, gaze = images.to(device), gaze.to(device)
            
            output = model(images)
            loss = criterion(output, gaze)
            angle_err = angle_error(output, gaze)
            
            running_loss += loss.item()
            running_angle_error += angle_err.item()
    
    avg_loss = running_loss / len(val_loader)
    avg_angle = running_angle_error / len(val_loader)
    
    return avg_loss, avg_angle


def load_config(config_path='config.json'):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train SwinUNet for Gaze Estimation')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to normalized data (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu, overrides config)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of workers for data loading (overrides config)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save checkpoints (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (overrides config)')
    parser.add_argument('--person_specific', action='store_true',
                        help='Use person-specific evaluation protocol')
    parser.add_argument('--person_id', type=str, default='p00',
                        help='Person ID for person-specific evaluation')
    
    args = parser.parse_args()
    
    # Load config
    print(f'Loading configuration from {args.config}...')
    config = load_config(args.config)
    
    # Override with command-line arguments
    data_dir = args.data_dir if args.data_dir is not None else config['data']['data_dir']
    batch_size = args.batch_size if args.batch_size is not None else config['data']['batch_size']
    epochs = args.epochs if args.epochs is not None else config['training']['epochs']
    lr = args.lr if args.lr is not None else config['training']['lr']
    num_workers = args.num_workers if args.num_workers is not None else config['data']['num_workers']
    save_dir = args.save_dir if args.save_dir is not None else config['training']['save_dir']
    resume = args.resume if args.resume is not None else config['training']['resume']
    
    # Set device
    if args.device is not None:
        device_str = args.device
    elif config['hardware']['device'] == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_str = config['hardware']['device']
    
    device = torch.device(device_str)
    print(f'Using device: {device}')
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    print('Loading data...')
    if args.person_specific:
        train_loader, val_loader, test_loader = get_person_specific_loaders(
            data_dir, args.person_id, batch_size=batch_size, num_workers=num_workers
        )
    else:
        train_loader, val_loader, test_loader = get_data_loaders(
            data_dir, batch_size=batch_size, num_workers=num_workers, 
            train_augment=config['data']['train_augment']
        )
    
    # Create model
    print('Creating model...')
    model_params = config['model']
    model = SwinUNet(
        img_size=tuple(model_params['img_size']),
        in_chans=model_params['in_chans'],
        embed_dim=model_params['embed_dim'],
        depths=model_params['depths'],
        num_heads=model_params['num_heads'],
        window_size=model_params['window_size'],
        drop_rate=model_params['drop_rate']
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss() if config['training']['loss'] == 'MSE' else nn.MSELoss()
    
    # Optimizer
    optim_params = config['training']['optimizer_params'].copy()
    if 'lr' in optim_params:
        optim_params['lr'] = lr
    else:
        optim_params['lr'] = lr
    
    if config['training']['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), **optim_params)
    elif config['training']['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), **optim_params)
    elif config['training']['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), **optim_params)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Scheduler
    scheduler_params = config['training']['lr_scheduler_params'].copy()
    if config['training']['lr_scheduler'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, **scheduler_params)
    elif config['training']['lr_scheduler'] == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    print(f'Configuration summary:')
    print(f'  Model: SwinUNet with {sum(p.numel() for p in model.parameters()):,} parameters')
    print(f'  Optimizer: {config["training"]["optimizer"]} with lr={lr}')
    print(f'  Scheduler: {config["training"]["lr_scheduler"]}')
    print(f'  Batch size: {batch_size}')
    print(f'  Epochs: {epochs}')
    print(f'  Save dir: {save_dir}')
    print('-' * 80)
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_angle = float('inf')
    
    if resume:
        print(f'Resuming from {resume}')
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_angle = checkpoint['best_val_angle']
    
    # Training loop
    print('Starting training...')
    train_losses = []
    val_losses = []
    train_angles = []
    val_angles = []
    log_interval = config['training']['log_interval']
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_angle = train_epoch(model, train_loader, criterion, optimizer, device, epoch, log_interval)
        
        # Validate
        val_loss, val_angle = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_angle)
        
        # Statistics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_angles.append(train_angle)
        val_angles.append(val_angle)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{epochs} Summary:')
        print(f'  Train Loss: {train_loss:.6f}, Train Angle: {train_angle:.2f}°')
        print(f'  Val Loss: {val_loss:.6f}, Val Angle: {val_angle:.2f}°')
        print(f'  Time: {epoch_time:.2f}s')
        print('-' * 80)
        
        # Save checkpoint
        is_best = val_angle < best_val_angle
        if is_best:
            best_val_angle = val_angle
            print(f'New best validation angle: {best_val_angle:.2f}°')
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_angle': best_val_angle,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_angles': train_angles,
            'val_angles': val_angles
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_latest.pth'))
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_best.pth'))
            print(f'Best model saved to {save_dir}/checkpoint_best.pth')
    
    print(f'\nTraining completed! Best validation angle: {best_val_angle:.2f}°')


if __name__ == '__main__':
    main()

