"""
FFDNet Training Script
Train FFDNet model on BSD400 dataset
"""

import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import FFDNet
from utils import (
    DenoisingDataset, 
    ValidationDataset,
    compute_psnr,
    compute_ssim,
    AverageMeter,
    save_checkpoint
)
import config


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train for one epoch
    
    Parameters
    ----------
    model : FFDNet
        Model to train
    train_loader : DataLoader
        Training data loader
    criterion : nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimizer
    device : torch.device
        Device to train on
    epoch : int
        Current epoch number
    
    Returns
    -------
    float
        Average training loss for this epoch
    """
    model.train()
    
    loss_meter = AverageMeter()
    
    for batch_idx, (noisy, clean, noise_sigma) in enumerate(train_loader):
        # Move to device and ensure float32
        noisy = noisy.float().to(device)
        clean = clean.float().to(device)
        
        # Convert noise_sigma to float (handle batch of different sigmas)
        if isinstance(noise_sigma, torch.Tensor):
            noise_sigma = noise_sigma[0].item()  # Take first value
        noise_sigma = float(noise_sigma)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(noisy, noise_sigma)
        
        # Calculate loss
        loss = criterion(output, clean)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update statistics
        loss_meter.update(loss.item(), noisy.size(0))
        
        # Print progress
        if (batch_idx + 1) % config.PRINT_EVERY == 0:
            print(f"Epoch [{epoch}/{config.NUM_EPOCHS}] "
                  f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.6f} "
                  f"Avg Loss: {loss_meter.avg:.6f}")
    
    return loss_meter.avg


def validate(model, val_loader, criterion, device):
    """
    Validate model on validation set
    
    Parameters
    ----------
    model : FFDNet
        Model to validate
    val_loader : DataLoader
        Validation data loader
    criterion : nn.Module
        Loss function
    device : torch.device
        Device
    
    Returns
    -------
    tuple
        (avg_loss, avg_psnr, avg_ssim)
    """
    model.eval()
    
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    
    with torch.no_grad():
        for noisy, clean, _ in val_loader:
            noisy = noisy.float().to(device)
            clean = clean.float().to(device)
            
            # Forward pass
            output = model(noisy, config.VAL_NOISE_SIGMA)
            
            # Calculate loss
            loss = criterion(output, clean)
            loss_meter.update(loss.item(), noisy.size(0))
            
            # Calculate metrics
            if config.COMPUTE_PSNR:
                psnr = compute_psnr(output, clean)
                psnr_meter.update(psnr)
            
            if config.COMPUTE_SSIM:
                ssim_val = compute_ssim(output, clean)
                ssim_meter.update(ssim_val)
    
    return loss_meter.avg, psnr_meter.avg, ssim_meter.avg


def train_ffdnet():
    """Main training function"""
    
    print("=" * 70)
    print("FFDNet Training")
    print("=" * 70)
    
    # Print configuration
    config.print_config()
    
    # Setup device
    device = config.get_device()
    
    # Create datasets
    print("\nCreating datasets...")
    
    train_dataset = DenoisingDataset(
        root_dir=config.TRAIN_DIRS if hasattr(config, 'TRAIN_DIRS') else config.TRAIN_DIR,
        patch_size=config.PATCH_SIZE,
        noise_sigma_range=(config.NOISE_SIGMA_MIN, config.NOISE_SIGMA_MAX),
        augment=config.USE_AUGMENTATION,
        num_patches=config.PATCHES_PER_IMAGE
    )
    
    val_dataset = ValidationDataset(
        root_dir=config.VAL_DIR,
        noise_sigma=config.VAL_NOISE_SIGMA
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print("\nCreating model...")
    model = FFDNet(
        num_channels=config.NUM_CHANNELS,
        num_features=config.NUM_FEATURES,
        num_layers=config.NUM_LAYERS,
        kernel_size=config.KERNEL_SIZE
    )
    model = model.to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Loss function
    if config.LOSS_FUNCTION == 'mse':
        criterion = nn.MSELoss()
    elif config.LOSS_FUNCTION == 'l1':
        criterion = nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss: {config.LOSS_FUNCTION}")
    
    # Optimizer
    if config.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=0.9,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.LR_DECAY_EPOCHS,
        gamma=config.LR_DECAY_RATE
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_psnr': [],
        'val_ssim': [],
        'learning_rate': []
    }
    
    # Best model tracking
    best_psnr = 0
    best_epoch = 0
    patience_counter = 0
    start_epoch = 1
    
    # Resume from checkpoint if specified
    if config.RESUME_TRAINING and config.RESUME_CHECKPOINT:
        checkpoint_path = os.path.join(config.BASE_DIR, config.RESUME_CHECKPOINT)
        if os.path.exists(checkpoint_path):
            print(f"\nResuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_psnr = checkpoint.get('val_psnr', 0)
            
            print(f"Resuming from epoch {start_epoch}")
            print(f"Best PSNR so far: {best_psnr:.2f} dB")
        else:
            print(f"\nWarning: Checkpoint not found: {checkpoint_path}")
            print("Starting from scratch...")
    
    # Start training
    print("\nStarting training...")
    print("=" * 70)
    
    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        if epoch % config.VALIDATE_EVERY == 0:
            val_loss, val_psnr, val_ssim = validate(
                model, val_loader, criterion, device
            )
        else:
            val_loss, val_psnr, val_ssim = 0, 0, 0
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)
        history['learning_rate'].append(current_lr)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print("\n" + "=" * 70)
        print(f"Epoch {epoch}/{config.NUM_EPOCHS} Summary:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Val PSNR: {val_psnr:.2f} dB")
        print(f"  Val SSIM: {val_ssim:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.1f}s")
        print("=" * 70 + "\n")
        
        # Save checkpoint
        if epoch % config.SAVE_CHECKPOINT_EVERY == 0:
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR,
                f'ffdnet_epoch_{epoch}.pth'
            )
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_psnr': val_psnr,
                'val_ssim': val_ssim
            }, checkpoint_path)
        
        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_epoch = epoch
            patience_counter = 0
            
            best_path = os.path.join(
                config.CHECKPOINT_DIR,
                'ffdnet_best.pth'
            )
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_psnr': val_psnr,
                'val_ssim': val_ssim
            }, best_path)
            
            print(f"New best model! PSNR: {val_psnr:.2f} dB\n")
        else:
            patience_counter += 1
        
        # Early stopping
        if config.USE_EARLY_STOPPING and patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"Best PSNR: {best_psnr:.2f} dB at epoch {best_epoch}")
            break
    
    # Save final model
    final_path = os.path.join(config.CHECKPOINT_DIR, 'ffdnet_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved: {final_path}")
    
    # Save training history
    if config.SAVE_HISTORY:
        with open(config.HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved: {config.HISTORY_FILE}")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best validation PSNR: {best_psnr:.2f} dB (epoch {best_epoch})")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train FFDNet')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    # Override config if arguments provided
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    
    # Start training
    train_ffdnet()