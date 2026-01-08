"""
DnCNN Training Script - FFDNet Style Output
Optimized for Windows + GPU Speed
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler # PyTorch Mixed Precision

import config
from Filtres_IA.DnCNN.dncnn_model import DnCNN
from utils import InMemoryDataset, ValidationDataset, compute_psnr

def train():
    config.print_config()
    
    # 1. Donanım Hazırlığı
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 2. Veri Seti (RAM'e Yükleme - Hızın Sırrı Burada)
    train_dataset = InMemoryDataset(
        root_dirs=config.TRAIN_DIRS,
        patch_size=config.PATCH_SIZE,
        num_patches=config.PATCHES_PER_IMAGE,
        sigma_range=(config.NOISE_SIGMA_MIN, config.NOISE_SIGMA_MAX)
    )
    
    val_dataset = ValidationDataset(config.VAL_DIR, sigma=config.VAL_NOISE_SIGMA)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
                              num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
                              
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # 3. Model Kurulumu
    model = DnCNN(num_layers=config.NUM_LAYERS, num_features=config.NUM_FEATURES).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # Learning Rate Planlayıcı
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.LR_STEPS, gamma=config.LR_GAMMA)
    scaler = GradScaler('cuda') # FP16 Hızlandırma
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Starting training for {config.NUM_EPOCHS} epochs...\n")

    best_psnr = 0.0

    # 4. Eğitim Döngüsü
    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        epoch_start = time.time()
        epoch_loss = 0
        
        for i, (noisy, clean, _) in enumerate(train_loader):
            noisy, clean = noisy.to(device), clean.to(device)
            
            optimizer.zero_grad()
            
            # Mixed Precision Forward
            with autocast(device_type='cuda', enabled=config.USE_MIXED_PRECISION):
                output = model(noisy)
                loss = criterion(output, clean)
            
            # Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            # Terminal Çıktısı (FFDNet Stili)
            if (i + 1) % config.PRINT_EVERY == 0:
                print(f"[Epoch {epoch}/{config.NUM_EPOCHS}][Batch {i+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.6f} LR: {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step()
        
        # Validation
        model.eval()
        psnr_avg = 0
        with torch.no_grad():
            for v_noisy, v_clean in val_loader:
                v_noisy, v_clean = v_noisy.to(device), v_clean.to(device)
                v_out = model(v_noisy)
                # Orijinal aralığa (0-1) clamp et
                v_out = torch.clamp(v_out, 0., 1.)
                psnr_avg += compute_psnr(v_out, v_clean)
        
        psnr_avg /= len(val_dataset)
        time_elapsed = time.time() - epoch_start
        
        print(f"\n----------------------------------------------------------------")
        print(f"Epoch: {epoch} Complete in {time_elapsed:.0f}s")
        print(f"Avg Train Loss: {epoch_loss / len(train_loader):.6f}")
        print(f"Validation PSNR: {psnr_avg:.2f} dB (Target: {config.TARGET_PSNR} dB)")
        print(f"----------------------------------------------------------------\n")
        
        # Checkpoint
        if psnr_avg > best_psnr:
            best_psnr = psnr_avg
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, "dncnn_best.pth"))
            print(f"★ New Best Model Saved! ({best_psnr:.2f} dB)")
            
        if epoch % config.SAVE_EVERY == 0:
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, f"dncnn_epoch_{epoch}.pth"))

if __name__ == "__main__":
    train()