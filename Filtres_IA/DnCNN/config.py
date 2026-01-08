"""
DnCNN Configuration - ACADEMIC HIGH PERFORMANCE MODE
Optimized for Windows + GTX 1650 (Memory Efficient)
Targeting Zhang et al. 2017 Results
"""

import os
import torch

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(os.path.dirname(BASE_DIR), 'denoising-datasets-main')

TRAIN_DIRS = [
    os.path.join(DATASET_ROOT, 'BSD400'),
    os.path.join(DATASET_ROOT, 'DIV2K_train_HR'),
]

VAL_DIR = os.path.join(DATASET_ROOT, 'BSD68', 'original')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# PERFORMANCE OPTIMIZATIONS
# =============================================================================
LOAD_DATA_TO_RAM = True  
NUM_WORKERS = 0          
PIN_MEMORY = True        
USE_MIXED_PRECISION = True 

# =============================================================================
# MODEL PARAMETERS
# =============================================================================
NUM_CHANNELS = 1      
NUM_LAYERS = 17       
NUM_FEATURES = 64     
KERNEL_SIZE = 3

# =============================================================================
# TRAINING SETTINGS
# =============================================================================
NUM_EPOCHS = 80       
BATCH_SIZE = 64       
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Learning Rate Schedule
LR_STEPS = [40, 60]   
LR_GAMMA = 0.1        

# Noise Settings
NOISE_SIGMA_MIN = 0
NOISE_SIGMA_MAX = 55
VAL_NOISE_SIGMA = 25

# Data Augmentation
PATCH_SIZE = 40
PATCHES_PER_IMAGE = 64 

# =============================================================================
# LOGGING
# =============================================================================
PRINT_EVERY = 50      
SAVE_EVERY = 5        
TARGET_PSNR = 29.19   

def estimate_parameters():
    # 1st layer: 3*3*1*64 = 576
    # Mid layers (15): 15 * 3*3*64*64 = 552960
    # Last layer: 3*3*64*1 = 576
    # BN params: 15 * 64 * 2 = 1920
    return 556032

def print_config():
    print("=" * 70)
    print("DnCNN Configuration (OPTIMIZED for Zhang et al. 2017)")
    print("=" * 70)
    
    # Check available datasets
    total_images = 0
    dataset_names = []
    for train_dir in TRAIN_DIRS:
        if os.path.exists(train_dir):
            num_imgs = len([f for f in os.listdir(train_dir) 
                           if f.endswith(('.png', '.jpg', '.bmp'))])
            dataset_names.append(os.path.basename(train_dir))
            total_images += num_imgs
    
    print(f"Training datasets   : {', '.join(dataset_names)}")
    print(f"Total training imgs : {total_images}")
    print(f"Validation dir      : {VAL_DIR}")
    print(f"Checkpoint dir      : {CHECKPOINT_DIR}")
    
    print(f"\nModel Architecture:")
    print(f"  Channels          : {NUM_CHANNELS}")
    print(f"  Features          : {NUM_FEATURES}")
    print(f"  Layers            : {NUM_LAYERS}")
    print(f"  Parameters        : ~{estimate_parameters():,}")
    
    print(f"\nTraining Settings (High Performance):")
    print(f"  Epochs            : {NUM_EPOCHS}")
    print(f"  Batch size        : {BATCH_SIZE}")
    print(f"  Learning rate     : {LEARNING_RATE}")
    print(f"  LR decay          : {LR_GAMMA} at epochs {LR_STEPS}")
    print(f"  Patches/image     : {PATCHES_PER_IMAGE}")
    print(f"  Total samples     : {total_images * PATCHES_PER_IMAGE:,} patches per epoch")
    
    print(f"\nHardware Optimization:")
    print(f"  Device            : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  RAM Pre-loading   : {LOAD_DATA_TO_RAM} (Bypasses disk I/O bottleneck)")
    print(f"  Mixed Precision   : {USE_MIXED_PRECISION} (FP16)")
    
    print(f"\nTarget Performance:")
    print(f"  Target PSNR       : {TARGET_PSNR} dB (BSD68, Sigma=25)")
    print("=" * 70)