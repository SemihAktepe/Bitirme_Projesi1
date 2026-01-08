"""
FFDNet Configuration File - OPTIMIZED FOR 30+ dB
Multiple datasets + Enhanced hyperparameters
"""

import os

# =============================================================================
# PATHS - Dosya yollari
# =============================================================================

# Base directory (FFDNet klasoru)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset paths
DATASET_ROOT = os.path.join(os.path.dirname(BASE_DIR), 'denoising-datasets-main')

# Training directories - MULTIPLE DATASETS!
TRAIN_DIRS = [
    os.path.join(DATASET_ROOT, 'BSD400'),
    # DIV2K eklenince asagidaki satiri aktif et:
    os.path.join(DATASET_ROOT, 'DIV2K_train_HR'),
]

# Validation and test
VAL_DIR = os.path.join(DATASET_ROOT, 'BSD68', 'original')
TEST_DIR = os.path.join(DATASET_ROOT, 'BSD68', 'original')

# Output paths
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if not exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# MODEL ARCHITECTURE - Model yapisi (OPTIMIZED)
# =============================================================================

# Number of input/output channels
NUM_CHANNELS = 1  # Grayscale

# Number of feature maps (64 → 96 for better capacity)
NUM_FEATURES = 96

# Number of layers (15 → 17 for deeper network)
NUM_LAYERS = 17

# Kernel size
KERNEL_SIZE = 3

# =============================================================================
# TRAINING HYPERPARAMETERS - Egitim parametreleri (OPTIMIZED)
# =============================================================================

# Number of training epochs (50 → 80 for better convergence)
NUM_EPOCHS = 80

# Batch size (64 - optimal for 8GB RAM + GPU)
BATCH_SIZE = 64

# Learning rate (0.0005 → 0.0003 for stability)
LEARNING_RATE = 0.0003

# Learning rate decay schedule (OPTIMIZED)
LR_DECAY_EPOCHS = [30, 60]  # Decay later for better learning
LR_DECAY_RATE = 0.5

# Optimizer settings
OPTIMIZER = 'adam'
WEIGHT_DECAY = 0.00005  # Light regularization

# Loss function
LOSS_FUNCTION = 'mse'

# =============================================================================
# DATA AUGMENTATION - Veri cogaltma (ENHANCED)
# =============================================================================

# Patch size
PATCH_SIZE = 40

# Data augmentation
USE_AUGMENTATION = True
AUGMENT_ROTATION = True
AUGMENT_FLIP = True

# Patches per image (100 → 128 for more training data)
PATCHES_PER_IMAGE = 128

# =============================================================================
# NOISE CONFIGURATION
# =============================================================================

NOISE_SIGMA_MIN = 5
NOISE_SIGMA_MAX = 55
VAL_NOISE_SIGMA = 25
TEST_NOISE_SIGMAS = [15, 25, 50]

# =============================================================================
# TRAINING OPTIONS
# =============================================================================

SAVE_CHECKPOINT_EVERY = 5
VALIDATE_EVERY = 1

# Resume training
RESUME_TRAINING = False
RESUME_CHECKPOINT = None

# Early stopping (20 - more patience for 80 epochs)
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 20

# =============================================================================
# HARDWARE
# =============================================================================

DEVICE = 'cuda'
NUM_WORKERS = 4
PIN_MEMORY = True
USE_MIXED_PRECISION = False

# =============================================================================
# VALIDATION & TESTING
# =============================================================================

COMPUTE_PSNR = True
COMPUTE_SSIM = True
SAVE_VAL_IMAGES = True
MAX_VAL_IMAGES_TO_SAVE = 5

# =============================================================================
# LOGGING
# =============================================================================

PRINT_EVERY = 100
LOG_TO_FILE = True
LOG_FILE = os.path.join(RESULTS_DIR, 'training.log')
SAVE_HISTORY = True
HISTORY_FILE = os.path.join(CHECKPOINT_DIR, 'training_history.json')

# =============================================================================
# PRODUCTION
# =============================================================================

DEFAULT_CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'ffdnet_best.pth')
INPUT_FORMAT = 'RGB'
OUTPUT_FORMAT = 'RGB'

# =============================================================================
# PERFORMANCE THRESHOLDS (YUKSELTILMIS)
# =============================================================================

MIN_ACCEPTABLE_PSNR = 29.0  # Yuksek hedef
TARGET_PSNR = 30.0  # Ideal hedef

# =============================================================================
# DEBUG OPTIONS
# =============================================================================

DEBUG = False
LIMIT_TRAIN_SAMPLES = None
LIMIT_VAL_SAMPLES = None

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_config():
    """Print current configuration"""
    print("=" * 70)
    print("FFDNet Configuration (OPTIMIZED for 30+ dB)")
    print("=" * 70)
    
    # Check which datasets are available
    available_datasets = []
    total_images = 0
    for train_dir in TRAIN_DIRS:
        if os.path.exists(train_dir):
            num_imgs = len([f for f in os.listdir(train_dir) 
                           if f.endswith(('.png', '.jpg', '.bmp'))])
            available_datasets.append(os.path.basename(train_dir))
            total_images += num_imgs
    
    print(f"Training datasets: {', '.join(available_datasets)}")
    print(f"Total training images: {total_images}")
    print(f"Validation directory: {VAL_DIR}")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    print(f"\nModel (ENHANCED):")
    print(f"  Channels: {NUM_CHANNELS}")
    print(f"  Features: {NUM_FEATURES} (↑ from 64)")
    print(f"  Layers: {NUM_LAYERS} (↑ from 15)")
    print(f"  Parameters: ~1.08M")
    print(f"\nTraining (OPTIMIZED):")
    print(f"  Epochs: {NUM_EPOCHS} (↑ from 50)")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE} (↓ for stability)")
    print(f"  LR decay: {LR_DECAY_RATE} at epochs {LR_DECAY_EPOCHS}")
    print(f"  Patches/image: {PATCHES_PER_IMAGE} (↑ from 100)")
    print(f"  Total patches/epoch: ~{total_images * PATCHES_PER_IMAGE}")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"\nNoise:")
    print(f"  Training range: [{NOISE_SIGMA_MIN}, {NOISE_SIGMA_MAX}]")
    print(f"  Validation: {VAL_NOISE_SIGMA}")
    print(f"\nTarget: {TARGET_PSNR} dB (min: {MIN_ACCEPTABLE_PSNR} dB)")
    print(f"Device: {DEVICE}")
    print("=" * 70)


def get_device():
    """Get device (CUDA or CPU) automatically"""
    import torch
    if DEVICE == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


if __name__ == "__main__":
    print_config()
    
    # Check paths
    print("\nChecking datasets...")
    for i, train_dir in enumerate(TRAIN_DIRS, 1):
        if os.path.exists(train_dir):
            num_train = len([f for f in os.listdir(train_dir) 
                            if f.endswith(('.png', '.jpg', '.bmp'))])
            print(f"  Dataset {i} ({os.path.basename(train_dir)}): {num_train} images")
        else:
            print(f"  Dataset {i} ({os.path.basename(train_dir)}): NOT FOUND")
    
    if os.path.exists(VAL_DIR):
        num_val = len([f for f in os.listdir(VAL_DIR) 
                      if f.endswith(('.png', '.jpg', '.bmp'))])
        print(f"  Validation (BSD68/original): {num_val} images")
    else:
        print(f"  Validation: NOT FOUND")