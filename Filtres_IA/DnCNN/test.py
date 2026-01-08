"""
DnCNN ACADEMIC BENCHMARK (BSD68)
Reference: Zhang et al. (2017) IEEE TIP
"""

import os
import glob
import time
import numpy as np
import torch
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# Import modules
try:
    from Filtres_IA.DnCNN.dncnn_model import DnCNN
    import config
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from Filtres_IA.DnCNN.dncnn_model import DnCNN
    import config

def load_checkpoint(model, device):
    """Loads the best model checkpoint (dncnn_best.pth)"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(base_dir, 'checkpoints', 'dncnn_best.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
        return model
    
    try:
        # Safe load for different PyTorch versions
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
            
    model.eval()
    return model

def add_noise(image, sigma):
    """Adds Gaussian noise to the image"""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.float32)

def run_benchmark():
    print("\n" + "="*65)
    print("DnCNN ACADEMIC BENCHMARK TEST (BSD68 DATASET)")
    print("Reference: Zhang et al. (2017)")
    print("="*65)

    # 1. Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Path Setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.abspath(os.path.join(current_dir, '..', 'denoising-datasets-main'))
    test_path = os.path.join(dataset_root, 'BSD68', 'original')
    
    files = glob.glob(os.path.join(test_path, '*.png')) + glob.glob(os.path.join(test_path, '*.jpg'))
    
    if len(files) == 0:
        print(f"[ERROR] Test images not found!")
        print(f"Path searched: {test_path}")
        return

    print(f"Test Device       : {device}")
    print(f"Total Images      : {len(files)} (BSD68)")
    print("-" * 65)

    # 3. Initialize Model
    model = DnCNN(num_channels=1, num_layers=17, num_features=64).to(device)
    model = load_checkpoint(model, device)

    # 4. Test Loop
    noise_levels = [15, 25, 50]
    
    print(f"\n{'Noise (Sigma)':<15} | {'Avg PSNR (dB)':<15} | {'Avg SSIM':<15} | {'Time (s)':<10}")
    print("-" * 65)

    for sigma in noise_levels:
        avg_psnr = 0
        avg_ssim = 0
        start_time = time.time()

        for file in files:
            img_org = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            if img_org is None: continue
            
            img_org = np.float32(img_org)
            img_noisy = add_noise(img_org, sigma)

            img_noisy_norm = img_noisy / 255.0
            img_tensor = torch.from_numpy(img_noisy_norm).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                # Correct PyTorch Mixed Precision syntax to avoid FutureWarning
                if hasattr(config, 'USE_MIXED_PRECISION') and config.USE_MIXED_PRECISION:
                    with torch.amp.autocast('cuda'):
                        prediction = model(img_tensor)
                else:
                    prediction = model(img_tensor)

            prediction = torch.clamp(prediction, 0., 1.)
            img_denoised = prediction.cpu().numpy().squeeze() * 255.0

            cur_psnr = compare_psnr(img_org, img_denoised, data_range=255)
            cur_ssim = compare_ssim(img_org, img_denoised, data_range=255)

            avg_psnr += cur_psnr
            avg_ssim += cur_ssim

        avg_psnr /= len(files)
        avg_ssim /= len(files)
        elapsed = time.time() - start_time

        print(f"Sigma = {sigma:<7} | {avg_psnr:<15.2f} | {avg_ssim:<15.4f} | {elapsed:<10.2f}")

    print("-" * 65)
    print("Test Completed.")

if __name__ == "__main__":
    run_benchmark()