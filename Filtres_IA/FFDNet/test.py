"""
FFDNet Testing Script
Test trained model on test sets
"""

import os
import argparse
import torch
import csv
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from model import FFDNet
from utils import ValidationDataset, compute_psnr, compute_ssim
import config


def test_model(checkpoint_path, test_dir, noise_sigma, save_images=False):
    """
    Test model on a test set
    
    Parameters
    ----------
    checkpoint_path : str
        Path to model checkpoint
    test_dir : str
        Directory containing test images
    noise_sigma : float
        Noise level to test
    save_images : bool
        Whether to save denoised images
    
    Returns
    -------
    dict
        Dictionary containing test results
    """
    print(f"\nTesting on {os.path.basename(test_dir)} with sigma={noise_sigma}")
    print("-" * 70)
    
    # Setup device
    device = config.get_device()
    
    # Load model
    model = FFDNet(
        num_channels=config.NUM_CHANNELS,
        num_features=config.NUM_FEATURES,
        num_layers=config.NUM_LAYERS
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from: {checkpoint_path}")
    
    # Create dataset
    test_dataset = ValidationDataset(test_dir, noise_sigma=noise_sigma)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Test results
    results = []
    total_psnr = 0
    total_ssim = 0
    
    # Create output directory
    if save_images:
        output_dir = os.path.join(
            config.RESULTS_DIR,
            f"{os.path.basename(test_dir)}_sigma{noise_sigma}"
        )
        os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    with torch.no_grad():
        for idx, (noisy, clean, filename) in enumerate(test_loader):
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            # Denoise
            denoised = model(noisy, noise_sigma)
            
            # Calculate metrics
            psnr = compute_psnr(denoised, clean)
            ssim_val = compute_ssim(denoised, clean)
            
            total_psnr += psnr
            total_ssim += ssim_val
            
            # Store results
            results.append({
                'filename': filename,
                'psnr': psnr,
                'ssim': ssim_val
            })
            
            print(f"{idx+1}/{len(test_loader)} {filename}: "
                  f"PSNR={psnr:.2f} dB, SSIM={ssim_val:.4f}")
            
            # Save denoised image
            if save_images:
                denoised_np = denoised.squeeze().cpu().numpy()
                denoised_np = (denoised_np * 255).clip(0, 255).astype(np.uint8)
                
                output_path = os.path.join(output_dir, filename)
                Image.fromarray(denoised_np).save(output_path)
    
    # Calculate averages
    avg_psnr = total_psnr / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)
    
    print("-" * 70)
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    
    return {
        'test_dir': test_dir,
        'noise_sigma': noise_sigma,
        'num_images': len(test_loader),
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
        'per_image': results
    }


def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description='Test FFDNet')
    parser.add_argument('--checkpoint', type=str,
                        default=config.DEFAULT_CHECKPOINT,
                        help='Path to model checkpoint')
    parser.add_argument('--test_dir', type=str, default=None,
                        help='Test directory (default: Set12 and BSD68)')
    parser.add_argument('--noise_sigma', type=float, default=None,
                        help='Noise level (default: test multiple levels)')
    parser.add_argument('--save_images', action='store_true',
                        help='Save denoised images')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("FFDNet Testing")
    print("=" * 70)
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    # Determine test directories
    if args.test_dir is not None:
        test_dirs = [args.test_dir]
    else:
        test_dirs = [config.VAL_DIR, config.TEST_DIR]
    
    # Determine noise levels
    if args.noise_sigma is not None:
        noise_sigmas = [args.noise_sigma]
    else:
        noise_sigmas = config.TEST_NOISE_SIGMAS
    
    # Test on all combinations
    all_results = []
    
    for test_dir in test_dirs:
        if not os.path.exists(test_dir):
            print(f"Warning: Test directory not found: {test_dir}")
            continue
        
        for noise_sigma in noise_sigmas:
            results = test_model(
                args.checkpoint,
                test_dir,
                noise_sigma,
                args.save_images
            )
            all_results.append(results)
    
    # Save results to CSV
    csv_path = os.path.join(config.RESULTS_DIR, 'test_results.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Test Set', 'Noise Sigma', 'Num Images', 
                        'Avg PSNR', 'Avg SSIM'])
        
        for result in all_results:
            writer.writerow([
                os.path.basename(result['test_dir']),
                result['noise_sigma'],
                result['num_images'],
                f"{result['avg_psnr']:.2f}",
                f"{result['avg_ssim']:.4f}"
            ])
    
    print(f"\nResults saved to: {csv_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    for result in all_results:
        print(f"{os.path.basename(result['test_dir'])} "
              f"(sigma={result['noise_sigma']}): "
              f"PSNR={result['avg_psnr']:.2f} dB, "
              f"SSIM={result['avg_ssim']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()