"""
Métriques de qualité pour W-DENet
Auteur: Semih Aktepe
"""

import numpy as np
from scipy.stats import pearsonr


def calculate_psnr(image1, image2):
    """Calcule le PSNR entre deux images"""
    mse = np.mean((image1.astype(float) - image2.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def calculate_ssim(image1, image2):
    """Calcule le SSIM entre deux images (version simplifiée)"""
    img1_flat = image1.flatten().astype(float)
    img2_flat = image2.flatten().astype(float)
    correlation, _ = pearsonr(img1_flat, img2_flat)
    return float(max(0.0, correlation))