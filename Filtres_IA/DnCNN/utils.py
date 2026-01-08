"""
DnCNN Utilities - Optimized for Speed
Includes InMemoryDataset to solve Windows/CPU bottlenecks.
Fixed: Negative stride error in PyTorch
"""

import os
import glob
import random
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from skimage.metrics import structural_similarity as ssim

class InMemoryDataset(Dataset):
    """
    Tüm veri setini RAM'e yükler. Windows'ta num_workers=0 darboğazını çözer.
    """
    def __init__(self, root_dirs, patch_size=40, num_patches=64, sigma_range=(0, 55), augment=True):
        super(InMemoryDataset, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.sigma_range = sigma_range
        self.augment = augment
        self.images = []

        print(f"Loading datasets into RAM (this may take a moment)...")
        
        # Dosya yollarını topla
        file_paths = []
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
            
        for d in root_dirs:
            # PNG, JPG, BMP destekle
            file_paths.extend(glob.glob(os.path.join(d, "*.png")))
            file_paths.extend(glob.glob(os.path.join(d, "*.jpg")))
            file_paths.extend(glob.glob(os.path.join(d, "*.bmp")))

        # Resimleri oku ve RAM'e kaydet (Grayscale, 0-1 float32)
        for fp in file_paths:
            img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Normalize et ve float32 yap
                img = img.astype(np.float32) / 255.0
                self.images.append(img)
        
        print(f"✓ Loaded {len(self.images)} images into RAM.")

    def __len__(self):
        # Her epochta toplam patch sayısı
        return len(self.images) * self.num_patches

    def __getitem__(self, idx):
        # Rastgele bir resim seç
        img = random.choice(self.images)
        h, w = img.shape

        # Rastgele Patch Çıkar
        if h < self.patch_size or w < self.patch_size:
            img = cv2.resize(img, (self.patch_size, self.patch_size))
            h, w = self.patch_size, self.patch_size
            
        py = random.randint(0, h - self.patch_size)
        px = random.randint(0, w - self.patch_size)
        patch = img[py:py+self.patch_size, px:px+self.patch_size]

        # Data Augmentation
        if self.augment:
            mode = random.randint(0, 7)
            patch = self._augment_patch(patch, mode)

        # --- KRİTİK DÜZELTME BURADA ---
        # .copy() ekleyerek "negative strides" hatasını çözüyoruz
        patch_contiguous = patch.copy()
        clean = torch.from_numpy(patch_contiguous).unsqueeze(0)
        # ------------------------------

        # Gürültü Ekle
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        noise = torch.randn_like(clean) * (sigma / 255.0)
        noisy = clean + noise

        return noisy, clean, sigma

    def _augment_patch(self, img, mode):
        """
        0: orj, 1: flipLR, 2: flipUD, 3: rot90, 
        4: rot90+flipLR, 5: rot180, 6: rot180+flipLR, 7: rot270
        """
        if mode == 0: return img
        elif mode == 1: return np.flipud(img)
        elif mode == 2: return np.rot90(img)
        elif mode == 3: return np.flipud(np.rot90(img))
        elif mode == 4: return np.rot90(img, k=2)
        elif mode == 5: return np.flipud(np.rot90(img, k=2))
        elif mode == 6: return np.rot90(img, k=3)
        elif mode == 7: return np.flipud(np.rot90(img, k=3))
        return img

class ValidationDataset(Dataset):
    def __init__(self, root_dir, sigma=25):
        self.files = glob.glob(os.path.join(root_dir, "*.png")) + \
                     glob.glob(os.path.join(root_dir, "*.jpg"))
        self.sigma = sigma

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx], cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        clean = torch.from_numpy(img).unsqueeze(0)
        
        noise = torch.randn_like(clean) * (self.sigma / 255.0)
        noisy = clean + noise
        
        return noisy, clean

def compute_psnr(img1, img2):
    if isinstance(img1, torch.Tensor): img1 = img1.cpu().detach().numpy()
    if isinstance(img2, torch.Tensor): img2 = img2.cpu().detach().numpy()
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(1.0 / np.sqrt(mse))