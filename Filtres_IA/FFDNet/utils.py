"""
FFDNet Utilities
Dataset loading, data augmentation, and metrics
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim


class DenoisingDataset(Dataset):
    """
    Dataset for image denoising with variable noise levels
    
    Parameters
    ----------
    root_dir : str
        Path to directory containing images
    patch_size : int
        Size of image patches to extract
    noise_sigma_range : tuple
        Range of noise levels (min, max) for training
    augment : bool
        Whether to apply data augmentation
    num_patches : int
        Number of patches to extract per image per epoch
    """
    
    def __init__(
        self,
        root_dir,
        patch_size=40,
        noise_sigma_range=(5, 55),
        augment=True,
        num_patches=100
    ):
        """
        root_dir can be a string (single directory) or list of strings (multiple directories)
        """
        self.patch_size = patch_size
        self.noise_sigma_range = noise_sigma_range
        self.augment = augment
        self.num_patches = num_patches
        
        # Support single directory or list of directories
        if isinstance(root_dir, str):
            root_dirs = [root_dir]
        else:
            root_dirs = root_dir
        
        # Find all image files from all directories
        self.image_files = []
        for directory in root_dirs:
            self.image_files.extend(self._find_images(directory))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {root_dirs}")
        
        self.to_tensor = transforms.ToTensor()
    
    def _find_images(self, directory):
        """Find all image files in directory"""
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
        image_files = []
        
        for filename in os.listdir(directory):
            if filename.lower().endswith(valid_extensions):
                image_files.append(os.path.join(directory, filename))
        
        return sorted(image_files)
    
    def __len__(self):
        """Total number of patches per epoch"""
        return len(self.image_files) * self.num_patches
    
    def __getitem__(self, idx):
        """
        Get a training sample
        
        Returns
        -------
        noisy_patch : torch.Tensor
            Noisy image patch, shape (1, H, W)
        clean_patch : torch.Tensor
            Clean image patch, shape (1, H, W)
        noise_sigma : float
            Noise level used for this sample
        """
        # Select random image
        img_idx = np.random.randint(0, len(self.image_files))
        img_path = self.image_files[img_idx]
        
        # Load image as grayscale
        image = Image.open(img_path).convert('L')
        image = self.to_tensor(image)
        
        # Extract random patch
        clean_patch = self._extract_random_patch(image)
        
        # Apply augmentation
        if self.augment:
            clean_patch = self._augment(clean_patch)
        
        # Generate random noise level
        noise_sigma = np.random.uniform(
            self.noise_sigma_range[0],
            self.noise_sigma_range[1]
        )
        
        # Add Gaussian noise
        noise = torch.randn_like(clean_patch) * (noise_sigma / 255.0)
        noisy_patch = torch.clamp(clean_patch + noise, 0, 1)
        
        # Ensure float32 dtype
        noisy_patch = noisy_patch.float()
        clean_patch = clean_patch.float()
        
        return noisy_patch, clean_patch, float(noise_sigma)
    
    def _extract_random_patch(self, image):
        """Extract random patch from image"""
        _, h, w = image.shape
        
        # Resize if image too small
        if h < self.patch_size or w < self.patch_size:
            new_h = max(h, self.patch_size)
            new_w = max(w, self.patch_size)
            image = transforms.Resize((new_h, new_w))(image)
            _, h, w = image.shape
        
        # Random crop
        top = np.random.randint(0, h - self.patch_size + 1)
        left = np.random.randint(0, w - self.patch_size + 1)
        
        patch = image[:, top:top+self.patch_size, left:left+self.patch_size]
        
        return patch
    
    def _augment(self, patch):
        """Apply data augmentation"""
        # Random rotation (0, 90, 180, 270 degrees)
        k = np.random.randint(0, 4)
        if k > 0:
            patch = torch.rot90(patch, k, [1, 2])
        
        # Random horizontal flip
        if np.random.rand() > 0.5:
            patch = torch.flip(patch, [2])
        
        return patch


class ValidationDataset(Dataset):
    """
    Validation dataset without patching or augmentation
    
    Parameters
    ----------
    root_dir : str
        Path to directory containing validation images
    noise_sigma : float
        Fixed noise level for validation
    """
    
    def __init__(self, root_dir, noise_sigma=25):
        self.root_dir = root_dir
        self.noise_sigma = noise_sigma
        
        # Find all images
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
        self.image_files = []
        
        for filename in os.listdir(root_dir):
            if filename.lower().endswith(valid_extensions):
                self.image_files.append(os.path.join(root_dir, filename))
        
        self.image_files = sorted(self.image_files)
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {root_dir}")
        
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get validation sample
        
        Returns
        -------
        noisy : torch.Tensor
            Noisy image, shape (1, H, W)
        clean : torch.Tensor
            Clean image, shape (1, H, W)
        filename : str
            Image filename
        """
        img_path = self.image_files[idx]
        filename = os.path.basename(img_path)
        
        # Load as grayscale
        image = Image.open(img_path).convert('L')
        clean = self.to_tensor(image)
        
        # Add noise
        noise = torch.randn_like(clean) * (self.noise_sigma / 255.0)
        noisy = torch.clamp(clean + noise, 0, 1)
        
        # Ensure float32 dtype
        noisy = noisy.float()
        clean = clean.float()
        
        return noisy, clean, filename


def compute_psnr(img1, img2):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    
    Parameters
    ----------
    img1 : torch.Tensor or np.ndarray
        First image
    img2 : torch.Tensor or np.ndarray
        Second image
    
    Returns
    -------
    float
        PSNR value in dB
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Calculate MSE
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    # Assume images are in range [0, 1]
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr


def compute_ssim(img1, img2):
    """
    Calculate Structural Similarity Index (SSIM)
    
    Parameters
    ----------
    img1 : torch.Tensor or np.ndarray
        First image
    img2 : torch.Tensor or np.ndarray
        Second image
    
    Returns
    -------
    float
        SSIM value between -1 and 1 (higher is better)
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Remove batch and channel dimensions if present
    if img1.ndim == 4:
        img1 = img1[0, 0]
    elif img1.ndim == 3:
        img1 = img1[0]
    
    if img2.ndim == 4:
        img2 = img2[0, 0]
    elif img2.ndim == 3:
        img2 = img2[0]
    
    # Calculate SSIM
    ssim_value = ssim(img1, img2, data_range=1.0)
    
    return ssim_value


class AverageMeter:
    """
    Computes and stores the average and current value
    
    Useful for tracking metrics during training
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        """
        Update statistics
        
        Parameters
        ----------
        val : float
            New value to add
        n : int
            Number of items this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename):
    """
    Save model checkpoint
    
    Parameters
    ----------
    state : dict
        Dictionary containing model state and training info
    filename : str
        Path to save checkpoint
    """
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(filename, model, optimizer=None):
    """
    Load model checkpoint
    
    Parameters
    ----------
    filename : str
        Path to checkpoint file
    model : torch.nn.Module
        Model to load weights into
    optimizer : torch.optim.Optimizer, optional
        Optimizer to load state into
    
    Returns
    -------
    dict
        Checkpoint dictionary containing epoch, metrics, etc.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint not found: {filename}")
    
    checkpoint = torch.load(filename)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded: {filename}")
    
    return checkpoint


if __name__ == "__main__":
    print("=" * 70)
    print("Testing FFDNet Utilities")
    print("=" * 70)
    
    # Test metrics
    print("\nTesting metrics...")
    img1 = torch.rand(1, 1, 64, 64)
    img2 = img1 + torch.randn_like(img1) * 0.1
    
    psnr = compute_psnr(img1, img2)
    ssim_val = compute_ssim(img1, img2)
    
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")
    
    # Test AverageMeter
    print("\nTesting AverageMeter...")
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"Average: {meter.avg:.2f}")
    
    print("\nAll tests passed!")