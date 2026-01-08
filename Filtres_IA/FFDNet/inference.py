"""
FFDNet Inference Module
Use trained model for denoising images
"""

import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from model import FFDNet
import config


class FFDNetDenoiser:
    """
    FFDNet denoiser for production use
    
    Parameters
    ----------
    checkpoint_path : str, optional
        Path to model checkpoint (default: use best model)
    device : str, optional
        Device to run on ('cuda' or 'cpu')
    
    Examples
    --------
    >>> denoiser = FFDNetDenoiser()
    >>> noisy_image = Image.open('noisy.png')
    >>> clean_image = denoiser.denoise(noisy_image, noise_sigma=25)
    >>> clean_image.save('clean.png')
    """
    
    def __init__(self, checkpoint_path=None, device=None):
        if checkpoint_path is None:
            checkpoint_path = config.DEFAULT_CHECKPOINT
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() 
                                      else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = FFDNet(
            num_channels=config.NUM_CHANNELS,
            num_features=config.NUM_FEATURES,
            num_layers=config.NUM_LAYERS
        )
        
        if os.path.exists(checkpoint_path):
            # FIX: Add weights_only=False for PyTorch 2.6+
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            print(f"Model loaded from: {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            print("Using randomly initialized model!")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
    
    def denoise(self, image, noise_sigma=25):
        """
        Denoise a single image
        
        Parameters
        ----------
        image : PIL.Image or np.ndarray
            Input image (grayscale or RGB)
        noise_sigma : float
            Noise level (0-255 scale)
        
        Returns
        -------
        PIL.Image
            Denoised image
        """
        # Convert to grayscale tensor
        if isinstance(image, Image.Image):
            is_rgb = (image.mode == 'RGB')
            if is_rgb:
                image_gray = image.convert('L')
            else:
                image_gray = image
            
            img_tensor = self.to_tensor(image_gray).unsqueeze(0)
        
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])
                is_rgb = True
            else:
                image_gray = image
                is_rgb = False
            
            img_tensor = torch.from_numpy(image_gray).float().unsqueeze(0).unsqueeze(0)
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor / 255.0
        
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        img_tensor = img_tensor.to(self.device)
        
        # Denoise
        with torch.no_grad():
            output = self.model(img_tensor, noise_sigma)
        
        # Convert back to PIL
        output = output.squeeze().cpu()
        output = torch.clamp(output, 0, 1)
        denoised = self.to_pil(output)
        
        return denoised
    
    def denoise_rgb(self, image_rgb, noise_sigma=25):
        """
        Denoise RGB image by processing each channel separately
        
        Parameters
        ----------
        image_rgb : PIL.Image or np.ndarray
            RGB input image
        noise_sigma : float
            Noise level
        
        Returns
        -------
        PIL.Image
            Denoised RGB image
        """
        if isinstance(image_rgb, Image.Image):
            r, g, b = image_rgb.split()
        elif isinstance(image_rgb, np.ndarray):
            r = Image.fromarray(image_rgb[:,:,0])
            g = Image.fromarray(image_rgb[:,:,1])
            b = Image.fromarray(image_rgb[:,:,2])
        else:
            raise TypeError("RGB image must be PIL.Image or numpy array")
        
        # Denoise each channel
        r_denoised = self.denoise(r, noise_sigma)
        g_denoised = self.denoise(g, noise_sigma)
        b_denoised = self.denoise(b, noise_sigma)
        
        # Merge channels
        return Image.merge('RGB', (r_denoised, g_denoised, b_denoised))
    
    def denoise_file(self, input_path, output_path, noise_sigma=25):
        """
        Denoise an image file
        
        Parameters
        ----------
        input_path : str
            Path to input noisy image
        output_path : str
            Path to save denoised image
        noise_sigma : float
            Noise level
        """
        # Load image
        image = Image.open(input_path)
        
        # Denoise
        if image.mode == 'RGB':
            denoised = self.denoise_rgb(image, noise_sigma)
        else:
            denoised = self.denoise(image, noise_sigma)
        
        # Save
        denoised.save(output_path)
        print(f"Denoised image saved: {output_path}")


def batch_denoise(input_dir, output_dir, noise_sigma=25, checkpoint_path=None):
    """
    Denoise all images in a directory
    
    Parameters
    ----------
    input_dir : str
        Directory containing noisy images
    output_dir : str
        Directory to save denoised images
    noise_sigma : float
        Noise level
    checkpoint_path : str, optional
        Path to model checkpoint
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize denoiser
    denoiser = FFDNetDenoiser(checkpoint_path)
    
    # Find all images
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(valid_extensions)]
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for idx, filename in enumerate(image_files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        print(f"Processing {idx+1}/{len(image_files)}: {filename}")
        denoiser.denoise_file(input_path, output_path, noise_sigma)
    
    print(f"\nAll images denoised and saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Denoise images with FFDNet')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output image or directory')
    parser.add_argument('--sigma', type=float, default=25,
                        help='Noise level (default: 25)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint path')
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        # Batch processing
        batch_denoise(args.input, args.output, args.sigma, args.checkpoint)
    else:
        # Single image
        denoiser = FFDNetDenoiser(args.checkpoint)
        denoiser.denoise_file(args.input, args.output, args.sigma)