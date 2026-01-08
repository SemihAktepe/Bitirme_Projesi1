"""
DnCNN Inference Module
Eğitilmiş modeli kullanarak görüntüleri temizler.
"""

import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# İsim çakışmasını önlemek için özelleştirilmiş modül adından import yapıyoruz
from dncnn_model import DnCNN


class DnCNNDenoiser:
    """
    Üretim ortamı için DnCNN gürültü giderici sınıfı.
    Sınıf ismi PascalCase standardına uygundur[cite: 11].
    
    Kullanım:
    >>> denoiser = DnCNNDenoiser()
    >>> clean_image = denoiser.denoise(noisy_image)
    """
    
    def __init__(self, checkpoint_path=None, device=None):
        """
        Denoiser sınıfını başlatır ve modeli yükler.
        """
        # Varsayılan checkpoint yolu
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'checkpoints',
                'dncnn_best.pth'
            )
        
        # Cihaz seçimi (GPU/CPU)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() 
                                      else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Modeli yükle
        # Sabitler (num_layers vb.) burada doğrudan kullanılıyor
        self.model = DnCNN(
            num_channels=1,  # Gri tonlama
            num_layers=17,   # Standart DnCNN
            num_features=64  # Standart özellikler
        )
        
        if os.path.exists(checkpoint_path):
            try:
                # PyTorch sürüm uyumluluğu için try-except bloğu [cite: 20]
                checkpoint = torch.load(checkpoint_path, map_location=self.device,
                                       weights_only=False)
            except TypeError:
                # Eski PyTorch sürümleri için
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            print(f"DnCNN model loaded from: {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            print("Using randomly initialized model!")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
    
    def denoise(self, image):
        """
        Tek bir görüntüyü temizler.
        
        Parametreler:
        -----------
        image : PIL.Image veya np.ndarray
            Giriş görüntüsü
            
        Dönüş:
        --------
        PIL.Image
            Temizlenmiş görüntü
        """
        # Görüntüyü gri tonlamalı tensöre çevir
        # Mantıksal bloklar ayrılmıştır [cite: 20]
        if isinstance(image, Image.Image):
            is_rgb = (image.mode == 'RGB')
            if is_rgb:
                image_gray = image.convert('L')
            else:
                image_gray = image
            
            img_tensor = self.to_tensor(image_gray).unsqueeze(0)
        
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB'den griye dönüşüm
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
        
        # Inference (Gürültü giderme)
        with torch.no_grad():
            output = self.model(img_tensor)
        
        # Sonucu PIL formatına geri çevir
        output = output.squeeze().cpu()
        output = torch.clamp(output, 0, 1)
        denoised = self.to_pil(output)
        
        return denoised
    
    def denoise_rgb(self, image_rgb):
        """
        RGB görüntüyü her kanalı ayrı ayrı işleyerek temizler.
        Fonksiyon adı küçük harf ve alt çizgi standardına uygundur[cite: 10].
        """
        if isinstance(image_rgb, Image.Image):
            r, g, b = image_rgb.split()
        elif isinstance(image_rgb, np.ndarray):
            r = Image.fromarray(image_rgb[:,:,0])
            g = Image.fromarray(image_rgb[:,:,1])
            b = Image.fromarray(image_rgb[:,:,2])
        else:
            raise TypeError("RGB image must be PIL.Image or numpy array")
        
        # Her kanalı ayrı ayrı temizle
        r_denoised = self.denoise(r)
        g_denoised = self.denoise(g)
        b_denoised = self.denoise(b)
        
        # Kanalları birleştir
        return Image.merge('RGB', (r_denoised, g_denoised, b_denoised))
    
    def denoise_file(self, input_path, output_path):
        """
        Dosya yolundan görüntü okur, temizler ve kaydeder.
        """
        # Görüntüyü yükle
        image = Image.open(input_path)
        
        # Temizle
        if image.mode == 'RGB':
            denoised = self.denoise_rgb(image)
        else:
            denoised = self.denoise(image)
        
        # Kaydet
        denoised.save(output_path)
        print(f"Denoised image saved: {output_path}")


def batch_denoise(input_dir, output_dir, checkpoint_path=None):
    """
    Bir klasördeki tüm görüntüleri toplu olarak işler.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Denoiser nesnesini başlat
    denoiser = DnCNNDenoiser(checkpoint_path)
    
    # Dosyaları bul
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(valid_extensions)]
    
    print(f"Found {len(image_files)} images to process")
    
    # Her görüntüyü işle
    for idx, filename in enumerate(image_files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        print(f"Processing {idx+1}/{len(image_files)}: {filename}")
        denoiser.denoise_file(input_path, output_path)
    
    print(f"\nAll images denoised and saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Denoise images with DnCNN')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output image or directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint path')
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        batch_denoise(args.input, args.output, args.checkpoint)
    else:
        denoiser = DnCNNDenoiser(args.checkpoint)
        denoiser.denoise_file(args.input, args.output)