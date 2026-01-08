"""
Bruit Speckle (Multiplicatif)
"""

import cv2
import numpy as np


def ajouter_bruit_speckle(image, variance=0.1):
    """
    Speckle gürültüsü: Görüntü * Gürültü.
    Uniform'dan farkı: Siyah bölgelerde (değer 0) gürültü oluşmaz.
    """
    
    if image is None: return None
    
    # Variance 0-1.0 arası geliyor.
    scale = np.clip(variance, 0.0, 1.0)
    
    image_float = image.astype(np.float32)
    
    # 1. Tek kanal Gauss gürültüsü oluştur (Makale tarzı monochromatic)
    noise = np.zeros(image.shape[:2], dtype=np.float32)
    cv2.randn(noise, 0, 1.0) # Ort:0, Std:1
    
    # 2. Renkli ise gürültüyü 3 kanala yay
    if len(image.shape) == 3:
        noise = cv2.merge([noise, noise, noise])
        
    # 3. Çarpımsal İşlem (I + I * n * scale)
    speckle = image_float * noise * scale
    output = image_float + speckle
        
    return np.clip(output, 0, 255).astype(np.uint8)