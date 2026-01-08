"""
Bruit Shot / Poisson (Photon Counting Simulation)
Referans Prensip: https://www.whatmatrix.com/portal/intro-to-python-image-processing-in-computational-photography/
"""

import numpy as np
import cv2

def ajouter_bruit_shot(image, intensite=20):
    """
    Shot (Poisson) gürültüsü ekler.
    Makaledeki prensip: Düşük ışık = Az Foton = Yüksek Gürültü.
    
    Parameters
    ----------
    image : np.ndarray
        Giriş resmi (0-255)
    intensite : float
        0-100 arası. 
        Yüksek değer -> Düşük ışık simülasyonu (Çok gürültü).
        Düşük değer -> Yüksek ışık simülasyonu (Az gürültü).
    """
    
    if image is None or image.size == 0:
        raise ValueError("Image invalide")
        
    if intensite <= 0:
        return image.copy()
    
    image_float = image.astype(np.float32)
    
    # Intensite arttıkça ışık (scale) azalır.
    # %100 intensite -> Çok az ışık (Resim kararır).
    # Not: Veriyi tamamen öldürmemek için katsayı dengelendi (0.03).
    divisor = 1.0 + (intensite * 0.03)
    
    # 1. Işığı azalt (Fotonları düşür)
    simulated_photons = image_float / divisor
    
    # 2. Gürültüyü ekle (Poisson)
    # Poisson dağılımı, düşük foton sayılarında gürültüyü belirginleştirir.
    noisy_photons = np.random.poisson(simulated_photons)
    
    # 3. Çıktıyı hazırla
    # Gain (Parlatma) uygulanmaz, görüntü referanstaki gibi karanlık kalır.
    output = noisy_photons
    
    # Değerleri 0-255 arasına sıkıştır ve uint8 formatına çevir
    return np.clip(output, 0, 255).astype(np.uint8)