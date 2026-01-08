"""
Bruit Gaussien (Medium Article Style)
Referans: https://medium.com/@muyandiritujr/noise-in-image-processing-and-how-to-add-it-to-images-in-python-456a434dcc3f
"""

import cv2
import numpy as np


def ajouter_bruit_gaussien(image, intensite=20):
    """
    Medium makalesindeki yöntemle Gauss gürültüsü ekler.
    Gürültü tek kanalda üretilir ve RGB'ye birleştirilir (Monochromatic).
    
    Parameters
    ----------
    image : np.ndarray
        Giriş resmi
    intensite : float
        Gürültü şiddeti (0-100) -> Makaledeki 'stddev' ve 'gamma' parametrelerini kontrol eder.
    """
    
    if image is None: return None
    
    # Parametre ayarı
    mean = 0
    # Intensite'yi makul bir standart sapmaya dönüştür (0-100 -> 0-50 arası idealdir)
    stddev = intensite * 0.5
    gamma = 1  # Makaledeki gamma çarpanı
    
    # 1. Gürültü matrisi oluştur (Sadece yükseklik ve genişlik, kanal yok)
    gauss_noise = np.zeros(image.shape[:2], dtype=np.float32)
    
    # 2. OpenCV ile gürültü üret
    cv2.randn(gauss_noise, mean, stddev)
    
    # 3. Gamma ile ölçekle (Makale mantığı)
    gauss_noise = (gauss_noise * gamma)
    
    # Görüntüyü float'a çevir (Taşmaları önlemek için)
    image_float = image.astype(np.float32)
    output = np.zeros_like(image_float)
    
    # 4. Kanal birleştirme ve Ekleme
    if len(image.shape) == 2:
        # Gri seviye ise direkt ekle
        output = cv2.add(image_float, gauss_noise)
    elif len(image.shape) == 3:
        # Renkli ise gürültüyü 3 kanala kopyala (Merge)
        # Bu işlem gürültünün "renkli" değil "siyah-beyaz" karakterli olmasını sağlar
        merged_noise = cv2.merge([gauss_noise, gauss_noise, gauss_noise])
        output = cv2.add(image_float, merged_noise)
        
    # 5. Sonuç döndür
    return np.clip(output, 0, 255).astype(np.uint8)