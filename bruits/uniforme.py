"""
Bruit Uniforme (Medium Article Style)
Referans: https://medium.com/@muyandiritujr/noise-in-image-processing-and-how-to-add-it-to-images-in-python-456a434dcc3f
"""

import cv2
import numpy as np


def ajouter_bruit_uniforme(image, intensite=20):
    """
    Medium makalesindeki yöntemle Uniform gürültü ekler.
    Gürültü 'Quantization Noise' gibi davranır.
    """
    
    if image is None: return None
    
    # Intensite (0-100) makaledeki 'gamma' parametresi olarak kullanılacak.
    # 0 -> 0.0, 100 -> 0.4 (Çok yüksek değer görüntüyü bembeyaz yapar)
    gamma = (intensite / 100.0) * 0.5
    
    # 1. Gürültü matrisi
    uni_noise = np.zeros(image.shape[:2], dtype=np.float32)
    
    # 2. OpenCV ile Uniform dağılım (0 ile 256 arası)
    cv2.randu(uni_noise, 0, 256)
    
    # 3. Gamma ile ölçekle (Görünürlüğü ayarlar)
    # Makalede: uni_noise = (uni_noise * gamma).astype(np.uint8)
    # Biz hassasiyet için float devam ediyoruz
    uni_noise = uni_noise * gamma
    
    image_float = image.astype(np.float32)
    output = np.zeros_like(image_float)
    
    # 4. Kanal işlemleri
    if len(image.shape) == 2:
        output = cv2.add(image_float, uni_noise)
    elif len(image.shape) == 3:
        merged_noise = cv2.merge([uni_noise, uni_noise, uni_noise])
        output = cv2.add(image_float, merged_noise)
        
    return np.clip(output, 0, 255).astype(np.uint8)