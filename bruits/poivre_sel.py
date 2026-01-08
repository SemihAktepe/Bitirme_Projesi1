"""
Bruit Poivre et Sel (Medium Article Style)
Referans: https://medium.com/@muyandiritujr/noise-in-image-processing-and-how-to-add-it-to-images-in-python-456a434dcc3f
"""

import numpy as np


def ajouter_bruit_poivre_sel(image, densite=0.05):
    """
    Medium makalesindeki yöntemle Salt & Pepper gürültüsü.
    """
    
    if image is None: return None
    
    # Makaledeki 'prob' parametresi bizim 'densite' parametremiz.
    prob = np.clip(densite, 0.0, 1.0)
    
    output = image.copy()
    
    # 1. Siyah ve Beyaz değerlerini belirle
    if len(image.shape) == 2:
        black = 0
        white = 255
    else:
        # Renkli resimler için array formatında
        black = np.array([0, 0, 0], dtype='uint8')
        white = np.array([255, 255, 255], dtype='uint8')
        
    # 2. Olasılık matrisi oluştur (Tek katman)
    probs = np.random.random(image.shape[:2])
    
    # 3. Pikselleri değiştir (Impulsive Noise)
    # Eşik değerler: prob/2 kadar siyah, prob/2 kadar beyaz
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    
    return output