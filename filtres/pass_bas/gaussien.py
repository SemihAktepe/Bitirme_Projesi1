"""
Filtre Passe-Bas Gaussien (domaine fréquentiel)
Auteur: Semih Aktepe
Date: 5 Décembre 2025
"""

import numpy as np
import cv2


def appliquer_gaussien_frequentiel(image, cutoff_frequency=30):
    """
    Applique un filtre passe-bas gaussien dans le domaine fréquentiel
    
    Parameters
    ----------
    image : np.ndarray
        Image d'entrée
    cutoff_frequency : float
        Fréquence de coupure (plus petit = plus de flou)
        Recommandé: 10-50
    
    Returns
    -------
    np.ndarray
        Image filtrée
    """
    if image is None or image.size == 0:
        raise ValueError("Image invalide")
    
    # Traiter chaque canal séparément pour images couleur
    if len(image.shape) == 3:
        filtered = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[2]):
            filtered[:, :, c] = _apply_gaussian_lowpass_single(
                image[:, :, c], cutoff_frequency
            )
        
        # Normaliser TOUS les canaux ensemble
        filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
        return filtered.astype(np.uint8)
    else:
        result = _apply_gaussian_lowpass_single(image, cutoff_frequency)
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        return result.astype(np.uint8)


def _apply_gaussian_lowpass_single(image, cutoff_frequency):
    """Applique le filtre gaussien sur un canal unique"""
    
    # Convertir en float
    img_float = image.astype(np.float32)
    
    # FFT
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Créer masque gaussien
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    mask = np.zeros((rows, cols, 2), np.float32)
    
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow)**2 + (j - ccol)**2)
            mask[i, j] = np.exp(-(distance**2) / (2 * (cutoff_frequency**2)))
    
    # Appliquer masque
    fshift = dft_shift * mask
    
    # IFFT
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # Retourner en float32 (normalisation sera faite après pour tous les canaux)
    return img_back


# Options pour interface
CUTOFF_OPTIONS = {
    'Très fort (10)': 10,
    'Fort (20)': 20,
    'Moyen (30)': 30,
    'Léger (40)': 40,
    'Très léger (50)': 50
}