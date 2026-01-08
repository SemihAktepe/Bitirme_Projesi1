"""
Filtre Médian avec taille de kernel paramétrable
Auteur: Semih Aktepe
Date: 5 Décembre 2025
"""

import numpy as np
from scipy.ndimage import median_filter


def appliquer_median(image, kernel_size=3):
    """
    Applique un filtre médian avec taille de kernel paramétrable
    
    Parameters
    ----------
    image : np.ndarray
        Image d'entrée (H, W) ou (H, W, C)
    kernel_size : int
        Taille du kernel (2, 3, 5, 6, etc.)
    
    Returns
    -------
    np.ndarray
        Image filtrée
    """
    
    if image is None or image.size == 0:
        raise ValueError("Image invalide")
    
    # Limites
    kernel_size = max(2, min(kernel_size, 10))
    
    # Image couleur
    if len(image.shape) == 3:
        filtered = np.zeros_like(image)
        for c in range(image.shape[2]):
            filtered[:, :, c] = median_filter(image[:, :, c], size=kernel_size)
        return filtered
    else:
        return median_filter(image, size=kernel_size)


# Options pour interface
KERNEL_OPTIONS = {
    '2x2 (4 pixels)': 2,
    '3x3 (9 pixels)': 3,
    '5x5 (25 pixels)': 5,
    '6x6 (36 pixels)': 6
}