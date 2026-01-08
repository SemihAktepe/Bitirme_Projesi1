"""
Filtre Bilatéral
Auteur: Semih Aktepe
"""

import numpy as np
import cv2


def appliquer_bilateral(image, d=9, sigma_color=75, sigma_space=75):
    """
    Applique un filtre bilatéral (préserve les contours)
    
    Parameters
    ----------
    image : np.ndarray
        Image d'entrée
    d : int
        Diameter of pixel neighborhood
    sigma_color : float
        Filter sigma in color space
    sigma_space : float
        Filter sigma in coordinate space
    
    Returns
    -------
    np.ndarray
        Image filtrée
    """
    if image is None or image.size == 0:
        raise ValueError("Image invalide")
    
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)