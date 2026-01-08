"""
Filtre Passe-Bas Butterworth (domaine fréquentiel)
Auteur: Semih Aktepe
Date: 5 Décembre 2025
"""

import numpy as np
import cv2


def appliquer_butterworth(image, cutoff_frequency=30, order=2):
    """
    Applique un filtre passe-bas Butterworth dans le domaine fréquentiel
    
    Le filtre Butterworth offre une transition plus douce que le filtre idéal
    et évite les effets de ringing (ondulations).
    
    Parameters
    ----------
    image : np.ndarray
        Image d'entrée
    cutoff_frequency : float
        Fréquence de coupure D0 (plus petit = plus de flou)
        Recommandé: 10-50
    order : int
        Ordre du filtre (n)
        - n=1: transition douce
        - n=2: standard (recommandé)
        - n=5+: transition abrupte (proche idéal)
    
    Returns
    -------
    np.ndarray
        Image filtrée
    
    Notes
    -----
    Fonction de transfert: H(u,v) = 1 / (1 + (D(u,v)/D0)^(2n))
    où D(u,v) est la distance au centre fréquentiel
    """
    if image is None or image.size == 0:
        raise ValueError("Image invalide")
    
    # Traiter chaque canal séparément
    if len(image.shape) == 3:
        filtered = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[2]):
            filtered[:, :, c] = _apply_butterworth_single(
                image[:, :, c], cutoff_frequency, order
            )
        
        # Normaliser TOUS les canaux ensemble
        filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
        return filtered.astype(np.uint8)
    else:
        result = _apply_butterworth_single(image, cutoff_frequency, order)
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        return result.astype(np.uint8)


def _apply_butterworth_single(image, cutoff_frequency, order):
    """Applique le filtre Butterworth sur un canal unique"""
    
    # Convertir en float
    img_float = image.astype(np.float32)
    
    # FFT
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Créer masque Butterworth
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    mask = np.zeros((rows, cols, 2), np.float32)
    
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow)**2 + (j - ccol)**2)
            
            # Formule Butterworth: H = 1 / (1 + (D/D0)^(2n))
            if distance == 0:
                mask[i, j] = 1.0
            else:
                butterworth_value = 1.0 / (1.0 + (distance / cutoff_frequency)**(2 * order))
                mask[i, j] = butterworth_value
    
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

ORDER_OPTIONS = {
    'Ordre 1 (doux)': 1,
    'Ordre 2 (standard)': 2,
    'Ordre 3': 3,
    'Ordre 5 (abrupt)': 5
}


if __name__ == "__main__":
    # Test
    print("🧪 Test Butterworth")
    
    test_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    for label, freq in CUTOFF_OPTIONS.items():
        result = appliquer_butterworth(test_image, cutoff_frequency=freq)
        print(f"✅ {label}: {result.shape}")
    
    print("✅ Tests réussis!")