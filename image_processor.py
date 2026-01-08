"""
Traitement d'images pour W-DENet
Auteur: Semih Aktepe
"""

import cv2
from PyQt5.QtGui import QPixmap, QImage


def load_image(file_path):
    """Charge une image"""
    return cv2.imread(file_path)


def save_image(file_path, image):
    """Sauvegarde une image"""
    return cv2.imwrite(file_path, image)


def to_qpixmap(image, max_size=None):
    """
    Convertit une image en QPixmap pour PyQt5.
    
    Parameters:
    image: numpy array (OpenCV image)
    max_size: int or None. 
              Si None, l'image conserve sa résolution originale (Haute Qualité).
              Si int, l'image est redimensionnée pour s'adapter (Thumbnail).
    """
    if image is None:
        return QPixmap()

    # BGR -> RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Redimensionner SEULEMENT SI max_size est fourni (Thumbnail modu)
    if max_size is not None:
        h, w = rgb_image.shape[:2]
        # Sadece resim max_size'dan büyükse küçült
        if w > max_size or h > max_size:
            scale = min(max_size/w, max_size/h)
            new_w, new_h = int(w*scale), int(h*scale)
            # INTER_AREA küçültme işlemi için daha kalitelidir
            rgb_image = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Convertir en QPixmap
    height, width, channel = rgb_image.shape
    bytes_per_line = 3 * width
    q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    
    return QPixmap.fromImage(q_image)