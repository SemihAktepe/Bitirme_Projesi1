"""
Gestionnaire de bruits avec support standard (sans X/Y)
Auteur: Semih Aktepe
Date: 21 Decembre 2025 (Updated)
"""

# Not: Bu importların çalışması için shot.py ve diğer dosyaların 
# 'bruits' klasöründe veya aynı dizinde olması gerekir.
try:
    from shot import ajouter_bruit_shot
    # Diğer gürültü fonksiyonlarının da kendi dosyalarından veya tek bir dosyadan geldiğini varsayıyoruz
    from bruits import (
        ajouter_bruit_gaussien,
        ajouter_bruit_poivre_sel,
        ajouter_bruit_speckle,
        ajouter_bruit_uniforme
    )
except ImportError:
    # Eğer hepsi tek bir 'bruits.py' dosyasındaysa:
    from bruits import (
        ajouter_bruit_gaussien,
        ajouter_bruit_poivre_sel,
        ajouter_bruit_shot,
        ajouter_bruit_speckle,
        ajouter_bruit_uniforme
    )

def appliquer_bruit_mixte(image, bruits_config):
    """
    Applique une configuration de bruits standard
    
    Parameters
    ----------
    image : np.ndarray
        Image d'entree
    bruits_config : list of dict
        Configuration des bruits. 
        Exemple: [{'type': 'shot', 'intensite': 50}, {'type': 'gaussien', 'intensite': 10}]
    
    Returns
    -------
    np.ndarray
        Image bruitee
    """
    
    if not bruits_config:
        return image.copy()
    
    noisy_image = image.copy()
    
    for bruit in bruits_config:
        bruit_type = bruit.get('type')
        
        if bruit_type == 'gaussien':
            intensite = bruit.get('intensite', 20)
            noisy_image = ajouter_bruit_gaussien(noisy_image, intensite)
        
        elif bruit_type == 'poivre_sel':
            densite = bruit.get('densite', 0.05)
            noisy_image = ajouter_bruit_poivre_sel(noisy_image, densite)
        
        elif bruit_type == 'shot':
            # Shot gürültüsü için özel parametre (intensite)
            intensite = bruit.get('intensite', 20)
            noisy_image = ajouter_bruit_shot(noisy_image, intensite)
        
        elif bruit_type == 'speckle':
            variance = bruit.get('variance', 0.1)
            noisy_image = ajouter_bruit_speckle(noisy_image, variance)
        
        elif bruit_type == 'uniforme':
            intensite = bruit.get('intensite', 20)
            noisy_image = ajouter_bruit_uniforme(noisy_image, intensite)
    
    return noisy_image


def get_bruit_description(bruits_config):
    """Genere une description textuelle des bruits"""
    
    if not bruits_config:
        return "Aucun"
    
    descriptions = []
    
    for bruit in bruits_config:
        bruit_type = bruit.get('type', 'unknown')
        
        if bruit_type == 'gaussien':
            intensite = bruit.get('intensite', 20)
            descriptions.append(f"Gaussien({intensite}%)")
        
        elif bruit_type == 'poivre_sel':
            densite = bruit.get('densite', 0.05)
            descriptions.append(f"Poivre&Sel({densite*100:.0f}%)")
        
        elif bruit_type == 'shot':
            intensite = bruit.get('intensite', 20)
            descriptions.append(f"Shot({intensite}%)")
        
        elif bruit_type == 'speckle':
            variance = bruit.get('variance', 0.1)
            descriptions.append(f"Speckle({variance*100:.0f}%)")
        
        elif bruit_type == 'uniforme':
            intensite = bruit.get('intensite', 20)
            descriptions.append(f"Uniforme({intensite}%)")
    
    return " + ".join(descriptions)