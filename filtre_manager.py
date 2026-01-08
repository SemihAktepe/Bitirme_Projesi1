"""
Filtre Manager - Gestion des filtres KF, AI et Hybrides
W-DENet Arayüzü için FFDNet ve DnCNN desteğini içerir.

Bu modül, farklı filtre türlerinin (Klasik ve Yapay Zeka) yönetimini sağlar.
Bitirme projesi kod standartlarına uygun olarak düzenlenmiştir.
"""

import cv2
import numpy as np
import sys
import os

# ============================================================================
# AI MODEL IMPORTS (YAPAY ZEKA MODEL İÇE AKTARIMLARI)
# ============================================================================

# Modül yollarını tanımla
# Değişken isimleri küçük harf ve alt çizgi ile yazılmıştır [cite: 9]
current_dir = os.path.dirname(os.path.abspath(__file__))
ffdnet_path = os.path.join(current_dir, 'Filtres_IA', 'FFDNet')
dncnn_path = os.path.join(current_dir, 'Filtres_IA', 'DnCNN')

# ----------------------------------------------------------------------------
# FFDNet Yükleme Bloğu
# ----------------------------------------------------------------------------
if os.path.exists(ffdnet_path):
    print(f"FFDNet path added: {ffdnet_path}")
    try:
        # FFDNet yolunu ekle
        if ffdnet_path not in sys.path:
            sys.path.insert(0, ffdnet_path)
        
        import importlib.util
        
        # Inference modülünü yükle
        spec = importlib.util.spec_from_file_location(
            "ffdnet_inference", 
            os.path.join(ffdnet_path, "inference.py")
        )
        ffdnet_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ffdnet_module)
        
        FFDNetDenoiser = ffdnet_module.FFDNetDenoiser
        print("FFDNetDenoiser imported successfully!")
        
    except Exception as e:
        print(f"Failed to import FFDNetDenoiser: {e}")
        import traceback
        traceback.print_exc()
        FFDNetDenoiser = None
else:
    print(f"Warning: FFDNet path not found: {ffdnet_path}")
    FFDNetDenoiser = None

# ----------------------------------------------------------------------------
# DnCNN Yükleme Bloğu
# ----------------------------------------------------------------------------
if os.path.exists(dncnn_path):
    print(f"DnCNN path added: {dncnn_path}")
    
    # dncnn_inference.py dosyasını kontrol et
    dncnn_inference_path = os.path.join(dncnn_path, "dncnn_inference.py")
    
    if os.path.exists(dncnn_inference_path):
        try:
            # DnCNN yolunu sys.path'e ekle
            if dncnn_path not in sys.path:
                sys.path.insert(0, dncnn_path)
            
            import importlib.util
            
            # Inference modülünü yükle
            spec = importlib.util.spec_from_file_location(
                "dncnn_inference", 
                dncnn_inference_path
            )
            dncnn_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dncnn_module)
            
            DnCNNDenoiser = dncnn_module.DnCNNDenoiser
            print("DnCNNDenoiser imported successfully!")
            
        except Exception as e:
            print(f"Failed to import DnCNNDenoiser: {e}")
            import traceback
            traceback.print_exc()
            DnCNNDenoiser = None
    else:
        print(f"Warning: DnCNN inference module not found in {dncnn_path}")
        DnCNNDenoiser = None
else:
    print(f"Warning: DnCNN path not found: {dncnn_path}")
    DnCNNDenoiser = None

# Global denoiser örnekleri (Singleton deseni - modellerin tekrar yüklenmesini önler)
_ffdnet_denoiser = None
_dncnn_denoiser = None


# ============================================================================
# DENOISER GETTERS (SINGLETON PATTERN)
# ============================================================================

def get_ffdnet_denoiser():
    """
    FFDNet denoiser örneğini döndürür veya oluşturur (Singleton).
    Fonksiyon adı küçük harf ve alt çizgi kuralına uygundur[cite: 10].
    """
    global _ffdnet_denoiser
    
    if FFDNetDenoiser is None:
        print("Error: FFDNetDenoiser not available (import failed)")
        return None
    
    if _ffdnet_denoiser is None:
        try:
            print("Loading FFDNet model...")
            _ffdnet_denoiser = FFDNetDenoiser()
            print("FFDNet loaded successfully")
        except Exception as e:
            print(f"Error loading FFDNet: {e}")
            import traceback
            traceback.print_exc()
            _ffdnet_denoiser = None
    return _ffdnet_denoiser


def get_dncnn_denoiser():
    """
    DnCNN denoiser örneğini döndürür veya oluşturur (Singleton).
    """
    global _dncnn_denoiser
    
    if DnCNNDenoiser is None:
        print("Error: DnCNNDenoiser not available (import failed)")
        return None
    
    if _dncnn_denoiser is None:
        try:
            print("Loading DnCNN model...")
            _dncnn_denoiser = DnCNNDenoiser()
            print("DnCNN loaded successfully")
        except Exception as e:
            print(f"Error loading DnCNN: {e}")
            import traceback
            traceback.print_exc()
            _dncnn_denoiser = None
    return _dncnn_denoiser


# ============================================================================
# CLASSICAL FILTERS (KF) - KLASİK FİLTRELER
# ============================================================================

def apply_kf_filter(image, kf_choice, **params):
    """
    Klasik filtreleri (Kalman-bazlı / standart) uygular.
    
    Parametreler:
    -----------
    image : numpy.ndarray
        Gürültülü görüntü (uint8)
    kf_choice : str
        Filtre tipi seçimi
    **params : dict
        Ek parametreler (kernel_size, vb.)
        
    Dönüş:
    --------
    numpy.ndarray
        Filtrelenmiş görüntü
    """
    # Görüntü kontrolü
    if image is None:
        return None
    
    print(f"Applying KF filter: {kf_choice}")
    
    # Filtre ismini normalize et
    kf_choice_normalized = kf_choice.lower()
    
    # Mantıksal bloklar ayrılmıştır [cite: 20]
    if "median" in kf_choice_normalized or "médian" in kf_choice_normalized:
        # Medyan filtre
        kernel_size = params.get('kernel_size', 5)
        if kernel_size % 2 == 0:
            kernel_size += 1  # Tek sayı olmalı
        filtered = cv2.medianBlur(image, kernel_size)
        
    elif "gaussian" in kf_choice_normalized or "gaussien" in kf_choice_normalized:
        # Gauss filtresi
        if "pass-bas" in kf_choice_normalized:
            # Frekans domeni alçak geçiren Gauss
            cutoff = params.get('cutoff_frequency', 30)
            filtered = apply_lowpass_gaussian(image, cutoff)
        else:
            # Standart Gauss bulanıklığı
            filtered = cv2.GaussianBlur(image, (5, 5), 0)
        
    elif "bilateral" in kf_choice_normalized or "bilatéral" in kf_choice_normalized:
        # Bilateral filtre (Kenar koruyan)
        filtered = cv2.bilateralFilter(image, 9, 75, 75)
    
    elif "butterworth" in kf_choice_normalized:
        # Butterworth alçak geçiren filtre
        cutoff = params.get('cutoff_frequency', 30)
        order = params.get('order', 2)
        filtered = apply_lowpass_butterworth(image, cutoff, order)
    
    else:
        print(f"Unknown KF filter: {kf_choice}, returning original")
        filtered = image
    
    return filtered


def apply_lowpass_gaussian(image, cutoff_frequency=30):
    """
    Frekans domeninde Gauss alçak geçiren filtre uygular.
    
    Parametreler:
    -----------
    image : numpy.ndarray
        Giriş görüntüsü
    cutoff_frequency : float
        Kesim frekansı
    """
    import numpy as np
    from scipy import fftpack
    
    # Çok kanallı görüntü kontrolü
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        for c in range(3):
            result[:,:,c] = apply_lowpass_gaussian(image[:,:,c], cutoff_frequency)
        return result
    
    # FFT (Hızlı Fourier Dönüşümü)
    f = fftpack.fft2(image.astype(float))
    fshift = fftpack.fftshift(f)
    
    # Gauss filtresi oluşturma
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    x = np.linspace(-ccol, ccol, cols)
    y = np.linspace(-crow, crow, rows)
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    H = np.exp(-(D**2) / (2 * (cutoff_frequency**2)))
    
    # Filtreyi uygulama ve Ters FFT
    fshift_filtered = fshift * H
    f_filtered = fftpack.ifftshift(fshift_filtered)
    img_filtered = fftpack.ifft2(f_filtered)
    img_filtered = np.abs(img_filtered)
    
    return np.clip(img_filtered, 0, 255).astype(np.uint8)


def apply_lowpass_butterworth(image, cutoff_frequency=30, order=2):
    """
    Frekans domeninde Butterworth alçak geçiren filtre uygular.
    """
    import numpy as np
    from scipy import fftpack
    
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        for c in range(3):
            result[:,:,c] = apply_lowpass_butterworth(
                image[:,:,c], cutoff_frequency, order
            )
        return result
    
    # FFT işlemi
    f = fftpack.fft2(image.astype(float))
    fshift = fftpack.fftshift(f)
    
    # Butterworth filtresi oluşturma
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    x = np.linspace(-ccol, ccol, cols)
    y = np.linspace(-crow, crow, rows)
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    H = 1 / (1 + (D / cutoff_frequency)**(2 * order))
    
    # Filtreyi uygulama
    fshift_filtered = fshift * H
    f_filtered = fftpack.ifftshift(fshift_filtered)
    img_filtered = fftpack.ifft2(f_filtered)
    img_filtered = np.abs(img_filtered)
    
    return np.clip(img_filtered, 0, 255).astype(np.uint8)


# ============================================================================
# AI FILTERS - YAPAY ZEKA FİLTRELERİ
# ============================================================================

def apply_ai_filter(image, ai_choice):
    """
    Yapay zeka tabanlı filtreleri (DnCNN, FFDNet) uygular.
    
    Parametreler:
    -----------
    image : numpy.ndarray
        Gürültülü görüntü
    ai_choice : str
        Model seçimi ("DnCNN" veya "FFDNet")
    """
    if image is None:
        return None
    
    print(f"Applying AI filter: {ai_choice}")
    
    # DnCNN Bloğu
    if ai_choice == "DnCNN":
        try:
            from PIL import Image
            
            denoiser = get_dncnn_denoiser()
            if denoiser is None:
                print("ERROR: DnCNN denoiser not available")
                return image
            
            print(f"Processing with DnCNN... Image shape: {image.shape}")
            
            # Renkleri koru - kopya üzerinde çalış
            result = image.copy()
            
            if len(image.shape) == 3 and image.shape[2] == 3:
                print("Applying DnCNN to RGB channels...")
                r_channel = Image.fromarray(image[:,:,0].astype('uint8'), mode='L')
                g_channel = Image.fromarray(image[:,:,1].astype('uint8'), mode='L')
                b_channel = Image.fromarray(image[:,:,2].astype('uint8'), mode='L')
                
                r_denoised = np.array(denoiser.denoise(r_channel))
                g_denoised = np.array(denoiser.denoise(g_channel))
                b_denoised = np.array(denoiser.denoise(b_channel))
                
                result = np.stack([r_denoised, g_denoised, b_denoised], axis=2)
            else:
                # Gri tonlama
                pil_gray = Image.fromarray(image.astype('uint8'), mode='L')
                denoised_pil = denoiser.denoise(pil_gray)
                result = np.array(denoised_pil)
            
            return result
            
        except Exception as e:
            print(f"ERROR in DnCNN: {e}")
            import traceback
            traceback.print_exc()
            return image
    
    # FFDNet Bloğu
    elif ai_choice == "FFDNet":
        try:
            from PIL import Image
            
            denoiser = get_ffdnet_denoiser()
            if denoiser is None:
                print("ERROR: FFDNet denoiser not available")
                return image
            
            print(f"Processing with FFDNet... Image shape: {image.shape}")
            
            result = image.copy()
            
            if len(image.shape) == 3 and image.shape[2] == 3:
                print("Applying FFDNet to RGB channels...")
                r_channel = Image.fromarray(image[:,:,0].astype('uint8'), mode='L')
                g_channel = Image.fromarray(image[:,:,1].astype('uint8'), mode='L')
                b_channel = Image.fromarray(image[:,:,2].astype('uint8'), mode='L')
                
                # FFDNet noise_sigma parametresi ister
                r_denoised = np.array(denoiser.denoise(r_channel, noise_sigma=25))
                g_denoised = np.array(denoiser.denoise(g_channel, noise_sigma=25))
                b_denoised = np.array(denoiser.denoise(b_channel, noise_sigma=25))
                
                result = np.stack([r_denoised, g_denoised, b_denoised], axis=2)
            else:
                pil_gray = Image.fromarray(image.astype('uint8'), mode='L')
                denoised_pil = denoiser.denoise(pil_gray, noise_sigma=25)
                result = np.array(denoised_pil)
            
            return result
            
        except Exception as e:
            print(f"ERROR in FFDNet: {e}")
            import traceback
            traceback.print_exc()
            return image
    
    else:
        print(f"Unknown AI model: {ai_choice}")
        return image


# ============================================================================
# HYBRID FILTERS (KF + AI)
# ============================================================================

def apply_hybrid_filter(image, kf_choice, ai_choice, **kf_params):
    """
    W-DENet Hibrit Modu: KF ve AI filtrelerini sıralı olarak uygular.
    
    Parametreler:
    -----------
    image : numpy.ndarray
        Gürültülü görüntü
    kf_choice : str
        Klasik filtre seçimi
    ai_choice : str
        AI model seçimi
    **kf_params : dict
        Klasik filtre parametreleri
        
    Dönüş:
    --------
    dict
        Farklı aşamaların sonuçlarını içeren sözlük.
    """
    if image is None:
        return {'kf_result': None, 'ai_result': None, 'hybrid_result': None}
    
    print(f"Applying hybrid filter: {kf_choice} + {ai_choice}")
    
    # Adım 1: KF filtresi uygula
    print("Step 1: KF filtering...")
    kf_output = apply_kf_filter(image, kf_choice, **kf_params)
    
    # Adım 2: AI filtresini doğrudan gürültülü görüntüye uygula (Kıyaslama için)
    print("Step 2: AI filtering (direct on noisy)...")
    ai_output = apply_ai_filter(image, ai_choice)
    
    # Adım 3: AI filtresini KF çıktısına uygula (HİBRİT)
    print("Step 3: AI filtering (on KF output - hybrid)...")
    
    # DnCNN Hibrit İşlemi
    if ai_choice == "DnCNN":
        try:
            from PIL import Image
            denoiser = get_dncnn_denoiser()
            
            if denoiser is not None:
                print("Processing hybrid with DnCNN...")
                if len(kf_output.shape) == 3 and kf_output.shape[2] == 3:
                    r_ch = Image.fromarray(kf_output[:,:,0].astype('uint8'), mode='L')
                    g_ch = Image.fromarray(kf_output[:,:,1].astype('uint8'), mode='L')
                    b_ch = Image.fromarray(kf_output[:,:,2].astype('uint8'), mode='L')
                    
                    r_den = np.array(denoiser.denoise(r_ch))
                    g_den = np.array(denoiser.denoise(g_ch))
                    b_den = np.array(denoiser.denoise(b_ch))
                    
                    hybrid_output = np.stack([r_den, g_den, b_den], axis=2)
                else:
                    pil_img = Image.fromarray(kf_output.astype('uint8'), mode='L')
                    hybrid_output = np.array(denoiser.denoise(pil_img))
            else:
                hybrid_output = kf_output
                
        except Exception as e:
            print(f"Error in hybrid DnCNN: {e}")
            hybrid_output = kf_output
            
    # FFDNet Hibrit İşlemi
    elif ai_choice == "FFDNet":
        try:
            from PIL import Image
            denoiser = get_ffdnet_denoiser()
            
            if denoiser is not None:
                # Hibrit modda daha düşük sigma kullanılır çünkü KF gürültüyü azalttı
                print("Processing hybrid with FFDNet (sigma=15, post-KF)...")
                
                if len(kf_output.shape) == 3 and kf_output.shape[2] == 3:
                    r_ch = Image.fromarray(kf_output[:,:,0].astype('uint8'), mode='L')
                    g_ch = Image.fromarray(kf_output[:,:,1].astype('uint8'), mode='L')
                    b_ch = Image.fromarray(kf_output[:,:,2].astype('uint8'), mode='L')
                    
                    r_den = np.array(denoiser.denoise(r_ch, noise_sigma=15))
                    g_den = np.array(denoiser.denoise(g_ch, noise_sigma=15))
                    b_den = np.array(denoiser.denoise(b_ch, noise_sigma=15))
                    
                    hybrid_output = np.stack([r_den, g_den, b_den], axis=2)
                else:
                    pil_img = Image.fromarray(kf_output.astype('uint8'), mode='L')
                    hybrid_output = np.array(denoiser.denoise(pil_img, noise_sigma=15))
            else:
                hybrid_output = kf_output
        except Exception as e:
            print(f"Error in hybrid FFDNet: {e}")
            hybrid_output = kf_output
    else:
        # Fallback
        hybrid_output = apply_ai_filter(kf_output, ai_choice)
    
    return {
        'kf_result': kf_output,
        'ai_result': ai_output,
        'hybrid_result': hybrid_output
    }


def test_filters():
    """
    Filtre yöneticisini test eder.
    """
    print("=" * 60)
    print("TESTING FILTRE_MANAGER (FFDNet + DnCNN)")
    print("=" * 60)
    
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    print(f"Test image: shape={test_image.shape}")
    
    # Test KF
    apply_kf_filter(test_image, "Median")
    
    # Test AI
    apply_ai_filter(test_image, "DnCNN")
    
    print("Tests complete.")


if __name__ == "__main__":
    test_filters()