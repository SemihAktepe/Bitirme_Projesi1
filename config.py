"""
W-DENet Sistem Konfigürasyonu
Tüm path'ler, dataset ayarları ve veritabanı bağlantıları buradan yönetilir.
"""
import os
from pathlib import Path

# ============================================================================
# PROJE KÖKLERI
# ============================================================================
# Bu dosyanın (config.py) bulunduğu yer proje köküdür
PROJECT_ROOT = Path(__file__).parent.absolute()
DATABASE_PATH = PROJECT_ROOT / "wdenet_database.db"

# ============================================================================
# DATASET PATH'LERİ (LS ÇIKTISINA GÖRE DÜZENLENDİ)
# ============================================================================

# "ls" çıktılarına göre datasetler bu yolun altında:
# .../Proje/Filtres_IA/denoising-datasets-main/Experiments/
DATASET_ROOT = PROJECT_ROOT / "Filtres_IA" / "denoising-datasets-main" / "Experiments"

if DATASET_ROOT.exists():
    # Klasör isimleri ls çıktısına göre:
    BSD68_PATH = DATASET_ROOT / "BSD68"
    DIV2K_PATH = DATASET_ROOT / "DIV2K" 
else:
    print(f"⚠️ UYARI: Dataset ana klasörü bulunamadı: {DATASET_ROOT}")
    BSD68_PATH = None
    DIV2K_PATH = None

# ============================================================================
# DENEY AYARLARI (Bot için)
# ============================================================================
# Bu kısım bot_gtx1650.py içinde hardcoded olsa da burada tutmak iyidir.
EXPERIMENT_CONFIG = {
    "max_bsd68_images": None,    # None = Hepsi
    "max_div2k_images": None,    # None = Hepsi
}

# ============================================================================
# VERİTABANI AYARLARI
# ============================================================================
DB_CONFIG = {
    "auto_commit": True,
    "check_same_thread": False,  # SQLite multithread hatasını önlemek için
    "timeout": 30.0,
}

# ============================================================================
# SİSTEM BİLGİLERİ (Test için)
# ============================================================================
def print_config():
    """Mevcut konfigürasyonu yazdır"""
    print("=" * 70)
    print("W-DENet Sistem Konfigürasyonu (Fixed Paths)")
    print("=" * 70)
    print(f"📁 Proje Kökü: {PROJECT_ROOT}")
    print(f"💾 Veritabanı: {DATABASE_PATH}")
    print(f"📊 Dataset Kökü: {DATASET_ROOT}")
    
    bsd_ok = '✓' if BSD68_PATH and BSD68_PATH.exists() else '✗'
    div_ok = '✓' if DIV2K_PATH and DIV2K_PATH.exists() else '✗'
    
    print(f"  └─ BSD68 Path: {BSD68_PATH} {bsd_ok}")
    print(f"  └─ DIV2K Path: {DIV2K_PATH} {div_ok}")
    print("=" * 70)

if __name__ == "__main__":
    print_config()
