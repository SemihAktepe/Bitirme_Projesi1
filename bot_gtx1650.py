"""
W-DENet Deney Botu - GTX 1650 (G770)
=====================================
Görev: 6 görüntü işle (BSD68: 9-11, DIV2K: 9-11)
Toplam deney: ~1440 (6 img × 30 noise × 8 filter)
"""
import os
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

from config import BSD68_PATH, DIV2K_PATH, PROJECT_ROOT
from db_manager import db
from metrics import calculate_psnr, calculate_ssim
from bruit_manager import appliquer_bruit_mixte
from filtre_manager import apply_kf_filter


# ============================================================================
# CİHAZ KONFİGÜRASYONU
# ============================================================================
MACHINE_ID = "GTX_1650"
MACHINE_NAME = "Excalibur G770 GTX 1650"

# Görüntü aralıkları (0-indexed)
BSD68_START = 9
BSD68_END = 12  # 3 görüntü

DIV2K_START = 9
DIV2K_END = 12  # 3 görüntü

# ============================================================================
# GÜRÜLTÜ KONFİGÜRASYONLARI (30 config = 5 tip × 6 intensity)
# ============================================================================
NOISE_INTENSITIES = [5, 10, 25, 35, 50, 70]

NOISE_CONFIGS = []
for intensity in NOISE_INTENSITIES:
    NOISE_CONFIGS.append([{'type': 'gaussien', 'intensite': intensity}])
    NOISE_CONFIGS.append([{'type': 'poivre_sel', 'densite': intensity/100}])
    NOISE_CONFIGS.append([{'type': 'shot', 'intensite': intensity}])
    NOISE_CONFIGS.append([{'type': 'speckle', 'variance': intensity/100}])
    NOISE_CONFIGS.append([{'type': 'uniforme', 'intensite': intensity}])

# ============================================================================
# FİLTRE KONFİGÜRASYONLARI (8 config)
# ============================================================================
FILTER_CONFIGS = [
    {"name": "Median 3x3", "kf_choice": "median", "params": {"kernel_size": 3}},
    {"name": "Median 5x5", "kf_choice": "median", "params": {"kernel_size": 5}},
    {"name": "Median 7x7", "kf_choice": "median", "params": {"kernel_size": 7}},
    {"name": "Median 20x20", "kf_choice": "median", "params": {"kernel_size": 20}},
    {"name": "Median 50x50", "kf_choice": "median", "params": {"kernel_size": 50}},
    {"name": "Bilateral", "kf_choice": "bilateral", "params": {}},
    {"name": "Gaussian Lowpass", "kf_choice": "gaussian pass-bas", "params": {"cutoff_frequency": 30}},
    {"name": "Butterworth Lowpass", "kf_choice": "butterworth", "params": {"cutoff_frequency": 30, "order": 2}},
]


class GTX1650Bot:
    """GTX 1650 için özel deney botu"""

    def __init__(self):
        self.machine_id = MACHINE_ID
        self.stats = {
            "total_experiments": 0,
            "successful": 0,
            "failed": 0,
            "total_time_ms": 0
        }

    def get_dataset_images(self, dataset_path: Path, start_idx: int, end_idx: int):
        """Dataset'ten belirli aralıktaki görüntüleri al"""
        if not dataset_path or not dataset_path.exists():
            return []

        image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
        all_images = sorted([
            f for f in dataset_path.iterdir()
            if f.suffix.lower() in image_extensions
        ])

        return all_images[start_idx:end_idx]

    def run_single_experiment(self, image_path: Path, noise_cfg: list, filter_cfg: dict):
        """Tek bir deney çalıştır"""
        try:
            # Görüntüyü yükle
            original = np.array(Image.open(image_path))

            # Gürültü ekle
            start_time = time.time()
            noisy = appliquer_bruit_mixte(original, noise_cfg)

            # Filtre uygula
            filtered = apply_kf_filter(noisy, filter_cfg["kf_choice"], **filter_cfg["params"])
            end_time = time.time()

            processing_time_ms = (end_time - start_time) * 1000

            # Metrikler
            psnr = calculate_psnr(original, filtered)
            ssim = calculate_ssim(original, filtered)

            # DB'ye kaydet
            success = db.log_complete_experiment(
                image_path=str(image_path),
                image_array=original,
                noise_config={
                    "type": noise_cfg[0]['type'],
                    "intensity": noise_cfg[0].get('intensite', noise_cfg[0].get('densite', noise_cfg[0].get('variance', 0)))
                },
                filter_config={
                    "type": filter_cfg["name"],
                    **filter_cfg["params"]
                },
                results={
                    "psnr": psnr,
                    "ssim": ssim,
                    "processing_time_ms": processing_time_ms
                },
                mode="KF",
                output_type="Filtered"
            )

            if success:
                self.stats["successful"] += 1
                self.stats["total_time_ms"] += processing_time_ms
            else:
                self.stats["failed"] += 1

            self.stats["total_experiments"] += 1
            return success

        except Exception as e:
            print(f"\n❌ Hata: {image_path.name} - {e}")
            self.stats["failed"] += 1
            self.stats["total_experiments"] += 1
            return False

    def run_all_experiments(self):
        """Tüm deneyleri çalıştır"""
        print("\n" + "="*70)
        print(f"🚀 {MACHINE_NAME} - Deney Botu")
        print("="*70)
        print(f"📊 Görüntü aralığı:")
        print(f"   • BSD68: {BSD68_START}-{BSD68_END-1} ({BSD68_END-BSD68_START} görüntü)")
        print(f"   • DIV2K: {DIV2K_START}-{DIV2K_END-1} ({DIV2K_END-DIV2K_START} görüntü)")
        print(f"📊 Deney matrisi:")
        print(f"   • {len(NOISE_CONFIGS)} gürültü konfigürasyonu")
        print(f"   • {len(FILTER_CONFIGS)} filtre konfigürasyonu")
        print(f"   • Toplam: {6 * len(NOISE_CONFIGS) * len(FILTER_CONFIGS)} deney")
        print("="*70)

        # BSD68 görüntüleri
        bsd68_images = self.get_dataset_images(BSD68_PATH, BSD68_START, BSD68_END)
        print(f"\n✓ BSD68: {len(bsd68_images)} görüntü yüklendi")

        # DIV2K görüntüleri
        div2k_images = self.get_dataset_images(DIV2K_PATH, DIV2K_START, DIV2K_END)
        print(f"✓ DIV2K: {len(div2k_images)} görüntü yüklendi")

        all_images = bsd68_images + div2k_images
        total_experiments = len(all_images) * len(NOISE_CONFIGS) * len(FILTER_CONFIGS)

        print(f"\n🔬 Toplam deney sayısı: {total_experiments}")
        print(f"⏱️  Tahmini süre: ~{total_experiments * 1.6 / 3600:.1f} saat (1.6s/deney, GTX 1650)")

        # Kullanıcı onayı
        response = input("\n  Başlatmak istiyor musunuz? (e/h): ").strip().lower()
        if response != 'e':
            print(" İptal edildi.")
            return

        # Deneyleri çalıştır
        start_time = time.time()

        with tqdm(total=total_experiments, desc="GTX 1650", unit="exp") as pbar:
            for image_path in all_images:
                pbar.set_description(f"📸 {image_path.name[:20]}")

                for noise_cfg in NOISE_CONFIGS:
                    for filter_cfg in FILTER_CONFIGS:
                        self.run_single_experiment(image_path, noise_cfg, filter_cfg)
                        pbar.update(1)
                        pbar.set_postfix({
                            "✓": self.stats["successful"],
                            "✗": self.stats["failed"]
                        })

        end_time = time.time()
        total_time = end_time - start_time

        # Rapor
        print("\n" + "="*70)
        print(" DENEY RAPORU")
        print("="*70)
        print(f" Başarılı: {self.stats['successful']}")
        print(f" Başarısız: {self.stats['failed']}")
        print(f" Toplam: {self.stats['total_experiments']}")
        print(f"  Süre: {total_time/3600:.2f} saat")
        print(f" Ortalama: {total_time/max(self.stats['total_experiments'], 1):.3f} s/deney")
        print(f" Veritabanı: wdenet_database.db")
        print("="*70)


def main():

    bot = GTX1650Bot()
    bot.run_all_experiments()

    print("\n GTX 1650 görevi tamamlandı!")
    print(" DB dosyasını G770'de saklayın ve merge için hazır tutun!")


if __name__ == "__main__":
    main()
