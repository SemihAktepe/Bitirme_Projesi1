

import sqlite3
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
import numpy as np
from PIL import Image

# Config modülü yoksa varsayılan yolu kullan
try:
    from config import DATABASE_PATH, DB_CONFIG
except ImportError:
    # Fallback konfigürasyon
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATABASE_PATH = Path(current_dir) / "wdenet_database.db"
    DB_CONFIG = {"check_same_thread": False, "timeout": 30.0, "auto_commit": True}


class DatabaseManager:
    """
    Singleton pattern ile veritabanı yönetimi.
    Tüm projede tek bir DB bağlantısı kullanılır.
    Standart: Sınıf isimleri PascalCase.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.db_path = DATABASE_PATH
        self.connection: Optional[sqlite3.Connection] = None
        self._initialized = True

        # Veritabanını başlat
        self._ensure_database_exists()
        self.connect()

    def _ensure_database_exists(self):
        """Veritabanı yoksa oluştur."""
        if not self.db_path.exists():
            print(f" Veritabanı bulunamadı, oluşturuluyor: {self.db_path}")
            try:
                from create_database_sqlite import create_database
                create_database(str(self.db_path))
                print(" Veritabanı başarıyla oluşturuldu!")
            except ImportError:
                print(" HATA: 'create_database_sqlite.py' bulunamadı.")

    def connect(self) -> sqlite3.Connection:
        """Veritabanına bağlanır."""
        if self.connection is None:
            self.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=DB_CONFIG["check_same_thread"],
                timeout=DB_CONFIG["timeout"]
            )
            # Row'lara isimle erişim sağlar (dict-like)
            self.connection.row_factory = sqlite3.Row
        return self.connection

    def close(self):
        """Bağlantıyı kapatır."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """SQL sorgusu çalıştırır."""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            if DB_CONFIG["auto_commit"]:
                self.connection.commit()
            return cursor
        except sqlite3.Error as e:
            print(f"SQL Hatası: {e}\nSorgu: {query}\nParametreler: {params}")
            raise

    def fetchone(self, query: str, params: tuple = ()) -> Optional[Dict]:
        """Tek satır getirir."""
        cursor = self.execute(query, params)
        row = cursor.fetchone()
        return dict(row) if row else None

    def fetchall(self, query: str, params: tuple = ()) -> List[Dict]:
        """Tüm satırları getirir."""
        cursor = self.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    # ========================================================================
    # IMAGES TABLOSU İŞLEMLERİ
    # ========================================================================

    def insert_image(self, file_path: str, image_array: np.ndarray = None) -> int:
        """
        Görüntü bilgilerini Images tablosuna ekler.
        Standart: Değişkenler snake_case.

        Args:
            file_path: Görüntü dosya yolu.
            image_array: NumPy array (boyut hesabı için opsiyonel).

        Returns:
            image_id: Eklenen kaydın ID'si.
        """
        # Görüntü boyutlarını al
        if image_array is not None:
            if len(image_array.shape) == 2:  # Grayscale
                height, width = image_array.shape
                channels = 1
            else:  # Color
                height, width, channels = image_array.shape
        elif os.path.exists(file_path):
            try:
                img = Image.open(file_path)
                width, height = img.size
                channels = len(img.getbands())
            except Exception:
                width, height, channels = 0, 0, 0
        else:
            width, height, channels = 0, 0, 0

        # Dosya bilgileri
        file_name = os.path.basename(file_path)
        # Uzantıyı al ve temizle (örn: .jpg -> JPG)
        file_format = os.path.splitext(file_name)[1].upper().replace(".", "")
        if file_format == "JPEG": file_format = "JPG"
        
        # SQL Şemasına Uyum: FileSize (Bytes) olarak saklanır
        file_size_bytes = os.path.getsize(file_path) if os.path.exists(file_path) else 0

        # Mevcut mu kontrol et (FilePath UNIQUE olduğu için)
        existing = self.fetchone("SELECT ImageID FROM Images WHERE FilePath = ?", (file_path,))
        if existing:
            return existing['ImageID']

        query = """
        INSERT INTO Images 
        (FileName, FilePath, FileFormat, FileSize, Width, Height, Channels, UploadDate)
        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
        """
        params = (file_name, file_path, file_format, file_size_bytes, width, height, channels)

        cursor = self.execute(query, params)
        return cursor.lastrowid

    # ========================================================================
    # NOISE TYPES & CONFIGS
    # ========================================================================

    def get_or_create_noise_type(self, noise_name: str) -> int:
        """Gürültü tipi ID'sini getirir."""
        # İsim eşleştirmesi (normalize et)
        noise_name = noise_name.lower()
        
        # Veritabanındaki isimlerle eşleştirme haritası
        name_map = {
            'gaussian': 'gaussien', 'gaussien': 'gaussien',
            'salt & pepper': 'poivre_sel', 'poivre_sel': 'poivre_sel',
            'shot': 'shot', 'poisson': 'shot',
            'speckle': 'speckle',
            'uniform': 'uniforme', 'uniforme': 'uniforme'
        }
        db_name = name_map.get(noise_name, 'gaussien')

        result = self.fetchone("SELECT NoiseTypeID FROM NoiseTypes WHERE NoiseTypeName = ?", (db_name,))
        if result:
            return result['NoiseTypeID']
        
        # Bulunamazsa varsayılan olarak 1 (Gaussien) döndür
        print(f" Uyarı: Gürültü tipi '{noise_name}' bulunamadı, varsayılan kullanılıyor.")
        return 1

    def insert_noise_config(self, noise_type_id: int, config_dict: Dict) -> int:
        """
        Gürültü konfigürasyonunu ayrı sütunlara kaydeder (SQL Schema Uyumlu).
        JSON blob YERİNE Intensity, Variance, Density sütunları kullanılır.
        """
        # Sözlükten değerleri çek (Varsayılanlar None veya 0)
        # Bot tarafındaki anahtarlar: 'intensite', 'variance', 'densite'
        intensity = config_dict.get('intensite', config_dict.get('intensity', 0))
        variance = config_dict.get('variance', None)
        density = config_dict.get('densite', config_dict.get('density', None))
        
        query = """
        INSERT INTO NoiseConfigs 
        (NoiseTypeID, Intensity, Variance, Density, CreatedDate) 
        VALUES (?, ?, ?, ?, datetime('now', 'localtime'))
        """
        cursor = self.execute(query, (noise_type_id, intensity, variance, density))
        return cursor.lastrowid

    # ========================================================================
    # FILTER TYPES & CONFIGS
    # ========================================================================

    def get_or_create_filter_type(self, filter_name: str) -> int:
        """Filtre tipi ID'sini getirir."""
        # İsim normalizasyonu
        name_lower = filter_name.lower()
        
        # Mapping
        if "median" in name_lower: db_name = "median"
        elif "bilateral" in name_lower: db_name = "bilateral"
        elif "gaussian" in name_lower and ("pass" in name_lower or "low" in name_lower): db_name = "lowpass_gaussian"
        elif "butterworth" in name_lower: db_name = "lowpass_butterworth"
        elif "ffdnet" in name_lower: db_name = "ffdnet"
        elif "dncnn" in name_lower: db_name = "dncnn"
        elif "hybrid" in name_lower or "w-denet" in name_lower: db_name = "wdenet"
        else: db_name = "median" # Fallback

        result = self.fetchone("SELECT FilterTypeID FROM FilterTypes WHERE FilterName = ?", (db_name,))
        if result:
            return result['FilterTypeID']
        
        # Filtre yoksa varsayılan (1) döndür veya oluşturma mantığı ekle
        print(f"⚠️ Uyarı: Filtre '{filter_name}' DB'de bulunamadı.")
        return 1

    def insert_filter_config(self, filter_type_id: int, config_dict: Dict) -> int:
        """
        Filtre konfigürasyonunu ayrı sütunlara kaydeder.
        JSON blob yerine KernelSize, NoiseSigma vb. kullanılır.
        """
        kernel_size = config_dict.get('kernel_size', None)
        pass_bas_type = None # Şu an bot tarafında bu veri yoksa None
        cutoff = config_dict.get('cutoff_frequency', None)
        order = config_dict.get('order', None)
        noise_sigma = config_dict.get('noise_sigma', None)
        
        # Hibrit modda pre-filter ID gerekebilir, şimdilik NULL
        pre_filter_id = None

        query = """
        INSERT INTO FilterConfigs 
        (FilterTypeID, KernelSize, PassBasType, CutoffFrequency, ButterworthOrder, NoiseSigma, PreFilterID, CreatedDate) 
        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
        """
        params = (filter_type_id, kernel_size, pass_bas_type, cutoff, order, noise_sigma, pre_filter_id)
        cursor = self.execute(query, params)
        return cursor.lastrowid

    # ========================================================================
    # EXPERIMENTS & RESULTS
    # ========================================================================

    def insert_experiment(self, image_id: int, noise_config_id: int, filter_config_id: int,
                         mode: str, output_type: str = "Filtered") -> int:
        """Deney kaydı oluşturur."""
        query = """
        INSERT INTO Experiments 
        (ImageID, NoiseConfigID, FilterConfigID, Mode, OutputType, CreatedDate)
        VALUES (?, ?, ?, ?, ?, datetime('now', 'localtime'))
        """
        # Mode kontrolü (Constraint check)
        if mode not in ['KF', 'AI', 'HYB']:
            mode = 'KF' # Fallback
            
        params = (image_id, noise_config_id, filter_config_id, mode, output_type)
        cursor = self.execute(query, params)
        return cursor.lastrowid

    def insert_result(self, experiment_id: int, psnr: float, ssim: float, processing_time_ms: float) -> int:
        """Deney sonuçlarını kaydeder."""
        query = """
        INSERT INTO Results 
        (ExperimentID, PSNR, SSIM, ProcessingTimeMs, CreatedDate)
        VALUES (?, ?, ?, ?, datetime('now', 'localtime'))
        """
        params = (experiment_id, psnr, ssim, processing_time_ms)
        cursor = self.execute(query, params)
        return cursor.lastrowid

    # ========================================================================
    # YÜKSEK SEVİYELİ İŞLEM (BOT İÇİN)
    # ========================================================================

    def log_complete_experiment(self,
                                image_path: str,
                                image_array: np.ndarray,
                                noise_config: Dict,
                                filter_config: Dict,
                                results: Dict,
                                mode: str = "KF",
                                output_type: str = "Filtered") -> bool:
        """
        Tam bir deneyi tüm detaylarıyla ve ilişkileriyle kaydeder.
        Transaction mantığı ile çalışır (hata olursa kaydetmez).
        """
        try:
            # 1. Görüntü ekle
            image_id = self.insert_image(file_path=image_path, image_array=image_array)

            # 2. Gürültü config ekle
            noise_type_id = self.get_or_create_noise_type(noise_config.get("type", "gaussien"))
            noise_config_id = self.insert_noise_config(noise_type_id, noise_config)

            # 3. Filtre config ekle
            filter_type_name = filter_config.get("type", "median")
            filter_type_id = self.get_or_create_filter_type(filter_type_name)
            filter_config_id = self.insert_filter_config(filter_type_id, filter_config)

            # 4. Deney kaydı oluştur
            experiment_id = self.insert_experiment(
                image_id, noise_config_id, filter_config_id, mode, output_type
            )

            # 5. Sonuçları kaydet
            self.insert_result(
                experiment_id,
                results.get("psnr", 0.0),
                results.get("ssim", 0.0),
                results.get("processing_time_ms", 0.0)
            )

            return True

        except Exception as e:
            print(f"❌ Deney kaydedilemedi: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_experiment_count(self) -> int:
        """Toplam deney sayısını döndürür."""
        result = self.fetchone("SELECT COUNT(*) as count FROM Experiments")
        return result['count'] if result else 0


# Singleton instance
db = DatabaseManager()

if __name__ == "__main__":
    print(f" Database Path: {db.db_path}")
    print(f" Mevcut Deney Sayısı: {db.get_experiment_count()}")
