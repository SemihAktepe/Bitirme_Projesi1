

import sqlite3
import os
from datetime import datetime

def create_database(db_path='wdenet_database.db'):
    """
    SQL Server semasina (wdenet_db.sql) tam uyumlu SQLite veritabani olusturur.
    
    Args:
        db_path (str): Olusturulacak veritabani dosyasinin yolu.
    
    Returns:
        str: Olusturulan veritabaninin yolu.
    """
    
    # 1. Mevcut veritabanini yedekle veya sil
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"⚠️  Eski veritabani silindi ve yenisi olusturuluyor: {db_path}")
        except PermissionError:
            print("❌ HATA: Veritabani dosyasi kullanimda! Lutfen botu durdurun.")
            return None

    # Veritabani baglantisini kur
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print(f"\n{'='*70}")
    print(" W-DENet Database Schema Creation (Fixed Constraints)")
    print(f"{'='*70}\n")

    # Foreign key destegini aktif et
    cursor.execute("PRAGMA foreign_keys = ON;")

    # =========================================================================
    # TABLO: Images
    # =========================================================================
    print(" Creating Images table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Images (
            ImageID INTEGER PRIMARY KEY AUTOINCREMENT,

            -- File Information
            FileName TEXT NOT NULL,
            FilePath TEXT NOT NULL UNIQUE,
            OriginalImagePath TEXT,

            -- Image Dimensions
            Width INTEGER NOT NULL CHECK (Width > 0),
            Height INTEGER NOT NULL CHECK (Height > 0),
            Channels INTEGER NOT NULL CHECK (Channels IN (1, 3)),

            -- File Properties
            FileSize INTEGER,
            FileFormat TEXT NOT NULL CHECK (FileFormat IN ('PNG', 'JPG', 'JPEG', 'BMP')),

            -- Metadata
            UploadDate TEXT DEFAULT (datetime('now', 'localtime')),
            Description TEXT
        );
    """)

    # =========================================================================
    # TABLO: NoiseTypes
    # =========================================================================
    print("🔊 Creating NoiseTypes table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS NoiseTypes (
            NoiseTypeID INTEGER PRIMARY KEY,
            NoiseTypeName TEXT NOT NULL UNIQUE,
            DisplayName TEXT NOT NULL,
            Description TEXT,
            Category TEXT,
            IsActive INTEGER DEFAULT 1 CHECK (IsActive IN (0, 1))
        );
    """)

    # =========================================================================
    # TABLO: NoiseConfigs
    # =========================================================================
    print("  Creating NoiseConfigs table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS NoiseConfigs (
            NoiseConfigID INTEGER PRIMARY KEY AUTOINCREMENT,
            NoiseTypeID INTEGER NOT NULL,

            -- Parameters
            Intensity REAL NOT NULL CHECK (Intensity >= 0 AND Intensity <= 100),
            Variance REAL,
            Density REAL,
            AxisX REAL,
            AxisY REAL,

            -- Metadata
            ConfigName TEXT,
            CreatedDate TEXT DEFAULT (datetime('now', 'localtime')),

            FOREIGN KEY (NoiseTypeID) REFERENCES NoiseTypes(NoiseTypeID)
        );
    """)

    # =========================================================================
    # TABLO: FilterTypes
    # =========================================================================
    print(" Creating FilterTypes table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS FilterTypes (
            FilterTypeID INTEGER PRIMARY KEY,
            FilterName TEXT NOT NULL UNIQUE,
            DisplayName TEXT NOT NULL,
            Category TEXT NOT NULL CHECK (Category IN ('KF', 'AI', 'HYB')),
            Description TEXT,
            ModelPath TEXT,
            IsActive INTEGER DEFAULT 1 CHECK (IsActive IN (0, 1))
        );
    """)

    # =========================================================================
    # TABLO: FilterConfigs
    # DUZELTME BURADA YAPILDI: KernelSize IN (3,5,7,9) yerine KernelSize > 0
    # =========================================================================
    print("  Creating FilterConfigs table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS FilterConfigs (
            FilterConfigID INTEGER PRIMARY KEY AUTOINCREMENT,
            FilterTypeID INTEGER NOT NULL,

            -- Classical Filter Parameters
            -- DÜZELTME: Botun 20x20 ve 50x50 deneylerine izin verilmeli
            KernelSize INTEGER CHECK (KernelSize IS NULL OR KernelSize > 0),
            
            PassBasType TEXT,
            CutoffFrequency REAL,
            ButterworthOrder INTEGER,

            -- AI Model Parameters
            NoiseSigma REAL,

            -- Hybrid Parameters
            PreFilterID INTEGER,

            -- Metadata
            ConfigName TEXT,
            CreatedDate TEXT DEFAULT (datetime('now', 'localtime')),

            FOREIGN KEY (FilterTypeID) REFERENCES FilterTypes(FilterTypeID),
            FOREIGN KEY (PreFilterID) REFERENCES FilterConfigs(FilterConfigID)
        );
    """)

    # =========================================================================
    # TABLO: Experiments
    # =========================================================================
    print(" Creating Experiments table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Experiments (
            ExperimentID INTEGER PRIMARY KEY AUTOINCREMENT,
            ImageID INTEGER NOT NULL,
            NoiseConfigID INTEGER,
            FilterConfigID INTEGER NOT NULL,

            -- Experiment Type
            Mode TEXT NOT NULL CHECK (Mode IN ('KF', 'AI', 'HYB')),
            OutputType TEXT NOT NULL,

            -- Metadata
            ExperimentName TEXT,
            Description TEXT,
            CreatedDate TEXT DEFAULT (datetime('now', 'localtime')),

            FOREIGN KEY (ImageID) REFERENCES Images(ImageID) ON DELETE CASCADE,
            FOREIGN KEY (NoiseConfigID) REFERENCES NoiseConfigs(NoiseConfigID),
            FOREIGN KEY (FilterConfigID) REFERENCES FilterConfigs(FilterConfigID)
        );
    """)

    # =========================================================================
    # TABLO: Results
    # =========================================================================
    print(" Creating Results table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Results (
            ResultID INTEGER PRIMARY KEY AUTOINCREMENT,
            ExperimentID INTEGER NOT NULL,

            -- Quality Metrics
            PSNR REAL CHECK (PSNR IS NULL OR PSNR >= 0),
            SSIM REAL CHECK (SSIM IS NULL OR (SSIM >= 0 AND SSIM <= 1)),

            -- Performance Metrics
            ProcessingTimeMs REAL CHECK (ProcessingTimeMs IS NULL OR ProcessingTimeMs >= 0),

            -- Output Paths
            NoisyImagePath TEXT,
            OutputImagePath TEXT,

            -- Metadata
            Notes TEXT,
            CreatedDate TEXT DEFAULT (datetime('now', 'localtime')),

            FOREIGN KEY (ExperimentID) REFERENCES Experiments(ExperimentID) ON DELETE CASCADE
        );
    """)

    # =========================================================================
    # TABLOLAR: ComparisonSets & ComparisonSetExperiments
    # =========================================================================
    print(" Creating Comparison tables...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ComparisonSets (
            ComparisonSetID INTEGER PRIMARY KEY AUTOINCREMENT,
            SetName TEXT NOT NULL,
            Description TEXT,
            CreatedDate TEXT DEFAULT (datetime('now', 'localtime'))
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ComparisonSetExperiments (
            ComparisonSetID INTEGER NOT NULL,
            ExperimentID INTEGER NOT NULL,
            DisplayOrder INTEGER NOT NULL DEFAULT 0,

            PRIMARY KEY (ComparisonSetID, ExperimentID),
            FOREIGN KEY (ComparisonSetID) REFERENCES ComparisonSets(ComparisonSetID) ON DELETE CASCADE,
            FOREIGN KEY (ExperimentID) REFERENCES Experiments(ExperimentID) ON DELETE CASCADE
        );
    """)

    # =========================================================================
    # INDEXLER
    # =========================================================================
    print(" Creating indexes...")
    cursor.execute("CREATE INDEX IF NOT EXISTS IX_Images_FileName ON Images(FileName);")
    cursor.execute("CREATE INDEX IF NOT EXISTS IX_Images_FileFormat ON Images(FileFormat);")
    cursor.execute("CREATE INDEX IF NOT EXISTS IX_NoiseConfigs_NoiseTypeID ON NoiseConfigs(NoiseTypeID);")
    cursor.execute("CREATE INDEX IF NOT EXISTS IX_FilterConfigs_FilterTypeID ON FilterConfigs(FilterTypeID);")
    cursor.execute("CREATE INDEX IF NOT EXISTS IX_Experiments_ImageID ON Experiments(ImageID);")
    cursor.execute("CREATE INDEX IF NOT EXISTS IX_Experiments_NoiseConfigID ON Experiments(NoiseConfigID);")
    cursor.execute("CREATE INDEX IF NOT EXISTS IX_Experiments_FilterConfigID ON Experiments(FilterConfigID);")
    cursor.execute("CREATE INDEX IF NOT EXISTS IX_Results_ExperimentID ON Results(ExperimentID);")
    cursor.execute("CREATE INDEX IF NOT EXISTS IX_Results_PSNR ON Results(PSNR);")

    # =========================================================================
    # VIEW: vw_CompleteResults
    # =========================================================================
    print(" Creating vw_CompleteResults view...")
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS vw_CompleteResults AS
        SELECT
            r.ResultID,
            e.ExperimentID,
            i.ImageID,
            i.FileName AS ImageName,
            i.Width AS ImageWidth,
            i.Height AS ImageHeight,
            i.FileSize AS FileSizeBytes,
            CAST(i.FileSize / 1024.0 AS REAL) AS FileSizeKB,
            CAST(i.FileSize / (1024.0 * 1024.0) AS REAL) AS FileSizeMB,
            (i.Width * i.Height) AS TotalPixels,
            i.Channels,
            i.FileFormat AS ImageType,
            i.UploadDate,
            nt.NoiseTypeName,
            nt.DisplayName AS NoiseDisplayName,
            nc.Intensity AS NoiseIntensity,
            nc.Variance AS NoiseVariance,
            nc.Density AS NoiseDensity,
            ft.FilterName,
            ft.DisplayName AS FilterDisplayName,
            ft.Category AS FilterCategory,
            fc.KernelSize,
            e.Mode,
            e.OutputType,
            e.ExperimentName,
            r.PSNR,
            r.SSIM,
            r.ProcessingTimeMs,
            i.FilePath AS OriginalImagePath,
            r.NoisyImagePath,
            r.OutputImagePath,
            r.CreatedDate AS ResultDate
        FROM Results r
        INNER JOIN Experiments e ON r.ExperimentID = e.ExperimentID
        INNER JOIN Images i ON e.ImageID = i.ImageID
        LEFT JOIN NoiseConfigs nc ON e.NoiseConfigID = nc.NoiseConfigID
        LEFT JOIN NoiseTypes nt ON nc.NoiseTypeID = nt.NoiseTypeID
        INNER JOIN FilterConfigs fc ON e.FilterConfigID = fc.FilterConfigID
        INNER JOIN FilterTypes ft ON fc.FilterTypeID = ft.FilterTypeID;
    """)

    # =========================================================================
    # VIEW: vw_ProcessingPerformance
    # =========================================================================
    print(" Creating vw_ProcessingPerformance view...")
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS vw_ProcessingPerformance AS
        SELECT
            ft.FilterName,
            ft.DisplayName AS FilterDisplayName,
            ft.Category,
            AVG(CAST(i.FileSize AS REAL) / (1024.0 * 1024.0)) AS AvgImageSizeMB,
            AVG(CAST(i.Width * i.Height AS REAL)) AS AvgTotalPixels,
            AVG(r.ProcessingTimeMs) AS AvgProcessingTimeMs,
            MIN(r.ProcessingTimeMs) AS MinProcessingTimeMs,
            MAX(r.ProcessingTimeMs) AS MaxProcessingTimeMs,
            AVG(r.PSNR) AS AvgPSNR,
            MAX(r.PSNR) AS MaxPSNR,
            AVG(r.SSIM) AS AvgSSIM,
            MAX(r.SSIM) AS MaxSSIM,
            COUNT(r.ResultID) AS TotalExperiments
        FROM Results r
        INNER JOIN Experiments e ON r.ExperimentID = e.ExperimentID
        INNER JOIN Images i ON e.ImageID = i.ImageID
        INNER JOIN FilterConfigs fc ON e.FilterConfigID = fc.FilterConfigID
        INNER JOIN FilterTypes ft ON fc.FilterTypeID = ft.FilterTypeID
        WHERE r.PSNR IS NOT NULL
        GROUP BY ft.FilterName, ft.DisplayName, ft.Category;
    """)

    # =========================================================================
    # VIEW: vw_BestResultsByFilter
    # =========================================================================
    print(" Creating vw_BestResultsByFilter view...")
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS vw_BestResultsByFilter AS
        SELECT
            ft.FilterName,
            ft.DisplayName AS FilterDisplayName,
            ft.Category,
            nt.NoiseTypeName,
            AVG(r.PSNR) AS AvgPSNR,
            AVG(r.SSIM) AS AvgSSIM,
            AVG(r.ProcessingTimeMs) AS AvgProcessingTimeMs,
            COUNT(r.ResultID) AS TotalExperiments,
            MAX(r.PSNR) AS BestPSNR,
            MAX(r.SSIM) AS BestSSIM
        FROM Results r
        INNER JOIN Experiments e ON r.ExperimentID = e.ExperimentID
        INNER JOIN FilterConfigs fc ON e.FilterConfigID = fc.FilterConfigID
        INNER JOIN FilterTypes ft ON fc.FilterTypeID = ft.FilterTypeID
        LEFT JOIN NoiseConfigs nc ON e.NoiseConfigID = nc.NoiseConfigID
        LEFT JOIN NoiseTypes nt ON nc.NoiseTypeID = nt.NoiseTypeID
        WHERE r.PSNR IS NOT NULL
        GROUP BY ft.FilterName, ft.DisplayName, ft.Category, nt.NoiseTypeName;
    """)

    # =========================================================================
    # VIEW: vw_ImageSizeDistribution
    # =========================================================================
    print(" Creating vw_ImageSizeDistribution view...")
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS vw_ImageSizeDistribution AS
        SELECT
            CASE
                WHEN i.Width * i.Height < 500000 THEN 'Small (<500K pixels)'
                WHEN i.Width * i.Height < 2000000 THEN 'Medium (500K-2M pixels)'
                WHEN i.Width * i.Height < 8000000 THEN 'Large (2M-8M pixels)'
                ELSE 'Very Large (>8M pixels)'
            END AS ImageSizeCategory,
            i.FileFormat AS ImageType,
            COUNT(DISTINCT i.ImageID) AS ImageCount,
            AVG(r.ProcessingTimeMs) AS AvgProcessingTimeMs,
            AVG(r.PSNR) AS AvgPSNR,
            AVG(r.SSIM) AS AvgSSIM
        FROM Images i
        INNER JOIN Experiments e ON i.ImageID = e.ImageID
        INNER JOIN Results r ON e.ExperimentID = r.ExperimentID
        WHERE r.PSNR IS NOT NULL
        GROUP BY ImageSizeCategory, i.FileFormat;
    """)

    # =========================================================================
    # SEED DATA
    # =========================================================================
    print(" Inserting seed data...")

    noise_types = [
        (1, 'gaussien', 'Gaussian Noise', 'Statistical noise', 'Statistical', 1),
        (2, 'poivre_sel', 'Salt & Pepper', 'Impulse noise', 'Transmission', 1),
        (3, 'shot', 'Shot/Poisson Noise', 'Quantum noise', 'Sensor', 1),
        (4, 'speckle', 'Speckle Noise', 'Multiplicative noise', 'Sensor', 1),
        (5, 'uniforme', 'Uniform Noise', 'Uniform distribution', 'Statistical', 1),
    ]
    cursor.executemany("""
        INSERT OR REPLACE INTO NoiseTypes 
        (NoiseTypeID, NoiseTypeName, DisplayName, Description, Category, IsActive) 
        VALUES (?, ?, ?, ?, ?, ?)
    """, noise_types)

    filter_types = [
        (1, 'median', 'Median Filter', 'KF', 'Non-linear', None, 1),
        (2, 'bilateral', 'Bilateral Filter', 'KF', 'Edge-preserving', None, 1),
        (3, 'lowpass_gaussian', 'Gaussian Low-pass', 'KF', 'Frequency domain', None, 1),
        (4, 'lowpass_butterworth', 'Butterworth Low-pass', 'KF', 'Frequency domain', None, 1),
        (5, 'ffdnet', 'FFDNet', 'AI', 'Deep Learning', 'Filtres_IA/ffdnet_best.pth', 1),
        (6, 'dncnn', 'DnCNN', 'AI', 'Deep Learning', 'Filtres_IA/dncnn_best.pth', 1),
        (7, 'wdenet', 'W-DENet (Hybrid)', 'HYB', 'Hybrid', None, 1),
    ]
    cursor.executemany("""
        INSERT OR REPLACE INTO FilterTypes 
        (FilterTypeID, FilterName, DisplayName, Category, Description, ModelPath, IsActive) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, filter_types)

    conn.commit()
    conn.close()

    print("\n" + "="*70)
    print(" DATABASE CREATED SUCCESSFULLY (KERNEL SIZE FIX APPLIED)!")
    print("="*70)
    print(f" Database: {os.path.abspath(db_path)}")
    return db_path

if __name__ == "__main__":
    create_database()
