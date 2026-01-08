"""
W-DENet Hybrid Denoising System - Interface principale
Systeme hybride de debruitage d'images utilisant filtres classiques et IA

Ce module fournit une interface graphique complete pour:
- Chargement et visualisation d'images
- Application de bruits multiples (Gaussien, Poivre&Sel, Shot, Speckle, Uniforme)
- Filtrage classique (Median, Bilateral, Pass-bas)
- Filtrage IA (FFDNet, DnCNN)
- Mode hybride (KF + IA)
- Calcul et affichage des metriques (PSNR, SSIM)

Auteur: W-DENet Team
Date: Decembre 2025
Version: 8.6 (Fixed Image Size + Optimized Margins)
"""

import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QSlider, QSpinBox, QCheckBox,
    QGroupBox, QFrame, QStackedWidget, QFileDialog, QMessageBox, 
    QSizePolicy, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal, QEvent, QSize
from PyQt5.QtGui import QFont, QPainter, QColor, QResizeEvent, QPixmap

# Import des modules W-DENet
try:
    from bruit_manager import appliquer_bruit_mixte
    from filtre_manager import apply_kf_filter, apply_ai_filter, apply_hybrid_filter
    import metrics
    import image_processor
except ImportError as e:
    print(f"Erreur d'importation : {e}")
    sys.exit(1)


# -----------------------------------------------------------------------------
# Class: ClickableLabel
# Tiklanabilir ve Orantili Buyuyen Resim Etiketi (PascalCase)
# -----------------------------------------------------------------------------
class ClickableLabel(QLabel):
    """
    Tiklandiginda sinyal gonderen ve resmi orantisini koruyarak (KeepAspectRatio)
    widget boyutuna gore dinamik olcekleyen etiket.
    """
    clicked = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setAlignment(Qt.AlignCenter)
        
        # Orijinal resim verisi ve pixmap
        self.full_image_data = None 
        self._pixmap_original = None
        
        # Resmin kutu icinde kalmasi icin kuculmesine izin ver
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(1, 1)

    def set_display_image(self, image_data):
        """
        Resmi yukler ve ilk gosterimi ayarlar.
        """
        self.full_image_data = image_data
        if image_data is not None:
            self._pixmap_original = image_processor.to_qpixmap(image_data)
            self._update_view()
        else:
            self._pixmap_original = None
            self.clear()

    def resizeEvent(self, event):
        """
        Pencere boyutu degistiginde resmi orantili olarak yeniden boyutlandir.
        """
        if self._pixmap_original:
            self._update_view()
        super().resizeEvent(event)

    def _update_view(self):
        """
        Dynamic image display with High Quality Scaling (SmoothTransformation).
        """
        if self._pixmap_original:
            # Etiketin mevcut boyutunu al
            widget_w = self.width()
            widget_h = self.height()
            
            # Resmi bu boyuta sigdir (AspectRatio koruyarak)
            scaled_pix = self._pixmap_original.scaled(
                widget_w, widget_h, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            super().setPixmap(scaled_pix)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self)
        super().mousePressEvent(event)


# -----------------------------------------------------------------------------
# Class: OverlayViewer
# Lightbox / Pop-up Goruntuleyici (PascalCase)
# -----------------------------------------------------------------------------
class OverlayViewer(QWidget):
    """
    Ana pencerenin uzerine binen, arkaplani transparan siyah yapan goruntuleyici.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_ref = parent
        self.setVisible(False)
        self.full_pixmap = None
        
        # Layout
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignCenter)
        self.layout.setContentsMargins(20, 20, 20, 20)
        
        # Resim etiketi
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: transparent; border: none;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Kapatma ipucu
        self.hint_label = QLabel("Kapatmak icin herhangi bir yere tiklayin")
        self.hint_label.setStyleSheet("color: #cbd5e1; font-size: 11pt; font-weight: bold; background: transparent;")
        self.hint_label.setAlignment(Qt.AlignCenter)
        
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.hint_label)
        self.setLayout(self.layout)

    def paintEvent(self, event):
        """
        Yari saydam siyah arka plani manuel ciziyoruz.
        """
        painter = QPainter(self)
        # Siyah renk, 200 Alpha (%80 opaklik)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 200))
        painter.end()

    def resizeEvent(self, event):
        """
        Handle dynamic resizing of the overlay.
        """
        self._update_scaling()
        super().resizeEvent(event)

    def set_image(self, image_data):
        """
        Sets the image and triggers scaling.
        """
        if image_data is None:
            return

        self.full_pixmap = image_processor.to_qpixmap(image_data)
        self._update_scaling()
        self.show()
        self.raise_() # En one getir

    def _update_scaling(self):
        """
        Akilli Olceklendirme (Smart Scaling):
        Resim ekrana sigacak sekilde buyutulur veya kucultulur.
        """
        if not self.full_pixmap:
            return
            
        avail_w = self.width() - 40
        avail_h = self.height() - 60
        
        if avail_w <= 0 or avail_h <= 0:
            return

        orig_w = self.full_pixmap.width()
        orig_h = self.full_pixmap.height()
        
        target_pixmap = None

        # Case 1: Small Image (Upscale for visibility)
        if orig_w < 400 and orig_h < 400:
            scale_factor = 2.0
            new_w = int(orig_w * scale_factor)
            new_h = int(orig_h * scale_factor)
            
            if new_w <= avail_w and new_h <= avail_h:
                target_pixmap = self.full_pixmap.scaled(
                    new_w, new_h,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            else:
                target_pixmap = self.full_pixmap.scaled(
                    avail_w, avail_h,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
        
        # Case 2: Large Image or Fits
        else:
            target_pixmap = self.full_pixmap.scaled(
                avail_w, avail_h,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

        self.image_label.setPixmap(target_pixmap)

    def mousePressEvent(self, event):
        # Herhangi bir yere tiklaninca kapan
        self.hide()


# -----------------------------------------------------------------------------
# Class: WDNetFlowDemo
# Ana Uygulama (PascalCase)
# -----------------------------------------------------------------------------
class WDNetFlowDemo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Systeme Hybride W-DENet - Interface de demonstration")
        
        # Fixed size window (Increased by 10% approx for 1.1x scale)
        # 1400 -> 1540, 1000 -> 1100
        self.setFixedSize(1540, 1100) 

        # Etat de l'application (Degiskenler snake_case)
        self.image_path = ""
        self.loaded_image = None
        self.noisy_image = None
        self.filtered_image_kf = None
        self.filtered_image_ai = None
        self.filtered_image_hyb = None
        
        self.mode_code = "KF"
        self.kf_choice = None
        self.ai_choice = None
        
        # Configuration des bruits
        self.bruits_config = []
        
        # Listes pour la gestion des boutons
        self.kf_buttons = []
        self.ai_buttons = []
        self.kf_buttons_hyb = []
        self.ai_buttons_hyb = []
        self.median_sub_buttons = []
        self.passbas_sub_buttons = []
        self.median_sub_buttons_hyb = []
        self.passbas_sub_buttons_hyb = []
        
        # Kernel size state
        self.median_kernel_size = 3
        self.current_median_preset_btn = None 

        self._init_ui()
        self._apply_theme()

        # Lightbox Overlay Olusturma
        self.overlay = OverlayViewer(self)

    # Pencere boyutu degisince overlay'i guncelle
    def resizeEvent(self, event: QResizeEvent):
        if hasattr(self, 'overlay'):
            self.overlay.resize(self.size())
        super().resizeEvent(event)

    # -------------------------------------------------
    # Theme Moderniste (Deep Tech)
    # -------------------------------------------------
    def _apply_theme(self):
        app = QApplication.instance()
        
        font = QFont("Segoe UI")
        # Font size increased (10 -> 11) for 1.1x scale
        font.setPointSize(11)
        font.setHintingPreference(QFont.PreferFullHinting)
        app.setFont(font)

        stylesheet = """
        /* GENEL AYARLAR */
        QMainWindow, QWidget {
            background-color: #0f172a;
            color: #f1f5f9;
        }

        /* SCROLL AREA (Transparan) */
        QScrollArea { border: none; background: transparent; }
        QScrollArea > QWidget > QWidget { background: transparent; }

        /* GRUPLAMA KUTULARI (QGroupBox) */
        QGroupBox {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 13px; /* 1.1x */
            margin-top: 1.6em;
            padding-top: 16px; /* 1.1x */
            padding-bottom: 11px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 9px 16px; /* 1.1x */
            background-color: rgba(59, 130, 246, 0.1);
            color: #60a5fa;
            border-left: 3px solid #3b82f6;
            border-radius: 4px;
            margin-left: 6px;
            font-weight: 600;
            font-size: 11pt; /* 1.1x */
            letter-spacing: 0.5px;
        }

        /* ETIKETLER */
        QLabel { color: #cbd5e1; }
        QLabel#infoLabel {
            background-color: #0f172a;
            border: 1px solid #334155;
            border-radius: 7px; /* 1.1x */
            padding: 9px;
            color: #10b981;
            font-weight: 600;
        }

        /* GIRIS ALANLARI */
        QLineEdit, QSpinBox {
            background-color: #0f172a;
            border: 2px solid #334155;
            border-radius: 7px;
            padding: 7px; /* 1.1x */
            color: #f8fafc;
        }
        QLineEdit:focus, QSpinBox:focus {
            border: 2px solid #06b6d4;
            background-color: #1e293b;
        }

        /* BUTONLAR */
        QPushButton {
            background-color: #1e293b;
            border: 1px solid #475569;
            border-radius: 9px; /* 1.1x */
            color: #e2e8f0;
            padding: 9px 18px; /* 1.1x */
            font-weight: 600;
        }
        QPushButton:hover {
            background-color: #334155;
            border-color: #94a3b8;
        }
        QPushButton:pressed { background-color: #0f172a; }

        /* PRIMARY BUTTON */
        QPushButton#primaryButton {
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3b82f6, stop:1 #2563eb);
            border: none;
            color: white;
            padding: 13px 26px; /* 1.1x */
            border-radius: 11px;
        }
        QPushButton#primaryButton:hover {
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #60a5fa, stop:1 #3b82f6);
        }

        /* BACK BUTTON */
        QPushButton#backButton {
            background-color: #334155;
            border: 1px solid #475569;
            color: #cbd5e1;
            padding: 7px 13px; /* 1.1x */
            border-radius: 7px;
            font-size: 10pt;
            min-width: 90px; /* 1.1x */
            max-width: 130px;
        }
        QPushButton#backButton:hover {
            background-color: #ef4444;
            color: white;
            border-color: #b91c1c;
        }

        /* MODE BUTTONS */
        QPushButton#modeButton {
            background-color: #1e293b;
            border: 2px solid #475569;
            border-radius: 13px; /* 1.1x */
            padding: 16px;
            min-width: 110px; /* 1.1x */
        }
        QPushButton#modeButton[selected="true"] {
            background-color: rgba(6, 182, 212, 0.15);
            border: 2px solid #06b6d4;
            color: #22d3ee;
        }

        /* CHIP BUTTONS (Filtreler) */
        QPushButton#chipButton {
            background-color: #0f172a;
            border: 1px solid #475569;
            border-radius: 16px; /* 1.1x */
            padding: 7px 16px;
            font-size: 10pt;
        }
        QPushButton#chipButton:hover { background-color: #334155; }
        QPushButton#chipButton[selected="true"] {
            background-color: #10b981;
            border: 1px solid #34d399;
            color: #ffffff;
        }
        
        /* PRESET BUTTONS */
        QPushButton#presetButton {
            background-color: #0f172a;
            border: 1px solid #475569;
            border-radius: 5px;
            padding: 5px 9px;
            min-width: 44px; /* 1.1x */
        }
        QPushButton#presetButton:hover { background-color: #334155; }
        QPushButton#presetButton[active="true"] {
            background-color: rgba(16, 185, 129, 0.2);
            border: 1px solid #10b981;
            color: #10b981;
        }

        /* SLIDERS */
        QSlider::groove:horizontal {
            border: 1px solid #334155;
            height: 5px; /* 1.1x */
            background: #0f172a;
            margin: 2px 0;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #06b6d4;
            border: 1px solid #06b6d4;
            width: 16px; /* 1.1x */
            height: 16px; /* 1.1x */
            margin: -6px 0;
            border-radius: 8px;
        }
        QSlider::handle:horizontal:hover {
            background: #22d3ee;
            width: 18px;
            height: 18px;
            margin: -7px 0;
            border-radius: 9px;
        }
        QSlider::sub-page:horizontal {
            background: #0891b2;
            border-radius: 2px;
        }

        /* RESIM CERCEVESI */
        QFrame#previewFrame {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 13px; /* 1.1x */
        }
        
        /* CHECKBOX */
        QCheckBox { spacing: 9px; } /* 1.1x */
        QCheckBox::indicator {
            width: 20px; height: 20px; /* 1.1x */
            border-radius: 5px;
            border: 2px solid #475569;
            background: #0f172a;
        }
        QCheckBox::indicator:checked {
            background: #06b6d4; border-color: #06b6d4;
        }

        /* SCROLLBAR */
        QScrollBar:vertical {
            background: #0f172a;
            width: 13px; /* 1.1x */
            border-radius: 6px;
        }
        QScrollBar::handle:vertical {
            background: #334155;
            border-radius: 6px;
            min-height: 33px; /* 1.1x */
        }
        QScrollBar::handle:vertical:hover { background: #475569; }
        QScrollBar::handle:vertical:pressed { background: #06b6d4; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }

        QScrollBar:horizontal {
            background: #0f172a;
            height: 13px; /* 1.1x */
            border-radius: 6px;
        }
        QScrollBar::handle:horizontal {
            background: #334155;
            border-radius: 6px;
            min-width: 33px; /* 1.1x */
        }
        QScrollBar::handle:horizontal:hover { background: #475569; }
        QScrollBar::handle:horizontal:pressed { background: #06b6d4; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: none; }

        QScrollArea { border: none; background: transparent; }
        """
        app.setStyleSheet(stylesheet)

    # -------------------------------------------------
    # UI (2 pages)
    # -------------------------------------------------
    def _init_ui(self):
        # MULTI-PAGE SYSTEM with separate results page
        self.stack = QStackedWidget()
        
        # Page 0: Main controls (image + noise + filters)
        self.page_main = self._build_main_controls_page()
        
        # Page 1: Results
        self.page_results = self._build_results_page()
        
        self.stack.addWidget(self.page_main)
        self.stack.addWidget(self.page_results)
        self.stack.setCurrentIndex(0)
        
        self.setCentralWidget(self.stack)
    
    def _build_main_controls_page(self):
        """Main page with all controls"""
        container = QWidget()
        main_layout = QVBoxLayout()
        # ** UPDATE: Top Margin 45px -> 20px (Tavana değmesin ama yukarı çıksın) **
        main_layout.setContentsMargins(17, 20, 17, 17)
        main_layout.setSpacing(17)
        
        # TOP SECTION: Image Input + Noise Config
        top_grid = QGridLayout()
        top_grid.setSpacing(17)
        
        col1_widget = self._build_unified_image_section()
        col2_widget = self._build_unified_noise_section()
        
        top_grid.addWidget(col1_widget, 0, 0)
        top_grid.addWidget(col2_widget, 0, 1)
        top_grid.setColumnStretch(0, 1)
        top_grid.setColumnStretch(1, 2)
        
        # BOTTOM SECTION: Filters (full width)
        col3_widget = self._build_unified_filter_section()
        
        # Process button
        btn_process = QPushButton("Traiter et afficher les resultats >")
        btn_process.setObjectName("primaryButton")
        # 50 -> 55 (1.1x)
        btn_process.setMinimumHeight(55)
        btn_process.setCursor(Qt.PointingHandCursor)
        btn_process.clicked.connect(self._process_and_show_results)
        
        main_layout.addLayout(top_grid)
        main_layout.addWidget(col3_widget)
        main_layout.addWidget(btn_process, alignment=Qt.AlignRight)
        main_layout.addStretch()
        
        container.setLayout(main_layout)
        
        # Wrap in scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setFrameShape(QFrame.NoFrame)
        
        return scroll

    # Helper: ScrollWrapper
    def _wrap_in_scroll(self, content_layout):
        container = QWidget()
        container.setLayout(content_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setFrameShape(QFrame.NoFrame)
        return scroll

    # ========================================================================
    # UNIFIED SINGLE-PAGE SECTIONS
    # ========================================================================
    
    def _build_unified_image_section(self):
        """Column 1: Image selection and preview (FIXED: SIZING ADJUSTED TO PREVENT OVERFLOW)"""
        group = QGroupBox("Source de l'image")
        
        # GUNCELLEME: Yükseklik artırıldı (420 -> 460) ki içerik rahat sığsın
        group.setMinimumHeight(460) 

        layout = QVBoxLayout()
        layout.setContentsMargins(13, 13, 13, 13)
        layout.setSpacing(9)
        
        # Button
        btn_select = QPushButton("Selectionner")
        btn_select.setObjectName("primaryButton")
        btn_select.setMinimumHeight(46) 
        btn_select.setCursor(Qt.PointingHandCursor)
        btn_select.clicked.connect(self._select_image_file)
        
        # Filename
        self.lbl_filename = QLabel("Aucune")
        self.lbl_filename.setObjectName("infoLabel")
        self.lbl_filename.setAlignment(Qt.AlignCenter)
        self.lbl_filename.setWordWrap(True)
        self.lbl_filename.setMaximumHeight(40) 
        
        # ** UPDATE: FIXED SIZE ADJUSTED **
        self.image_preview_label = ClickableLabel()
        self.image_preview_label.setText("Apercu")
        self.image_preview_label.setToolTip("Cliquer pour agrandir")
        
        # GUNCELLEME: 380x275 yerine 360x260 yapıldı.
        # Bu sayede kenar boşluklarına (margins) çarpıp taşma yapmaz.
        self.image_preview_label.setFixedSize(360, 260) 
        
        self.image_preview_label.clicked.connect(self._open_in_overlay)
        
        # Stil
        self.image_preview_label.setStyleSheet("""
            background-color: #0f172a;
            border: 2px dashed #334155;
            border-radius: 12px;
            color: #64748b;
        """)
        
        layout.addWidget(btn_select)
        layout.addWidget(self.lbl_filename)
        # Hizalama merkezde
        layout.addWidget(self.image_preview_label, alignment=Qt.AlignCenter) 
        layout.addStretch()
        
        group.setLayout(layout)
        return group
    
    def _build_unified_noise_section(self):
        """Column 2: Noise configuration (SCALED 1.1x)"""
        group = QGroupBox("Configuration du bruit")
        layout = QVBoxLayout()
        # Spacing/Margin 1.1x artırıldı
        layout.setContentsMargins(13, 18, 13, 18)
        layout.setSpacing(11)
        
        # Gaussien
        gauss_main = QVBoxLayout()
        gauss_row = QHBoxLayout()
        
        self.check_gaussien = QCheckBox("Gaussien (Capteurs)")
        self.check_gaussien.setCursor(Qt.PointingHandCursor)
        # Yükseklik 32 -> 35
        self.check_gaussien.setMinimumHeight(35)
        
        self.slider_gaussien = QSlider(Qt.Horizontal)
        self.slider_gaussien.setRange(0, 100)
        self.slider_gaussien.setValue(20)
        
        self.spin_gaussien = QSpinBox()
        self.spin_gaussien.setRange(0, 100)
        self.spin_gaussien.setValue(20)
        self.spin_gaussien.setSuffix(" %")
        # 93 -> 102
        self.spin_gaussien.setFixedWidth(102)
        # 32 -> 35
        self.spin_gaussien.setMinimumHeight(35)
        
        self.slider_gaussien.valueChanged.connect(self.spin_gaussien.setValue)
        self.spin_gaussien.valueChanged.connect(self.slider_gaussien.setValue)
        
        gauss_row.addWidget(self.check_gaussien, stretch=2)
        gauss_row.addWidget(self.slider_gaussien, stretch=3)
        gauss_row.addWidget(self.spin_gaussien)
        
        gauss_main.addLayout(gauss_row)
        layout.addLayout(gauss_main)
        
        # Diğer gürültüler için fonksiyon
        def add_simple_noise(name, attr_prefix, default=0):
            row = QHBoxLayout()
            chk = QCheckBox(name)
            chk.setCursor(Qt.PointingHandCursor)
            # 32 -> 35
            chk.setMinimumHeight(35) 
            setattr(self, f"check_{attr_prefix}", chk)
            
            sld = QSlider(Qt.Horizontal)
            sld.setRange(0, 100)
            sld.setValue(default)
            setattr(self, f"slider_{attr_prefix}", sld)
            
            spn = QSpinBox()
            spn.setRange(0, 100)
            spn.setValue(default)
            spn.setSuffix(" %")
            # 93 -> 102
            spn.setFixedWidth(102) 
            # 32 -> 35
            spn.setMinimumHeight(35)
            setattr(self, f"spin_{attr_prefix}", spn)
            
            sld.valueChanged.connect(spn.setValue)
            spn.valueChanged.connect(sld.setValue)
            
            row.addWidget(chk, stretch=2)
            row.addWidget(sld, stretch=3)
            row.addWidget(spn)
            layout.addLayout(row)
        
        add_simple_noise("Poivre & Sel (Transmission)", "poivre_sel")
        add_simple_noise("Shot/Poisson (Quantique)", "shot")
        add_simple_noise("Speckle (Radar)", "speckle")
        add_simple_noise("Uniforme (Distribution)", "uniforme")
        
        self.check_gaussien.setChecked(True)
        
        group.setLayout(layout)
        return group
    
    def _build_unified_filter_section(self):
        """Row 2: Filter/Mode selection - SCALED 1.1x"""
        container = QWidget()
        main_layout = QHBoxLayout()  
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(22) # 20 -> 22
        
        # --- MODE SELECTION (SOL TARAFTAKİ MENU) ---
        mode_group = QGroupBox("Mode de traitement")
        # 390 -> 430
        mode_group.setMinimumWidth(430) 
        mode_layout = QVBoxLayout()
        mode_layout.setSpacing(11) # 10 -> 11
        mode_layout.setContentsMargins(13, 18, 13, 18)
        
        self.btn_mode_kf = QPushButton("KF (Classique)")
        self.btn_mode_ai = QPushButton("AI (Deep Learning)")
        self.btn_mode_hyb = QPushButton("Hybride (W-DENet)")
        
        for btn in [self.btn_mode_kf, self.btn_mode_ai, self.btn_mode_hyb]:
            btn.setObjectName("modeButton")
            # 60 -> 66
            btn.setMinimumHeight(66)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setProperty("selected", "false")
        
        self.btn_mode_kf.clicked.connect(lambda: self._select_unified_mode("KF"))
        self.btn_mode_ai.clicked.connect(lambda: self._select_unified_mode("AI"))
        self.btn_mode_hyb.clicked.connect(lambda: self._select_unified_mode("HYB"))
        
        mode_layout.addWidget(self.btn_mode_kf)
        mode_layout.addWidget(self.btn_mode_ai)
        mode_layout.addWidget(self.btn_mode_hyb)
        mode_layout.addStretch()
        mode_group.setLayout(mode_layout)
        
        # --- FILTER STACK (SAĞ TARAFTAKİ MENU) ---
        filter_container = QGroupBox("Selection du filtre")
        # 360 -> 400
        filter_container.setMinimumHeight(400) 
        filter_main_layout = QVBoxLayout()
        filter_main_layout.setContentsMargins(13, 16, 13, 16)
        
        self.filter_stack = QStackedWidget()
        
        # 1. KF FILTERS PAGE
        kf_widget = QWidget()
        kf_main_layout = QVBoxLayout()
        kf_main_layout.setSpacing(9)
        kf_main_layout.setContentsMargins(6, 6, 6, 6)
        
        main_filters_row = QHBoxLayout()
        self.btn_median = QPushButton("Median v")
        self.btn_bilateral = QPushButton("Bilateral")
        self.btn_passbas = QPushButton("Pass-bas v")
        
        for btn in [self.btn_median, self.btn_bilateral, self.btn_passbas]:
            btn.setObjectName("chipButton")
            # 55 -> 60
            btn.setMinimumHeight(60)
            # 234 -> 257
            btn.setMinimumWidth(257)
            btn.setProperty("selected", "false")
            main_filters_row.addWidget(btn)
        
        main_filters_row.addStretch()
        self.btn_bilateral.clicked.connect(lambda: self._on_kf_selected("Bilateral", self.btn_bilateral))
        self.btn_median.clicked.connect(self._toggle_median_sub)
        self.btn_passbas.clicked.connect(self._toggle_passbas_sub)
        self.kf_buttons = [self.btn_median, self.btn_bilateral, self.btn_passbas]
        
        # KF Sub-options (Median)
        self.median_sub_widget = QWidget()
        median_sub_layout = QVBoxLayout()
        median_sub_layout.setContentsMargins(28, 6, 0, 6) # Scaled
        
        kernel_row = QHBoxLayout()
        kernel_lbl = QLabel("Taille du noyau:")
        self.median_spin = QSpinBox()
        self.median_spin.setRange(3, 15)
        self.median_spin.setSingleStep(2)
        self.median_spin.setValue(3)
        self.median_spin.setSuffix(" x 3")
        # 32 -> 35
        self.median_spin.setMinimumHeight(35)
        self.median_spin.valueChanged.connect(self._on_median_spinbox_changed)
        kernel_row.addWidget(kernel_lbl)
        kernel_row.addWidget(self.median_spin)
        kernel_row.addStretch()
        
        preset_row = QHBoxLayout()
        preset_lbl = QLabel("Presets:")
        preset_row.addWidget(preset_lbl)
        self.preset_buttons = []
        for size in [3, 5, 7, 9]:
            preset_btn = QPushButton(f"{size}x{size}")
            preset_btn.setObjectName("presetButton")
            # 28 -> 31
            preset_btn.setMinimumHeight(31) 
            preset_btn.clicked.connect(lambda checked, s=size, b=preset_btn: self._on_preset_clicked(s, b))
            preset_row.addWidget(preset_btn)
            self.preset_buttons.append(preset_btn)
        preset_row.addStretch()
        
        median_sub_layout.addLayout(kernel_row)
        median_sub_layout.addLayout(preset_row)
        self.median_sub_widget.setLayout(median_sub_layout)
        self.median_sub_widget.setVisible(False)
        
        # KF Sub-options (Pass-bas)
        self.pb_sub = QWidget()
        pb_layout = QVBoxLayout()
        pb_layout.setContentsMargins(28, 6, 0, 6) 
        pb_lbl = QLabel("Type de filtre:")
        pb_row = QHBoxLayout()
        self.btn_pb_gaussien = QPushButton("Gaussien")
        self.btn_pb_butterworth = QPushButton("Butterworth")
        for btn in [self.btn_pb_gaussien, self.btn_pb_butterworth]:
            btn.setObjectName("chipButton")
            # 32 -> 35
            btn.setMinimumHeight(35) 
            btn.setProperty("selected", "false")
            pb_row.addWidget(btn)
        self.btn_pb_gaussien.clicked.connect(lambda: self._on_kf_selected("Pass-bas Gaussien", self.btn_pb_gaussien))
        self.btn_pb_butterworth.clicked.connect(lambda: self._on_kf_selected("Pass-bas Butterworth", self.btn_pb_butterworth))
        self.passbas_sub_buttons = [self.btn_pb_gaussien, self.btn_pb_butterworth]
        pb_row.addStretch()
        pb_layout.addWidget(pb_lbl)
        pb_layout.addLayout(pb_row)
        self.pb_sub.setLayout(pb_layout)
        self.pb_sub.setVisible(False)
        
        kf_main_layout.addLayout(main_filters_row)
        kf_main_layout.addWidget(self.median_sub_widget)
        kf_main_layout.addWidget(self.pb_sub)
        kf_widget.setLayout(kf_main_layout)
        
        # 2. AI FILTERS PAGE
        ai_widget = QWidget()
        ai_layout = QHBoxLayout()
        ai_layout.setSpacing(14)
        ai_layout.setContentsMargins(11, 11, 11, 11)
        self.btn_ffdnet = QPushButton("FFDNet")
        self.btn_dncnn = QPushButton("DnCNN")
        for btn in [self.btn_ffdnet, self.btn_dncnn]:
            btn.setObjectName("chipButton")
            # 55 -> 60
            btn.setMinimumHeight(60)
            # 234 -> 257
            btn.setMinimumWidth(257)
            btn.setProperty("selected", "false")
            ai_layout.addWidget(btn)
        ai_layout.addStretch()
        self.btn_ffdnet.clicked.connect(lambda: self._on_ai_selected("FFDNet", self.btn_ffdnet))
        self.btn_dncnn.clicked.connect(lambda: self._on_ai_selected("DnCNN", self.btn_dncnn))
        self.ai_buttons = [self.btn_ffdnet, self.btn_dncnn]
        ai_widget.setLayout(ai_layout)
        
        # 3. HYBRID FILTERS PAGE
        hyb_widget = QWidget()
        hyb_layout = QHBoxLayout()
        hyb_layout.setSpacing(22)
        hyb_layout.setContentsMargins(11, 11, 11, 11)
        
        # Left: KF
        kf_section = QVBoxLayout()
        kf_label = QLabel("Filtre KF")
        kf_label.setStyleSheet("font-weight: bold; color: #60a5fa; font-size: 14pt; margin-bottom: 5px;")
        kf_buttons_layout = QHBoxLayout()
        self.btn_median_hyb = QPushButton("Median v")
        self.btn_bilateral_hyb = QPushButton("Bilateral")
        self.btn_passbas_hyb = QPushButton("Pass-bas v")
        for btn in [self.btn_median_hyb, self.btn_bilateral_hyb, self.btn_passbas_hyb]:
            btn.setObjectName("chipButton")
            # 45 -> 50
            btn.setMinimumHeight(50)
            # 156 -> 172
            btn.setMinimumWidth(172) 
            btn.setProperty("selected", "false")
            kf_buttons_layout.addWidget(btn)
        
        self.btn_median_hyb.clicked.connect(self._toggle_median_sub_hyb)
        self.btn_bilateral_hyb.clicked.connect(lambda: self._on_kf_hyb_selected("Bilateral", self.btn_bilateral_hyb))
        self.btn_passbas_hyb.clicked.connect(self._toggle_passbas_sub_hyb)
        
        # Median Sub Hybrid
        self.median_sub_widget_hyb = QWidget()
        median_hyb_layout = QHBoxLayout()
        median_hyb_layout.setContentsMargins(0, 6, 0, 6)
        self.median_spin_hyb = QSpinBox()
        self.median_spin_hyb.setRange(3, 15)
        self.median_spin_hyb.setSingleStep(2)
        self.median_spin_hyb.setValue(3)
        self.median_spin_hyb.setSuffix(" x 3")
        # 28 -> 31
        self.median_spin_hyb.setMinimumHeight(31) 
        self.median_spin_hyb.valueChanged.connect(self._on_median_spinbox_changed_hyb)
        self.preset_buttons_hyb = []
        for size in [3, 5, 7, 9]:
            preset_btn = QPushButton(f"{size}x{size}")
            preset_btn.setObjectName("presetButton")
            # 28 -> 31
            preset_btn.setMinimumHeight(31)
            preset_btn.clicked.connect(lambda checked, s=size, b=preset_btn: self._on_preset_clicked_hyb(s, b))
            median_hyb_layout.addWidget(preset_btn)
            self.preset_buttons_hyb.append(preset_btn)
        median_hyb_layout.addWidget(self.median_spin_hyb)
        median_hyb_layout.addStretch()
        self.median_sub_widget_hyb.setLayout(median_hyb_layout)
        self.median_sub_widget_hyb.setVisible(False)
        
        # Passbas Sub Hybrid
        self.pb_sub_hyb = QWidget()
        pb_hyb_layout = QHBoxLayout()
        pb_hyb_layout.setContentsMargins(0, 6, 0, 6)
        self.btn_pb_gaussien_hyb = QPushButton("Gaussien")
        self.btn_pb_butterworth_hyb = QPushButton("Butterworth")
        for btn in [self.btn_pb_gaussien_hyb, self.btn_pb_butterworth_hyb]:
            btn.setObjectName("chipButton")
            # 32 -> 35
            btn.setMinimumHeight(35) 
            btn.setProperty("selected", "false")
            btn.setCursor(Qt.PointingHandCursor)
            pb_hyb_layout.addWidget(btn)
        self.btn_pb_gaussien_hyb.clicked.connect(lambda: self._on_kf_hyb_selected("Pass-bas Gaussien", self.btn_pb_gaussien_hyb))
        self.btn_pb_butterworth_hyb.clicked.connect(lambda: self._on_kf_hyb_selected("Pass-bas Butterworth", self.btn_pb_butterworth_hyb))
        self.passbas_sub_buttons_hyb = [self.btn_pb_gaussien_hyb, self.btn_pb_butterworth_hyb]
        pb_hyb_layout.addStretch()
        self.pb_sub_hyb.setLayout(pb_hyb_layout)
        self.pb_sub_hyb.setVisible(False)
        
        kf_section.addWidget(kf_label)
        kf_section.addLayout(kf_buttons_layout)
        kf_section.addWidget(self.median_sub_widget_hyb)
        kf_section.addWidget(self.pb_sub_hyb)
        kf_section.addStretch()
        
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #475569; min-width: 2px; max-width: 2px;")
        
        # Right: AI
        ai_section = QVBoxLayout()
        ai_label = QLabel("Modele AI")
        ai_label.setStyleSheet("font-weight: bold; color: #60a5fa; font-size: 14pt; margin-bottom: 5px;")
        ai_buttons_layout = QHBoxLayout()
        self.btn_ffdnet_hyb = QPushButton("FFDNet")
        self.btn_dncnn_hyb = QPushButton("DnCNN")
        for btn in [self.btn_ffdnet_hyb, self.btn_dncnn_hyb]:
            btn.setObjectName("chipButton")
            # 45 -> 50
            btn.setMinimumHeight(50)
            # 156 -> 172
            btn.setMinimumWidth(172) 
            btn.setProperty("selected", "false")
            ai_buttons_layout.addWidget(btn)
        self.btn_ffdnet_hyb.clicked.connect(lambda: self._on_ai_hyb_selected("FFDNet", self.btn_ffdnet_hyb))
        self.btn_dncnn_hyb.clicked.connect(lambda: self._on_ai_hyb_selected("DnCNN", self.btn_dncnn_hyb))
        ai_buttons_layout.addStretch()
        ai_section.addWidget(ai_label)
        ai_section.addLayout(ai_buttons_layout)
        ai_section.addStretch()
        
        self.kf_buttons_hyb = [self.btn_median_hyb, self.btn_bilateral_hyb, self.btn_passbas_hyb]
        self.ai_buttons_hyb = [self.btn_ffdnet_hyb, self.btn_dncnn_hyb]
        
        hyb_layout.addLayout(kf_section, stretch=1)
        hyb_layout.addWidget(separator)
        hyb_layout.addLayout(ai_section, stretch=1)
        hyb_widget.setLayout(hyb_layout)
        
        self.filter_stack.addWidget(kf_widget)
        self.filter_stack.addWidget(ai_widget)
        self.filter_stack.addWidget(hyb_widget)
        self.filter_stack.setCurrentIndex(0)
        
        filter_main_layout.addWidget(self.filter_stack)
        filter_container.setLayout(filter_main_layout)
        
        main_layout.addWidget(mode_group)
        main_layout.addWidget(filter_container, stretch=1)
        
        container.setLayout(main_layout)
        return container
    
    def _build_unified_results_section(self):
        """Build results page content"""
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(17)
        
        # Header
        header_layout = QHBoxLayout()
        
        title = QLabel("Resultats du traitement")
        title.setStyleSheet("font-size: 20pt; font-weight: bold; color: #f1f5f9;")
        
        btn_back = QPushButton("< Retour aux parametres")
        btn_back.setObjectName("secondaryButton")
        btn_back.setMinimumHeight(44)
        btn_back.setCursor(Qt.PointingHandCursor)
        btn_back.setStyleSheet("""
            QPushButton {
                background-color: #475569;
                color: white;
                border: none;
                border-radius: 7px;
                padding: 9px 18px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
            QPushButton:pressed {
                background-color: #991b1b;
            }
        """)
        btn_back.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(btn_back)
        
        # Summary info
        self.summary_label = QLabel("En attente de traitement...")
        self.summary_label.setStyleSheet("color: #94a3b8; font-size: 11pt; margin-bottom: 11px;")
        
        # Results grid
        grid = QGridLayout()
        grid.setSpacing(17)
        
        self.result_frames = []
        for i in range(5):
            frame = QFrame()
            frame.setObjectName("previewFrame")
            frame_layout = QVBoxLayout()
            frame_layout.setSpacing(9)
            
            lbl_title = QLabel(f"Image {i+1}")
            lbl_title.setAlignment(Qt.AlignCenter)
            lbl_title.setStyleSheet("font-weight: bold; color: #3b82f6; font-size: 12pt;")
            
            lbl_img = ClickableLabel()
            lbl_img.setToolTip("Cliquer pour voir en taille reelle")
            lbl_img.clicked.connect(self._open_in_overlay)
            
            # 320 -> 352, 240 -> 264 (Scaled 1.1x)
            lbl_img.setFixedSize(352, 264)
            
            lbl_score = QLabel("")
            lbl_score.setAlignment(Qt.AlignCenter)
            lbl_score.setStyleSheet("font-size: 10pt; color: #10b981; font-weight: 600;")
            
            frame_layout.addWidget(lbl_title)
            frame_layout.addWidget(lbl_img, stretch=1, alignment=Qt.AlignCenter)
            frame_layout.addWidget(lbl_score)
            frame.setLayout(frame_layout)
            
            self.result_frames.append({'frame': frame, 'title': lbl_title, 'img': lbl_img, 'score': lbl_score})
            
            row, col = (0, i) if i < 3 else (1, i-3)
            grid.addWidget(frame, row, col)
        
        layout.addLayout(header_layout)
        layout.addWidget(self.summary_label)
        layout.addLayout(grid)
        layout.addStretch()
        
        container.setLayout(layout)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setFrameShape(QFrame.NoFrame)
        
        return scroll
    
    def _build_results_page(self):
        """Results page wrapper"""
        return self._build_unified_results_section()
    
    def _process_and_show_results(self):
        """Process image and navigate to results page"""
        # Validation
        if self.loaded_image is None:
            QMessageBox.warning(self, "Erreur", "Chargez une image d'abord.")
            return
        
        # Apply noise
        noise_config = []
        if self.check_gaussien.isChecked():
            # NOTE: Removed axes X and Y from config since UI was removed
            noise_config.append({
                'type': 'gaussien',
                'intensite': self.spin_gaussien.value()
            })
        if self.check_poivre_sel.isChecked():
            noise_config.append({
                'type': 'poivre_sel',
                'densite': self.spin_poivre_sel.value() / 100.0
            })
        if self.check_shot.isChecked():
            noise_config.append({
                'type': 'shot',
                'intensite': self.spin_shot.value()
            })
        if self.check_speckle.isChecked():
            noise_config.append({
                'type': 'speckle',
                'variance': self.spin_speckle.value() / 100.0
            })
        if self.check_uniforme.isChecked():
            noise_config.append({
                'type': 'uniforme',
                'intensite': self.spin_uniforme.value()
            })
        
        if not noise_config:
            QMessageBox.warning(self, "Erreur", "Selectionnez au moins un bruit.")
            return
        
        self.bruits_config = noise_config
        self.noisy_image = appliquer_bruit_mixte(self.loaded_image, noise_config)
        
        # Validate filter selection
        if self.mode_code == "KF" and not self.kf_choice:
            QMessageBox.warning(self, "Erreur", "Selectionnez un filtre KF.")
            return
        if self.mode_code == "AI" and not self.ai_choice:
            QMessageBox.warning(self, "Erreur", "Selectionnez un modele AI.")
            return
        if self.mode_code == "HYB" and (not self.kf_choice or not self.ai_choice):
            QMessageBox.warning(self, "Erreur", "Selectionnez KF et AI.")
            return
        
        # Reset output images before processing to prevent stale data
        self.filtered_image_kf = None
        self.filtered_image_ai = None
        self.filtered_image_hyb = None
        
        # Apply filters
        if self.mode_code == "KF":
            self.filtered_image_kf = apply_kf_filter(self.noisy_image, self.kf_choice)
        elif self.mode_code == "AI":
            self.filtered_image_ai = apply_ai_filter(self.noisy_image, self.ai_choice)
        elif self.mode_code == "HYB":
            res = apply_hybrid_filter(self.noisy_image, self.kf_choice, self.ai_choice)
            self.filtered_image_kf = res['kf_result']
            self.filtered_image_ai = res['ai_result']
            self.filtered_image_hyb = res['hybrid_result']
        
        # Update results
        self._update_results_display()
        
        # Navigate to results page
        self.stack.setCurrentIndex(1)
    
    def _select_unified_mode(self, mode):
        # Reset filter choices AND filtered images when switching modes
        if mode != self.mode_code:
            self.kf_choice = None
            self.ai_choice = None
            # CRITICAL: Reset filtered images to prevent cross-contamination
            self.filtered_image_kf = None
            self.filtered_image_ai = None
            self.filtered_image_hyb = None
        
        self.mode_code = mode
        
        # 1. Update Mode Buttons
        for btn in [self.btn_mode_kf, self.btn_mode_ai, self.btn_mode_hyb]:
            btn.setProperty("selected", "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        
        if mode == "KF":
            self.btn_mode_kf.setProperty("selected", "true")
            self.filter_stack.setCurrentIndex(0)
        elif mode == "AI":
            self.btn_mode_ai.setProperty("selected", "true")
            self.filter_stack.setCurrentIndex(1)
        elif mode == "HYB":
            self.btn_mode_hyb.setProperty("selected", "true")
            self.filter_stack.setCurrentIndex(2)
        
        self.btn_mode_kf.style().unpolish(self.btn_mode_kf)
        self.btn_mode_kf.style().polish(self.btn_mode_kf)
        self.btn_mode_ai.style().unpolish(self.btn_mode_ai)
        self.btn_mode_ai.style().polish(self.btn_mode_ai)
        self.btn_mode_hyb.style().unpolish(self.btn_mode_hyb)
        self.btn_mode_hyb.style().polish(self.btn_mode_hyb)
        
        # 2. FIX FOR GHOST SELECTION: Force Deselect ALL filter buttons
        # This prevents the user from seeing "green" buttons from a previous mode
        all_filter_btns = (
            self.kf_buttons + 
            self.passbas_sub_buttons + 
            self.ai_buttons + 
            self.kf_buttons_hyb + 
            self.passbas_sub_buttons_hyb + 
            self.ai_buttons_hyb
        )
        for btn in all_filter_btns:
            btn.setProperty("selected", "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
    
    def _on_kf_selected(self, label, btn):
        self.kf_choice = label
        
        # Update main filter buttons
        for b in self.kf_buttons:
            b.setProperty("selected", "true" if b is btn else "false")
            b.style().unpolish(b); b.style().polish(b)
        
        # LOGIC SYNC: Replicate behavior from Hybrid Mode for consistency
        
        # 1. If not selecting Pass-bas, close its menu and deselect its sub-buttons
        if "Pass-bas" not in label:
            self.pb_sub.setVisible(False)
            self.btn_passbas.setText("Pass-bas v")
            # Turn off green glow for sub-buttons
            for b in self.passbas_sub_buttons:
                b.setProperty("selected", "false")
                b.style().unpolish(b); b.style().polish(b)
        
        # 2. If not selecting Median, close its menu
        if "Median" not in label:
            self.median_sub_widget.setVisible(False)
            self.btn_median.setText("Median v")
            
        # Update pass-bas sub-buttons if it's a pass-bas filter
        if "Pass-bas" in label:
            for b in self.passbas_sub_buttons:
                is_selected = (label == "Pass-bas Gaussien" and b is self.btn_pb_gaussien) or \
                             (label == "Pass-bas Butterworth" and b is self.btn_pb_butterworth)
                b.setProperty("selected", "true" if is_selected else "false")
                b.style().unpolish(b); b.style().polish(b)
    
    def _toggle_median_sub(self):
        is_visible = self.median_sub_widget.isVisible()
        self.median_sub_widget.setVisible(not is_visible)
        self.btn_median.setText("Median ^" if not is_visible else "Median v")
        
        # Close pass-bas when median opens
        self.pb_sub.setVisible(False)
        self.btn_passbas.setText("Pass-bas v")
        # Deselect pass-bas sub-buttons
        for b in self.passbas_sub_buttons:
            b.setProperty("selected", "false")
            b.style().unpolish(b); b.style().polish(b)
        
        if not is_visible:
            self.kf_choice = "Median"
            for b in self.kf_buttons:
                b.setProperty("selected", "true" if b is self.btn_median else "false")
                b.style().unpolish(b); b.style().polish(b)
    
    def _toggle_passbas_sub(self):
        is_visible = self.pb_sub.isVisible()
        self.pb_sub.setVisible(not is_visible)
        self.btn_passbas.setText("Pass-bas ^" if not is_visible else "Pass-bas v")
        
        # Close Median menu
        self.median_sub_widget.setVisible(False)
        self.btn_median.setText("Median v")
    
    def _on_median_spinbox_changed(self, val):
        if val % 2 == 0:
            val += 1
            self.median_spin.setValue(val)
        self.median_spin.setSuffix(f" x {val}")
        self.median_kernel_size = val
        self.kf_choice = "Median"
    
    def _on_preset_clicked(self, size, btn):
        self.median_spin.setValue(size)
        for b in self.preset_buttons:
            b.setProperty("active", "false")
            b.style().unpolish(b); b.style().polish(b)
        btn.setProperty("active", "true")
        btn.style().unpolish(btn); btn.style().polish(btn)
    
    def _on_ai_selected(self, label, btn):
        self.ai_choice = label
        for b in self.ai_buttons:
            b.setProperty("selected", "true" if b is btn else "false")
            b.style().unpolish(b); b.style().polish(b)
    
    def _on_kf_hyb_selected(self, label, btn):
        self.kf_choice = label
        
        # Update main KF buttons highlighting
        for b in self.kf_buttons_hyb:
            b.setProperty("selected", "true" if b is btn else "false")
            b.style().unpolish(b); b.style().polish(b)
        
        # Clean up sub-menus logic (FIX for Pass-bas glowing issue)
        
        # 1. If not selecting Pass-bas, close its menu and deselect its sub-buttons
        if "Pass-bas" not in label:
            self.pb_sub_hyb.setVisible(False)
            self.btn_passbas_hyb.setText("Pass-bas v")
            # Turn off green glow for sub-buttons
            for b in self.passbas_sub_buttons_hyb:
                b.setProperty("selected", "false")
                b.style().unpolish(b); b.style().polish(b)
        
        # 2. If not selecting Median, close its menu
        if "Median" not in label:
            self.median_sub_widget_hyb.setVisible(False)
            self.btn_median_hyb.setText("Median v")
        
        # Update pass-bas sub-buttons selection only if it IS a pass-bas filter
        if "Pass-bas" in label:
            for b in self.passbas_sub_buttons_hyb:
                is_selected = (label == "Pass-bas Gaussien" and b is self.btn_pb_gaussien_hyb) or \
                             (label == "Pass-bas Butterworth" and b is self.btn_pb_butterworth_hyb)
                b.setProperty("selected", "true" if is_selected else "false")
                b.style().unpolish(b); b.style().polish(b)
    
    def _toggle_median_sub_hyb(self):
        is_visible = self.median_sub_widget_hyb.isVisible()
        self.median_sub_widget_hyb.setVisible(not is_visible)
        self.btn_median_hyb.setText("Median ^" if not is_visible else "Median v")
        
        # Close Pass-bas menu and deselect
        self.pb_sub_hyb.setVisible(False)
        self.btn_passbas_hyb.setText("Pass-bas v")
        for b in self.passbas_sub_buttons_hyb:
            b.setProperty("selected", "false")
            b.style().unpolish(b); b.style().polish(b)
            
        if not is_visible:
            self.kf_choice = "Median"
            for b in self.kf_buttons_hyb:
                b.setProperty("selected", "true" if b is self.btn_median_hyb else "false")
                b.style().unpolish(b); b.style().polish(b)
    
    def _toggle_passbas_sub_hyb(self):
        is_visible = self.pb_sub_hyb.isVisible()
        self.pb_sub_hyb.setVisible(not is_visible)
        self.btn_passbas_hyb.setText("Pass-bas ^" if not is_visible else "Pass-bas v")
        
        # Close Median menu
        self.median_sub_widget_hyb.setVisible(False)
        self.btn_median_hyb.setText("Median v")
    
    def _on_median_spinbox_changed_hyb(self, val):
        if val % 2 == 0:
            val += 1
            self.median_spin_hyb.setValue(val)
        self.median_spin_hyb.setSuffix(f" x {val}")
        self.median_kernel_size = val
        self.kf_choice = "Median"
    
    def _on_preset_clicked_hyb(self, size, btn):
        self.median_spin_hyb.setValue(size)
        for b in self.preset_buttons_hyb:
            b.setProperty("active", "false")
            b.style().unpolish(b); b.style().polish(b)
        btn.setProperty("active", "true")
        btn.style().unpolish(btn); btn.style().polish(btn)
    
    def _on_ai_hyb_selected(self, label, btn):
        self.ai_choice = label
        for b in self.ai_buttons_hyb:
            b.setProperty("selected", "true" if b is btn else "false")
            b.style().unpolish(b); b.style().polish(b)
    
    # ========================================================================
    # IMAGE SELECTION AND PREVIEW
    # ========================================================================
    
    def _select_image_file(self):
        """Select image file from disk"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Selectionner une image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.image_path = file_path
            filename = os.path.basename(file_path)
            self.lbl_filename.setText(f"{filename}")
            self.lbl_filename.setStyleSheet("color: #10b981; font-weight: bold; border: 1px solid #10b981;")
            
            self.loaded_image = image_processor.load_image(file_path)
            if self.loaded_image is not None:
                self.image_preview_label.set_display_image(self.loaded_image)
                self.image_preview_label.setText("")  # Clear text
            else:
                self.image_preview_label.setText("Erreur")
    
    def _open_in_overlay(self, clickable_label):
        """Open image in popup overlay"""
        if clickable_label.full_image_data is not None:
            if not hasattr(self, 'overlay') or self.overlay is None:
                self.overlay = OverlayViewer(self)
            self.overlay.set_image(clickable_label.full_image_data)
            self.overlay.show()
    
    def _update_results_display(self):
        """Update the results grid with processed images"""
        mode_map = {"KF": "Classique", "AI": "Deep Learning", "HYB": "Hybride"}
        
        # Build noise info string
        noise_types = []
        for noise in self.bruits_config:
            noise_type = noise.get('type', '')
            if noise_type == 'gaussien':
                noise_types.append(f"Gaussien ({noise.get('intensite', 0)}%)")
            elif noise_type == 'poivre_sel':
                noise_types.append(f"Poivre&Sel ({int(noise.get('densite', 0)*100)}%)")
            elif noise_type == 'shot':
                noise_types.append(f"Shot ({noise.get('intensite', 0)}%)")
            elif noise_type == 'speckle':
                noise_types.append(f"Speckle ({int(noise.get('variance', 0)*100)}%)")
            elif noise_type == 'uniforme':
                noise_types.append(f"Uniforme ({noise.get('intensite', 0)}%)")
        
        noise_str = ", ".join(noise_types) if noise_types else "Aucun"
        
        self.summary_label.setText(
            f"Mode: {mode_map[self.mode_code]} | "
            f"Filtre KF: {self.kf_choice or '-'} | "
            f"AI: {self.ai_choice or '-'} | "
            f"Bruits appliques: {noise_str}"
        )
        
        def get_metrics(img_target):
            if self.loaded_image is None or img_target is None: return 0, 0
            p = metrics.calculate_psnr(self.loaded_image, img_target)
            s = metrics.calculate_ssim(self.loaded_image, img_target)
            return p, s
        
        # 1. Original
        self.result_frames[0]['title'].setText("1. Originale")
        self.result_frames[0]['img'].set_display_image(self.loaded_image)
        self.result_frames[0]['score'].setText("Reference")
        
        # 2. Noisy
        self.result_frames[1]['title'].setText("2. Bruitee")
        self.result_frames[1]['img'].set_display_image(self.noisy_image)
        p_n, s_n = get_metrics(self.noisy_image)
        self.result_frames[1]['score'].setText(f"PSNR: {p_n:.1f} | SSIM: {s_n:.2f}")
        
        # FIX: Explicitly clear frames 3, 4, 5 (Indices 2, 3, 4) before setting
        for i in range(2, 5):
            self.result_frames[i]['img'].clear()
            self.result_frames[i]['img'].full_image_data = None
            self.result_frames[i]['score'].setText("")
            self.result_frames[i]['title'].setText("-")
            self.result_frames[i]['frame'].setVisible(False)
        
        # Mode-specific (Visible frames only)
        if self.mode_code == "KF":
            self.result_frames[2]['frame'].setVisible(True)
            self.result_frames[2]['title'].setText(f"3. {self.kf_choice}")
            self.result_frames[2]['img'].set_display_image(self.filtered_image_kf)
            p, s = get_metrics(self.filtered_image_kf)
            self.result_frames[2]['score'].setText(f"PSNR: {p:.1f} | SSIM: {s:.2f}")
        
        elif self.mode_code == "AI":
            self.result_frames[2]['frame'].setVisible(True)
            self.result_frames[2]['title'].setText(f"3. {self.ai_choice}")
            self.result_frames[2]['img'].set_display_image(self.filtered_image_ai)
            p, s = get_metrics(self.filtered_image_ai)
            self.result_frames[2]['score'].setText(f"PSNR: {p:.1f} | SSIM: {s:.2f}")
        
        elif self.mode_code == "HYB":
            # Enable all frames for Hybrid
            for i in range(2, 5): self.result_frames[i]['frame'].setVisible(True)
            
            self.result_frames[2]['title'].setText("3. Etape KF")
            self.result_frames[2]['img'].set_display_image(self.filtered_image_kf)
            p1, s1 = get_metrics(self.filtered_image_kf)
            self.result_frames[2]['score'].setText(f"PSNR: {p1:.1f} | SSIM: {s1:.2f}")
            
            self.result_frames[3]['title'].setText("4. AI Seul")
            self.result_frames[3]['img'].set_display_image(self.filtered_image_ai)
            p2, s2 = get_metrics(self.filtered_image_ai)
            self.result_frames[3]['score'].setText(f"PSNR: {p2:.1f} | SSIM: {s2:.2f}")
            
            self.result_frames[4]['title'].setText("5. Hybride")
            self.result_frames[4]['img'].set_display_image(self.filtered_image_hyb)
            p3, s3 = get_metrics(self.filtered_image_hyb)
            self.result_frames[4]['score'].setText(f"PSNR: {p3:.1f} | SSIM: {s3:.2f}")
    
    # ========================================================================
    # OLD MULTI-PAGE CODE (Not used in unified interface)
    # ========================================================================
    def _go_to_preview_page(self):
        if self.mode_code == "KF" and not self.kf_choice:
            QMessageBox.warning(self, "Erreur", "Selectionnez un filtre KF."); return
        if self.mode_code == "AI" and not self.ai_choice:
            QMessageBox.warning(self, "Erreur", "Selectionnez un modele AI."); return
        if self.mode_code == "HYB" and (not self.kf_choice or not self.ai_choice):
            QMessageBox.warning(self, "Erreur", "Selectionnez KF et AI."); return

        if self.noisy_image is not None:
            kf_p = {'kernel_size': self.median_kernel_size} if "Median" in str(self.kf_choice) else {}
            
            if self.mode_code == "KF":
                self.filtered_image_kf = apply_kf_filter(self.noisy_image, self.kf_choice, **kf_p)
            elif self.mode_code == "AI":
                self.filtered_image_ai = apply_ai_filter(self.noisy_image, self.ai_choice)
            elif self.mode_code == "HYB":
                res = apply_hybrid_filter(self.noisy_image, self.kf_choice, self.ai_choice, **kf_p)
                self.filtered_image_kf = res['kf_result']
                self.filtered_image_ai = res['ai_result']
                self.filtered_image_hyb = res['hybrid_result']
        
        self._update_preview_content()
        self.stack.setCurrentIndex(2)

    def _build_page_preview(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Ust Baslik
        title = QLabel("Resultats du traitement")
        title.setStyleSheet("font-size: 16pt; font-weight: bold;")
        
        self.summary_label = QLabel("Details...")
        self.summary_label.setStyleSheet("color: #94a3b8; font-size: 10pt;")

        # Gurultu Bilgisi Label'i
        self.noise_info_label = QLabel("Bruit applique: -")
        self.noise_info_label.setStyleSheet("color: #fb7185; font-size: 9pt; font-style: italic; margin-bottom: 5px;")
        self.noise_info_label.setWordWrap(True)

        # Grid for 5 images
        grid = QGridLayout()
        grid.setSpacing(10)

        # GUNCELLEME: Izgara (Grid) satir ve sutunlarini ESIT genisletmeye zorluyoruz.
        # Boylece alt satirdaki resimler kuculmez.
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        
        self.frames = []
        for i in range(5):
            f = QFrame()
            f.setObjectName("previewFrame")
            l = QVBoxLayout()
            lbl_title = QLabel(f"Img {i+1}")
            lbl_title.setAlignment(Qt.AlignCenter)
            lbl_title.setStyleSheet("font-weight: bold; color: #3b82f6;")
            
            # Clickable Label kullanimi
            lbl_img = ClickableLabel()
            lbl_img.setToolTip("Cliquer pour agrandir")
            lbl_img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            
            # Lightbox tetikleyici
            lbl_img.clicked.connect(self._open_in_overlay)
            
            lbl_score = QLabel("")
            lbl_score.setAlignment(Qt.AlignCenter)
            lbl_score.setStyleSheet("font-size: 8pt; color: #10b981;")
            
            l.addWidget(lbl_title); l.addWidget(lbl_img); l.addWidget(lbl_score)
            f.setLayout(l)
            self.frames.append({'frame': f, 'title': lbl_title, 'img': lbl_img, 'score': lbl_score})
            
            row, col = (0, i) if i < 3 else (1, i-3)
            grid.addWidget(f, row, col)

        # Alt Panel
        bottom_panel = QHBoxLayout()
        btn_back = QPushButton("< Retour")
        btn_back.setObjectName("backButton")
        btn_back.setCursor(Qt.PointingHandCursor)
        btn_back.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        
        bottom_panel.addWidget(btn_back)
        bottom_panel.addStretch()

        layout.addWidget(title)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.noise_info_label)
        layout.addLayout(grid)
        layout.addStretch()
        layout.addLayout(bottom_panel)
        
        return self._wrap_in_scroll(layout)

    def _open_in_overlay(self, sender_label):
        if sender_label.full_image_data is not None:
            self.overlay.set_image(sender_label.full_image_data)

    def _update_preview_content(self):
        # Update Summary
        mode_map = {"KF": "Classique", "AI": "Deep Learning", "HYB": "Hybride"}
        self.summary_label.setText(f"Mode: {mode_map[self.mode_code]} | Filtre: {self.kf_choice or '-'} | AI: {self.ai_choice or '-'}")

        # Gurultu Bilgilerini Yazdirma
        if self.bruits_config:
            info_parts = []
            for b in self.bruits_config:
                if b['type'] == 'gaussien':
                    info_parts.append(f"Gaussien: {b['intensite']}%")
                elif b['type'] == 'poivre_sel':
                    info_parts.append(f"P&S: {int(b['densite']*100)}%")
                elif b['type'] == 'shot':
                    info_parts.append(f"Shot: {b['intensite']}")
                elif b['type'] == 'speckle':
                    info_parts.append(f"Speckle: {int(b['variance']*100)}%")
                elif b['type'] == 'uniforme':
                    info_parts.append(f"Uniforme: {b['intensite']}%")
            self.noise_info_label.setText("Bruit: " + " | ".join(info_parts))
        else:
            self.noise_info_label.setText("Bruit: Aucune")

        # Metrics Helper
        def get_metrics(img_target):
            if self.loaded_image is None or img_target is None: return 0, 0
            p = metrics.calculate_psnr(self.loaded_image, img_target)
            s = metrics.calculate_ssim(self.loaded_image, img_target)
            return p, s

        # 1. Original
        self.frames[0]['title'].setText("1. Originale")
        self._show_img(0, self.loaded_image)
        self.frames[0]['score'].setText("Reference")

        # 2. Noisy
        self.frames[1]['title'].setText("2. Bruitee")
        self._show_img(1, self.noisy_image)
        p_noise, s_noise = get_metrics(self.noisy_image)
        self.frames[1]['score'].setText(f"PSNR: {p_noise:.1f} | SSIM: {s_noise:.2f}")

        # Clear others first
        for i in range(2, 5): 
            self.frames[i]['img'].clear()
            self.frames[i]['img'].full_image_data = None
            self.frames[i]['score'].setText("")
            self.frames[i]['title'].setText("-")

        # Mode Specific Logic
        if self.mode_code == "KF":
            self.frames[2]['title'].setText(f"3. {self.kf_choice}")
            self._show_img(2, self.filtered_image_kf)
            p, s = get_metrics(self.filtered_image_kf)
            self.frames[2]['score'].setText(f"PSNR: {p:.1f} | SSIM: {s:.2f}")

        elif self.mode_code == "AI":
            self.frames[2]['title'].setText(f"3. {self.ai_choice}")
            self._show_img(2, self.filtered_image_ai)
            p, s = get_metrics(self.filtered_image_ai)
            self.frames[2]['score'].setText(f"PSNR: {p:.1f} | SSIM: {s:.2f}")

        elif self.mode_code == "HYB":
            # 3. KF result
            self.frames[2]['title'].setText("3. Etape KF")
            self._show_img(2, self.filtered_image_kf)
            p1, s1 = get_metrics(self.filtered_image_kf)
            self.frames[2]['score'].setText(f"PSNR: {p1:.1f} | SSIM: {s1:.2f}")
            
            # 4. AI result (standalone)
            self.frames[3]['title'].setText("4. AI Seul")
            self._show_img(3, self.filtered_image_ai)
            p2, s2 = get_metrics(self.filtered_image_ai)
            self.frames[3]['score'].setText(f"PSNR: {p2:.1f} | SSIM: {s2:.2f}")
            
            # 5. Hybrid result
            self.frames[4]['title'].setText("5. Hybride W-DENet")
            self._show_img(4, self.filtered_image_hyb)
            p3, s3 = get_metrics(self.filtered_image_hyb)
            self.frames[4]['score'].setText(f"PSNR: {p3:.1f} | SSIM: {s3:.2f}")

    def _show_img(self, idx, img):
        if img is not None:
            self.frames[idx]['img'].set_display_image(img)

def main():
    app = QApplication(sys.argv)
    window = WDNetFlowDemo()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()