"""
DnCNN Model Mimarisi (PyTorch)
Referans: Zhang et al. (2017)

Bu modül DnCNN ağ yapısını tanımlar.
"""

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """
    DnCNN (Denoising Convolutional Neural Network) Model Sınıfı.
    
    Bitirme projesi standartlarına uygun olarak sınıf ismi PascalCase
    formatında yazılmıştır[cite: 11].
    """
    
    def __init__(self, num_channels=1, num_layers=17, num_features=64):
        """
        Modeli başlatır ve katmanları oluşturur.
        
        Parametreler:
        -----------
        num_channels : int
            Giriş görüntü kanalı sayısı (1: Gri tonlama, 3: RGB)
        num_layers : int
            Ağ derinliği (Katman sayısı)
        num_features : int
            Her katmandaki özellik haritası sayısı
        """
        super(DnCNN, self).__init__()
        
        layers = []
        
        # 1. Katman: Conv + ReLU
        # Değişken isimlendirme standardı: küçük harf ve alt çizgi [cite: 9]
        layers.append(nn.Conv2d(num_channels, num_features, 
                                kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        # Orta Katmanlar: Conv + BN + ReLU
        # Mantıksal bloklar bir arada tutulmuştur [cite: 20]
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, 
                                    kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
            
        # Son Katman: Conv (Residual tahmini)
        layers.append(nn.Conv2d(num_features, num_channels, 
                                kernel_size=3, padding=1, bias=False))
        
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        """
        İleri yayılım (Forward pass) fonksiyonu.
        
        DnCNN residual (kalıntı) öğrenir: Temiz = Gürültülü - Gürültü
        """
        noise_prediction = self.dncnn(x)
        return x - noise_prediction

    def _initialize_weights(self):
        """
        Ağırlık başlatma fonksiyonu.
        
        Fonksiyon isimlendirme standardı: küçük harf ve alt çizgi[cite: 10].
        Ortogonal başlatma kullanır.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)