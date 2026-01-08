"""
FFDNet Model Architecture
Fast and Flexible Denoising Network

Reference:
    Zhang et al., "FFDNet: Toward a Fast and Flexible Solution for 
    CNN-Based Image Denoising", TIP 2018


"""

import torch
import torch.nn as nn


class FFDNet(nn.Module):
    """
    FFDNet: Fast and Flexible Denoising Network
    
    This network can handle variable noise levels by taking the noise
    level as an additional input (noise map).
    
    Architecture:
        Input: [Noisy image + Noise level map] concatenated
        Output: Denoised image
    
    Parameters
    ----------
    num_channels : int
        Number of image channels (1 for grayscale, 3 for RGB)
    num_features : int
        Number of feature maps in hidden layers (default: 64)
    num_layers : int
        Number of convolutional layers (default: 15)
    kernel_size : int
        Convolution kernel size (default: 3)
    
    Attributes
    ----------
    layers : nn.Sequential
        Sequential container of all network layers
    
    Examples
    --------
    >>> model = FFDNet(num_channels=1, num_features=64, num_layers=15)
    >>> noisy = torch.randn(1, 1, 256, 256)
    >>> noise_sigma = 25.0
    >>> clean = model(noisy, noise_sigma)
    >>> print(clean.shape)  # (1, 1, 256, 256)
    """
    
    def __init__(
        self, 
        num_channels=1, 
        num_features=64, 
        num_layers=15,
        kernel_size=3
    ):
        super(FFDNet, self).__init__()
        
        self.num_channels = num_channels
        self.num_features = num_features
        self.num_layers = num_layers
        
        # Input has one extra channel for noise level map
        input_channels = num_channels + 1
        
        padding = kernel_size // 2
        
        # Build network layers
        layers = []
        
        # First layer: Conv + ReLU
        layers.append(
            nn.Conv2d(
                input_channels, 
                num_features, 
                kernel_size=kernel_size,
                padding=padding,
                bias=False
            )
        )
        layers.append(nn.ReLU(inplace=True))
        
        # Middle layers: Conv + BatchNorm + ReLU
        for _ in range(num_layers - 2):
            layers.append(
                nn.Conv2d(
                    num_features,
                    num_features,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False
                )
            )
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        
        # Final layer: Conv (no activation)
        layers.append(
            nn.Conv2d(
                num_features,
                num_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False
            )
        )
        
        self.layers = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, noisy_image, noise_sigma):
        """
        Forward pass through the network
        
        Parameters
        ----------
        noisy_image : torch.Tensor
            Noisy input image with shape (B, C, H, W)
            Values should be in range [0, 1]
        noise_sigma : float or torch.Tensor
            Noise level (standard deviation)
            - If float: same noise level for all images in batch
            - If Tensor: shape (B,) for different noise per image
            Values should be in range [0, 255] (will be normalized)
        
        Returns
        -------
        torch.Tensor
            Denoised image with shape (B, C, H, W)
        
        Notes
        -----
        The noise level is converted to a noise map that has the same
        spatial dimensions as the input image but with a single channel.
        """
        batch_size, channels, height, width = noisy_image.shape
        
        # Create noise level map
        if isinstance(noise_sigma, float) or isinstance(noise_sigma, int):
            # Single noise level for entire batch
            noise_map = torch.full(
                (batch_size, 1, height, width),
                noise_sigma,
                dtype=torch.float32,
                device=noisy_image.device
            )
        else:
            # Different noise level per image
            # Reshape from (B,) to (B, 1, 1, 1) and expand
            noise_map = noise_sigma.view(batch_size, 1, 1, 1).expand(
                batch_size, 1, height, width
            )
        
        # Normalize noise map to [0, 1] if needed
        if noise_map.max() > 1.0:
            noise_map = noise_map / 255.0
        
        # Concatenate image and noise map along channel dimension
        # Shape: (B, C+1, H, W)
        network_input = torch.cat([noisy_image, noise_map], dim=1)
        
        # Pass through network
        denoised = self.layers(network_input)
        
        return denoised
    
    def _initialize_weights(self):
        """
        Initialize network weights using Kaiming initialization
        
        This initialization is designed for ReLU activations and helps
        prevent vanishing/exploding gradients during training.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Kaiming initialization for conv layers
                nn.init.kaiming_normal_(
                    module.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                # Initialize bias if present
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
            elif isinstance(module, nn.BatchNorm2d):
                # Initialize batch norm parameters
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def count_parameters(self):
        """
        Count total number of trainable parameters
        
        Returns
        -------
        int
            Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_model():
    """Test the FFDNet model"""
    print("=" * 70)
    print("FFDNet Model Test")
    print("=" * 70)
    
    # Create model
    model = FFDNet(num_channels=1, num_features=64, num_layers=15)
    print(f"\n✅ Model created")
    print(f"   Parameters: {model.count_parameters():,}")
    
    # Test with single noise level
    batch_size = 4
    height, width = 128, 128
    
    noisy = torch.randn(batch_size, 1, height, width)
    noise_sigma = 25.0
    
    print(f"\n📥 Input:")
    print(f"   Noisy image: {noisy.shape}")
    print(f"   Noise level: {noise_sigma}")
    
    # Forward pass
    with torch.no_grad():
        clean = model(noisy, noise_sigma)
    
    print(f"\n📤 Output:")
    print(f"   Clean image: {clean.shape}")
    
    # Test with variable noise levels
    noise_levels = torch.tensor([15.0, 25.0, 35.0, 45.0])
    
    print(f"\n🔍 Testing with variable noise levels:")
    print(f"   {noise_levels.tolist()}")
    
    with torch.no_grad():
        clean_variable = model(noisy, noise_levels)
    
    print(f"   Output shape: {clean_variable.shape}")
    
    # Test forward/backward
    print(f"\n🧪 Testing backward pass...")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Simulate training step
    optimizer.zero_grad()
    output = model(noisy, noise_sigma)
    target = torch.randn_like(output)  # Dummy target
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    print(f"   Loss: {loss.item():.6f}")
    print(f"   ✅ Backward pass successful")
    
    # Print model architecture
    print(f"\n{'='*70}")
    print("Model Architecture:")
    print(f"{'='*70}")
    print(model)
    
    print(f"\n✅ All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_model()