"""
Noise Generator for Friendly Noise Defense.

This module implements a CNN-based noise generator that takes an image as input
and outputs additive noise of the same dimensions. The generator is designed to
be trained in an alternating fashion with the main classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseGenerator(nn.Module):
    """
    A simple CNN-based noise generator using an encoder-decoder architecture.
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB images)
        base_channels: Number of base channels for the network (default: 32)
        num_layers: Number of encoder/decoder layers (default: 3)
        noise_clamp: Maximum absolute value for generated noise (default: 32/255)
    """
    
    def __init__(self, in_channels=3, base_channels=32, num_layers=3, noise_clamp=32/255):
        super(NoiseGenerator, self).__init__()
        self.noise_clamp = noise_clamp
        
        # Encoder layers
        encoder_layers = []
        curr_channels = in_channels
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(curr_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            curr_channels = out_channels
        self.encoder = nn.ModuleList(encoder_layers)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(curr_channels, curr_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(curr_channels),
            nn.ReLU(inplace=True)
        )
        
        # Decoder layers
        decoder_layers = []
        for i in range(num_layers):
            in_dec_channels = curr_channels
            out_dec_channels = curr_channels // 2 if i < num_layers - 1 else in_channels
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_dec_channels, out_dec_channels, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_dec_channels) if i < num_layers - 1 else nn.Identity(),
                    nn.ReLU(inplace=True) if i < num_layers - 1 else nn.Identity()
                )
            )
            curr_channels = out_dec_channels
        self.decoder = nn.ModuleList(decoder_layers)
        
        # Final layer to produce noise (tanh activation to bound output)
        self.output_layer = nn.Tanh()
        
    def forward(self, x):
        """
        Generate noise for input image.
        
        Args:
            x: Input image tensor of shape (B, C, H, W)
            
        Returns:
            Noise tensor of same shape as input, clamped to [-noise_clamp, noise_clamp]
        """
        # Encode
        encoded = x
        for encoder_block in self.encoder:
            encoded = encoder_block(encoded)
        
        # Bottleneck
        encoded = self.bottleneck(encoded)
        
        # Decode
        decoded = encoded
        for decoder_block in self.decoder:
            decoded = decoder_block(decoded)
        
        # Generate noise with tanh activation and scale by noise_clamp
        noise = self.output_layer(decoded) * self.noise_clamp
        
        return noise


class SimpleNoiseGenerator(nn.Module):
    """
    A simpler, lightweight noise generator with fewer parameters.
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB images)
        hidden_channels: Number of hidden channels (default: 64)
        noise_clamp: Maximum absolute value for generated noise (default: 32/255)
    """
    
    def __init__(self, in_channels=3, hidden_channels=64, noise_clamp=32/255):
        super(SimpleNoiseGenerator, self).__init__()
        self.noise_clamp = noise_clamp
        
        self.net = nn.Sequential(
            # First conv block
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            # Second conv block
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            # Third conv block
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            
            # Output layer
            nn.Conv2d(hidden_channels // 2, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        """
        Generate noise for input image.
        
        Args:
            x: Input image tensor of shape (B, C, H, W)
            
        Returns:
            Noise tensor of same shape as input, clamped to [-noise_clamp, noise_clamp]
        """
        noise = self.net(x) * self.noise_clamp
        return noise


def generator_picker(arch='simple', in_channels=3, noise_clamp=32/255, **kwargs):
    """
    Factory function to create a noise generator.
    
    Args:
        arch: Architecture type ('simple', 'encoder-decoder', or 'unet')
        in_channels: Number of input channels
        noise_clamp: Maximum absolute value for generated noise
        **kwargs: Additional architecture-specific arguments
        
    Returns:
        Noise generator model
    """
    if arch == 'simple':
        return SimpleNoiseGenerator(
            in_channels=in_channels,
            hidden_channels=kwargs.get('hidden_channels', 64),
            noise_clamp=noise_clamp
        )
    elif arch == 'encoder-decoder':
        return NoiseGenerator(
            in_channels=in_channels,
            base_channels=kwargs.get('base_channels', 32),
            num_layers=kwargs.get('num_layers', 3),
            noise_clamp=noise_clamp
        )
    else:
        raise ValueError(f'Invalid noise generator architecture: {arch}')
