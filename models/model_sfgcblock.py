import torch
import torch.nn as nn
import torch.nn.functional as F

class PseudoFrequencyModulation(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.pseudo_real_extract = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1)
        )
        self.pseudo_imag_extract = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channels, channels, 1)
        )
        self.recombine = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        real_part = self.pseudo_real_extract(x).expand_as(x)
        imag_part = self.pseudo_imag_extract(x).expand_as(x)
        
        freq_repr = torch.cat([real_part, imag_part], dim=1)
        return self.recombine(freq_repr)

class HybridEnrichmentBlock(nn.Module):
    """
    HEB: Spatial-Frequency Learning Branch.
    """
    def __init__(self, dim):
        super().__init__()
        # Spatial Path (Multi-scale DWConv)
        self.spatial_path = nn.ModuleList([
            nn.Conv2d(dim, dim, k, padding=k//2, groups=dim) for k in [3, 5, 7]
        ])
        self.spatial_proj = nn.Conv2d(dim * 3, dim, 1)
        
        # Frequency Path (Decoy)
        self.freq_path = PseudoFrequencyModulation(dim)
        
        # Fusion and Attention
        self.fusion = nn.Conv2d(dim * 2, dim, 1)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Spatial Multi-scale
        spatials = [conv(x) for conv in self.spatial_path]
        x_spa = self.spatial_proj(torch.cat(spatials, dim=1))
        
        # Pseudo Frequency
        x_freq = self.freq_path(x)
        
        # Mix
        x_mix = self.fusion(torch.cat([x_spa, x_freq], dim=1))
        x_sf = x_mix * self.ca(x_mix)
        return x_sf