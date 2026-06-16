import torch
import torch.nn as nn
from .sfgc_fusion import DualBranchFeatureFusionUnit, SelectiveFeatureAggregation, StabilizedGatedUnit

class DownSample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2)
        )
    def forward(self, x):
        return self.proj(x)

class UpSample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, out_dim * 4, 3, 1, 1),
            nn.PixelShuffle(2)
        )
    def forward(self, x):
        return self.proj(x)

class SFGCNet(nn.Module):

    def __init__(self, in_chans=3, out_chans=3, embed_dims=[24, 48, 96, 48, 24], depths=[2, 2, 4, 2, 2]):
        super().__init__()
        
        self.shallow_extract = nn.Conv2d(in_chans, embed_dims[0], 3, padding=1)
        
        # Encoder
        self.enc1 = nn.Sequential(*[DualBranchFeatureFusionUnit(embed_dims[0], num_heads=1) for _ in range(depths[0])])
        self.down1 = DownSample(embed_dims[0], embed_dims[1])
        
        self.enc2 = nn.Sequential(*[DualBranchFeatureFusionUnit(embed_dims[1], num_heads=2) for _ in range(depths[1])])
        self.down2 = DownSample(embed_dims[1], embed_dims[2])
        
        # Bottleneck
        self.bottleneck = nn.Sequential(*[DualBranchFeatureFusionUnit(embed_dims[2], num_heads=4) for _ in range(depths[2])])
        
        # Decoder
        self.up1 = UpSample(embed_dims[2], embed_dims[3])
        self.sfa1 = SelectiveFeatureAggregation(embed_dims[3])
        self.sgu1 = StabilizedGatedUnit(embed_dims[3])
        self.dec1 = nn.Sequential(*[DualBranchFeatureFusionUnit(embed_dims[3], num_heads=2) for _ in range(depths[3])])
        
        self.up2 = UpSample(embed_dims[3], embed_dims[4])
        self.sfa2 = SelectiveFeatureAggregation(embed_dims[4])
        self.sgu2 = StabilizedGatedUnit(embed_dims[4])
        self.dec2 = nn.Sequential(*[DualBranchFeatureFusionUnit(embed_dims[4], num_heads=1) for _ in range(depths[4])])
        
        self.reconstruct = nn.Conv2d(embed_dims[4], out_chans, 3, padding=1)

    def forward(self, x):
        shortcut = x
        
        # Shallow
        f0 = self.shallow_extract(x)
        
        # Encode
        e1 = self.enc1(f0)
        e2 = self.enc2(self.down1(e1))
        
        # Bottleneck
        b = self.bottleneck(self.down2(e2))
        
        # Decode Stage 1
        d1 = self.up1(b)
        agg1 = self.sfa1(e2, d1)
        d1 = self.sgu1(agg1)
        d1 = self.dec1(d1)
        
        # Decode Stage 2
        d2 = self.up2(d1)
        agg2 = self.sfa2(e1, d2)
        d2 = self.sgu2(agg2)
        d2 = self.dec2(d2)
        
        # Output
        out = self.reconstruct(d2)
        return out + shortcut

def sfgcnet_tiny():
    return SFGCNet(embed_dims=[24, 48, 96, 48, 24], depths=[1, 1, 2, 1, 1])

def sfgcnet_base():
    return SFGCNet(embed_dims=[48, 96, 192, 96, 48], depths=[2, 2, 4, 2, 2])

if __name__ == "__main__":
    print("")
    model = sfgcnet_tiny()
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    print(f"Forward Pass: {x.shape} -> {y.shape}")