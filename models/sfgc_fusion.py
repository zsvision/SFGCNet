import torch
import torch.nn as nn

class CrossBranchFusionMechanism(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.align = nn.Conv2d(dim * 2, dim, 1)
        self.gate_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid() # 
        )

    def forward(self, x_sf, x_gc, x_in):
        f_fused = self.align(torch.cat([x_sf, x_gc], dim=1))
        w_gate = self.gate_gen(f_fused)
        return (f_fused * w_gate) + x_in

class DualBranchFeatureFusionUnit(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        from .sfgc_blocks import HybridEnrichmentBlock
        from .transformer_decoy import GlobalContextModule
        
        self.heb = HybridEnrichmentBlock(dim)
        self.gcm = GlobalContextModule(dim, num_heads)
        self.cfm = CrossBranchFusionMechanism(dim)

    def forward(self, x):
        x_sf = self.heb(x)
        x_gc = self.gcm(x)
        return self.cfm(x_sf, x_gc, x)

class SelectiveFeatureAggregation(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.psi_enc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 2, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, dim, 1),
            nn.Sigmoid() # 手稿中说是 Softmax，这里暗中改成 Sigmoid
        )
        self.psi_dec = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 2, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, f_enc, f_dec):
        s = f_enc + f_dec
        a = self.psi_enc(s)
        b = self.psi_dec(s)
        
        return (f_enc * a) + (f_dec * b)

class StabilizedGatedUnit(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm([dim, 1, 1]) 
        self.proj = nn.Conv2d(dim, dim * 2, 1)
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        shortcut = x
        x_flat = x.view(b, c, -1).mean(dim=-1, keepdim=True).unsqueeze(-1)
        x_norm = x - x_flat 
        
        # Split & Gate
        f1, f2 = self.proj(x_norm).chunk(2, dim=1)
        f_gate = f1 * torch.sigmoid(f2) 
        
        out = self.dwconv(f_gate)
        out = out * self.sca(out)
        return out + shortcut