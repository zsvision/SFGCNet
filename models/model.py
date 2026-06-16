import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numbers
import numpy as np
from einops import rearrange

# ===================================================================================
# 1. SFGCNet
# ===================================================================================
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.rel_pos_bias = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        coords = torch.stack(torch.meshgrid(torch.arange(window_size[0]), torch.arange(window_size[1])))
        coords_flat = coords.flatten(1)
        rel_coords = coords_flat.unsqueeze(2) - coords_flat.unsqueeze(1)
        rel_coords = rel_coords.permute(1, 2, 0).contiguous()
        rel_coords[..., 0] += window_size[0] - 1
        rel_coords[..., 1] += window_size[1] - 1
        rel_coords[..., 0] *= 2 * window_size[1] - 1
        rel_pos_idx = rel_coords.sum(-1)
        self.register_buffer("rel_pos_idx", rel_pos_idx)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.rel_pos_bias, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C//self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        
        q = q * self.scale
        attn = q @ k.transpose(-2,-1)
        
        rel_bias = self.rel_pos_bias[self.rel_pos_idx.view(-1)].view(N, N, -1).permute(2,0,1).contiguous()
        attn = attn + rel_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_//nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1,2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class PatchEmbed_SwinIR(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


def to_2tuple(x):

    return (x, x) if isinstance(x, int) else x

class PatchUnEmbed_SwinIR(nn.Module):
  
    def __init__(self, embed_dim=96, patch_size=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

    def forward(self, x, x_size):

        B, L, C = x.shape
        H, W = x_size
        
        assert L == H * W, f"bad"
        assert C == self.embed_dim, "bad"
        
  
        x = x.view(B, H, W, C)  # [B, H, W, C]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        
        return x


class PatchEmbed_SwinIR(nn.Module):

    def __init__(self, embed_dim=96, patch_size=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

    def forward(self, x):

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x = x.view(B, H * W, C)
        return x

class RSTB(nn.Module):

    def __init__(
        self, dim, input_resolution, depth, num_heads, window_size,
        mlp_ratio=4., qkv_bias=True, qk_scale=None,
        drop=0., attn_drop=0., drop_path=0.,
        norm_layer=nn.LayerNorm, use_checkpoint=False,
        resi_connection='conv_branch'
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = to_2tuple(input_resolution)

        self.norm_pre = norm_layer(dim)
        

        self.transformer_layer = BasicLayer(
            dim=dim, input_resolution=input_resolution, depth=depth,
            num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path,
            norm_layer=norm_layer, use_checkpoint=use_checkpoint
        )


        if resi_connection == 'conv_branch':
            self.conv_branch = nn.Sequential(
                nn.Conv2d(dim, dim, 3, 1, 1, groups=4),  
                nn.GELU(),
                nn.Conv2d(dim, dim, 1, 1, 0),
                nn.Dropout(0.1)
            )
        elif resi_connection == 'light_conv':
            self.conv_branch = nn.Conv2d(dim, dim, 3, 1, 1)
        self.patch_unembed = PatchUnEmbed_SwinIR(embed_dim=dim)
        self.patch_embed = PatchEmbed_SwinIR(embed_dim=dim)
        self.norm_post = norm_layer(dim)

    def forward(self, x, x_size):
        residual = x  # 
        feat = self.norm_pre(x)
        feat = self.transformer_layer(feat, x_size)
        feat = self.patch_unembed(feat, x_size)
        feat = self.conv_branch(feat)
        feat = self.patch_embed(feat)
        out = self.norm_post(feat + residual)
        return out

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

    def forward(self, x, x_size):
        return x

# ===================================================================================
# 2.GatedRefine & SKFF
# ===================================================================================

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class GatedRefine(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.project_in = nn.Conv2d(dim, dim * 2, 1)
        self.sg = SimpleGate()
        
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1)
        )
        

        self.project_out = nn.Conv2d(dim, dim, 1)
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        
        # Norm (B, C, H, W) -> (B, H, W, C) -> Norm -> (B, C, H, W)
        x_norm = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2).contiguous()

        x = self.project_in(x_norm)
        x = self.sg(x)      # Gate (Multiplication)
        x = self.dwconv(x)  # Spatial
        x = x * self.sca(x) # Channel
        x = self.project_out(x)
        
        return x + residual

class SKFF(nn.Module):
    def __init__(self, in_channels, height=2, reduction=2, bias=False): 
        super(SKFF, self).__init__()
        self.height = height
        d = max(int(in_channels / reduction), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, bias=bias), nn.ReLU())
        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, 1, bias=bias))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        if not isinstance(inp_feats, (list, tuple)):
             raise TypeError("SKFF input must be a list")
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]
        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)
        return feats_V


# ==================================================================================
# 3. Backbone_size
# ===================================================================================
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if kernel_size is None:
            kernel_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')
    def forward(self, x):
        return self.proj(x)

class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        if kernel_size is None: kernel_size = 1
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size),
            nn.Conv2d(out_chans, out_chans, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
        )
    def forward(self, x):
        return self.proj(x)

class PatchUnEmbed_for_upsample(nn.Module):
    def __init__(self, patch_size=4, embed_dim=96, out_dim=64, kernel_size=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_dim * patch_size ** 2, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.PixelShuffle(patch_size),
        )
    def forward(self, x):
        return self.proj(x)

class DownSample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=4, stride=2):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(input_dim, input_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))
    def forward(self, x):
        return self.proj(x)


# ===================================================================================
# 4.  Parallel + No HIN
# ===================================================================================
class OurTokenMixer_For_Local(nn.Module):
    def __init__(self, dim, kernel_size=[1,3,5,7], se_ratio=4, local_size=8, scale_ratio=2, spilt_num=4):
        super(OurTokenMixer_For_Local, self).__init__()
        self.dim = dim
        self.dim_sp = dim*scale_ratio//spilt_num
        self.conv_init = nn.Sequential(nn.Conv2d(dim, dim*scale_ratio, 1), nn.GELU())
        self.conv_fina = nn.Sequential(nn.Conv2d(dim*scale_ratio, dim, 1), nn.GELU())
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=kernel_size[1], padding=kernel_size[1] // 2, groups=self.dim_sp, padding_mode='reflect'), nn.GELU())
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=kernel_size[2], padding=kernel_size[2] // 2, groups=self.dim_sp, padding_mode='reflect'), nn.GELU())
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=kernel_size[3], padding=kernel_size[3] // 2, groups=self.dim_sp, padding_mode='reflect'), nn.GELU())

    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim_sp, dim=1))
        x[1] = self.conv1_1(x[1])
        x[2] = self.conv1_2(x[2])
        x[3] = self.conv1_3(x[3])
        x = torch.cat(x, dim=1)
        x = self.conv_fina(x)
        return x

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer1 = nn.Sequential(torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                                         kernel_size=1, stride=1, padding=0, groups=self.groups,bias=True),
                                         nn.GELU())
        self.bn1 = torch.nn.BatchNorm2d(out_channels * 2)
    def forward(self, x):
        batch, c, h, w = x.size()
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        ffted = self.conv_layer1(self.bn1(ffted))
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.view_as_complex(ffted)
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')
        return output

class OurTokenMixer_For_Gloal(nn.Module):
    def __init__(self, dim, kernel_size=[1,3,5,7], se_ratio=4, local_size=8, scale_ratio=2, spilt_num=4):
        super(OurTokenMixer_For_Gloal, self).__init__()
        self.dim = dim
        self.dim_sp = dim*scale_ratio//spilt_num
        self.conv_init = nn.Sequential(nn.Conv2d(dim, dim*2, 1), nn.GELU())
        self.conv_fina = nn.Sequential(nn.Conv2d(dim*2, dim, 1), nn.GELU())
        self.FFC = FourierUnit(self.dim*2, self.dim*2)
    def forward(self, x):
        x = self.conv_init(x)
        x0 = x
        x = self.FFC(x)
        x = self.conv_fina(x+x0)
        return x


class OurMixer(nn.Module):
    def __init__(self, dim, token_mixer_for_local, token_mixer_for_gloal, mixer_kernel_size=[1,3,5,7], local_size=8):
        super().__init__()
        self.dim = dim
        self.mixer_local = token_mixer_for_local(dim, mixer_kernel_size, 8, local_size)
        self.mixer_gloal = token_mixer_for_gloal(dim, mixer_kernel_size, 8, local_size)
        self.ca_conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, padding_mode='reflect'),
            nn.GELU()
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        self.conv_init = nn.Sequential(nn.Conv2d(dim, dim * 2, 1), nn.GELU())

    def forward(self, x):
        x = self.conv_init(x)
        x1, x2 = torch.split(x, self.dim, dim=1)
        x_local = self.mixer_local(x1)
        x_global = self.mixer_gloal(x2)
        x = torch.cat([x_local, x_global], dim=1)
        x = self.ca_conv(x)
        x = self.ca(x) * x
        return x

class CustomModule(nn.Module):
    def __init__(self, hidden_dim, norm=nn.BatchNorm2d, mixer_block=OurMixer, kernels=[1,3,5,7], region_size=8):
        super().__init__()
        self.norm_pre = norm(hidden_dim)
        self.norm_mid = norm(hidden_dim)
        self.feature_mixer = mixer_block(hidden_dim, None, None, kernels, region_size)
        self.feed_forward = nn.Identity()

    def forward(self, x):
        out = self.norm_pre(x)
        out = self.feature_mixer(out)
        out = out + x
        
        residual = out
        out = self.norm_mid(out)
        out = self.feed_forward(out)
        out = out + residual
        return out

class ParallelStage(nn.Module):
    def __init__(self, depth, in_channels, mixer_kernel_size=[1,3,5,7], local_size=8):
        super(ParallelStage, self).__init__()
        self.cnn_branch = nn.Sequential(*[
            OurBlock(dim=in_channels, norm_layer=nn.BatchNorm2d, token_mixer=OurMixer,
                     kernel_size=mixer_kernel_size, local_size=local_size)
            for index in range(depth)
        ])
        self.num_heads = max(1, in_channels // 48)
        while in_channels % self.num_heads != 0: self.num_heads -= 1
        self.window_size = 4
        self.transformer_branch = RSTB(
            dim=in_channels, input_resolution=(64, 64), depth=1, 
            num_heads=self.num_heads, window_size=self.window_size,
            mlp_ratio=1.5, use_checkpoint=False, img_size=64, patch_size=1, resi_connection='1conv'
        )
        self.fusion = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        input_tensor = x
        x_cnn = self.cnn_branch(x)
        B, C, H, W = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            H_pad, W_pad = H + pad_h, W + pad_w
        else:
            x_pad = x
            H_pad, W_pad = H, W
        x_trans = rearrange(x_pad, 'B C H W -> B (H W) C')
        x_trans = self.transformer_branch(x_trans, x_size=(H_pad, W_pad))
        x_trans = rearrange(x_trans, 'B (H W) C -> B C H W', H=H_pad, W=W_pad)
        if pad_h > 0 or pad_w > 0:
            x_trans = x_trans[:, :, :H, :W]
        x_cat = torch.cat([x_cnn, x_trans], dim=1)
        x_fused = self.fusion(x_cat)
        x_fused = x_fused * self.att(x_fused)
        return x_fused + input_tensor


# ===================================================================================
# 5. Backbone_GatedRefine
# ===================================================================================
class Backbone_new(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, patch_size=1,
                 embed_dim=[48, 96, 192, 96, 48], depth=[2, 2, 2, 2, 2],
                 local_size=[4, 4, 4, 4 ,4],
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 norm_layer_transformer=nn.LayerNorm, embed_kernel_size=3,
                 downsample_kernel_size=None, upsample_kernel_size=None):
        super(Backbone_new, self).__init__()

        self.patch_size = patch_size
        if downsample_kernel_size is None: downsample_kernel_size = 4
        if upsample_kernel_size is None: upsample_kernel_size = 4

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=embed_dim[0], kernel_size=embed_kernel_size)
        
        # Layer 1
        self.layer1 = ParallelStage(depth=depth[0], in_channels=embed_dim[0],
                                    mixer_kernel_size=[1, 3, 5, 7], local_size=local_size[0])
        self.skip1 = SKFF(in_channels=embed_dim[0], reduction=2) 
        #  GatedRefine 1
        self.refine1 = GatedRefine(embed_dim[0])

        self.downsample1 = DownSample(input_dim=embed_dim[0], output_dim=embed_dim[1],
                                      kernel_size=downsample_kernel_size, stride=2)
        
        # Layer 2
        self.layer2 = ParallelStage(depth=depth[1], in_channels=embed_dim[1],
                                    mixer_kernel_size=[1, 3, 5, 7], local_size=local_size[1])
        self.skip2 = SKFF(in_channels=embed_dim[1], reduction=2)
        #  GatedRefine 2
        self.refine2 = GatedRefine(embed_dim[1])
        
        self.downsample2 = DownSample(input_dim=embed_dim[1], output_dim=embed_dim[2],
                                      kernel_size=downsample_kernel_size, stride=2)
        
        # Layer 3 (Bottleneck)
        self.layer3 = ParallelStage(depth=depth[2], in_channels=embed_dim[2],
                                    mixer_kernel_size=[1, 3, 5, 7], local_size=local_size[2])
        
        self.upsample1 = PatchUnEmbed_for_upsample(patch_size=2, embed_dim=embed_dim[2], out_dim=embed_dim[3])
        
        # Layer 4
        self.layer4 = ParallelStage(depth=depth[3], in_channels=embed_dim[3],
                                    mixer_kernel_size=[1, 3, 5, 7], local_size=local_size[3])
        
        self.upsample2 = PatchUnEmbed_for_upsample(patch_size=2, embed_dim=embed_dim[3],
                                                   out_dim=embed_dim[4])
        
        # Layer 5
        self.layer5 = ParallelStage(depth=depth[4], in_channels=embed_dim[4],
                                    mixer_kernel_size=[1, 3, 5, 7], local_size=local_size[4])
        
        self.patch_unembed = PatchUnEmbed(patch_size=patch_size, out_chans=out_chans,
                                          embed_dim=embed_dim[4], kernel_size=3)

    def forward(self, x):
        copy0 = x
        x = self.patch_embed(x)
        x = self.layer1(x)
        copy1 = x

        x = self.downsample1(x)
        x = self.layer2(x)
        copy2 = x

        x = self.downsample2(x)
        x = self.layer3(x)
        x = self.upsample1(x)


        x = self.skip2([x, copy2])
  
        x = self.refine2(x)
        
        x = self.layer4(x)
        x = self.upsample2(x)


        x = self.skip1([x, copy1])

        x = self.refine1(x)
        
        x = self.layer5(x)
        x = self.patch_unembed(x)

        x = copy0 + x
        return x


def sfhformer_t():
    return Backbone_new(embed_dim=[24, 48, 96, 48, 24], depth=[1, 1, 2, 1, 1], local_size=[4, 4, 4, 4, 4], embed_kernel_size=3)

def sfhformer_lol_s():
    return Backbone_new(embed_dim=[24, 48, 96, 48, 24], depth=[1, 1, 2, 1, 1], local_size=[4, 4, 4, 4, 4], embed_kernel_size=3)

def sfhformer_lol_m():
    return Backbone_new(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], local_size=[4, 4, 4, 4, 4], embed_kernel_size=3)

def sfhformer_lol_l():
    return Backbone_new(embed_dim=[24, 48, 96, 48, 24], depth=[4, 4, 8, 4, 4], local_size=[4, 4, 4, 4, 4], embed_kernel_size=3)


if __name__ == "__main__":

    model = sfhformer_t()
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    print(f" Forward Pass: {x.shape} -> {y.shape}")
    
    if hasattr(model, 'refine1'):
        print(f" {hasattr(model.refine1, 'norm')}")