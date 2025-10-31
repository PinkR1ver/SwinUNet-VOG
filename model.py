"""
SwinUNet-based model for gaze estimation from eye images.
Adapted for the MPIIGaze dataset with 36x60 eye images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwinUNet(nn.Module):
    """SwinUNet-inspired architecture for Gaze Estimation."""

    def __init__(self, img_size=(36, 60), in_chans=3, embed_dim=96, depths=[2, 2, 2],
                 num_heads=[3, 6, 12], window_size=7, drop_rate=0.1):
        super().__init__()

        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=2, stride=2)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        in_dim = embed_dim
        for i, depth in enumerate(depths):
            for _ in range(depth):
                self.encoder_blocks.append(SwinBlock(in_dim, num_heads[i], window_size, drop_rate))
            if i < len(depths) - 1:
                out_dim = in_dim * 2
                self.encoder_blocks.append(ConvPatchMerging(in_dim, out_dim))
                in_dim = out_dim
        
        # Bottleneck
        self.bottleneck = SwinBlock(in_dim, num_heads[-1], window_size, drop_rate)
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(depths) - 1):
            out_dim = in_dim // 2
            # Upsample with channel adjustment
            self.decoder_blocks.append(nn.Sequential(
                SwinBlock(in_dim, num_heads[len(depths)-2-i], window_size, drop_rate),
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)
            ))
            in_dim = out_dim
        
        # Final upsampling to original size
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(in_dim, in_dim, kernel_size=2, stride=2),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1)
        )
        
        # Regression head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(128, 3)  # 3D gaze vector
        )
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # B, C, 18, 30
        
        # Encoder
        for block in self.encoder_blocks:
            x = block(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for block in self.decoder_blocks:
            x = block(x)
        
        # Final upsampling
        x = self.final_up(x)
        
        # Regression head
        x = self.head(x)
        return x


class SwinBlock(nn.Module):
    """Simplified Swin Transformer Block with depthwise separable convolution."""
    
    def __init__(self, dim, num_heads, window_size, drop_rate):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        
        # Simplified attention using depthwise conv
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=window_size, padding=window_size//2, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        
        # MLP
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(drop_rate),
            nn.Conv2d(dim * 4, dim, kernel_size=1),
            nn.Dropout2d(drop_rate)
        )
        
        self.drop_path = DropPath(drop_rate) if drop_rate > 0 else nn.Identity()
    
    def forward(self, x):
        # Attention
        shortcut = x
        x = self.norm1(x)
        x_attn = self.attn(x)
        x = x + self.drop_path(x_attn)
        
        # MLP
        x_mlp = self.mlp(self.norm2(x))
        x = x + self.drop_path(x_mlp)
        
        return x


class ConvPatchMerging(nn.Module):
    """Convolution-based patch merging (downsampling)."""
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.merging = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_dim)
        )
    
    def forward(self, x):
        return self.merging(x)


class DropPath(nn.Module):
    """Stochastic Depth."""
    def __init__(self, drop_prob=0.1):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob + torch.rand((x.size()[0], *([1] * (len(x.size()) - 1))), 
                                               dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


if __name__ == "__main__":
    # Test model
    print("Testing SwinUNet Model...")
    
    model = SwinUNet(
        img_size=(36, 60),
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 2],
        num_heads=[3, 6, 12],
        window_size=7
    )
    
    print(f"\nModel created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 36, 60)
    with torch.no_grad():
        output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Model test completed!")
