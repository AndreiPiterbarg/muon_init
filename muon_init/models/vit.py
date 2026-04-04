"""Vision Transformer (ViT-Tiny) for CIFAR-10.

6 layers, 4 heads, embed dim 192, patch size 4 (yields 8x8=64 patches for 32x32 images).
~1.2M params. Encoder-only (no causal mask). Tests cross-domain init sensitivity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)

    def forward(self, x):
        # (B, C, H, W) -> (B, num_patches, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, n_head, dropout=0.0):
        super().__init__()
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = (att @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class ViTBlock(nn.Module):
    def __init__(self, dim, n_head, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_head, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ViTTiny(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 n_layer=6, n_head=4, embed_dim=192, dropout=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, n_head, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize pos embedding and cls token.
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.drop(x + self.pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x[:, 0])  # CLS token
        return self.head(x)
