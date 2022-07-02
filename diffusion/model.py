from typing import Tuple
from torch import nn
import torch

from einops import rearrange
from einops.layers.torch import Rearrange

class BaseNet(nn.Module):
    def __init__(self) -> None:
        super(BaseNet, self).__init__()

        self.core = nn.Sequential(
            nn.Conv2d(3, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, input, t):
        x = self.core(input)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(ResBlock, self).__init__()

        self.core = nn.Sequential(
            nn.Conv2d(in_features, out_features, 3, 1, 1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.Conv2d(out_features, out_features, 3, 1, 1),
            nn.BatchNorm2d(out_features)
        )

        self.act = nn.ReLU()
    
    def forward(self, input):
        x = self.core(input)
        return self.act(x + input)


class PreNorm(nn.Module):
    def __init__(self, dim, fn) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, input, **kwargs):
        return self.fn(self.norm(input), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 64, dropout = 0.) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.dropout(self.attend(dots))

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0.) -> None:
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
    
    def forward(self, input):
        x = self.attn(input) + input
        x = self.ff(x) + x
        return x


class ViTBlock(nn.Module):
    def __init__(self, dim=128, heads=4, dim_head=32, mlp_dim=64, dropout=0.) -> None:
        super().__init__()
        self.tf = Transformer(dim, heads, dim_head, mlp_dim, dropout=dropout)


    def forward(self, x: torch.Tensor, t, T=1000):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        embed = torch.sin(torch.ones_like(x, device=x.device) * 2 * torch.pi * t / T)
        x = self.tf(x + embed)
        return rearrange(x, 'b (h w) c -> b c h w', h = h)


class BaseUnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(3, 128, 1, 1)
        self.core1 = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128)
        )
        self.avgpool1 = nn.AvgPool2d(2, 2)
        self.core2 = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128)
        )
        self.avgpool2 = nn.AvgPool2d(2, 2)
        self.core3 = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128)
        )
        self.avgpool3 = nn.AvgPool2d(2, 2)
        self.core4 = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128)
        )
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upcore1 = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128)
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upcore2 = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128)
        )
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upcore3 = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128)
        )
        self.conv = nn.Conv2d(128, 3, 3, 1, 1)
    
    def forward(self, input, t):
        x = self.in_conv(input)
        x_1 = self.core1(x)
        x_2 = self.core2(self.avgpool1(x_1))
        x_3 = self.core3(self.avgpool2(x_2))
        x = self.core4(self.avgpool3(x_3))

        x = self.upcore1(self.upsample1(x) + x_3)
        x = self.upcore2(self.upsample1(x) + x_2)
        x = self.upcore3(self.upsample1(x) + x_1)

        return self.conv(x)


class AttentionUnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(3, 128, 1, 1)
        self.core1 = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128)
        )
        self.avgpool1 = nn.AvgPool2d(2, 2)
        self.core21 = ResBlock(128, 128)
        self.core22 = ViTBlock()
        self.core23 = ResBlock(128, 128)

        self.avgpool2 = nn.AvgPool2d(2, 2)
        self.core3 = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128)
        )
        self.avgpool3 = nn.AvgPool2d(2, 2)
        self.core4 = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128)
        )
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upcore1 = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128)
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upcore2 = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128)
        )
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upcore3 = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128)
        )
        self.conv = nn.Conv2d(128, 3, 3, 1, 1)
    
    def forward(self, input, t):
        x = self.in_conv(input)
        x_1 = self.core1(x)
        x_2 = self.core21(self.avgpool1(x_1))
        x_2 = self.core23(self.core22(x_2, t))
        x_3 = self.core3(self.avgpool2(x_2))
        x = self.core4(self.avgpool3(x_3))

        x = self.upcore1(self.upsample1(x) + x_3)
        x = self.upcore2(self.upsample1(x) + x_2)
        x = self.upcore3(self.upsample1(x) + x_1)

        return self.conv(x)
