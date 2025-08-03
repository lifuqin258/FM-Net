import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels = 256, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.mlp(self.max_pool(x).view(x.size(0), -1))
        channel_weights = self.sigmoid(avg_out + max_out)
        return channel_weights.unsqueeze(2).unsqueeze(3)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        spatial_weights = self.sigmoid(self.conv(combined))
        return spatial_weights


class Aggregation(nn.Module):
    def __init__(self, in_channels = 256, reduction_ratio=16, kernel_size=7):
        super(Aggregation, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(in_channels)

    def forward(self, vision, tactile):

        vision = vision.permute(0, 3, 1, 2)
        tactile = tactile.permute(0, 3, 1, 2)

        channel_weights = self.channel_att(vision)
        vision_channel = vision * channel_weights

        spatial_weights = self.spatial_att(tactile)
        tactile_final = tactile * spatial_weights

        sigma = self.act(self.layer_norm((vision_channel * tactile_final).permute(0, 2, 3, 1)).contiguous()
            ).permute(0, 3, 1, 2).contiguous()
        x_final = sigma * vision_channel + (1 - sigma) * tactile_final

        return x_final