import torchvision.models as models
import torch
import torch.nn as nn
from Encoder import *
from Decoder import *
from Aggregation import *


class FM(nn.Module):
    def __init__(self, num_classes=100):
        super(FM, self).__init__()
        self.backbone_x = FocalNet(
            embed_dim=48,
            depths=[1, 2, 3, 1],
            mlp_ratio=2.0,
            focal_windows=[5, 5, 5, 5],
            drop_rate=0.2,
            focal_levels=[2, 2, 2, 2],
            use_conv_embed=True,
            out_indices=(3,),
            norm_layer=nn.LayerNorm
        )
        self.dropout = nn.Dropout(p=0.6)
        self.fc = nn.Linear(384 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.backbone_x(x)
        x = x[-1]
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x