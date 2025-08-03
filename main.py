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
            patch_size = 1,
            embed_dim=24,  # 减小通道基数
            depths=[1, 1, 1, 1],  # 大幅减少层数
            focal_levels=[1, 2, 2, 1],  # 降低聚焦层级
            focal_windows=[3, 3, 3, 3],  # 减小窗口尺寸
            drop_rate=0.5,  # 增加Dropout
            drop_path_rate=0.5,  # 增加DropPath
            use_conv_embed=True,  # 使用卷积嵌入
            mlp_ratio=2,  # 减少MLP扩展比例
            use_layerscale=True,  # 启用层缩放
            use_checkpoint=False
        )
        self.backbone_y = FocalNet(
            patch_size=1,
            embed_dim=24,  # 减小通道基数
            depths=[1, 1, 2, 1],  # 大幅减少层数
            focal_levels=[1, 2, 2, 1],  # 降低聚焦层级
            focal_windows=[3, 3, 3, 3],  # 减小窗口尺寸
            drop_rate=0.5,  # 增加Dropout
            drop_path_rate=0.5,  # 增加DropPath
            use_conv_embed=True,  # 使用卷积嵌入
            mlp_ratio=2,  # 减少MLP扩展比例
            use_layerscale=True,  # 启用层缩放
            use_checkpoint=False
        )

        self.decoder = Decoder(dim=192)

        self.aggregation = Aggregation()

        self.focal_x = FocalM(dim=192)
        self.focal_y = FocalM(dim=192)
        self.dropout = nn.Dropout(p=0.6)
        self.fc = nn.Linear(192 * 7 * 7, num_classes)

    def forward(self, x, y):
        x = self.backbone_x(x)
        y = self.backbone_y(y)

        q = self.decoder(x, y)

        x_focal = self.focal_x(x, q)
        y_focal = self.focal_y(y, q)

        aggregation = self.aggregation(x_focal, y_focal)


        aggregation = self.dropout(aggregation)
        aggregation = aggregation.reshape(aggregation.size(0), -1)
        aggregation = self.dropout(aggregation)
        aggregation = self.fc(aggregation)
        return aggregation