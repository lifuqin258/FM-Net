import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalM(nn.Module):
    def __init__(self, dim, proj_drop=0., focal_level=2, focal_window=7, focal_factor=2,
                 use_postln=False):
        super().__init__()
        self.dim = dim
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.focal_factor = focal_factor

        self.use_postln = use_postln
        self.f = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=True)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.act = nn.GELU()

        self.ctx_layers = nn.ModuleList()
        self.gate_layers = nn.ModuleList()
        self.layer_norm = nn.LayerNorm(dim)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.use_postln:
            self.ln = nn.LayerNorm(dim)

        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.ctx_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, groups=dim,
                              padding=kernel_size // 2, bias=False),
                    nn.GELU(),
                )
            )
            self.gate_layers.append(
                nn.Sequential(
                    nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, groups=1,
                              padding=kernel_size // 2, bias=False),
                    nn.GELU()
                )
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, q):
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()
        x_initial = x
        x = self.f(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        _, ctx, gates = torch.split(x, (C, C, self.focal_level + 1), 1)
        ctx_all = 0

        for l in range(self.focal_level):
            ctx = self.ctx_layers[l](ctx)
            ft_gates = self.gate_layers[l](gates[:, l:l + 1])
            sigma = self.sigmoid(
                self.layer_norm((ctx * ft_gates).permute(0, 2, 3, 1)).contiguous()
            ).permute(0, 3, 1, 2).contiguous()
            ctx_all = ctx_all + sigma * ctx + (1 - sigma) * gates[:, l:l + 1]

        ctx_global = ctx.mean(2, keepdim=True).mean(3, keepdim=True)
        ctx_global = self.act(ctx_global)
        fv_global_ctx = self.ctx_layers[-1](ctx_global)
        ft_global_gate = self.gate_layers[-1](gates[:, self.focal_level:self.focal_level + 1])
        sigma_global = self.sigmoid(
            self.layer_norm((fv_global_ctx * ft_global_gate).permute(0, 2, 3, 1)).contiguous()
        ).permute(0, 3, 1, 2).contiguous()
        ctx_global_updated = sigma_global * ctx_global + (1 - sigma_global) * gates[:,
                                                                              self.focal_level:self.focal_level + 1]
        ctx_all = ctx_all + ctx_global_updated

        x_out = q * self.h(ctx_all)
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_postln:
            x_out = self.ln(x_out)

        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        avg_pooled = F.avg_pool2d(
            x_initial.permute(0, 3, 1, 2),
            kernel_size=3,
            stride=1,
            padding=1
        )
        avg_pooled = avg_pooled.permute(0, 2, 3, 1).contiguous()
        x_out = x_out + avg_pooled

        return x_out


class PyramidFusionDecoder(nn.Module):
    def __init__(self,
                 in_dim=256,
                 num_scales=3,  # 金字塔层级数（P3, P4, P5）
                 target_sizes=[28, 14, 6],  # 目标分辨率 [P3, P4, P5]
                 focal_window=5):
        super().__init__()
        self.num_scales = num_scales
        self.in_dim = in_dim
        self.target_sizes = target_sizes

        # 多尺度卷积组（x和y分支）
        self.x_layers = nn.ModuleList()
        self.y_layers = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()

        # 初始化各尺度处理层
        for scale in range(num_scales):
            # 动态计算卷积核大小（随尺度增大）
            kernel_size = focal_window + 2 * scale

            # x模态处理分支
            self.x_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_dim, in_dim, kernel_size=kernel_size,
                              padding=kernel_size // 2, groups=in_dim, bias=False),
                    nn.GELU()
                )
            )

            # y模态处理分支
            self.y_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_dim, in_dim, kernel_size=kernel_size,
                              padding=kernel_size // 2, groups=in_dim, bias=False),
                    nn.GELU()
                )
            )

            # 融合层（保持通道数不变）
            self.fusion_layers.append(
                nn.Sequential(
                    nn.GroupNorm(num_groups=1, num_channels=in_dim),  # 显式指定通道数
                    nn.Conv2d(in_dim, in_dim, kernel_size=1),
                    nn.Sigmoid()
                )
            )

    def forward(self, x, y):
        """
        输入:
            x -> (b,256,6,6)   # 高层特征
            y -> (b,256,6,6)   # 另一模态特征
        输出:
            [p3, p4, p5] -> [(b,256,28,28), (b,256,14,14), (b,256,6,6)]
        """
        pyramid_features = []
        current_x, current_y = x, y

        # 从低层到高层构建金字塔（P5→P4→P3）
        for scale in reversed(range(self.num_scales)):
            # Step 1: 动态上采样到目标尺寸
            if scale < self.num_scales - 1:  # 最高层P5不需要上采样
                target_size = (self.target_sizes[scale], self.target_sizes[scale])
                current_x = F.interpolate(current_x, size=target_size, mode='bilinear', align_corners=False)
                current_y = F.interpolate(current_y, size=target_size, mode='bilinear', align_corners=False)

            # Step 2: 跨模态融合
            q = self.x_layers[scale](current_x)
            k = self.y_layers[scale](current_y)

            # 计算注意力权重（无需permute）
            fused = q * k
            sigma = self.fusion_layers[scale](fused)

            # 加权融合
            fused_feature = sigma * q + (1 - sigma) * k
            pyramid_features.append(fused_feature)

        # 调整顺序为 [P3, P4, P5]
        return pyramid_features[::-1]