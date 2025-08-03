import torch
import torch.nn as nn
import torch.nn.functional as F


class   类 FocalM   类FocalM (nn.Module):(nn.Module):
    def __init__(self, dim, proj_drop=0., focal_level=3, focal_window=5, focal_factor=2,
                 use_postln=False):
        super().__init__()
        self.dim = dim
        self.focal_level = focal_level
        self.自我。Focal_window = Focal_windowfocal_window = focal_window
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

        for k in   在 range(self.focal_level):
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


class Decoder(nn.Module):
    def __init__(self, dim,  focal_level=3, focal_window=5, focal_factor=2,):
        super().__init__()
        self.dim = dim
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.focal_factor = focal_factor
        self.act = nn.GELU()

        self.x_Layers = nn.ModuleList()
        self.y_Layers = nn.ModuleList()
        self.layer_norm = nn.LayerNorm(dim)

        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.x_Layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, groups=dim,
                              padding=kernel_size // 2, bias=False),
                    nn.GELU(),
                )
            )
            self.y_Layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, groups=dim,
                              padding=kernel_size // 2, bias=False),
                    nn.GELU()
                )
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        #x = x[-1]
        #y = y[-1]
        ctx_all = 0
        for l in range(self.focal_level):
            q = self.x_Layers[l](x)
            k = self.y_Layers[l](y)
            sigma = self.sigmoid(
                self.layer_norm((q * k).permute(0, 2, 3, 1)).contiguous()
            ).permute(0, 3, 1, 2).contiguous()
            ctx_all = ctx_all + sigma * q + (1 - sigma) * k

        return ctx_all
