import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torchvision.ops import SqueezeExcitation
import math
import numpy as np
from functools import partial
from typing import Optional, Callable, Optional, Dict, Union
from collections import OrderedDict
from ..modules.conv import Conv, DWConv, DSConv, RepConv, GhostConv, autopad, LightConv, ConvTranspose
from ..modules.block import get_activation, ConvNormLayer, WTConvNormLayer,BasicBlock, BottleNeck, RepC3, C3, C2f, Bottleneck
from timm.layers import DropPath
from einops import rearrange, reduce

__all__ = ['DySample','SPDConv','MFFF','MFCF','FrequencyFocusedDownSampling','FEDown','SemanticAlignmenCalibration','FFAE']
        
######################################## DySample start ########################################
class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            self.constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def normal_init(self, module, mean=0, std=1, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def constant_init(self, module, val, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)
    
    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).contiguous().repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.reshape(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale).reshape(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W).contiguous(), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset.contiguous())

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

######################################## DySample end ########################################

class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x


class FFM(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv = nn.Conv2d(dim, dim*2, 3, 1, 1, groups=dim)

        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        input_dtype = x.dtype
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)

        x2_fft = torch.fft.fft2(x2.float(), norm='backward')

        out = x1.float() * x2_fft

        out = torch.fft.ifft2(out, dim=(-2,-1), norm='backward')
        out = torch.abs(out).to(input_dtype)

        return out * self.alpha + x * self.beta


class ImprovedFFTKernel(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        ker = 31
        pad = ker // 2
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.GELU()
        )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

        self.act = nn.SiLU()

        # 改进后的 SCA 部分
        self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=True)
        self.conv5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, stride=1, groups=dim, bias=True)

        # self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.ffm = FFM(dim)

        #通道注意力
        self.channel_attention = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        input_dtype = x.dtype
        out = self.in_conv(x)
        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out.float(), norm='backward')
        x_fft = x_att.float() * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2, -1), norm='backward')
        x_fca = torch.abs(x_fca).to(input_dtype)

        # 公式2
        x_sca1 = self.conv1x1(x_fca)
        x_sca2 = self.conv3x3(x_fca)
        x_sca3 = self.conv5x5(x_fca)
        x_sca = x_sca1 + x_sca2 + x_sca3
        #公式2结束

        # 使用通道注意力机制
        channel_weights = self.channel_attention(x_att)
        x_sca = x_sca * channel_weights

        #FF的公式
        x_sca = self.ffm(x_sca)

        # 最终融合 公式4
        out = x + self.dw_33(out) + self.dw_11(out) + x_sca
        out = self.act(out)
        return self.out_conv(out)

class MFFF(nn.Module): 
    def __init__(self, dim, e=0.25):
        super().__init__()
        c1 = int(round(dim * e))
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.m = ImprovedFFTKernel(c1)

    def forward(self, x):
        c1 = self.m.in_conv[0].in_channels
        c2 = x.size(1) - c1
        ok_branch, identity = torch.split(self.cv1(x), [c1, c2], dim=1)
        return self.cv2(torch.cat((self.m(ok_branch), identity), 1))


#####################################################################
class FEM(nn.Module):
    def __init__(self, nc):
        super(FEM, self).__init__()
        self.fpre = nn.Conv2d(nc, nc, 1, 1, 0)
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.process2 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.alpha = nn.Parameter(torch.zeros(nc, 1, 1))
        self.beta = nn.Parameter(torch.ones(nc, 1, 1))

    def forward(self, x):
        input_dtype = x.dtype
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(self.fpre(x).float(), norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        pha = self.process2(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward').to(input_dtype)

        return x_out * self.alpha + x * self.beta

class LargeFFTKernel(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        ker = 31
        ker1 = 15
        ker2 = 17
        pad = ker // 2
        pad1 = ker1 // 2
        pad2 = ker2 // 2
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.GELU()
        )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_55 = nn.Conv2d(dim, dim, kernel_size=ker1, padding=pad1, stride=1, groups=dim)
        self.dw_77 = nn.Conv2d(dim, dim, kernel_size=ker2, padding=pad2, stride=1, groups=dim)

        self.act = nn.SiLU()

        # 改进后的 SCA 部分
        self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=True)
        self.conv5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, stride=1, groups=dim, bias=True)

        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fem = FEM(dim)

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        input_dtype = x.dtype
        out = self.in_conv(x)
        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out.float(), norm='backward')
        x_fft = x_att.float() * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2, -1), norm='backward')
        x_fca = torch.abs(x_fca).to(input_dtype)

        # 公式2  修改，残差连接
        x_sca1 = self.conv1x1(x_fca)
        x_sca2 = self.conv3x3(x_fca + x_sca1 / 2)
        x_sca3 = self.conv5x5(x_fca + x_sca2 / 2 + x_sca1 / 4)
        x_sca = x_sca1 + x_sca2 + x_sca3
        # 公式2结束

        # 使用通道注意力机制
        channel_weights = self.channel_attention(x_att)
        x_sca = x_sca * channel_weights

        # FF的公式
        x_sca = self.fem(x_sca)

        # 最终融合 公式4
        out = x + self.dw_55(out)+ self.dw_77(out) + self.dw_11(out) + x_sca
        out = self.act(out)
        return self.out_conv(out)


class MFCF(nn.Module):
    def __init__(self, dim, e=0.25):
        super().__init__()
        c1 = int(round(dim * e))
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.m = LargeFFTKernel(c1)

    def forward(self, x):
        c1 = self.m.in_conv[0].in_channels
        c2 = x.size(1) - c1
        ok_branch, identity = torch.split(self.cv1(x), [c1, c2], dim=1)
        return self.cv2(torch.cat((self.m(ok_branch), identity), 1))

##############################################################
class ADown(nn.Module): # Downsample x2分支
    def __init__(self, c1, c2):  
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1,x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)

class FrequencyFocusedDownSampling(nn.Module):  # Downsample x2分支 with parallel FGM
    def __init__(self, c1, c2):  
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)
        self.ffm = FFM(self.c)  # FGM 模块处理 x2 分支

        # 1x1 卷积用于在拼接后减少通道数
        self.conv_reduce = Conv(self.c * 2, self.c, 1, 1)

        # 新增的卷积层用于调整 fgm_out 的空间尺寸
        self.conv_resize = Conv(self.c, self.c, 3, 2, 1)
    #经过池化后分成两个分支，一个分支经过 cv1 处理，另一个分支经过 fgm + maxpool cv2 处理，然后将两个分支拼接在一起，最后使用 1x1 卷积将通道数减少到预期的值。
    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)

        # 并联处理 x2 分支
        fgm_out = self.ffm(x2)  # FGM 处理的输出
        fgm_out = self.conv_resize(fgm_out)  # 调整 fgm_out 的空间尺寸
        pooled_out = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        pooled_out = self.cv2(pooled_out)

        # 将 FGM 输出和 MaxPool2d + Conv 输出拼接
        x2 = torch.cat((fgm_out, pooled_out), 1)
        
        # 使用 1x1 卷积将通道数减少到预期的值
        x2 = self.conv_reduce(x2)

        return torch.cat((x1, x2), 1)
    
class FEDown(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.c = c2 // 2
        self.cv0 = Conv(c1, c2, 3, 2, 1)  # 用于处理 x 分支
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)  # 用于处理 x1 分支
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)
        self.fem = FEM(self.c)  # FEM 模块处理 x2 分支

        # 1x1 卷积用于在拼接后减少通道数
        self.conv_reduce = Conv(self.c * 2, self.c, 1, 1)

        # 新增的卷积层用于调整 fem_out 的空间尺寸
        self.conv_resize = Conv(self.c, self.c, 3, 2, 1)
        
        # 这里的alpha和beta是可学习的参数
        self.alpha = nn.Parameter(torch.zeros(c2, 1, 1))
        self.beta = nn.Parameter(torch.ones(c2, 1, 1))

    # 经过池化后分成两个分支，一个分支经过 cv1 处理，另一个分支经过 fem + maxpool cv2 处理，然后将两个分支拼接在一起，最后使用 1x1 卷积将通道数减少到预期的值。
    def forward(self, x):
        x0 = self.cv0(x)  # 处理 x 分支
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)

        # 并联处理 x2 分支
        fem_out = self.fem(x2)  # FEM 处理的输出
        fem_out = self.conv_resize(fem_out)  # 调整 fem_out 的空间尺寸
        pooled_out = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        pooled_out = self.cv2(pooled_out)

        # 将 FEM 输出和 MaxPool2d + Conv 输出拼接
        x2 = torch.cat((fem_out, pooled_out), 1)

        # 使用 1x1 卷积将通道数减少到预期的值
        x2 = self.conv_reduce(x2)

        return torch.cat((x1, x2), 1) * self.alpha + x0 * self.beta # 将处理后的 x1 和 x2 分支与 x 分支拼接


######################################################################################
class SemanticAlignmenCalibration(nn.Module):  # 
    def __init__(self, inc):
        super(SemanticAlignmenCalibration, self).__init__()
        hidden_channels = inc[0]

        self.groups = 2
        self.spatial_conv = Conv(inc[0], hidden_channels, 3)  # 用于处理高分辨率的空间特征
        self.semantic_conv = Conv(inc[1], hidden_channels, 3)  # 用于处理低分辨率的语义特征

        # FGM模块：用于在频域中增强特征
        self.frequency_enhancer = FFM(hidden_channels)
        # 门控卷积：结合空间和频域特征
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=0, bias=True)
        
        # 用于生成偏移量的卷积序列
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64),  # 处理拼接后的特征
            nn.Conv2d(64, self.groups * 4 + 2, kernel_size=3, padding=1, bias=False)  # 生成偏移量
        )

        self.init_weights()
        self.offset_conv[1].weight.data.zero_()  # 初始化最后一层卷积的权重为零

    def init_weights(self):
        # 初始化卷积层的权重
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        coarse_features, semantic_features = x
        batch_size, _, out_h, out_w = coarse_features.size()

        # 处理低分辨率的语义特征 (1/8 下采样)
        semantic_features = self.semantic_conv(semantic_features)
        semantic_features = F.interpolate(semantic_features, coarse_features.size()[2:], mode='bilinear', align_corners=True)

        # 频域增强特征
        enhanced_frequency = self.frequency_enhancer(semantic_features)
        
        # 门控机制融合频域和空间域的特征
        gate = torch.sigmoid(self.gating_conv(semantic_features))
        fused_features = semantic_features * (1 - gate) + enhanced_frequency * gate

        # 处理高分辨率的空间特征 (1/8 下采样)
        coarse_features = self.spatial_conv(coarse_features)

        # 拼接处理后的空间特征和融合后的特征
        conv_results = self.offset_conv(torch.cat([coarse_features, fused_features], 1))

        # 调整特征维度以适应分组
        fused_features = fused_features.reshape(batch_size * self.groups, -1, out_h, out_w)
        coarse_features = coarse_features.reshape(batch_size * self.groups, -1, out_h, out_w)

        # 获取偏移量
        offset_low = conv_results[:, 0:self.groups * 2, :, :].reshape(batch_size * self.groups, -1, out_h, out_w)
        offset_high = conv_results[:, self.groups * 2:self.groups * 4, :, :].reshape(batch_size * self.groups, -1, out_h, out_w)

        # 生成归一化网格用于偏移校正
        normalization_factors = torch.tensor([[[[out_w, out_h]]]]).type_as(fused_features).to(fused_features.device)
        grid_w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        grid_h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        base_grid = torch.cat((grid_h.unsqueeze(2), grid_w.unsqueeze(2)), 2)
        base_grid = base_grid.repeat(batch_size * self.groups, 1, 1, 1).type_as(fused_features).to(fused_features.device)

        # 使用生成的偏移量对网格进行调整
        adjusted_grid_l = base_grid + offset_low.permute(0, 2, 3, 1) / normalization_factors
        adjusted_grid_h = base_grid + offset_high.permute(0, 2, 3, 1) / normalization_factors

        # 进行特征采样
        coarse_features = F.grid_sample(coarse_features, adjusted_grid_l, align_corners=True)
        fused_features = F.grid_sample(fused_features, adjusted_grid_h, align_corners=True)

        # 调整维度回到原始形状
        coarse_features = coarse_features.reshape(batch_size, -1, out_h, out_w)
        fused_features = fused_features.reshape(batch_size, -1, out_h, out_w)

        # 融合增强后的特征
        attention_weights = 1 + torch.tanh(conv_results[:, self.groups * 4:, :, :])
        final_features = fused_features * attention_weights[:, 0:1, :, :] + coarse_features * attention_weights[:, 1:2, :, :]

        return final_features



######################################################################################
class FFAE(nn.Module):
    def __init__(self, inc, groups=2, reduction=16):
        """
        :param inc: 输入特征的通道数信息，格式为[高分辨率特征通道数, 低分辨率特征通道数]
        :param groups: 分组数，默认为2，用于后续特征处理和维度调整等操作，增加灵活性
        :param reduction: SE模块中的通道压缩比例，默认为16，用于控制特征选择的强度
        """
        super().__init__()
        hidden_channels = inc[0]
        self.groups = groups

        # 处理高分辨率空间特征的卷积层
        self.spatial_conv = Conv(inc[0], hidden_channels, 3)
        # 处理低分辨率语义特征的卷积层
        self.semantic_conv = Conv(inc[1], hidden_channels, 3)

        # 频域增强模块
        self.frequency_enhancer = FEM(hidden_channels)
        # 门控卷积：融合空间和频域特征
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=0, bias=True)

        # 生成偏移量的卷积序列，调整第一个卷积层输入通道数
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64),  # 这里保持为 hidden_channels * 2，因为拼接后通道数是其两倍
            nn.Conv2d(64, self.groups * 4 + 2, kernel_size=3, padding=1, bias=False)  # 生成偏移量
        )

        # 预计算归一化因子，避免每次前向传播时重复计算
        self.register_buffer('normalization_factors', torch.tensor([[[[1.0, 1.0]]]]), persistent=False)

        # 初始化卷积层权重
        self.init_weights()
        self.offset_conv[1].weight.data.zero_()  # 初始化最后一层卷积的权重为零

        # 使用官方SE模块替代自定义实现
        self.se_module = SqueezeExcitation(hidden_channels, reduction)
        
        # 这里的alpha和beta是可学习的参数
        self.alpha = nn.Parameter(torch.zeros(hidden_channels, 1, 1))
        self.beta = nn.Parameter(torch.ones(hidden_channels, 1, 1))

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        coarse_features, semantic_features = x
        batch_size, _, out_h, out_w = coarse_features.size()

        # 处理低分辨率语义特征 (1/8 下采样)
        semantic_features = self.semantic_conv(semantic_features)
        semantic_features = F.interpolate(semantic_features, coarse_features.size()[2:], mode='bilinear',
                                          align_corners=True)

        # 频域增强特征
        enhanced_frequency = self.frequency_enhancer(semantic_features)

        # 门控机制融合频域和空间域特征
        gate = torch.sigmoid(self.gating_conv(semantic_features))
        fused_features = semantic_features * (1 - gate) + enhanced_frequency * gate

        # 处理高分辨率空间特征 (1/8 下采样)
        coarse_features = self.spatial_conv(coarse_features)

        # 调整特征维度，确保拼接时通道数匹配
        fused_features = fused_features.reshape(batch_size, -1, out_h, out_w)
        coarse_features = coarse_features.reshape(batch_size, -1, out_h, out_w)

        # 拼接处理后的空间特征和融合后的特征
        conv_results = self.offset_conv(torch.cat([coarse_features, fused_features], 1))

        # 获取偏移量并调整维度以适应分组
        offset_low = conv_results[:, 0:self.groups * 2, :, :].reshape(batch_size, self.groups, -1, out_h, out_w)
        offset_high = conv_results[:, self.groups * 2:self.groups * 4, :, :].reshape(batch_size, self.groups, -1, out_h, out_w)

        # 调整归一化因子
        normalization_factors = self.normalization_factors * torch.tensor([[out_w, out_h]]).view(1, 1, 1, 2).type_as(
            fused_features)

        # 生成归一化网格用于偏移校正
        with torch.no_grad():
            h, w = out_h, out_w
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid = torch.stack([grid_x, grid_y], dim=-1).float()
            grid = grid * 2.0 / torch.tensor([w, h]).float() - 1.0  # 归一化到[-1,1]
            base_grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1).to(fused_features.device)

        # 使用生成的偏移量对网格进行调整
        adjusted_grid_l = base_grid.unsqueeze(1) + offset_low.permute(0, 1, 3, 4, 2) / normalization_factors
        adjusted_grid_h = base_grid.unsqueeze(1) + offset_high.permute(0, 1, 3, 4, 2) / normalization_factors

        # 调整偏移后的网格维度
        adjusted_grid_l = adjusted_grid_l.reshape(batch_size * self.groups, out_h, out_w, 2)
        adjusted_grid_h = adjusted_grid_h.reshape(batch_size * self.groups, out_h, out_w, 2)

        # 调整特征维度以适应分组
        coarse_features = coarse_features.reshape(batch_size * self.groups, -1, out_h, out_w)
        fused_features = fused_features.reshape(batch_size * self.groups, -1, out_h, out_w)

        # 进行特征采样
        coarse_features = F.grid_sample(coarse_features, adjusted_grid_l, align_corners=True)
        fused_features = F.grid_sample(fused_features, adjusted_grid_h, align_corners=True)

        # 调整维度回到原始形状
        coarse_features = coarse_features.reshape(batch_size, -1, out_h, out_w)
        fused_features = fused_features.reshape(batch_size, -1, out_h, out_w)
        # 应用SE模块增强特征选择
        final_features1 = self.se_module(fused_features * 0.25  + coarse_features * 0.75)

        # 生成注意力权重并融合特征
        attention_weights = 1 + torch.tanh(conv_results[:, self.groups * 4:, :, :])
        final_features = fused_features * attention_weights[:, 0:1, :, :] + coarse_features * attention_weights[:, 1:2, :, :]
        
        # 返回最终融合的特征
        final_features = final_features1 * self.alpha + final_features * self.beta

        return final_features      
