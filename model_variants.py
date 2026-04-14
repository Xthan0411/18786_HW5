# CMU 18-780/6 Homework 4 — Part 2 模型变体
# 在这里实现 Part 2 要求的 GAN 变体: Spectral Norm, WGAN-GP, LSGAN, Custom
# 本文件禁止使用 torch.nn.utils.spectral_norm 等一行封装, 必须从零实现

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 通用卷积/上采样构造器 (沿用 Part 1 的写法)
# ============================================================

def up_conv(in_channels, out_channels, kernel_size, stride=1, padding=1,
            scale_factor=2, norm='batch', activ=None):
    """Upsample + Conv2d, 可选 normalization 和激活"""
    layers = []
    layers.append(nn.Upsample(scale_factor=scale_factor, mode='nearest'))
    layers.append(nn.Conv2d(
        in_channels, out_channels,
        kernel_size, stride, padding, bias=norm is None
    ))
    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    if activ == 'relu':
        layers.append(nn.ReLU())
    elif activ == 'leaky':
        layers.append(nn.LeakyReLU(0.2))
    elif activ == 'tanh':
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1,
         norm='batch', init_zero_weights=False, activ=None):
    """普通 Conv2d, 可选 normalization 和激活"""
    layers = []
    conv_layer = nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels,
        kernel_size=kernel_size, stride=stride, padding=padding,
        bias=norm is None
    )
    if init_zero_weights:
        conv_layer.weight.data = 0.001 * torch.randn(
            out_channels, in_channels, kernel_size, kernel_size
        )
    layers.append(conv_layer)

    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    if activ == 'relu':
        layers.append(nn.ReLU())
    elif activ == 'leaky':
        layers.append(nn.LeakyReLU(0.2))
    elif activ == 'tanh':
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


# ============================================================
# 生成器 (所有变体共用, 与 Part 1 相同)
# ============================================================

class DCGenerator(nn.Module):
    """DCGAN 生成器: z -> 64x64 RGB. 与 Part 1 结构完全一致"""
    def __init__(self, noise_size, conv_dim=64):
        super().__init__()
        # 第一层: 1x1 噪声 -> 4x4 特征图, 用 conv(k=4, s=1, p=3) 实现
        self.up_conv1 = conv(
            in_channels=noise_size, out_channels=conv_dim * 8,
            kernel_size=4, stride=1, padding=3,
            norm='instance', activ='relu'
        )
        # up_conv2-4: 逐步上采样 4->8->16->32, 通道 256->128->64->32
        self.up_conv2 = up_conv(conv_dim * 8, conv_dim * 4, 3, 1, 1, 2, 'instance', 'relu')
        self.up_conv3 = up_conv(conv_dim * 4, conv_dim * 2, 3, 1, 1, 2, 'instance', 'relu')
        self.up_conv4 = up_conv(conv_dim * 2, conv_dim,     3, 1, 1, 2, 'instance', 'relu')
        # 最后一层: 32x32 -> 64x64, 输出 RGB, Tanh 把像素压到 [-1, 1]
        self.up_conv5 = up_conv(conv_dim, 3, 3, 1, 1, 2, None, 'tanh')

    def forward(self, z):
        out = self.up_conv1(z)
        out = self.up_conv2(out)
        out = self.up_conv3(out)
        out = self.up_conv4(out)
        out = self.up_conv5(out)
        return out


# ============================================================
# 判别器 (Part 1 原版, 作为基线)
# ============================================================

class DCDiscriminator(nn.Module):
    """原始 DCGAN 判别器, 与 Part 1 相同"""
    def __init__(self, conv_dim=64, norm='instance'):
        super().__init__()
        self.conv1 = conv(3, 32, 4, 2, 1, norm, False, 'relu')
        self.conv2 = conv(32, 64, 4, 2, 1, norm, False, 'relu')
        self.conv3 = conv(64, 128, 4, 2, 1, norm, False, 'relu')
        self.conv4 = conv(128, 256, 4, 2, 1, norm, False, 'relu')
        self.conv5 = conv(256, 1, 4, 1, 0, norm=None, activ=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x.squeeze()


class ResnetBlock(nn.Module):
    """残差块, 留给 Custom Architecture 使用"""
    def __init__(self, conv_dim, norm, activ):
        super().__init__()
        self.conv_layer = conv(
            in_channels=conv_dim, out_channels=conv_dim,
            kernel_size=3, stride=1, padding=1, norm=norm, activ=activ
        )

    def forward(self, x):
        return x + self.conv_layer(x)


# ============================================================
# Part 2 — 变体 1: Spectral Normalization (Miyato et al., 2018)
# ============================================================
# 思路:
#   对判别器的每一层权重 W, 强制 W <- W / sigma(W), 其中 sigma(W) 是最大奇异值。
#   这使得 D 的每层算子范数 <= 1, 从而 D 整体 1-Lipschitz, 有助于稳定训练。
#   sigma(W) 用 power iteration 近似: 维护单位向量 u, v, 迭代
#       v = W^T u / ||W^T u||
#       u = W v   / ||W v||
#   然后 sigma ≈ u^T W v. 实际上每次 forward 做 1 次 iteration 就够了。

class SpectralNorm(nn.Module):
    """
    给一个 Conv2d 模块套上谱归一化 (从零实现, 不使用 torch.nn.utils.spectral_norm)
    """
    def __init__(self, module: nn.Conv2d, n_power_iterations: int = 1):
        super().__init__()
        assert isinstance(module, nn.Conv2d), "SpectralNorm 此处只支持 Conv2d"
        self.module = module
        self.n_power_iterations = n_power_iterations

        # 把 weight 看作 (out_channels, in_channels * kH * kW) 的矩阵
        W = module.weight
        out_ch = W.shape[0]
        in_flat = W.view(out_ch, -1).shape[1]

        # u 的长度等于输出通道数; v 的长度等于 in_channels*kH*kW
        # 把它们注册为 buffer, 这样: (1) 不参与梯度 (2) 会随模型一起 .to(device)
        u = F.normalize(torch.randn(out_ch), dim=0, eps=1e-12)
        v = F.normalize(torch.randn(in_flat), dim=0, eps=1e-12)
        self.register_buffer('u', u)
        self.register_buffer('v', v)

    def _compute_sigma(self):
        """做 power iteration 并返回最大奇异值的估计 sigma (对 W 可导)"""
        W = self.module.weight
        W_mat = W.view(W.shape[0], -1)  # (out_ch, in_flat)

        # power iteration 不参与反传, 只用来刷新 u, v 的估计
        with torch.no_grad():
            u = self.u
            v = self.v
            for _ in range(self.n_power_iterations):
                # v_new ∝ W^T u
                v = F.normalize(torch.mv(W_mat.t(), u), dim=0, eps=1e-12)
                # u_new ∝ W v
                u = F.normalize(torch.mv(W_mat, v), dim=0, eps=1e-12)
            # 就地写回 buffer, 下次 forward 接着用, 收敛更快
            self.u.copy_(u)
            self.v.copy_(v)

        # sigma = u^T W v, 这里 u, v 已 detach, 梯度只沿 W 回传
        sigma = torch.dot(u, torch.mv(W_mat, v))
        return sigma

    def forward(self, x):
        # 1) 估算当前 W 的最大奇异值 sigma
        sigma = self._compute_sigma()
        # 2) 用归一化后的权重做卷积: W_SN = W / sigma
        W_sn = self.module.weight / (sigma + 1e-12)
        return F.conv2d(
            x, W_sn, bias=self.module.bias,
            stride=self.module.stride, padding=self.module.padding,
            dilation=self.module.dilation, groups=self.module.groups,
        )


def sn_conv_block(in_channels, out_channels, kernel_size, stride, padding,
                  use_norm=False, activ='leaky'):
    """
    构造一个 "SN-Conv ( + Norm) + Activ" 块, 专供 SNDiscriminator 使用。
    注: Miyato 原论文里只对 D 的权重加 SN, 不再使用 BatchNorm/InstanceNorm。
         这里默认 use_norm=False, 以贴合原论文做法。
    """
    layers = [SpectralNorm(nn.Conv2d(
        in_channels, out_channels, kernel_size, stride, padding, bias=True
    ))]
    if use_norm:
        layers.append(nn.InstanceNorm2d(out_channels))
    if activ == 'leaky':
        layers.append(nn.LeakyReLU(0.2))
    elif activ == 'relu':
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class SNDiscriminator(nn.Module):
    """
    把 DCDiscriminator 的所有 Conv2d 都换成 SpectralNorm(Conv2d), 去掉 norm 层。
    仍然保留 stride=2 下采样的结构: 64 -> 32 -> 16 -> 8 -> 4 -> 1.
    """
    def __init__(self, conv_dim=64):
        super().__init__()
        # 用 LeakyReLU(0.2) 替代 ReLU, 这是 SNGAN 论文里的惯例
        self.conv1 = sn_conv_block(3,   32,  4, 2, 1, use_norm=False)
        self.conv2 = sn_conv_block(32,  64,  4, 2, 1, use_norm=False)
        self.conv3 = sn_conv_block(64,  128, 4, 2, 1, use_norm=False)
        self.conv4 = sn_conv_block(128, 256, 4, 2, 1, use_norm=False)
        # 最后一层直接输出 logit, 无激活
        self.conv5 = SpectralNorm(nn.Conv2d(256, 1, 4, 1, 0))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x.squeeze()


# ============================================================
# Part 2 — 变体 2: Wasserstein GAN with Gradient Penalty
#   (Arjovsky et al. 2017 + Gulrajani et al. 2017)
# ============================================================
# 思路概述:
#   WGAN 不再让 D 输出 0/1 的真假概率, 而是输出一个实数, 作为 Wasserstein 距离
#   的 Kantorovich–Rubinstein 对偶形式里的 "critic" 函数 f.
#
#   原始 WGAN 要求 f 是 1-Lipschitz, 最初用 weight clipping 保证, 但 clipping
#   会破坏优化。WGAN-GP 改用 "梯度惩罚":
#       L_D = E[f(fake)] - E[f(real)] + lambda * E[(||∇_x̂ f(x̂)||_2 - 1)^2]
#   其中 x̂ = eps*real + (1-eps)*fake, eps ~ U[0,1] 随机插值点。
#   G 的损失: L_G = -E[f(fake)]
#
#   关键实现细节:
#     1) Critic 不使用 BatchNorm (GP 是 per-sample 的, BN 会搅乱它); 用 LayerNorm
#        或干脆不加 norm. 这里采用 LayerNorm 风格的做法, 在 pytorch 里用
#        nn.GroupNorm(num_groups=1) 做等效实现, 保持 per-sample 独立。
#     2) Critic 训练频率通常 5 倍于 Generator (n_critic=5, 不在本文件管).
#     3) Adam 推荐 beta1=0.0, beta2=0.9 (训练脚本里设定).
#
# 本文件只负责提供模型 + gradient_penalty 工具函数, 训练循环在单独脚本里。

class WGANCritic(nn.Module):
    """
    WGAN-GP 使用的 critic: 结构和 DCDiscriminator 类似, 但
      - 不使用 BatchNorm/InstanceNorm (BN 会打断 per-sample 梯度惩罚)
      - 输出不加激活 (Wasserstein 距离估计是实数, 可正可负)
      - 激活函数用 LeakyReLU(0.2), 和 WGAN-GP 原论文一致
    """
    def __init__(self, conv_dim=64):
        super().__init__()
        # 用 LayerNorm 替代 BN: GroupNorm(num_groups=1, C) 等价于 LayerNorm over (C,H,W)
        def ln(c):
            return nn.GroupNorm(num_groups=1, num_channels=c)

        # 下采样路径 64 -> 32 -> 16 -> 8 -> 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,   32,  4, 2, 1),
            # 第一层按惯例不加 norm
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,  64,  4, 2, 1),
            ln(64),
            nn.LeakyReLU(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,  128, 4, 2, 1),
            ln(128),
            nn.LeakyReLU(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            ln(256),
            nn.LeakyReLU(0.2),
        )
        # 最后一层: 4x4 -> 1x1, 输出一个标量 (每张图一个 "critic score")
        self.conv5 = nn.Conv2d(256, 1, 4, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x.squeeze()


def gradient_penalty(critic: nn.Module,
                     real: torch.Tensor,
                     fake: torch.Tensor,
                     lambda_gp: float = 10.0) -> torch.Tensor:
    """
    计算 WGAN-GP 的梯度惩罚项:
        GP = lambda * E[(||∇_{x̂} D(x̂)||_2 - 1)^2]
    其中 x̂ = eps*real + (1-eps)*fake, eps ~ U[0,1] per-sample.

    参数
    ----
    critic : 判别器 (critic) 网络
    real   : 真实图像 tensor, 形状 (B, C, H, W)
    fake   : 生成图像 tensor, 与 real 同形状
    lambda_gp : 惩罚系数, 默认 10 (原论文推荐值)

    返回
    ----
    标量 tensor: lambda_gp * E[(|grad| - 1)^2], 已乘好系数, 可直接加到 D loss 上。
    """
    # 1) 为每个样本生成独立的 eps, 形状 (B, 1, 1, 1) 便于广播
    batch_size = real.size(0)
    eps = torch.rand(batch_size, 1, 1, 1, device=real.device, dtype=real.dtype)

    # 2) 插值点 x̂. requires_grad_=True 以便对它求梯度
    x_hat = eps * real + (1.0 - eps) * fake
    x_hat.requires_grad_(True)

    # 3) 计算 critic 在 x̂ 上的输出
    d_hat = critic(x_hat)

    # 4) 对 d_hat 的每个样本求它对 x̂ 的梯度
    #    create_graph=True 因为 GP 本身要参与 D 参数的二阶反传
    grads = torch.autograd.grad(
        outputs=d_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # 5) 把梯度展平后求每个样本的 L2 范数
    grads = grads.view(batch_size, -1)
    grad_norm = grads.norm(2, dim=1)

    # 6) 惩罚 ||grad|| 偏离 1 的程度 (平方)
    gp = ((grad_norm - 1.0) ** 2).mean()
    return lambda_gp * gp


# ============================================================
# Part 2 — 变体 3: LSGAN (Mao et al., 2017)
# ============================================================
# 思路概述:
#   把原始 GAN 的 BCE (sigmoid + cross-entropy) 替换为 MSE (least-squares).
#   目标:
#       L_D = 0.5 * E[(D(real) - 1)^2] + 0.5 * E[(D(fake) - 0)^2]
#       L_G = 0.5 * E[(D(G(z)) - 1)^2]
#   动机:
#     - BCE 在 D 几乎判定正确时梯度饱和 (sigmoid 进入平坦区); MSE 不饱和,
#       即使假样本被判在正确一侧, 只要离决策面越远, 梯度越大, G 会被持续"推回来"。
#     - 在实践中 LSGAN 训练更稳定, 生成图像更锐利。
#
#   架构上 LSGAN 和 DCGAN 完全相同 — 只改损失, 不改模型。
#   因此这里只提供 loss 函数封装; 模型直接复用本文件顶部的 DCGenerator / DCDiscriminator。
#   注: Part 1 的 vanilla_gan.py 实际上已经使用了 LSGAN 形式的 MSE loss,
#       这里单独列出来是为了在 Part 2 有一个独立可对比的 run。

def lsgan_d_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """
    LSGAN 判别器损失:
        L_D = 0.5 * E[(D(real) - 1)^2] + 0.5 * E[D(fake)^2]
    传入的 d_real / d_fake 都是 D 的原始输出 (logit), 无需 sigmoid.
    """
    # 让 D(real) 向 1 靠拢
    loss_real = 0.5 * torch.mean((d_real - 1.0) ** 2)
    # 让 D(fake) 向 0 靠拢
    loss_fake = 0.5 * torch.mean(d_fake ** 2)
    return loss_real + loss_fake


def lsgan_g_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """
    LSGAN 生成器损失:
        L_G = 0.5 * E[(D(G(z)) - 1)^2]
    生成器希望 D 把假图误判为真 (即输出靠近 1)。
    """
    return 0.5 * torch.mean((d_fake - 1.0) ** 2)


# 给 LSGAN 提供语义上独立的"别名"类, 便于训练脚本里按 --model lsgan 分派
# (实际结构与 DCGenerator/DCDiscriminator 完全相同)
class LSGANGenerator(DCGenerator):
    """LSGAN 使用的生成器: 结构与 DCGenerator 相同, 仅在训练损失上不同"""
    pass


class LSGANDiscriminator(DCDiscriminator):
    """LSGAN 使用的判别器: 结构与 DCDiscriminator 相同, 最后一层输出 logit (无 sigmoid)"""
    pass


# ============================================================
# Part 2 — 变体 4: Custom Architecture
# ============================================================
# 设计思路:
#   目标是在 Grumpy-Cat 这种非常小的数据集 (204 张) 上拿到更稳定、更锐利的生成结果。
#   综合前三种变体的经验, 本自定义架构同时采用以下三点改进:
#
#   (a) 生成器插入残差块 (ResnetBlock):
#       - 原 DCGenerator 是纯前馈的 5 层 upsample. 在每个分辨率后加一个 ResBlock,
#         可以在不增加尺度的情况下增加"深度可塑性", 让 G 学到更精细的纹理。
#       - 残差连接缓解深层网络的梯度消失, 有利于 64x64 这种较高分辨率的细节。
#
#   (b) 判别器使用 Spectral Normalization:
#       - 直接复用变体 1 的 SpectralNorm, 让 D 成为 1-Lipschitz, 训练更稳。
#       - 在 Grumpy-Cat 这种小数据集上, SN 比 BatchNorm 更能抑制 D 过拟合。
#
#   (c) 判别器使用 SN-ResBlock:
#       - 在 stride-2 下采样之间再插入一个带 SN 的 3x3 卷积残差块, 等效于加深 D
#         但不改变空间尺寸, 提升判别力。
#
#   训练损失: 搭配 LSGAN (MSE) 使用, 因为 SN + MSE 的组合在实践中比 SN + BCE 更稳。
#   训练脚本应当用 advanced 数据增强 (RandomCrop / HFlip / ColorJitter)。

class CustomGenerator(nn.Module):
    """
    Custom 生成器: DCGenerator 的每个尺度后插入一个 ResnetBlock。
    结构:
        z (100x1x1)
          -> up_conv1 -> 256x4x4  -> ResBlock
          -> up_conv2 -> 128x8x8  -> ResBlock
          -> up_conv3 -> 64x16x16 -> ResBlock
          -> up_conv4 -> 32x32x32
          -> up_conv5 -> 3x64x64 (tanh)
    """
    def __init__(self, noise_size, conv_dim=64):
        super().__init__()
        # 第一层: 1x1 -> 4x4, 通道 noise -> 256 (直接 conv, 不 upsample)
        self.up_conv1 = conv(
            in_channels=noise_size, out_channels=conv_dim * 8,
            kernel_size=4, stride=1, padding=3,
            norm='instance', activ='relu'
        )
        # 4x4 分辨率上的残差块: 强化 coarse 结构特征
        self.res1 = ResnetBlock(conv_dim * 8, norm='instance', activ='relu')

        # 4x4 -> 8x8
        self.up_conv2 = up_conv(conv_dim * 8, conv_dim * 4, 3, 1, 1, 2, 'instance', 'relu')
        # 8x8 分辨率上的残差块: 学习中层特征 (比如脸部整体形状)
        self.res2 = ResnetBlock(conv_dim * 4, norm='instance', activ='relu')

        # 8x8 -> 16x16
        self.up_conv3 = up_conv(conv_dim * 4, conv_dim * 2, 3, 1, 1, 2, 'instance', 'relu')
        # 16x16 分辨率上的残差块: 学习局部纹理 (毛发、眼睛轮廓)
        self.res3 = ResnetBlock(conv_dim * 2, norm='instance', activ='relu')

        # 16x16 -> 32x32, 32x32 -> 64x64. 最后两级不加 ResBlock 以控制参数量
        self.up_conv4 = up_conv(conv_dim * 2, conv_dim,     3, 1, 1, 2, 'instance', 'relu')
        # 最后一层用 tanh 把像素压到 [-1, 1], 与真图 Normalize 后的范围对齐
        self.up_conv5 = up_conv(conv_dim,     3,            3, 1, 1, 2, None, 'tanh')

    def forward(self, z):
        out = self.up_conv1(z)
        out = self.res1(out)
        out = self.up_conv2(out)
        out = self.res2(out)
        out = self.up_conv3(out)
        out = self.res3(out)
        out = self.up_conv4(out)
        out = self.up_conv5(out)
        return out


class SNResBlock(nn.Module):
    """
    带 SpectralNorm 的残差块, 专供 CustomDiscriminator 使用。
    两层 3x3 SN-Conv, 不改变空间尺寸, 不加 norm, 用 LeakyReLU(0.2) 激活。
    """
    def __init__(self, channels):
        super().__init__()
        # 两层 SN 卷积, stride=1 保持分辨率; padding=1 保持空间尺寸
        self.sn_conv1 = SpectralNorm(nn.Conv2d(channels, channels, 3, 1, 1))
        self.sn_conv2 = SpectralNorm(nn.Conv2d(channels, channels, 3, 1, 1))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        # 残差路径: x -> LReLU -> SNConv -> LReLU -> SNConv
        h = self.act(self.sn_conv1(x))
        h = self.sn_conv2(h)
        # 主路径 + 残差, 再过一次激活
        return self.act(x + h)


class CustomDiscriminator(nn.Module):
    """
    Custom 判别器: 在 SNDiscriminator 的基础上插入 SN-ResBlock。
    结构:
        3x64x64
          -> SN-Conv 3->32 (stride 2)   -> 32x32x32
          -> SN-Conv 32->64 (stride 2)  -> 64x16x16  -> SNResBlock
          -> SN-Conv 64->128 (stride 2) -> 128x8x8   -> SNResBlock
          -> SN-Conv 128->256 (stride 2)-> 256x4x4
          -> SN-Conv 256->1 (4x4)       -> 1x1x1 (logit)
    """
    def __init__(self, conv_dim=64):
        super().__init__()
        # 每层均使用 sn_conv_block (SN-Conv + LeakyReLU), 不加 norm
        self.conv1 = sn_conv_block(3,   32,  4, 2, 1, use_norm=False, activ='leaky')
        self.conv2 = sn_conv_block(32,  64,  4, 2, 1, use_norm=False, activ='leaky')
        # 在 16x16 尺度加一层残差块, 增加判别深度
        self.res2 = SNResBlock(64)
        self.conv3 = sn_conv_block(64,  128, 4, 2, 1, use_norm=False, activ='leaky')
        # 在 8x8 尺度再加一个残差块
        self.res3 = SNResBlock(128)
        self.conv4 = sn_conv_block(128, 256, 4, 2, 1, use_norm=False, activ='leaky')
        # 最后 4x4 -> 1x1 输出 logit, 无激活
        self.conv5 = SpectralNorm(nn.Conv2d(256, 1, 4, 1, 0))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res2(x)
        x = self.conv3(x)
        x = self.res3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x.squeeze()
