# CMU 18-780/6 Homework 5 - Part 2 GAN variants.
# Implements Spectral Norm, WGAN-GP, LSGAN and a Custom architecture.
# Spectral normalization is implemented from scratch
# (torch.nn.utils.spectral_norm is NOT used).

import torch
import torch.nn as nn
import torch.nn.functional as F


def up_conv(in_channels, out_channels, kernel_size, stride=1, padding=1,
            scale_factor=2, norm='batch', activ=None):
    """Upsample + Conv2d with optional norm / activation."""
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
    """Plain Conv2d with optional norm / activation."""
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


class DCGenerator(nn.Module):
    """DCGAN generator: z -> 64x64 RGB. Same architecture as Part 1."""
    def __init__(self, noise_size, conv_dim=64):
        super().__init__()
        self.up_conv1 = conv(
            in_channels=noise_size, out_channels=conv_dim * 8,
            kernel_size=4, stride=1, padding=3,
            norm='instance', activ='relu'
        )
        self.up_conv2 = up_conv(conv_dim * 8, conv_dim * 4, 3, 1, 1, 2, 'instance', 'relu')
        self.up_conv3 = up_conv(conv_dim * 4, conv_dim * 2, 3, 1, 1, 2, 'instance', 'relu')
        self.up_conv4 = up_conv(conv_dim * 2, conv_dim,     3, 1, 1, 2, 'instance', 'relu')
        self.up_conv5 = up_conv(conv_dim, 3, 3, 1, 1, 2, None, 'tanh')

    def forward(self, z):
        out = self.up_conv1(z)
        out = self.up_conv2(out)
        out = self.up_conv3(out)
        out = self.up_conv4(out)
        out = self.up_conv5(out)
        return out


class DCDiscriminator(nn.Module):
    """Baseline DCGAN discriminator, same as Part 1."""
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
    """Residual block used by the Custom architecture."""
    def __init__(self, conv_dim, norm, activ):
        super().__init__()
        self.conv_layer = conv(
            in_channels=conv_dim, out_channels=conv_dim,
            kernel_size=3, stride=1, padding=1, norm=norm, activ=activ
        )

    def forward(self, x):
        return x + self.conv_layer(x)


# ============================================================
# Variant 1: Spectral Normalization (Miyato et al., 2018)
# ============================================================
# For each weight W in D, replace W with W / sigma(W) where sigma(W) is
# the largest singular value, estimated by power iteration:
#     v = W^T u / ||W^T u||
#     u = W v   / ||W v||
#     sigma ~= u^T W v
# One iteration per forward pass is typically enough in practice.

class SpectralNorm(nn.Module):
    """Spectral-norm wrapper around a Conv2d module, implemented from scratch
    (does not use torch.nn.utils.spectral_norm)."""
    def __init__(self, module: nn.Conv2d, n_power_iterations: int = 1):
        super().__init__()
        assert isinstance(module, nn.Conv2d), "SpectralNorm only supports Conv2d here"
        self.module = module
        self.n_power_iterations = n_power_iterations

        # Treat weight as a (out_channels, in_channels * kH * kW) matrix.
        W = module.weight
        out_ch = W.shape[0]
        in_flat = W.view(out_ch, -1).shape[1]

        # Register u, v as buffers so they do not take gradients
        # and automatically move with the module via .to(device).
        u = F.normalize(torch.randn(out_ch), dim=0, eps=1e-12)
        v = F.normalize(torch.randn(in_flat), dim=0, eps=1e-12)
        self.register_buffer('u', u)
        self.register_buffer('v', v)

    def _compute_sigma(self):
        """Run power iteration and return sigma, differentiable wrt W."""
        W = self.module.weight
        W_mat = W.view(W.shape[0], -1)  # (out_ch, in_flat)

        # Power iteration is only used to refresh u / v; it has no grad.
        with torch.no_grad():
            u = self.u
            v = self.v
            for _ in range(self.n_power_iterations):
                v = F.normalize(torch.mv(W_mat.t(), u), dim=0, eps=1e-12)
                u = F.normalize(torch.mv(W_mat, v), dim=0, eps=1e-12)
            self.u.copy_(u)
            self.v.copy_(v)

        # sigma = u^T W v; u, v are detached so the gradient flows only via W.
        sigma = torch.dot(u, torch.mv(W_mat, v))
        return sigma

    def forward(self, x):
        sigma = self._compute_sigma()
        W_sn = self.module.weight / (sigma + 1e-12)
        return F.conv2d(
            x, W_sn, bias=self.module.bias,
            stride=self.module.stride, padding=self.module.padding,
            dilation=self.module.dilation, groups=self.module.groups,
        )


def sn_conv_block(in_channels, out_channels, kernel_size, stride, padding,
                  use_norm=False, activ='leaky'):
    """SN-Conv (+ optional norm) + activation block for SNDiscriminator.
    Miyato et al. apply SN only to D's weights and drop BN / InstanceNorm,
    so use_norm defaults to False."""
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
    """DCDiscriminator with every Conv2d wrapped in SpectralNorm and no norm
    layers. Downsampling path: 64 -> 32 -> 16 -> 8 -> 4 -> 1."""
    def __init__(self, conv_dim=64):
        super().__init__()
        self.conv1 = sn_conv_block(3,   32,  4, 2, 1, use_norm=False)
        self.conv2 = sn_conv_block(32,  64,  4, 2, 1, use_norm=False)
        self.conv3 = sn_conv_block(64,  128, 4, 2, 1, use_norm=False)
        self.conv4 = sn_conv_block(128, 256, 4, 2, 1, use_norm=False)
        self.conv5 = SpectralNorm(nn.Conv2d(256, 1, 4, 1, 0))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x.squeeze()


# ============================================================
# Variant 2: Wasserstein GAN with Gradient Penalty
#   (Arjovsky et al. 2017 + Gulrajani et al. 2017)
# ============================================================
# D outputs a real-valued critic score f.
#     L_D = E[f(fake)] - E[f(real)] + lambda * E[(||grad_x_hat f(x_hat)||_2 - 1)^2]
#     L_G = -E[f(fake)]
# where x_hat = eps * real + (1 - eps) * fake, eps ~ U[0, 1] per sample.
#
# Implementation notes:
#   - The critic must NOT use BatchNorm (GP is per-sample; BN mixes samples).
#     We use GroupNorm(num_groups=1), which is LayerNorm over (C, H, W).
#   - n_critic = 5 and Adam betas (0.0, 0.9) are set in the training script.

class WGANCritic(nn.Module):
    """WGAN-GP critic. Similar to DCDiscriminator but with:
      - no BatchNorm / InstanceNorm (broken by GP);
      - no activation at the output (score is a real number);
      - LeakyReLU(0.2) as the hidden activation (WGAN-GP paper)."""
    def __init__(self, conv_dim=64):
        super().__init__()
        # GroupNorm(num_groups=1, C) is equivalent to LayerNorm over (C, H, W).
        def ln(c):
            return nn.GroupNorm(num_groups=1, num_channels=c)

        # Downsampling path: 64 -> 32 -> 16 -> 8 -> 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,   32,  4, 2, 1),
            # First block has no norm, per convention.
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
        # Final layer 4x4 -> 1x1: one scalar score per image.
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
    """Compute the WGAN-GP gradient penalty:
        GP = lambda * E[(||grad_{x_hat} D(x_hat)||_2 - 1)^2]
    with x_hat = eps * real + (1 - eps) * fake, eps ~ U[0, 1] per sample.

    Returns a scalar tensor pre-multiplied by lambda_gp, ready to be added
    to the critic loss."""
    batch_size = real.size(0)
    eps = torch.rand(batch_size, 1, 1, 1, device=real.device, dtype=real.dtype)

    x_hat = eps * real + (1.0 - eps) * fake
    x_hat.requires_grad_(True)

    d_hat = critic(x_hat)

    # create_graph=True: GP participates in second-order backprop through D's params.
    grads = torch.autograd.grad(
        outputs=d_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grads = grads.view(batch_size, -1)
    grad_norm = grads.norm(2, dim=1)

    gp = ((grad_norm - 1.0) ** 2).mean()
    return lambda_gp * gp


# ============================================================
# Variant 3: LSGAN (Mao et al., 2017)
# ============================================================
# Replace BCE with MSE:
#     L_D = 0.5 * E[(D(real) - 1)^2] + 0.5 * E[(D(fake) - 0)^2]
#     L_G = 0.5 * E[(D(G(z)) - 1)^2]
# Architecture is identical to DCGAN; only the loss changes.

def lsgan_d_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """LSGAN discriminator loss.
    d_real / d_fake are raw logits (no sigmoid)."""
    loss_real = 0.5 * torch.mean((d_real - 1.0) ** 2)
    loss_fake = 0.5 * torch.mean(d_fake ** 2)
    return loss_real + loss_fake


def lsgan_g_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """LSGAN generator loss: push D(G(z)) toward 1."""
    return 0.5 * torch.mean((d_fake - 1.0) ** 2)


# Semantic aliases for the LSGAN variant (same architecture as DCGAN).
class LSGANGenerator(DCGenerator):
    """LSGAN generator: identical architecture to DCGenerator."""
    pass


class LSGANDiscriminator(DCDiscriminator):
    """LSGAN discriminator: identical architecture to DCDiscriminator;
    final layer outputs a raw logit."""
    pass


# ============================================================
# Variant 4: Custom Architecture
# ============================================================
# Three design choices aimed at the small (204-image) Grumpy-Cat dataset:
#   (a) Insert ResnetBlocks into G at each resolution to add depth without
#       increasing spatial size.
#   (b) Apply SpectralNorm to D so it is 1-Lipschitz.
#   (c) Add SN-ResBlocks inside D to further deepen it without resolution change.
# Trained with LSGAN (MSE) loss and advanced data augmentation.

class CustomGenerator(nn.Module):
    """Custom generator: DCGenerator with a ResnetBlock after each upsample.
        z (100x1x1)
          -> up_conv1 -> 256x4x4  -> ResBlock
          -> up_conv2 -> 128x8x8  -> ResBlock
          -> up_conv3 -> 64x16x16 -> ResBlock
          -> up_conv4 -> 32x32x32
          -> up_conv5 -> 3x64x64 (tanh)"""
    def __init__(self, noise_size, conv_dim=64):
        super().__init__()
        self.up_conv1 = conv(
            in_channels=noise_size, out_channels=conv_dim * 8,
            kernel_size=4, stride=1, padding=3,
            norm='instance', activ='relu'
        )
        self.res1 = ResnetBlock(conv_dim * 8, norm='instance', activ='relu')

        self.up_conv2 = up_conv(conv_dim * 8, conv_dim * 4, 3, 1, 1, 2, 'instance', 'relu')
        self.res2 = ResnetBlock(conv_dim * 4, norm='instance', activ='relu')

        self.up_conv3 = up_conv(conv_dim * 4, conv_dim * 2, 3, 1, 1, 2, 'instance', 'relu')
        self.res3 = ResnetBlock(conv_dim * 2, norm='instance', activ='relu')

        self.up_conv4 = up_conv(conv_dim * 2, conv_dim,     3, 1, 1, 2, 'instance', 'relu')
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
    """Residual block with SpectralNorm-wrapped Conv2d layers, used inside
    CustomDiscriminator. Two 3x3 SN-Convs, no norm, LeakyReLU(0.2)."""
    def __init__(self, channels):
        super().__init__()
        self.sn_conv1 = SpectralNorm(nn.Conv2d(channels, channels, 3, 1, 1))
        self.sn_conv2 = SpectralNorm(nn.Conv2d(channels, channels, 3, 1, 1))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.act(self.sn_conv1(x))
        h = self.sn_conv2(h)
        return self.act(x + h)


class CustomDiscriminator(nn.Module):
    """Custom discriminator: SNDiscriminator backbone with two SN-ResBlocks
    inserted at intermediate resolutions.
        3x64x64
          -> SN-Conv 3->32 (stride 2)   -> 32x32x32
          -> SN-Conv 32->64 (stride 2)  -> 64x16x16  -> SNResBlock
          -> SN-Conv 64->128 (stride 2) -> 128x8x8   -> SNResBlock
          -> SN-Conv 128->256 (stride 2)-> 256x4x4
          -> SN-Conv 256->1 (4x4)       -> 1x1x1 (logit)"""
    def __init__(self, conv_dim=64):
        super().__init__()
        self.conv1 = sn_conv_block(3,   32,  4, 2, 1, use_norm=False, activ='leaky')
        self.conv2 = sn_conv_block(32,  64,  4, 2, 1, use_norm=False, activ='leaky')
        self.res2 = SNResBlock(64)
        self.conv3 = sn_conv_block(64,  128, 4, 2, 1, use_norm=False, activ='leaky')
        self.res3 = SNResBlock(128)
        self.conv4 = sn_conv_block(128, 256, 4, 2, 1, use_norm=False, activ='leaky')
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
