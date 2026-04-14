# Part 2 训练主文件, 统一入口支持 4 种变体:
#   --variant sn        : Spectral Normalization (+ LSGAN loss)
#   --variant wgan_gp   : Wasserstein GAN with Gradient Penalty
#   --variant lsgan     : 纯 LSGAN (架构同 DCGAN, 只改损失)
#   --variant custom    : Custom 架构 (SN + ResBlock, LSGAN loss)
#
# 用法示例:
#   python gan_variants.py --variant sn      --num_epochs 500
#   python gan_variants.py --variant wgan_gp --num_epochs 500
#   python gan_variants.py --variant lsgan   --num_epochs 500
#   python gan_variants.py --variant custom  --num_epochs 500

import argparse
import math   # 兼容 numpy>=2.0 (np.math 已废弃)
import os

import imageio
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import utils
from data_loader import get_data_loader

# 从 model_variants 导入 Part 2 所有变体模型与工具函数
from model_variants import (
    DCGenerator, DCDiscriminator,
    SNDiscriminator,
    WGANCritic, gradient_penalty,
    LSGANGenerator, LSGANDiscriminator,
    lsgan_d_loss, lsgan_g_loss,
    CustomGenerator, CustomDiscriminator,
)


SEED = 11
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# ============================================================
# 根据 --variant 分派对应的 (Generator, Discriminator) 类
# ============================================================

def build_models(variant: str, noise_size: int, conv_dim: int):
    """
    根据 variant 名称返回一对 (G, D) 实例。
    注意: WGAN-GP 的 D 其实是 critic (输出实数而非概率)。
    """
    if variant == 'sn':
        # SN: 生成器沿用 DCGenerator, 判别器换成 SN 版
        G = DCGenerator(noise_size=noise_size, conv_dim=conv_dim)
        D = SNDiscriminator(conv_dim=conv_dim)
    elif variant == 'wgan_gp':
        # WGAN-GP: 判别器无 sigmoid, 用 LayerNorm 式 GroupNorm
        G = DCGenerator(noise_size=noise_size, conv_dim=conv_dim)
        D = WGANCritic(conv_dim=conv_dim)
    elif variant == 'lsgan':
        # LSGAN: 结构与 DCGAN 完全相同, 仅损失函数为 MSE
        G = LSGANGenerator(noise_size=noise_size, conv_dim=conv_dim)
        D = LSGANDiscriminator(conv_dim=conv_dim)
    elif variant == 'custom':
        # Custom: 生成器加 ResBlock, 判别器加 SN + SN-ResBlock
        G = CustomGenerator(noise_size=noise_size, conv_dim=conv_dim)
        D = CustomDiscriminator(conv_dim=conv_dim)
    else:
        raise ValueError(f"未知的 variant: {variant}")

    if torch.cuda.is_available():
        G.cuda()
        D.cuda()
        print('Models moved to GPU.')
    return G, D


# ============================================================
# 图像网格 / 样本保存 / checkpoint (基本复用 vanilla_gan.py)
# ============================================================

def create_image_grid(array, ncols=None):
    num_images, channels, cell_h, cell_w = array.shape
    if not ncols:
        ncols = int(np.sqrt(num_images))
    nrows = int(math.floor(num_images / float(ncols)))
    result = np.zeros(
        (cell_h * nrows, cell_w * ncols, channels), dtype=array.dtype
    )
    for i in range(nrows):
        for j in range(ncols):
            result[i * cell_h:(i + 1) * cell_h,
                   j * cell_w:(j + 1) * cell_w, :] = array[i * ncols + j].transpose(1, 2, 0)
    if channels == 1:
        result = result.squeeze()
    return result


def save_samples(G, fixed_noise, iteration, opts):
    """用 fixed_noise 生成样本并保存为 png 网格"""
    G.eval()
    with torch.no_grad():
        generated_images = G(fixed_noise)
    G.train()
    generated_images = utils.to_data(generated_images)
    grid = create_image_grid(generated_images)
    grid = np.uint8(255 * (grid + 1) / 2)
    path = os.path.join(opts.sample_dir, f'sample-{iteration:06d}.png')
    imageio.imwrite(path, grid)
    print(f'Saved {path}')


def save_images(images, iteration, opts, name):
    grid = create_image_grid(utils.to_data(images))
    path = os.path.join(opts.sample_dir, f'{name}-{iteration:06d}.png')
    grid = np.uint8(255 * (grid + 1) / 2)
    imageio.imwrite(path, grid)


def checkpoint(iteration, G, D, opts):
    torch.save(G.state_dict(), os.path.join(opts.checkpoint_dir, f'G_iter{iteration}.pkl'))
    torch.save(D.state_dict(), os.path.join(opts.checkpoint_dir, f'D_iter{iteration}.pkl'))


def sample_noise(batch_size, dim):
    """
    采样 U(-1, 1) 噪声, shape = (B, dim, 1, 1).
    对 WGAN-GP 也使用同样的分布以便公平对比。
    """
    return utils.to_var(torch.rand(batch_size, dim) * 2 - 1).unsqueeze(2).unsqueeze(3)


# ============================================================
# 训练循环 A: 通用非 WGAN 分支 (sn / lsgan / custom)
# 三者都用 LSGAN (MSE) 损失; sn 和 custom 的判别器已经内置 SN
# ============================================================

def training_loop_lsgan(train_dataloader, G, D, opts, logger):
    """SN / LSGAN / Custom 共用的训练循环, 采用 LSGAN 形式的 MSE 损失"""
    # LSGAN 常用的 Adam 配置, 与 Part 1 一致
    g_optim = optim.Adam(G.parameters(), opts.lr, [opts.beta1, opts.beta2])
    d_optim = optim.Adam(D.parameters(), opts.lr, [opts.beta1, opts.beta2])

    # 固定噪声用于纵向可视化 G 的进步
    fixed_noise = sample_noise(opts.batch_size, opts.noise_size)

    iteration = 1
    total_iters = opts.num_epochs * len(train_dataloader)
    for _ in range(opts.num_epochs):
        for batch in train_dataloader:
            real_images = utils.to_var(batch)
            bs = real_images.size(0)

            # ---------- 更新 D ----------
            d_real = D(real_images)
            noise = sample_noise(bs, opts.noise_size)
            fake_images = G(noise)
            d_fake = D(fake_images.detach())        # detach 阻断梯度流向 G
            d_loss = lsgan_d_loss(d_real, d_fake)   # MSE 判别损失

            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            # ---------- 更新 G ----------
            noise = sample_noise(bs, opts.noise_size)
            fake_images = G(noise)
            g_loss = lsgan_g_loss(D(fake_images))   # 希望 D(G(z)) -> 1

            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

            # ---------- 日志 / 采样 / 存档 ----------
            if iteration % opts.log_step == 0:
                print(f'Iter [{iteration:5d}/{total_iters}] | '
                      f'D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}')
                logger.add_scalar('D/total', d_loss.item(), iteration)
                logger.add_scalar('G/total', g_loss.item(), iteration)

            if iteration % opts.sample_every == 0:
                save_samples(G, fixed_noise, iteration, opts)
                save_images(real_images, iteration, opts, 'real')
            if iteration % opts.checkpoint_every == 0:
                checkpoint(iteration, G, D, opts)

            iteration += 1


# ============================================================
# 训练循环 B: WGAN-GP 分支
# - Critic 训练 n_critic 次, 生成器训练 1 次
# - Adam 推荐 beta1=0.0, beta2=0.9
# ============================================================

def training_loop_wgan_gp(train_dataloader, G, D, opts, logger):
    """WGAN-GP 专用训练循环"""
    # 注意: WGAN-GP 原论文推荐的 Adam 超参与 LSGAN 不同
    g_optim = optim.Adam(G.parameters(), lr=opts.lr, betas=(0.0, 0.9))
    d_optim = optim.Adam(D.parameters(), lr=opts.lr, betas=(0.0, 0.9))

    fixed_noise = sample_noise(opts.batch_size, opts.noise_size)

    iteration = 1
    total_iters = opts.num_epochs * len(train_dataloader)
    # 将 dataloader 转为可重复取下一 batch 的迭代器
    data_iter = iter(train_dataloader)

    def _next_batch():
        """数据迭代器耗尽时自动重启, 保证 critic 的 n_critic 次更新都能拿到 batch"""
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            return next(data_iter)

    for _ in range(opts.num_epochs):
        for _ in range(len(train_dataloader)):
            # ---------- Critic 训练 n_critic 次 ----------
            for _ in range(opts.n_critic):
                real_images = utils.to_var(_next_batch())
                bs = real_images.size(0)

                noise = sample_noise(bs, opts.noise_size)
                fake_images = G(noise).detach()  # 更新 D 时切断 G 的梯度

                d_real = D(real_images)
                d_fake = D(fake_images)

                # Wasserstein 距离的对偶形式损失: minimize E[D(fake)] - E[D(real)]
                # 外加梯度惩罚 GP
                gp = gradient_penalty(D, real_images, fake_images,
                                      lambda_gp=opts.lambda_gp)
                d_loss = d_fake.mean() - d_real.mean() + gp

                d_optim.zero_grad()
                d_loss.backward()
                d_optim.step()

            # ---------- Generator 训练 1 次 ----------
            noise = sample_noise(opts.batch_size, opts.noise_size)
            fake_images = G(noise)
            # G 希望 D(G(z)) 越大越好 -> loss 取负
            g_loss = -D(fake_images).mean()

            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

            # ---------- 日志 / 采样 / 存档 ----------
            if iteration % opts.log_step == 0:
                # 记录 Wasserstein estimate (d_real - d_fake) 约等于真实 W 距离
                w_dist = (d_real.mean() - d_fake.mean()).item()
                print(f'Iter [{iteration:5d}/{total_iters}] | '
                      f'D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f} | '
                      f'W_dist: {w_dist:.4f} | GP: {gp.item():.4f}')
                logger.add_scalar('D/total', d_loss.item(), iteration)
                logger.add_scalar('G/total', g_loss.item(), iteration)
                logger.add_scalar('D/w_distance', w_dist, iteration)
                logger.add_scalar('D/gradient_penalty', gp.item(), iteration)

            if iteration % opts.sample_every == 0:
                save_samples(G, fixed_noise, iteration, opts)
                save_images(real_images, iteration, opts, 'real')
            if iteration % opts.checkpoint_every == 0:
                checkpoint(iteration, G, D, opts)

            iteration += 1


# ============================================================
# main
# ============================================================

def main(opts):
    # 创建数据加载器 (Part 2 建议全部使用 advanced 增强)
    dataloader = get_data_loader(opts.data, opts)

    # 创建输出目录
    utils.create_dir(opts.checkpoint_dir)
    utils.create_dir(opts.sample_dir)

    # 日志器写到 sample_dir, 方便 TensorBoard 一起查看
    logger = SummaryWriter(opts.sample_dir)

    # 构建模型
    G, D = build_models(opts.variant, opts.noise_size, opts.conv_dim)
    print('                    G                  '); print(G)
    print('                    D                  '); print(D)

    # 分派训练循环
    if opts.variant == 'wgan_gp':
        training_loop_wgan_gp(dataloader, G, D, opts, logger)
    else:
        training_loop_lsgan(dataloader, G, D, opts, logger)


def create_parser():
    parser = argparse.ArgumentParser()

    # 变体选择
    parser.add_argument('--variant', type=str, required=True,
                        choices=['sn', 'wgan_gp', 'lsgan', 'custom'],
                        help='要训练的 Part 2 变体')

    # 模型 / 训练超参
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--conv_dim', type=int, default=32)
    parser.add_argument('--noise_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)   # 仅对 LSGAN 分支有效
    parser.add_argument('--beta2', type=float, default=0.999)

    # WGAN-GP 专用
    parser.add_argument('--n_critic', type=int, default=5,
                        help='每次 G 更新前 D 连续更新的步数 (仅 WGAN-GP)')
    parser.add_argument('--lambda_gp', type=float, default=10.0,
                        help='梯度惩罚系数 (仅 WGAN-GP)')

    # 数据源
    parser.add_argument('--data', type=str, default='cat/grumpifyBprocessed')
    # Part 2 默认 advanced 增强 (Part 1 已证明小数据集上 advanced 明显更好)
    parser.add_argument('--data_preprocess', type=str, default='advanced')
    parser.add_argument('--ext', type=str, default='*.png')

    # 路径
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='若为 None 则自动按 variant 命名')
    parser.add_argument('--sample_dir', type=str, default=None,
                        help='若为 None 则自动按 variant 命名')
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=200)
    parser.add_argument('--checkpoint_every', type=int, default=400)

    return parser


if __name__ == '__main__':
    opts = create_parser().parse_args()

    # 如果用户没指定输出目录, 就按 variant 自动分目录, 避免互相覆盖
    if opts.sample_dir is None:
        opts.sample_dir = os.path.join('output', opts.variant,
                                       f'{os.path.basename(opts.data)}_{opts.data_preprocess}')
    if opts.checkpoint_dir is None:
        opts.checkpoint_dir = os.path.join('checkpoints', opts.variant)

    # 与 vanilla_gan.py 一致: 同目录重跑先清空旧样本
    if os.path.exists(opts.sample_dir):
        os.system(f'rm {opts.sample_dir}/*')

    print(opts)
    main(opts)
