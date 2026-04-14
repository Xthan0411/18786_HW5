# Unified Part 2 training entry point, supporting four variants:
#   --variant sn        : Spectral Normalization (+ LSGAN loss)
#   --variant wgan_gp   : Wasserstein GAN with Gradient Penalty
#   --variant lsgan     : plain LSGAN (DCGAN architecture, MSE loss only)
#   --variant custom    : Custom architecture (SN + ResBlock, LSGAN loss)
#
# Example usage:
#   python gan_variants.py --variant sn      --num_epochs 500
#   python gan_variants.py --variant wgan_gp --num_epochs 500
#   python gan_variants.py --variant lsgan   --num_epochs 500
#   python gan_variants.py --variant custom  --num_epochs 500

import argparse
import math   # numpy>=2.0 removed np.math
import os

import imageio
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import utils
from data_loader import get_data_loader

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


def build_models(variant: str, noise_size: int, conv_dim: int):
    """Return a (G, D) pair for the given variant name.
    Note: for WGAN-GP, D is a real-valued critic (no sigmoid)."""
    if variant == 'sn':
        G = DCGenerator(noise_size=noise_size, conv_dim=conv_dim)
        D = SNDiscriminator(conv_dim=conv_dim)
    elif variant == 'wgan_gp':
        G = DCGenerator(noise_size=noise_size, conv_dim=conv_dim)
        D = WGANCritic(conv_dim=conv_dim)
    elif variant == 'lsgan':
        G = LSGANGenerator(noise_size=noise_size, conv_dim=conv_dim)
        D = LSGANDiscriminator(conv_dim=conv_dim)
    elif variant == 'custom':
        G = CustomGenerator(noise_size=noise_size, conv_dim=conv_dim)
        D = CustomDiscriminator(conv_dim=conv_dim)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    if torch.cuda.is_available():
        G.cuda()
        D.cuda()
        print('Models moved to GPU.')
    return G, D


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
    """Generate samples with the fixed noise batch and save as a PNG grid."""
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
    """Sample U(-1, 1) noise with shape (B, dim, 1, 1).
    The same distribution is used for WGAN-GP for a fair comparison."""
    return utils.to_var(torch.rand(batch_size, dim) * 2 - 1).unsqueeze(2).unsqueeze(3)


def training_loop_lsgan(train_dataloader, G, D, opts, logger):
    """Shared training loop for the SN / LSGAN / Custom variants (MSE loss)."""
    g_optim = optim.Adam(G.parameters(), opts.lr, [opts.beta1, opts.beta2])
    d_optim = optim.Adam(D.parameters(), opts.lr, [opts.beta1, opts.beta2])

    fixed_noise = sample_noise(opts.batch_size, opts.noise_size)

    iteration = 1
    total_iters = opts.num_epochs * len(train_dataloader)
    for _ in range(opts.num_epochs):
        for batch in train_dataloader:
            real_images = utils.to_var(batch)
            bs = real_images.size(0)

            # ---------- update D ----------
            d_real = D(real_images)
            noise = sample_noise(bs, opts.noise_size)
            fake_images = G(noise)
            # detach so gradients do not flow back to G.
            d_fake = D(fake_images.detach())
            d_loss = lsgan_d_loss(d_real, d_fake)

            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            # ---------- update G ----------
            noise = sample_noise(bs, opts.noise_size)
            fake_images = G(noise)
            g_loss = lsgan_g_loss(D(fake_images))

            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

            # ---------- log / sample / checkpoint ----------
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


def training_loop_wgan_gp(train_dataloader, G, D, opts, logger):
    """Training loop for WGAN-GP.
    The critic is updated n_critic times per generator step.
    WGAN-GP paper recommends Adam with beta1=0.0, beta2=0.9."""
    g_optim = optim.Adam(G.parameters(), lr=opts.lr, betas=(0.0, 0.9))
    d_optim = optim.Adam(D.parameters(), lr=opts.lr, betas=(0.0, 0.9))

    fixed_noise = sample_noise(opts.batch_size, opts.noise_size)

    iteration = 1
    total_iters = opts.num_epochs * len(train_dataloader)
    data_iter = iter(train_dataloader)

    def _next_batch():
        """Restart the data iterator when exhausted so that all n_critic
        updates get a fresh batch."""
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            return next(data_iter)

    for _ in range(opts.num_epochs):
        for _ in range(len(train_dataloader)):
            # ---------- critic updates (n_critic times) ----------
            for _ in range(opts.n_critic):
                real_images = utils.to_var(_next_batch())
                bs = real_images.size(0)

                noise = sample_noise(bs, opts.noise_size)
                fake_images = G(noise).detach()

                d_real = D(real_images)
                d_fake = D(fake_images)

                # Wasserstein dual objective + gradient penalty.
                gp = gradient_penalty(D, real_images, fake_images,
                                      lambda_gp=opts.lambda_gp)
                d_loss = d_fake.mean() - d_real.mean() + gp

                d_optim.zero_grad()
                d_loss.backward()
                d_optim.step()

            # ---------- generator update (once) ----------
            noise = sample_noise(opts.batch_size, opts.noise_size)
            fake_images = G(noise)
            # G wants to maximize D(G(z)), so negate for minimization.
            g_loss = -D(fake_images).mean()

            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

            # ---------- log / sample / checkpoint ----------
            if iteration % opts.log_step == 0:
                # Wasserstein estimate: d_real - d_fake.
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


def main(opts):
    dataloader = get_data_loader(opts.data, opts)

    utils.create_dir(opts.checkpoint_dir)
    utils.create_dir(opts.sample_dir)

    logger = SummaryWriter(opts.sample_dir)

    G, D = build_models(opts.variant, opts.noise_size, opts.conv_dim)
    print('                    G                  '); print(G)
    print('                    D                  '); print(D)

    if opts.variant == 'wgan_gp':
        training_loop_wgan_gp(dataloader, G, D, opts, logger)
    else:
        training_loop_lsgan(dataloader, G, D, opts, logger)


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--variant', type=str, required=True,
                        choices=['sn', 'wgan_gp', 'lsgan', 'custom'],
                        help='Which Part 2 variant to train.')

    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--conv_dim', type=int, default=32)
    parser.add_argument('--noise_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)   # LSGAN branch only
    parser.add_argument('--beta2', type=float, default=0.999)

    parser.add_argument('--n_critic', type=int, default=5,
                        help='Critic updates per G step (WGAN-GP only).')
    parser.add_argument('--lambda_gp', type=float, default=10.0,
                        help='Gradient penalty coefficient (WGAN-GP only).')

    parser.add_argument('--data', type=str, default='cat/grumpifyBprocessed')
    parser.add_argument('--data_preprocess', type=str, default='advanced')
    parser.add_argument('--ext', type=str, default='*.png')

    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='If None, defaults to checkpoints/<variant>.')
    parser.add_argument('--sample_dir', type=str, default=None,
                        help='If None, defaults to output/<variant>/<data>_<preproc>.')
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=200)
    parser.add_argument('--checkpoint_every', type=int, default=400)

    return parser


if __name__ == '__main__':
    opts = create_parser().parse_args()

    if opts.sample_dir is None:
        opts.sample_dir = os.path.join('output', opts.variant,
                                       f'{os.path.basename(opts.data)}_{opts.data_preprocess}')
    if opts.checkpoint_dir is None:
        opts.checkpoint_dir = os.path.join('checkpoints', opts.variant)

    if os.path.exists(opts.sample_dir):
        os.system(f'rm {opts.sample_dir}/*')

    print(opts)
    main(opts)
