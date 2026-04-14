# Plot training loss curves for the four Part 2 variants.
#   log_lsgan.txt / log_sn.txt / log_custom.txt format:
#       Iter [ X/Y] | D_loss: A | G_loss: B
#   log_wgan_gp.txt format:
#       Iter [ X/Y] | D_loss: A | G_loss: B | W_dist: C | GP: D
#
# Outputs:
#   part2_results/loss_lsgan.png
#   part2_results/loss_sn.png
#   part2_results/loss_custom.png
#   part2_results/loss_wgan_gp.png   (with W_dist and GP subplots)
#   part2_results/loss_compare.png   (all four variants side-by-side)

import os
import re
import matplotlib.pyplot as plt

RESULT_DIR = "part2_results"

LSGAN_RE = re.compile(
    r"Iter\s*\[\s*(\d+)/\s*\d+\]\s*\|\s*"
    r"D_loss:\s*([-\d.]+)\s*\|\s*"
    r"G_loss:\s*([-\d.]+)"
)

WGAN_RE = re.compile(
    r"Iter\s*\[\s*(\d+)/\s*\d+\]\s*\|\s*"
    r"D_loss:\s*([-\d.]+)\s*\|\s*"
    r"G_loss:\s*([-\d.]+)\s*\|\s*"
    r"W_dist:\s*([-\d.]+)\s*\|\s*"
    r"GP:\s*([-\d.]+)"
)


def parse_lsgan(path):
    """Parse LSGAN-style logs, return (iter, D_loss, G_loss)."""
    it, d, g = [], [], []
    with open(path, "r") as f:
        for line in f:
            m = LSGAN_RE.search(line)
            if m:
                it.append(int(m.group(1)))
                d.append(float(m.group(2)))
                g.append(float(m.group(3)))
    return it, d, g


def parse_wgan(path):
    """Parse WGAN-GP logs, return (iter, D, G, W_dist, GP)."""
    it, d, g, w, gp = [], [], [], [], []
    with open(path, "r") as f:
        for line in f:
            m = WGAN_RE.search(line)
            if m:
                it.append(int(m.group(1)))
                d.append(float(m.group(2)))
                g.append(float(m.group(3)))
                w.append(float(m.group(4)))
                gp.append(float(m.group(5)))
    return it, d, g, w, gp


def plot_lsgan_run(name, title, save_path):
    """Plot D_loss and G_loss for one LSGAN-style run."""
    it, d, g = parse_lsgan(os.path.join(RESULT_DIR, f"log_{name}.txt"))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(it, d, label="D_loss", alpha=0.8)
    ax.plot(it, g, label="G_loss", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


def plot_wgan_run(save_path):
    """Plot WGAN-GP with three subplots: D/G loss, Wasserstein distance, GP."""
    it, d, g, w, gp = parse_wgan(os.path.join(RESULT_DIR, "log_wgan_gp.txt"))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(it, d, label="D_loss", alpha=0.8)
    axes[0].plot(it, g, label="G_loss", alpha=0.8)
    axes[0].set_title("WGAN-GP: D_loss / G_loss")
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Wasserstein distance estimate = E[D(real)] - E[D(fake)];
    # should decrease and plateau at a small value when training converges.
    axes[1].plot(it, w, color="tab:green", alpha=0.8)
    axes[1].set_title("Wasserstein distance estimate")
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("W_dist")
    axes[1].grid(alpha=0.3)

    # Gradient penalty (ideally ~ 0, i.e. ||grad|| ~ 1).
    axes[2].plot(it, gp, color="tab:red", alpha=0.8)
    axes[2].set_title("Gradient penalty")
    axes[2].set_xlabel("Iteration"); axes[2].set_ylabel("GP")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


def plot_compare(save_path):
    """Compare all four variants in a 2x2 grid (separate axes per variant
    since WGAN-GP loss has a different sign/scale from the MSE variants)."""
    variants = ["lsgan", "sn", "custom", "wgan_gp"]
    titles = ["LSGAN", "Spectral Norm", "Custom", "WGAN-GP"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, v, t in zip(axes, variants, titles):
        path = os.path.join(RESULT_DIR, f"log_{v}.txt")
        if v == "wgan_gp":
            it, d, g, _, _ = parse_wgan(path)
        else:
            it, d, g = parse_lsgan(path)
        ax.plot(it, d, label="D_loss", alpha=0.7)
        ax.plot(it, g, label="G_loss", alpha=0.7)
        ax.set_title(t)
        ax.set_xlabel("Iteration"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle("Part 2: Training losses across four variants", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    plot_lsgan_run("lsgan",  "LSGAN Training Loss",          f"{RESULT_DIR}/loss_lsgan.png")
    plot_lsgan_run("sn",     "Spectral Norm Training Loss",  f"{RESULT_DIR}/loss_sn.png")
    plot_lsgan_run("custom", "Custom GAN Training Loss",     f"{RESULT_DIR}/loss_custom.png")
    plot_wgan_run(f"{RESULT_DIR}/loss_wgan_gp.png")
    plot_compare(f"{RESULT_DIR}/loss_compare.png")
