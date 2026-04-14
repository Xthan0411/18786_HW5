# Plot training loss curves from log files (Part 1: basic vs advanced).
import re
import matplotlib.pyplot as plt

LINE_RE = re.compile(
    r"Iteration\s*\[\s*(\d+)/\s*\d+\]\s*\|\s*"
    r"D_real_loss:\s*([\d.]+)\s*\|\s*"
    r"D_fake_loss:\s*([\d.]+)\s*\|\s*"
    r"G_loss:\s*([\d.]+)"
)


def parse_log(path):
    """Parse a log file and return (iterations, D_real, D_fake, G_loss)."""
    iters, d_real, d_fake, g = [], [], [], []
    with open(path, "r") as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                iters.append(int(m.group(1)))
                d_real.append(float(m.group(2)))
                d_fake.append(float(m.group(3)))
                g.append(float(m.group(4)))
    return iters, d_real, d_fake, g


def plot_run(path, title, save_path):
    """Plot D_real / D_fake / G_loss for a single run."""
    it, dr, df, gl = parse_log(path)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(it, dr, label="D_real", alpha=0.8)
    ax.plot(it, df, label="D_fake", alpha=0.8)
    ax.plot(it, gl, label="G_loss", alpha=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


def plot_compare(basic_path, adv_path, save_path):
    """Compare basic vs advanced D_fake and G_loss side-by-side."""
    it_b, dr_b, df_b, g_b = parse_log(basic_path)
    it_a, dr_a, df_a, g_a = parse_log(adv_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(it_b, df_b, label="basic D_fake", alpha=0.8)
    axes[0].plot(it_a, df_a, label="advanced D_fake", alpha=0.8)
    axes[0].set_yscale("log")
    axes[0].set_title("D_fake: basic vs advanced")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss (log)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(it_b, g_b, label="basic G_loss", alpha=0.8)
    axes[1].plot(it_a, g_a, label="advanced G_loss", alpha=0.8)
    axes[1].set_title("G_loss: basic vs advanced")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    basic_log = "results/log_basic.txt"
    adv_log = "results/log_advanced.txt"

    plot_run(basic_log, "Vanilla GAN Training Loss (basic augmentation)",
             "results/loss_basic.png")
    plot_run(adv_log, "Vanilla GAN Training Loss (advanced augmentation)",
             "results/loss_advanced.png")
    plot_compare(basic_log, adv_log, "results/loss_compare.png")
