import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

matplotlib.use("pdf")

ICML_RC = {
    "figure.figsize": (3.25, 2.4),
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "text.usetex": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
plt.rcParams.update(ICML_RC)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(SCRIPT_DIR, "..", "..", "exp", "MsPacman_dqn_sd1_20260306_231117")


def make_plot(out_path):
    df = pd.read_csv(os.path.join(EXP_DIR, "log.csv"))

    eval_df = df.dropna(subset=["Eval_AverageReturn"])
    train_df = df.dropna(subset=["Train_EpisodeReturn"])

    fig, ax = plt.subplots()
    ax.plot(train_df["step"], train_df["Train_EpisodeReturn"], label="train return", alpha=0.7)
    ax.plot(eval_df["step"], eval_df["Eval_AverageReturn"], label="eval return")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Average Return")
    ax.legend(frameon=False)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k" if x < 1e6 else f"{x/1e6:.1f}M")
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    make_plot(os.path.join(SCRIPT_DIR, "eval_return.pdf"))


if __name__ == "__main__":
    main()
