import glob
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
EXP_ROOT = os.path.join(SCRIPT_DIR, "..", "..", "exp")

LABEL_MAP = {
    "sac clipq sd1":  "SAC (clipped Q)",
    "sac singleq sd1": "SAC (single Q)",
}


def load_runs():
    """Load all Hopper-v4 SAC runs, returning unfiltered dataframes."""
    runs = {}

    for exp_dir in sorted(glob.glob(os.path.join(EXP_ROOT, "Hopper-v4_sac_*"))):
        log_path = os.path.join(exp_dir, "log.csv")
        if not os.path.isfile(log_path):
            continue

        dir_name = os.path.basename(exp_dir)
        parts = dir_name.split("_")
        # parts: ['Hopper-v4', 'sac', 'clipq', 'sd1', '<date>', '<time>']
        label = " ".join(parts[1:-2])  # e.g. "sac clipq sd1"

        label = LABEL_MAP.get(label, label)
        runs[label] = pd.read_csv(log_path)

    return runs


def make_plot(runs, y_col, y_label, out_path):
    fig, ax = plt.subplots()
    for name, df in sorted(runs.items()):
        col_df = df.dropna(subset=[y_col])
        ax.plot(col_df["step"], col_df[y_col], label=name)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(y_label)
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
    runs = load_runs()
    if not runs:
        print("No Hopper-v4 SAC runs found.")
        return

    make_plot(runs, "Eval_AverageReturn", "Eval Average Return",
              os.path.join(SCRIPT_DIR, "eval_return.pdf"))
    make_plot(runs, "q_values", "Q Values",
              os.path.join(SCRIPT_DIR, "q_values.pdf"))


if __name__ == "__main__":
    main()
