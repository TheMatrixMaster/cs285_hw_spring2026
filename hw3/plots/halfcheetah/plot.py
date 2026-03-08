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

RUNS = {
    "fixed ($\\alpha$=0.1)": "HalfCheetah-v4_sac_sd1_20260307_131007",
    "autotuned $\\alpha$":   "HalfCheetah-v4_sac_autotune_sd1_20260307_131102",
}


def make_plot(runs, y_col, y_label, out_path):
    fig, ax = plt.subplots()
    for name, df in runs.items():
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


def load_runs():
    return {
        label: pd.read_csv(os.path.join(EXP_ROOT, exp_dir, "log.csv"))
        for label, exp_dir in RUNS.items()
    }


def main():
    runs = load_runs()
    make_plot(runs, "Eval_AverageReturn", "Eval Average Return",
              os.path.join(SCRIPT_DIR, "eval_return.pdf"))

    # Eval return for the fixed temperature run only
    fixed_label = "fixed ($\\alpha$=0.1)"
    make_plot({fixed_label: runs[fixed_label]}, "Eval_AverageReturn", "Eval Average Return",
              os.path.join(SCRIPT_DIR, "eval_return_fixed.pdf"))

    # Temperature over training for the autotuned run only
    autotune_label = "autotuned $\\alpha$"
    make_plot({autotune_label: runs[autotune_label]}, "temperature", r"Temperature ($\alpha$)",
              os.path.join(SCRIPT_DIR, "temperature.pdf"))


if __name__ == "__main__":
    main()
