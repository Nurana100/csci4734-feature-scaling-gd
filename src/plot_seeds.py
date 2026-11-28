import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT = "results/e1_e2_seeds.csv"
SUMMARY = "results/e1_e2_seeds_summary.csv"
MAX_ITER = 20000

SCALERS = ["none", "min-max", "standardize", "whiten", "pca-whiten"]
COLORS = {
    "none": "#7f7f7f",
    "min-max": "#d62728",
    "standardize": "#1f77b4",
    "whiten": "#2ca02c",
    "pca-whiten": "#ff7f0e",
}


def summarize(df):
    keys = ["experiment", "task", "scale_ratio", "rho", "scaler"]
    out = df.groupby(keys).agg(
        runs=("seed", "count"),
        converged_runs=("converged", "sum"),
        iterations_mean=("iterations", "mean"),
        iterations_std=("iterations", "std"),
        iterations_median=("iterations", "median"),
        iterations_q25=("iterations", lambda s: s.quantile(0.25)),
        iterations_q75=("iterations", lambda s: s.quantile(0.75)),
        condition_number_median=("condition_number", "median"),
    ).reset_index()
    return out


def draw(ax, data, xcol, categorical):
    for name in SCALERS:
        g = data[data.scaler == name].sort_values(xcol)
        if g.empty:
            continue
        x = np.arange(len(g)) if categorical else g[xcol].values
        style = "--" if name == "pca-whiten" else "-"
        ax.plot(x, g.iterations_median, style, color=COLORS[name], label=name,
                linewidth=1.8, marker="o", markersize=4)
        ax.fill_between(x, g.iterations_q25, g.iterations_q75, color=COLORS[name],
                        alpha=0.18, linewidth=0)
    ax.axhline(MAX_ITER, color="black", linestyle=":", linewidth=1)
    ax.set_yscale("log")
    ax.grid(alpha=0.25)


def figure(summary, experiment, xcol, xlabel, title, outfile, log_x):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, task in zip(axes, ["linear", "logistic"]):
        data = summary[(summary.experiment == experiment) & (summary.task == task)]
        draw(ax, data, xcol, categorical=not log_x)
        if log_x:
            ax.set_xscale("log")
        else:
            ticks = sorted(data[xcol].unique())
            ax.set_xticks(range(len(ticks)))
            ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlabel(xlabel)
        ax.set_title(f"{task} regression")
    axes[0].set_ylabel("steps to converge")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9,
               frameon=False)
    fig.suptitle(title + f" (median and 25-75% range over 10 seeds, dotted line = "
                 f"{MAX_ITER}-step limit)", fontsize=10)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print("saved", outfile)


if __name__ == "__main__":
    df = pd.read_csv(INPUT)
    summary = summarize(df)
    summary.to_csv(SUMMARY, index=False)
    print("saved", SUMMARY)
    os.makedirs("results", exist_ok=True)
    figure(summary, "E1", "scale_ratio", "feature scale ratio",
           "E1: steps vs feature scale ratio", "results/e1_seeds.png", True)
    figure(summary, "E2", "rho", "feature correlation",
           "E2: steps vs feature correlation", "results/e2_seeds.png", False)
