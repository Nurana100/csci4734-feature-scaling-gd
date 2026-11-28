import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

E1_CSV = "results/e1_scale_iterations.csv"
E2_CSV = "results/e2_correlation_iterations.csv"
MAX_ITER = 20000

SCALERS = ["none", "min-max", "standardize", "whiten", "pca-whiten"]
COLORS = {
    "none": "#7f7f7f",
    "min-max": "#d62728",
    "standardize": "#1f77b4",
    "whiten": "#2ca02c",
    "pca-whiten": "#ff7f0e",
}


def draw_lines(ax, data, xcol, positions=None):
    for name in SCALERS:
        g = data[data.scaler == name].sort_values(xcol)
        if g.empty:
            continue
        x = np.arange(len(g)) if positions else g[xcol].values
        style = "--" if name == "pca-whiten" else "-"
        ax.plot(x, g.iterations, style, color=COLORS[name], label=name,
                linewidth=1.8, zorder=2)
        done = g.converged.values
        ax.scatter(np.asarray(x)[done], g.iterations.values[done], color=COLORS[name],
                   s=36, zorder=3)
        ax.scatter(np.asarray(x)[~done], g.iterations.values[~done], color=COLORS[name],
                   marker="x", s=60, linewidths=2, zorder=3)
    ax.axhline(MAX_ITER, color="black", linestyle=":", linewidth=1)
    ax.set_yscale("log")
    ax.set_ylabel("steps to converge")
    ax.grid(alpha=0.25)


def figure_two_panels(df, xcol, xlabel, title, outfile, log_x=False):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, task in zip(axes, ["linear", "logistic"]):
        data = df[df.task == task]
        if log_x:
            draw_lines(ax, data, xcol)
            ax.set_xscale("log")
        else:
            draw_lines(ax, data, xcol, positions=True)
            ticks = sorted(data[xcol].unique())
            ax.set_xticks(range(len(ticks)))
            ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlabel(xlabel)
        ax.set_title(f"{task} regression")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9, frameon=False)
    axes[1].set_ylabel("")
    fig.suptitle(title + f" (x = did not converge, dotted line = {MAX_ITER}-step limit)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print("saved", outfile)


def figure_kappa(e1, e2, outfile):
    both = pd.concat([e1, e2])
    d = both[(both.task == "linear") & both.converged & (both.scaler != "none")]
    d = d.drop_duplicates(subset=["scale_ratio", "rho", "scaler"])
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for name in SCALERS[1:]:
        g = d[d.scaler == name]
        ax.scatter(g.condition_number, g.iterations, color=COLORS[name], label=name, s=40)
    k = np.logspace(0, 4, 50)
    ax.plot(k, 9 * k, "k--", linewidth=1, label="9 x condition number")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("condition number of the Hessian")
    ax.set_ylabel("steps to converge")
    ax.set_title("Linear regression: steps grow with the condition number", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print("saved", outfile)


if __name__ == "__main__":
    e1 = pd.read_csv(E1_CSV)
    e2 = pd.read_csv(E2_CSV)
    os.makedirs("results", exist_ok=True)
    figure_two_panels(e1, "scale_ratio", "feature scale ratio",
                      "E1: steps vs feature scale ratio", "results/e1_iterations.png",
                      log_x=True)
    figure_two_panels(e2, "rho", "feature correlation",
                      "E2: steps vs feature correlation", "results/e2_iterations.png")
    figure_kappa(e1, e2, "results/iterations_vs_condition_number.png")
