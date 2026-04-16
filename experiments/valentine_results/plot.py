import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def annotate_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=10
            )


if __name__ == "__main__":
    RESULTS_PATH = "."
    files = sorted([f for f in os.listdir(RESULTS_PATH) if f.endswith(".csv") and f not in('results_Cupid.csv')])

    recall_avgs = []
    runtime_avgs = []
    runtime_distributions = []

    # Collect metrics
    for file in files:
        path = os.path.join(RESULTS_PATH, file)
        df = pd.read_csv(path)

        recall_avgs.append(df["All_RecallAtSizeofGroundTruth"].mean())
        runtime_avgs.append(df["One2One_RecallAtSizeofGroundTruth"].mean())

        runtimes = df["runtime"].dropna().values
        runtimes = runtimes[runtimes > 0]  # remove zeros for log
        runtime_distributions.append((file, runtimes))

    x = np.arange(len(files))

    
    # 2x2 Subplots

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))

    # avg Recall@GT Bar Chart
    bars1 = axes[0, 0].bar(x, recall_avgs)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(files, rotation=45, ha="right", fontsize=15)
    axes[0, 0].set_ylabel("Avg Recall@GT", fontsize=15)
    axes[0, 0].set_title("Average Recall@GT per File", fontsize=15)
    annotate_bars(axes[0, 0], bars1)

    # avg runtimes bar
    bars2 = axes[0, 1].bar(x, runtime_avgs)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(files, rotation=45, ha="right", fontsize=15)
    axes[0, 1].set_ylabel("Avg Runtime", fontsize=15)
    axes[0, 1].set_title("Average Runtime per File", fontsize=15)
    annotate_bars(axes[0, 1], bars2)

    # Runtime sorted curve log scale
    for file, runtimes in runtime_distributions:
        sorted_rt = np.sort(runtimes)
        axes[1, 0].plot(
            np.arange(len(sorted_rt)),
            sorted_rt,
            linewidth=2.5,
            label=file
        )

    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("Sorted Instance Index", fontsize=15)
    axes[1, 0].set_ylabel("Runtime (log scale)", fontsize=15)
    axes[1, 0].set_title("Runtime Distribution (Sorted)", fontsize=15)
    axes[1, 0].legend(fontsize=12)

    # KDE on log runtime
    for file, runtimes in runtime_distributions:

        log_rt = np.log10(runtimes)

        kde = gaussian_kde(log_rt)

        x_vals = np.linspace(log_rt.min(), log_rt.max(), 500)
        density = kde(x_vals)

        # Convert back to runtime scale for plotting
        axes[1, 1].plot(
            12**x_vals,
            density,
            linewidth=3,
            label=file
        )

    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("Runtime (log scale)", fontsize=15)
    axes[1, 1].set_ylabel("Density", fontsize=15)
    axes[1, 1].set_title("Runtime Density (Bell Curves)", fontsize=15)
    axes[1, 1].legend(fontsize=12)

    plt.tight_layout()
    plt.show()