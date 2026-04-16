import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

plt.rcParams.update({
    "font.size": 50,
    "axes.titlesize": 50,
    "axes.labelsize": 50,
    "xtick.labelsize": 45,
    "ytick.labelsize": 50
})

fig, axes = plt.subplots(1, 2, figsize=(22, 8))

models = {
    "results_DistributionBased.csv": "D-Based",
    "results_JaccardDistanceMatcher_t0.7.csv": "JLDistMatcher",
    "results_Coma++.csv": "COMA++",
    "results_Coma.csv": "COMA",
    "results_SimilarityFlooding_formulaA_invprod.csv": "SimFlooding",
    "ddm_sm_lostbench_qn.csv": "DDM",
    "ddm_llm_sm_lostbench_qn.csv": "DDM_LLM",
    "magneto_sm_lostbench_qn_metrics.csv": "Magneto",
    "magnetogpt_sm_lostbench_qn_metrics.csv": "MagnetoGPT",
}

DIRECTORY = ""


def load_metric(file, col):
    df = pd.read_csv(os.path.join(DIRECTORY, file))
    if col not in df.columns:
        raise ValueError

    series = df[col].dropna()

    if col == "max_rank_for_ideal_recall":
        series = series.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        series = series.explode()
        series = pd.to_numeric(series, errors='coerce').dropna()

    return series.values


def plot_block(col_name, col_idx, ylabel):
    ax = axes[col_idx]

    data_all, labels_all = [], []

    for file, label in models.items():
        try:
            data = load_metric(file, col_name)
            if len(data) > 0:
                data_all.append(data)
                labels_all.append(label)
        except:
            pass

    if len(data_all) == 0:
        print(f"No data found for {col_name}")
        return

    bp = ax.boxplot(
        data_all,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        widths=0.7,
        meanprops=dict(color="black", linewidth=2.5)
    )

    # same style for all
    for box in bp['boxes']:
        box.set_facecolor("#f78fb3")   # pink
        box.set_hatch("//")

    ax.set_xticks(range(1, len(labels_all) + 1))
    ax.set_xticklabels(labels_all, rotation=60)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")

    means = [np.mean(d) for d in data_all]

    for i, (mean_val, data) in enumerate(zip(means, data_all)):
        top = np.max(data)
        offset = 0.02 * (top if top != 0 else 1)

        ax.text(
            i + 1,
            top + offset,
            f"{mean_val:.2f}",
            color="red",
            fontsize=35,
            fontweight='bold',
            ha='center',
            va='bottom'
        )


plot_block("max_rank_for_ideal_recall", 0, "MaxRankForIdealRecall")
plot_block("runtime", 1, "Runtime (s)")

plt.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.30, wspace=0.12)
plt.show()