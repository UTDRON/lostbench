import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15
})

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

metadata_based = {
    "results_SimilarityFlooding_formulaA_invprod.csv": "SimFlooding",
    "results_Coma.csv": "COMA",
    "ddm_sm_lostbench_qn.csv": "DDM",
    "ddm_llm_sm_lostbench_qn.csv": "DDM_LLM"
}

instance_based = {
    "results_DistributionBased.csv": "D-Based",
    "results_JaccardDistanceMatcher_t0.7.csv": "JLDistMatcher"
}

hybrid = {
    "results_Coma++.csv": "COMA++",
    "magneto_sm_lostbench_qn_metrics.csv": "Magneto",
    "magnetogpt_sm_lostbench_qn_metrics.csv": "MagnetoGPT",
}

DIRECTORY = "magneto/"

def load_metric(file, col):
    df = pd.read_csv(os.path.join(DIRECTORY, file))
    if col not in df.columns:
        raise ValueError
    return df[col].dropna().values


def plot_block(col_name, col_idx, ylabel):
    ax = axes[col_idx]

    data_all, labels_all = [], []
    colors, hatches = [], []

    # instnace based
    for file, label in instance_based.items():
        try:
            data = load_metric(file, col_name)
            data_all.append(data)
            labels_all.append(label)
            colors.append("#7fb3d5")
            hatches.append("**")
        except:
            pass

    split_1 = len(data_all)

    # metadata bsed
    for file, label in metadata_based.items():
        try:
            data = load_metric(file, col_name)
            data_all.append(data)
            labels_all.append(label)
            colors.append("#d4c44c")
            hatches.append("//")
        except:
            pass

    split_2 = len(data_all)

    # hybrid
    for file, label in hybrid.items():
        try:
            data = load_metric(file, col_name)
            data_all.append(data)
            labels_all.append(label)
            colors.append("#82e0aa")
            hatches.append("..")
        except:
            pass

    bp = ax.boxplot(
    data_all,
    patch_artist=True,
    showmeans=True,
    meanline=True,
    widths=0.7,
    meanprops=dict(color="black", linewidth=2.0)
    )

    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i])
        box.set_hatch(hatches[i])

    ax.set_xticks(range(1, len(labels_all)+1))
    ax.set_xticklabels(labels_all, rotation=20)
    ax.set_ylabel(ylabel)

    ax.axvline(split_1 + 0.5, linestyle="--", linewidth=2)
    ax.axvline(split_2 + 0.5, linestyle="--", linewidth=2)

    # mean annotations at top of boxes
    means = [np.mean(d) for d in data_all]

    for i, (mean_val, data) in enumerate(zip(means, data_all)):
        top = np.max(data)  # top of the boxplot whisker region
        ax.text(
            i + 1,
            top + 0.01 * (top if top != 0 else 1),  # small offset above
            f"{mean_val:.2f}",
            color="red",
            fontsize=15,
            fontweight='bold',
            ha='center',
            va='bottom'
        )
        ax.set_ylim(-0.05, 1.1)


# legend
legend_elements = [
    Patch(facecolor="#7fb3d5", hatch="**", label="Instance-based"),
    Patch(facecolor="#d4c44c", hatch="//", label="Metadata-based"),
    Patch(facecolor="#82e0aa", hatch="..", label="Hybrid"),
]

fig.legend(
    handles=legend_elements,
    loc='upper center',
    ncol=3,
    fontsize=15,
    frameon=False
)

# plot_block("All_RecallAtSizeofGroundTruth", 0, "Recall@SizeOfGroundTruth")
plot_block("One2One_RecallAtSizeofGroundTruth", 0, "R@GT")
plot_block("mrr", 1, "MRR")

# spacing
plt.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.15, wspace=0.12)
plt.show()