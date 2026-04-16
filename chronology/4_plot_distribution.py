import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

df = pd.read_csv("all_at_once_num_subgraphs_only.csv")
# df = pd.read_csv("all_at_once_all.csv")

df["max_ranks_4_ideal_recall"] = df["max_ranks_4_ideal_recall"].apply(ast.literal_eval)
df["gt_nums"] = df["gt_nums"].apply(ast.literal_eval)
df["recall@gt"] = df["recall@gt"].apply(ast.literal_eval)
df["total_recall"] = df["total_recall"].apply(ast.literal_eval)

# Flatten
flat_ranks = [x for sublist in df["max_ranks_4_ideal_recall"] for x in sublist]
flat_gt = [x for sublist in df["gt_nums"] for x in sublist]
flat_recall_gt = [x for sublist in df["recall@gt"] for x in sublist]
flat_total_recall = [x for sublist in df["total_recall"] for x in sublist]

data_ranks = np.sort(flat_ranks)
cdf_ranks = np.arange(1, len(data_ranks)+1) / len(data_ranks)

data_gt = np.sort(flat_gt)
cdf_gt = np.arange(1, len(data_gt)+1) / len(data_gt)

# stats
mean_ranks = np.mean(flat_ranks)
var_ranks = np.var(flat_ranks)

mean_gt = np.mean(flat_gt)
var_gt = np.var(flat_gt)

avg_recall_gt = np.mean(flat_recall_gt)
avg_total_recall = np.mean(flat_total_recall)
avg_time = df["time"].mean()

fig, axes = plt.subplots(1, 2, figsize=(14,5))

# CDF 
sns.lineplot(x=data_ranks, y=cdf_ranks, ax=axes[0],
             color="#4C72B0", linewidth=2, label="max_ranks_4_ideal_recall")

sns.lineplot(x=data_gt, y=cdf_gt, ax=axes[0],
             color="#DD8452", linewidth=2, label="gt_nums")

axes[0].set_xlabel("Value")
axes[0].set_ylabel("CDF")
axes[0].set_title("CDF Comparison")
axes[0].legend(fontsize=11)

axes[0].text(
    0.98, 0.05,
    (
        f"Max Ranks for ideal recall: μ={mean_ranks:.2f}, σ²={var_ranks:.2f}\n"
        f"GT(num relevants): μ={mean_gt:.2f}, σ²={var_gt:.2f}\n"
        f"Recall@SizeOfGT: μ={avg_recall_gt:.3f}\n"
        f"Total Recall: μ={avg_total_recall:.3f}\n"
        f"Time spent: μ={avg_time:.3f}s"
    ),
    transform=axes[0].transAxes,
    fontsize=15,
    ha="right",
    va="bottom",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
)

# Violin plot
plot_df = pd.DataFrame({
    "value": flat_ranks + flat_gt,
    "type": ["max_ranks_4_ideal_recall"] * len(flat_ranks) +
            ["gt_nums"] * len(flat_gt)
})

sns.violinplot(
    x="type",
    y="value",
    data=plot_df,
    palette=["#4C72B0", "#DD8452"],
    inner="box",
    ax=axes[1]
)

axes[1].set_title("Distribution Comparison")
axes[1].set_xlabel("")
axes[1].set_ylabel("Value")

plt.tight_layout()
plt.show()