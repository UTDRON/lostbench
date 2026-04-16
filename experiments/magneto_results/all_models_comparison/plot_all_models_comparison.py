import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

case_labels = {
    "ANES_450_verbose_e-s_qns_old.csv": "Original",
    "ANES_450_default_e_lbl.csv": "Lbl- Both",
    "ANES_450_verbose_s_qn-lbl.csv": "S-Q, L-lbl",
    "ANES_450_default_e-s_qns_lbl_new_prompt.csv": "Revised",
    "DDM_450_sm_qn_wo_w_llm.csv": "DDM"
}

# case_labels = {
#     "ANES_450_verbose_e-s_qns_old.csv": "Magneto: Original Prompt",
#     "ANES_450_default_e_lbl.csv": "Magneto: Labels-> Both",
#     "ANES_450_verbose_s_qn-lbl.csv": "Magneto: SLM-> Question, LLM-> Labels",
#     "ANES_450_default_e-s_qns_lbl_new_prompt.csv": "Magneto: Revised Prompt",
#     "DDM_450_sm_qn_wo_w_llm.csv": "DDM"
# }

bottom_legend_labels = {
    "Original": "Magneto: Original Prompt",
    "Lbl- Both": "Magneto: Labels-> Both",
    "SLM-Qn, LLM-Lbl": "Magneto: SLM-> Question, LLM-> Labels",
    "Revised": "Magneto: Revised Prompt",
}

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
                fontsize=9,
                # fontweight="bold",
                color="black",
                rotation = 20
            )

RESULTS_PATH = "."
files = list(case_labels.keys())

measures = [
    # "All_RecallAtSizeofGroundTruth",
    "One2One_RecallAtSizeofGroundTruth",
    "mrr"
]

measures_labels = {
    # "All_RecallAtSizeofGroundTruth": "Recall@SizeOfGroundTruth",
    # "One2One_RecallAtSizeofGroundTruth": "One2OneRecall@SizeOfGroundTruth",
    "One2One_RecallAtSizeofGroundTruth": "R@GT",
    "mrr": "MRR"
}

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
plt.subplots_adjust(wspace=0.06)
bar_width = 0.45

for idx, measure in enumerate(measures):
    magneto_avgs, magnetogpt_avgs = [], []

    for file in files:
        df = pd.read_csv(os.path.join(RESULTS_PATH, file))
        magneto_df = df[df["method"] == "Magneto"]
        magnetogpt_df = df[df["method"] == "MagnetoGPT"]
        magneto_avgs.append(magneto_df[measure].mean())
        magnetogpt_avgs.append(magnetogpt_df[measure].mean())

    x = np.arange(len(files))
    ax = axes[idx]

    bars1 = ax.bar(
        x - bar_width/2,
        magneto_avgs,
        bar_width,
        label="No LLM",
        color="#7fb3d5",
        hatch="//"
    )
    bars2 = ax.bar(
        x + bar_width/2,
        magnetogpt_avgs,
        bar_width,
        label="With LLM",
        color="#f78fb3",
        hatch=".."
    )

    ax.set_title(f"{measures_labels[measure]}", fontsize=15, fontweight="normal")
    ax.set_xticks(x)
    ax.set_xticklabels([case_labels.get(f, f) for f in files], rotation=30, ha="center", fontsize=15)
    
    # Make y-axis tick labels larger
    ax.tick_params(axis='y', labelsize=15)
    ax.set_ylim(0.0, 1.1)
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

    # annotate_bars(ax, bars1)
    # annotate_bars(ax, bars2)

# Top legend for bars
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=15)

# Bottom legend as key-value boxes
# num_files = len(files)
# for i, f in enumerate(files):
#     case_key = case_labels[f]
#     case_value = bottom_legend_labels.get(case_key, case_key)
#     label_text = f"{case_key} → {case_value}"  # key-value style
#     x_pos = 0.1 + i * (0.8 / (num_files - 2))
#     fig.text(
#         x_pos, 0.02, label_text,
#         ha="center", va="center",
#         fontsize=10,
#         bbox=dict(facecolor="#f0f0f0", edgecolor="black", boxstyle="square")
#     )

# plt.tight_layout(rect=[0, 0.05, 1, 0.92])
plt.subplots_adjust(left=0.05, right=0.98, top=0.8, bottom=0.2, wspace=0.12)
plt.show()