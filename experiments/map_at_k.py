'''
metaTUS_lbl = {
    "metaTus-LoStBench_lbl-topic-embedding-f1.json": "embedding-f1",
    "metaTus-LoStBench_lbl-topic-embedding-youden.json": "embedding-youden",
    "metaTus-LoStBench_lbl-topic-filtering-f1.json": "filtering-f1",
    "metaTus-LoStBench_lbl-topic-filtering-youden.json": "filtering-youden",
}

metaTUS_qn = {
    "metaTus-LoStBench_qn-topic-embedding-f1.json": "embedding-f1",
    "metaTus-LoStBench_qn-topic-embedding-youden.json": "embedding-youden",    # The best
    "metaTus-LoStBench_qn-topic-filtering-f1.json": "filtering-f1",
    "metaTus-LoStBench_qn-topic-filtering-youden.json": "filtering-youden",
}


ddm = {
    "DDM/tus_results/tus_result_qn.json" : "DDM_Qn",    # The best
    "DDM/tus_results/tus_result_lbl.json" : "DDM_Qn",     
}


starmie = {
    "starmie/starmie-0.2-naive.json" : "Starmie_Naive-0.2",   # Same result as below
    "starmie/starmie-0.6-naive.json" : "Starmie_Naive-0.6",   # Same result as above - we choose this
    "starmie/starmie-0.7-hnsw.json" : "Starmie_HNSW-0.7",
    "starmie/starmie-0.7-lsh.json" : "Starmie_LSH-0.7",
}

starmie = {
    "starmie/starmie-0.6-naive.json" : "Starmie_Naive", 
    "starmie/starmie-0.7-hnsw.json" : "Starmie_HNSW",
    "starmie/starmie-0.7-lsh.json" : "Starmie_LSH",
}


d3l_qn = {
    
    "d3l/prose_qn/ranked_list_d3l_output_name_index.json": "D3L_Name",   #1- chosen
    "d3l/prose_qn/ranked_list_d3l_output_value_index.json": "D3L_Value", #4 
    "d3l/prose_qn/ranked_list_d3l_output_format_index.json": "D3L_Format",  #3 - chosen
    "d3l/prose_qn/ranked_list_d3l_output_embedding_index.json": "D3L_Embedding", #5
    "d3l/prose_qn/ranked_list_d3l_output_distribution_index.json": "D3L_Distribution", #6
    "d3l/prose_qn/ranked_list_d3l_output_all_index.json": "D3L_All",   #2
    
}

d3l_lbl = {
    "d3l/var_labels/ranked_list_d3l_output_name_index.json": "D3L_Name", 
    "d3l/var_labels/ranked_list_d3l_output_value_index.json": "D3L_Value",
    "d3l/var_labels/ranked_list_d3l_output_format_index.json": "D3L_Format",
    "d3l/var_labels/ranked_list_d3l_output_embedding_index.json": "D3L_Embedding",
    "d3l/var_labels/ranked_list_d3l_output_distribution_index.json": "D3L_Distribution",
    "d3l/var_labels/ranked_list_d3l_output_all_index.json": "D3L_All",
}


'''

#--------------------------------------------------------------------------------------------------------------------

import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

GROUNDTRUTH_FILE = Path("../groundtruth/anesUnionBenchmark_self_added.json")

metadata_based = {
    "d3l/prose_qn/ranked_list_d3l_output_name_index.json": "D3L_Name",
    "DDM/tus_results/tus_result_qn_naive.json": "DDM",
    # "DDM/tus_results/tus_result_qn_hnsw.json": "DDM_HNSW",
    "metaTUS/metaTus-LoStBench_qn-topic-embedding-youden.json": "MetaTUS",
}


values_based = {
    "santos/anes_benchmark_result_by_santos_full.pickle": "SANTOS",
    "starmie/starmie-0.6-naive.json": "Starmie",
    "d3l/prose_qn/ranked_list_d3l_output_format_index.json": "D3L_Inst", #D3L_Instance_Format
}

MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '>']
LINESTYLES = ['-', '--', '-.', ':']

with open(GROUNDTRUTH_FILE, "r") as f:
    gt = json.load(f)

def load_preds(path):
    if path.suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)
    elif path.suffix == ".pickle":
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def precision_at_k(preds, gt_items, k):
    preds_k = preds[:k]
    rels = [1 if p in gt_items else 0 for p in preds_k]
    return sum(rels) / k

def mean_precision_at_k(preds_dict, gt_dict, k):
    values = []
    for query, gt_items in gt_dict.items():
        preds = preds_dict.get(query, [])
        values.append(precision_at_k(preds, gt_items, k))
    return sum(values) / len(values) if values else 0.0

def average_precision_at_k(preds, gt_items, k):
    ap = 0.0
    num_relevant = 0
    for i, p in enumerate(preds[:k], start=1):
        if p in gt_items:
            num_relevant += 1
            ap += num_relevant / i
    return ap / min(len(gt_items), k) if gt_items else 0.0

def mean_average_precision_at_k(preds_dict, gt_dict, k):
    ap_values = []
    for query, gt_items in gt_dict.items():
        preds = preds_dict.get(query, [])
        ap_values.append(average_precision_at_k(preds, gt_items, k))
    return sum(ap_values) / len(ap_values) if ap_values else 0.0

Ks = list(range(1, 26))

fig, axes = plt.subplots(1, 2, figsize=(4, 4), sharey=True)

# ---- Metadata-based ----
for i, (file, label) in enumerate(metadata_based.items()):
    path = Path(file)
    if not path.exists():
        continue

    preds = load_preds(path)
    if preds is None:
        continue

    precision_vals = [mean_precision_at_k(preds, gt, k) for k in Ks]
    map25 = mean_average_precision_at_k(preds, gt, 25)

    axes[0].plot(
        Ks,
        precision_vals,
        label=f"{label} (MAP@25:{map25:.2f})",
        linewidth=1.5,
        linestyle=LINESTYLES[i % len(LINESTYLES)],
        marker=MARKERS[i % len(MARKERS)],
        markersize=4
    )

axes[0].set_title("Metadata Based Methods", fontsize=15)
axes[0].set_xlabel("K", fontsize=15)
axes[0].set_ylabel("Precision@K", fontsize=15)
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend(fontsize=15)
axes[0].tick_params(axis='both', labelsize=15)

# ---- Instance-based ----
for i, (file, label) in enumerate(values_based.items()):
    path = Path(file)
    if not path.exists():
        continue

    preds = load_preds(path)
    if preds is None:
        continue

    precision_vals = [mean_precision_at_k(preds, gt, k) for k in Ks]
    map25 = mean_average_precision_at_k(preds, gt, 25)

    axes[1].plot(
        Ks,
        precision_vals,
        label=f"{label} (MAP@25:{map25:.2f})",
        linewidth=1.5,
        linestyle=LINESTYLES[i % len(LINESTYLES)],
        marker=MARKERS[i % len(MARKERS)],
        markersize=4
    )

axes[1].set_title("Instance Based Methods", fontsize=15)
axes[1].set_xlabel("K", fontsize=15)
# axes[1].set_ylabel("Precision@K", fontsize=17)
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend(fontsize=15)
axes[1].tick_params(axis='both', labelsize=15)
axes[1].tick_params(axis='y', labelleft=True)

plt.tight_layout()
plt.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.15, wspace=0.12)
plt.show()