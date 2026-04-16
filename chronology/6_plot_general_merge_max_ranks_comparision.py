import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AAO_CSV = "all_at_once_num_subgraphs_only.csv"
PAIRWISE_NO_HITL_CSV = "pairwise_no_hitl.csv"
PAIRWISE_HITL_CSV = "pairwise_hitl.csv"
PAIRWISE_HITL_RANDOM = "pairwise_hitl_random.csv"

colors = {
    "AAO": "#2ca02c",     
    "HITL Random": "#d62728",     
    "HITL Temporal": "#ffbf00",
    "No HITL": "#1f77b4"         
}

def pad_to_n(lst, n):
    return lst + [0] * (n - len(lst))
           

df_aao = pd.read_csv(AAO_CSV)
df_pair_no_hitl = pd.read_csv(PAIRWISE_NO_HITL_CSV)
df_pair_hitl = pd.read_csv(PAIRWISE_HITL_CSV)
df_pair_hitl_random = pd.read_csv(PAIRWISE_HITL_RANDOM)

indices_to_plot = [0,64]

# fig, axes = plt.subplots(1, len(indices_to_plot), figsize=(8 * len(indices_to_plot), 6))
fig, axes = plt.subplots(
    1,
    len(indices_to_plot),
    figsize=(4 * len(indices_to_plot), 4),
    sharey=True
)

if len(indices_to_plot) == 1:
    axes = [axes]

for idx, ax in zip(indices_to_plot, axes):

    data_aao = df_aao["max_ranks_4_ideal_recall"][idx]
    data_pair_no_hitl = df_pair_no_hitl["max_ranks_4_ideal_recall"][idx]
    data_pair_hitl = df_pair_hitl["max_ranks_4_ideal_recall"][idx]
    data_pair_hitl_random = df_pair_hitl_random["max_ranks_4_ideal_recall"][idx]

    data_aao_list = ast.literal_eval(data_aao)
    data_pair_no_hitl_list = ast.literal_eval(data_pair_no_hitl)
    data_pair_hitl_list = ast.literal_eval(data_pair_hitl)
    data_pair_hitl_random_list = ast.literal_eval(data_pair_hitl_random)

    flat_data_no_hitl = [item for sublist in data_pair_no_hitl_list for item in sublist]
    flat_data_hitl = [item for sublist in data_pair_hitl_list for item in sublist]
    flat_data_hitl_random = [item for sublist in data_pair_hitl_random_list for item in sublist]

    auc2 = sum(data_aao_list)
    auc1 = sum(flat_data_no_hitl)
    auc3 = sum(flat_data_hitl)
    auc4 = sum(flat_data_hitl_random)

    # n = len(flat_data_no_hitl)
    n = max(
        len(flat_data_no_hitl),
        len(flat_data_hitl),
        len(flat_data_hitl_random),
        len(data_aao_list)
    )
    x = np.arange(n)

    flat_data_no_hitl = pad_to_n(flat_data_no_hitl, n)
    flat_data_hitl = pad_to_n(flat_data_hitl, n)
    flat_data_hitl_random = pad_to_n(flat_data_hitl_random, n)
    data2_padded = pad_to_n(data_aao_list, n)

    data2_padded = data_aao_list + [0] * (n - len(data_aao_list))

    w = 1

    ax.bar(x, data2_padded, width=w, align='edge',
           alpha=0.5, label=f"AAO (AUC:{auc2})", color = colors['AAO'])
    # ax.bar(x, flat_data_no_hitl, width=w, align='edge',
    #        alpha=0.8, label=f"Pairwise(No HITL) Random (AUC={auc1})", color = colors['No HITL'])
    ax.bar(x, flat_data_hitl, width=w, align='edge',
           alpha=0.8, label=f"Temporal (AUC:{auc3})", color = colors['HITL Temporal'])
    ax.bar(x, flat_data_hitl_random, width=w, align='edge',
           alpha=0.8, label=f"Random (AUC:{auc4})", color = colors['HITL Random'])

    ax.set_xlabel("Number of Steps", fontsize=15)
    ax.set_ylabel("RIR", fontsize=15)
    ax.set_title(f"Instance {idx}", fontsize=15)

    ax.legend(fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
    ax.tick_params(labelleft=True)
    ax.set_xlim(0, n)

plt.tight_layout()
plt.subplots_adjust(left=0.07)
plt.subplots_adjust(wspace=0.17)
# plt.show()


#------ FOR CDF

data_aao = df_aao["max_ranks_4_ideal_recall"]
data_pair_no_hitl = df_pair_no_hitl["max_ranks_4_ideal_recall"]
data_pair_hitl = df_pair_hitl["max_ranks_4_ideal_recall"]
data_pair_hitl_random = df_pair_hitl_random["max_ranks_4_ideal_recall"]

def parse_list(s):
    return ast.literal_eval(s)

data_aao = data_aao.apply(parse_list)
data_pair_no_hitl = data_pair_no_hitl.apply(parse_list)
data_pair_hitl = data_pair_hitl.apply(parse_list)
data_pair_hitl_random = data_pair_hitl_random.apply(parse_list)

data_pair_no_hitl = data_pair_no_hitl.apply(lambda row: [item for sublist in row for item in sublist])
data_pair_hitl = data_pair_hitl.apply(lambda row: [item for sublist in row for item in sublist])
data_pair_hitl_random = data_pair_hitl_random.apply(lambda row: [item for sublist in row for item in sublist])

sum_aao = data_aao.apply(sum)
sum_pair_no_hitl = data_pair_no_hitl.apply(sum)
sum_pair_hitl = data_pair_hitl.apply(sum)
sum_pair_hitl_random = data_pair_hitl_random.apply(sum)

mean_aao, std_aao = np.mean(sum_aao), np.std(sum_aao)
mean_hitl, std_hitl = np.mean(sum_pair_hitl), np.std(sum_pair_hitl)
mean_hitl_random, std_hitl_random = np.mean(sum_pair_hitl_random), np.std(sum_pair_hitl_random)

def compute_cdf(values):
    sorted_vals = np.sort(values)
    cdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
    return sorted_vals, cdf

x_aao, cdf_aao = compute_cdf(sum_aao)
x_no_hitl, cdf_no_hitl = compute_cdf(sum_pair_no_hitl)
x_hitl, cdf_hitl = compute_cdf(sum_pair_hitl)
x_hitl_random, cdf_hitl_random = compute_cdf(sum_pair_hitl_random)

plt.figure(figsize=(6,6))
plt.plot(x_aao, cdf_aao, label=f"AAO (μ={mean_aao:.2f}, σ={std_aao:.2f})", linewidth=5)
# plt.plot(x_no_hitl, cdf_no_hitl, label="Pairwise (No HITL)", linewidth=5)
plt.plot(x_hitl, cdf_hitl, label=f"Temporal (μ={mean_hitl:.2f}, σ={std_hitl:.2f})", linewidth=5)
plt.plot(x_hitl_random, cdf_hitl_random, label=f"Random (μ={mean_hitl_random:.2f}, σ={std_hitl_random:.2f})", linewidth=5)

# plt.xlabel("Area Under the Curve(AUC) for Test Case Instances", fontsize=20)
plt.ylabel("CDF", fontsize=20)
# plt.title("CDF of AUC", fontsize=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)

textstr = (
    f"All-at-once: μ={mean_aao:.2f}, σ={std_aao:.2f}\n"
    f"Temporal HITL: μ={mean_hitl:.2f}, σ={std_hitl:.2f}\n"
    f"Random HITL: μ={mean_hitl_random:.2f}, σ={std_hitl_random:.2f}"
)



plt.show()