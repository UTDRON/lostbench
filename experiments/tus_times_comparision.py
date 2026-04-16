import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

query_times = {
    "SANTOS": 176,
    "D3L_Inst": 225.83, #D3L_Instance_Format
    "Starmie_LSH": 15.14,
    "Starmie_HNSW": 0.347,

    "D3L_Name": 5.439,
    "DDM_HNSW": 3.22,
    
    "Starmie_Naive": 162.54,
    "MetaTUS": 21.39,
    "DDM_Naive": 4.401,
}

index_times = {
    "SANTOS": 2162,
    "D3L_Inst": 271.10, #D3L_Instance_Format
    "Starmie_LSH": 2.077,
    "Starmie_HNSW": 0.076,

    "D3L_Name": 7.78,
    "DDM_HNSW": 1.69,
    
    "Starmie_Naive": 0,
    "MetaTUS": 0,
    "DDM_Naive": 0,
}

map_at_k = {
    "SANTOS": 0.25,
    "D3L_Inst": 0.09, #D3L_Instance_Format
    "Starmie_LSH": 0.03,
    "Starmie_HNSW": 0.09,

    "D3L_Name": 0.58,
    "DDM_HNSW": 0.58,
    
    "Starmie_Naive": 0.20,
    "MetaTUS": 0.56,
    "DDM_Naive": 0.58,
}

methods = list(query_times.keys())

index_vals = np.array([index_times[m] for m in methods])
query_vals = np.array([query_times[m] for m in methods])

x = np.arange(len(methods))
width = 0.35

plt.figure(figsize=(10, 5))

index_color = "#82e0aa"
query_color = "#7fb3d5"

plt.bar(
    x - width/2, index_vals, width,
    label="Index Time",
    hatch='//',
    alpha=0.85,
    edgecolor='black',
    color=index_color
)

plt.bar(
    x + width/2, query_vals, width,
    label="Query Time",
    hatch='**',
    alpha=0.85,
    edgecolor='black',
    color=query_color
)

plt.xticks(x, methods, rotation=45, ha='right', fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Time (seconds)", fontsize=15)
# plt.title("Indexing vs Query Times", fontsize=15)

plt.yscale('log')

yticks = [0.01, 0.1, 1, 10, 100, 1000, 10000]
plt.yticks(yticks, [str(t) for t in yticks])

max_val = max(np.max(index_vals), np.max(query_vals))
plt.ylim(0.01, max_val * 5)  

plt.legend(fontsize=15)
plt.grid(axis='y', linestyle='--', alpha=0.5)


sep_index = len(methods) - 3
plt.axvline(x=sep_index - 0.5, linestyle=':', linewidth=3, color='black')
plt.axvline(x=sep_index - 2.5, linestyle=':', linewidth=3, color='black')

for i, method in enumerate(methods):
    max_height = max(index_vals[i], query_vals[i])
    map_val = map_at_k[method]

    if map_val > 0.5:
        color = "green"
        weight = "bold"
    elif map_val < 0.1:
        color = "red"
        weight = "bold"
    else:
        color = "black"
        weight = "normal"

    plt.text(
        x[i],
        max_height * 1.5,
        f"{map_val:.2f}",
        ha='center',
        va='bottom',
        fontsize=15,
        color=color,
        fontweight=weight
    )

plt.tight_layout()
plt.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.3, wspace=0.12)
plt.show()