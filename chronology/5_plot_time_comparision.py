import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk") 

df1 = pd.read_csv("all_at_once_all.csv")
df2 = pd.read_csv("all_at_once_num_subgraphs_only.csv")
df3 = pd.read_csv("pairwise_no_hitl.csv")
df4 = pd.read_csv("pairwise_hitl.csv")

avg1 = df1["time"].mean()
avg2 = df2["time"].mean()
avg3 = df3["time"].mean()
avg4 = df4["time"].mean()

plt.figure(figsize=(12,9))

sns.ecdfplot(data=df1, x="time", linewidth=3,
             label=f"AAO_ALL (avg={avg1:.2f})")

sns.ecdfplot(data=df2, x="time", linewidth=3,
             label=f"AAO_NUM_SUBGRAPHS (avg={avg2:.2f})")

sns.ecdfplot(data=df3, x="time", linewidth=3,
             label=f"PAIRWISE_NO_HITL (avg={avg3:.2f})")

sns.ecdfplot(data=df4, x="time", linewidth=3,
             label=f"PAIRWISE_HITL (avg={avg3:.2f})")

# average lines
plt.axvline(avg1, linestyle="--", linewidth=2)
plt.axvline(avg2, linestyle="--", linewidth=2)
plt.axvline(avg3, linestyle="--", linewidth=2)
plt.axvline(avg4, linestyle="--", linewidth=2)

plt.text(avg1, 0.15, f"{avg1:.2f}", rotation=90, fontsize=18, weight="bold")
plt.text(avg2, 0.30, f"{avg2:.2f}", rotation=90, fontsize=18, weight="bold")
plt.text(avg3, 0.45, f"{avg3:.2f}", rotation=90, fontsize=18, weight="bold")
plt.text(avg3, 0.60, f"{avg3:.2f}", rotation=90, fontsize=18, weight="bold")

plt.xlabel("Time", fontsize=16)
plt.ylabel("CDF", fontsize=16)
plt.title("CDF of Runtime Comparison", fontsize=18)

plt.legend(fontsize=18)
plt.tight_layout()
plt.show()