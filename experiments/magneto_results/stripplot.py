import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from pathlib import Path
import ast
import os

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

PALETTE = {
    "D-Based": "#1f77b4",
    "JLDistMatcher": "#ff7f0e",
    "COMA++": "#2ca02c",
    "COMA": "#d62728",
    "SimFlooding": "#9467bd",
    "DDM": "#8c564b",
    "DDM_LLM": "#e377c2",
    "Magneto": "#7f7f7f",
    "MagnetoGPT": "#bcbd22",
}

PATH_TO_LOOKUP = Path(__file__).resolve().parent.parent / "magneto"

def load_column(file_path: Path, col_name: str):
    df = pd.read_csv(file_path)
    if col_name not in df.columns:
        return np.array([])
    
    series = df[col_name].dropna()
    
    if col_name == "max_rank_for_ideal_recall":
        # Flatten lists in the column
        series = series.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        series = series.explode()
        series = pd.to_numeric(series, errors='coerce').dropna()
    
    return series.values

def load_all_data(models_dict, col_name):
    data = {}
    for f, label in models_dict.items():
        path = PATH_TO_LOOKUP / f
        print(path)
        if path.exists():
            vals = load_column(path, col_name)
            if len(vals) > 0:
                data[label] = vals
                print(f"Loaded {label}: n={len(vals)}, mean={vals.mean():.4f}")
        else:
            print(f"File not found: {f}")
    return data

# strip plot
def plot_strip_panel(ax, data, title, palette, max_points=3000, row_spacing=1.0, kde_height=0.3):
    categories = list(data.keys())
    y_positions = {cat: i * row_spacing for i, cat in enumerate(categories)}
    
    all_vals = np.concatenate(list(data.values()))

    x_min = max(all_vals.min() * 0.9, 1e-3)
    x_max = np.percentile(all_vals, 99.99999) * 1.05
    
    for cat in categories:
        vals = np.array(data[cat])
        n_total = len(vals)
        y0 = y_positions[cat]
        color = palette.get(cat, '#555555')
        
        # Baseline
        ax.hlines(y0, x_min, x_max, color='#cccccc', linewidth=1.0, alpha=0.8)
        
        # KDE
        if len(vals) > 1 and np.std(vals) > 1e-9:
            kde = gaussian_kde(vals, bw_method=0.2)
            x_grid = np.linspace(x_min, x_max, 500)
            density = kde(x_grid)
            density_norm = density / density.max()
        else:
            x_grid = np.linspace(x_min, x_max, 2)
            density_norm = np.zeros_like(x_grid)
        
        y_top = y0 + kde_height * density_norm
        y_bot = y0 - kde_height * density_norm
        ax.fill_between(x_grid, y_bot, y_top, color=color, alpha=0.25, zorder=1)
        
        # Scatter points
        if n_total > max_points:
            idx = np.random.choice(n_total, max_points, replace=False)
            plot_vals = vals[idx]
        else:
            plot_vals = vals
        jitter = np.random.uniform(-0.06, 0.06, size=len(plot_vals))
        ax.scatter(plot_vals, y0 + jitter, c=color, alpha=0.6, s=35, linewidths=0, zorder=2)
        
        # Mean marker and label
        mean_val = vals.mean()
        ax.scatter(mean_val, y0, color=color, s=150, edgecolors='white', linewidths=2, zorder=5)
        bbox_props = dict(boxstyle="round,pad=0.15", fc="white", ec="lightgrey", lw=0.8, alpha=0.9)
        ax.text(mean_val, y0 + 0.08, f"{mean_val:.3f}", ha='center', va='bottom', fontsize=15, bbox=bbox_props)
        
        # N annotation
        # if cat == "COMA++" and title == "Runtime (s)":
        #     ax.text(x_max, y0, f" n=450", ha='left', va='center', fontsize=15, color='#555555', clip_on=False)
        # else:
        #     ax.text(x_max, y0, f" n={n_total}", ha='left', va='center', fontsize=15, color='#555555', clip_on=False)
    
    ax.set_yticks([y_positions[c] for c in categories])
    ax.set_yticklabels(categories, fontsize=15, fontweight='bold')
    ax.tick_params(axis='y', length=0,labelsize=15)
    
    ax.set_xlim(x_min, x_max)
    ax.set_xscale("log")
    ax.set_xlabel(title, fontsize=15, fontweight='bold')
    ax.set_ylim(min(y_positions.values()) - 0.5, max(y_positions.values()) + 0.5)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.tick_params(axis='x', labelsize=15)
    sns.despine(ax=ax, left=True, top=True, right=True)
    
    
def main():
    fields = ["max_rank_for_ideal_recall", "runtime"]
    titles = ["Rank For Ideal Recall", "Runtime (s)"]
    
    # fig, axes = plt.subplots(1, 2, figsize=(12, len(models) * 0.8 + 2))
    fig, axes = plt.subplots(1, 2, figsize=(12, len(models) * 0.8 + 2),
                         gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.90})
    
    for ax, field, title in zip(axes, fields, titles):
        data = load_all_data(models, field)
        if data:
            plot_strip_panel(ax, data, title=title, palette=PALETTE)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.97, bottom=0.15, wspace=0)

    plt.show()

if __name__ == "__main__":
    main()