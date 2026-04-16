import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import os
from pathlib import Path

PALETTE = {
    'Coma':       '#4E79A7',
    'Coma++':          '#E15759',
    'DistributionBased':        '#8C6D3F',
    'JaccardDistMatcher_t0.7':        '#B07AA1',
    'JaccardDistanceMatcher':      '#D62728',
    'SimilarityFlooding_basic_invavg':      '#2CA02C',
    'SimilarityFlooding_basic_invprod':  '#FF7F0E',
    'SimilarityFlooding_formulaA_invavg':  '#1F77B4',
    'SimilarityFlooding_formulaA_invprod':  "#073E3F",
    'SimilarityFlooding_formulaB_invavg':  "#A305C6",
    'SimilarityFlooding_formulaB_invprod':  "#EFFF08",
}

PATH_TO_LOOKUP = Path(__file__).resolve().parent.parent / "experiments/valentine"

def load_runtime_data(files_with_extension):
    data = {}
    
    for file_path in files_with_extension:
        stem = file_path.stem
        # Extract model name
        model_name = stem
        if stem.startswith("results_"):
            model_name = stem.replace("results_", "")
        try:
            # Read only runtime column
            df_src = pd.read_csv(file_path, usecols=["runtime"])
            # Get actual values, drop NaNs
            vals = df_src["runtime"].dropna().values
            if len(vals) > 0:
                data[model_name] = vals
                print(f"Loaded {model_name}: n={len(vals)}, mean={vals.mean():.4f}s")
            else:
                print(f"Skipping {model_name} (empty)")
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")

    return data

#
# PLOTTING FUNCTION
#
def plot_strip_panel(
    ax,
    data: dict,
    title: str,
    categories: list = None,
    palette: dict = None,
    kde_bandwidth: float = 0.2, # Bandwidth as factor. 0.2 smooths it a bit more than 0.04
    point_alpha: float = 0.6,   # Higher alpha since points might be sparse
    point_size: float = 20,     # Larger points
    max_points: int = 3000,
    row_spacing: float = 1.0,
    kde_height: float = 0.30,
):
    """
    Plot one horizontal strip panel for runtime distribution.
    Key adaptation: Autoscaling x-axis to data range instead of [0,1].
    """
    if categories is None:
        categories = list(data.keys())
        
    if palette is None:
        palette = {} 

    y_positions = {cat: i * row_spacing for i, cat in enumerate(categories)}
    
    # Determine global X-scale
    # Collect all valid data to find the 99th percentile for the plot limit
    all_values = []
    for cat in categories:
        if cat in data:
            all_values.extend(data[cat])
    
    if not all_values:
        return

    # Use 99th percentile to avoid extreme outliers squashing the plot
    # Or just use max() if data is clean
    x_max = np.percentile(all_values, 99) * 1.05
    x_min = 0 # Runtimes are positive
    print(f"Plotting with X-axis range: 0 - {x_max:.4f} seconds")
    for cat in categories:
        if cat not in data:
            continue
        vals = np.asarray(data[cat], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        y0 = y_positions[cat]
        color = palette.get(cat, '#555555') 
        n_total = len(vals)

        # Horizontal baseline
        ax.hlines(y0, x_min, x_max, color='#cccccc', linewidth=0.8,
                  linestyle='-', zorder=0, alpha=0.8)

        # KDE density band
        x_grid = np.linspace(x_min, x_max, 500)
        try:
            # Only do KDE if we have variance
            if len(vals) > 1 and np.std(vals) > 1e-9:
                kde = gaussian_kde(vals, bw_method=kde_bandwidth)
                density = kde(x_grid)
                density_norm = density / density.max()
            else:
                density_norm = np.zeros_like(x_grid)
        except Exception:
            density_norm = np.zeros_like(x_grid)

        # Draw density as a filled polygon
        y_top = y0 + kde_height * density_norm
        y_bot = y0 - kde_height * density_norm
        ax.fill_between(x_grid, y_bot, y_top,
                        color=color, alpha=0.25, zorder=1, linewidth=0)

        # Scatter dots (subsampled)
        if n_total > max_points:
            idx = np.random.choice(n_total, max_points, replace=False)
            plot_vals = vals[idx]
        else:
            plot_vals = vals

        # Jitter
        jitter = np.random.uniform(-0.06, 0.06, size=len(plot_vals))
        
        ax.scatter(plot_vals, y0 + jitter,
                   c=color, alpha=point_alpha, s=point_size,
                   linewidths=0, zorder=2)

        # Mean marker
        mean_val = vals.mean()
        ax.scatter(mean_val, y0,
                   color=color, s=100, edgecolors='white',
                   linewidths=1.8, zorder=5)

        # Mean label
        bbox_props = dict(boxstyle="round,pad=0.2", fc="white",
                          ec="lightgrey", lw=0.6, alpha=0.9)
        ax.text(mean_val, y0 + kde_height + 0.15,
                f"{mean_val:.3f}s",
                ha='center', va='bottom', fontsize=9,
                color='#333333', bbox=bbox_props, zorder=6)

        # n = annotation
        # Place it slightly outside the max range on the right
        ax.text(x_max, y0,
                f" n={n_total:,}",
                ha='left', va='center', fontsize=9,
                color='#555555', clip_on=False)

    # Y-axis labels
    ax.set_yticks([y_positions[c] for c in categories])
    ax.set_yticklabels(categories, fontsize=11, fontweight='bold')
    ax.tick_params(axis='y', length=0)

    # X-axis
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel('Runtime (seconds)', fontsize=13, labelpad=8)
    ax.tick_params(axis='x', labelsize=11)

    # Y limits
    y_vals = list(y_positions.values())
    if y_vals:
        ax.set_ylim(min(y_vals) - 0.55, max(y_vals) + 0.55)

    # Title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=14)
    ax.set_ylabel('')

    # Grid
    ax.xaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.yaxis.grid(False)
    sns.despine(ax=ax, left=True, top=True, right=True)


def filePathChecker_and_plot(path_to_lookup):
    # Resolve path
    base_path = Path(path_to_lookup).resolve()
    print(f"Searching in: {base_path}")
    
    # Recursively find results*.csv or just in that dir
    # Using glob across dir
    files_with_extension = list(base_path.glob("results*.csv"))
    
    if not files_with_extension:
        print("No CSV files found matching 'results*.csv'")
        return

    print(f"Found {len(files_with_extension)} files.")
    data = load_runtime_data(files_with_extension)
    
    if not data:
        print("No data loaded.")
        return

    # Setup Plot
    # Sort categories by mean runtime so the plot is ordered
    categories = sorted(data.keys(), key=lambda c: data[c].mean())
    categories = sorted(data.keys())

    # Create figure - height scales with number of categories
    fig_height = len(categories) * 1.0 + 2
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    # 3. Call Plot Function
    plot_strip_panel(
        ax,
        data=data,
        title='Runtime Distribution by Model',
        categories=categories,
        palette=PALETTE,
        # Parameters to answer "how do we determine the values":
        # - kde_bandwidth: 0.2 is a good heuristic factor for general distributions.
        # - point_alpha: 0.6 visibility.
        # - row_spacing: 1.0 standard vertical separation.
        kde_bandwidth=0.2, 
        point_alpha=0.6,
        row_spacing=1.0
    )

    plt.tight_layout()
    
    # Save
    save_path = Path('stripplot_runtime.png').resolve()
    fig.savefig(str(save_path), dpi=180, bbox_inches='tight')
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    filePathChecker_and_plot(PATH_TO_LOOKUP)
