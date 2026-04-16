import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import ast
from scipy.stats import gaussian_kde

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

def load_flattened_column(file_path: Path, col_name: str):
    df = pd.read_csv(file_path)
    if col_name not in df.columns:
        return np.array([])
    
    series = df[col_name].dropna()
    
    # Flatten list-like strings
    series = series.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    series = series.explode()
    series = pd.to_numeric(series, errors='coerce').dropna()
    
    return series.values

def load_all(models_dict, col_name):
    data = {}
    for f, label in models_dict.items():
        path = PATH_TO_LOOKUP / f
        if path.exists():
            vals = load_flattened_column(path, col_name)
            if len(vals) > 0:
                data[label] = vals
                print(f"{label}: n={len(vals)}, mean={vals.mean():.3f}")
        else:
            print(f"Missing: {f}")
    return data

def compute_smooth_cdf(vals):
    vals = np.array(vals)
    
    # Remove non-positive values for log scale
    vals = vals[vals > 0]
    
    log_vals = np.log10(vals)
    
    kde = gaussian_kde(log_vals, bw_method=0.3)
    
    x_log = np.linspace(log_vals.min(), log_vals.max(), 500)
    pdf = kde(x_log)
    
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    
    x = 10 ** x_log  # back to original scale
    
    return x, cdf

def plot_cdf_two_panels(data):
    labels = list(data.keys())
    
    first_group = labels[:5]
    second_group = labels[5:]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
    
    for ax, group in zip(axes, [first_group, second_group]):
        for label in group:
            x, cdf = compute_smooth_cdf(data[label])
            
            ax.plot(
                x, cdf,
                label=label,
                color=PALETTE.get(label, None),
                linewidth=3.0
            )
        
        ax.set_xscale("log")
        
        ax.set_ylabel("CDF", fontsize=30, fontweight='bold')
        ax.tick_params(axis='both', labelsize=30)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=20)
    
    fig.supxlabel("Max Ranks for Ideal Recall", fontsize=30, fontweight='bold')
    
    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.15, top=0.95, wspace=0.25)
    
    plt.show()

def main():
    data = load_all(models, "max_rank_for_ideal_recall")
    plot_cdf_two_panels(data)

if __name__ == "__main__":
    main()