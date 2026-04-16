import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
import numpy as np

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

MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '>']
LINESTYLES = ['-', '--', '-.', ':', (0, (3,1,1,1)), (0, (5,2)), (0, (1,1)), (0, (5,1,1,1)), (0, (2,2))]

PATH_TO_LOOKUP = Path(__file__).resolve().parent.parent / "magneto"

def load_data(file_path):
    df = pd.read_csv(file_path)
    if not {"runtime", "ncols_tgt", "nrows_tgt"}.issubset(df.columns):
        return None
    df = df[["runtime", "ncols_tgt", "nrows_tgt"]].dropna()
    df["size"] = df["ncols_tgt"] * df["nrows_tgt"]
    return df

def plot_runtime_vs_size():
    plt.figure(figsize=(14, 8))
    
    for (file, label), linestyle in zip(models.items(), LINESTYLES):
        path = PATH_TO_LOOKUP / file
        if not path.exists():
            continue
        
        df = load_data(path)
        if df is None or df.empty:
            continue
        
        x = df["size"].values
        y = df["runtime"].values
        
        mask = (x > 0) & (y > 0)
        x, y = x[mask], y[mask]
        
        if len(x) < 5:
            continue
        
        # ---- LOG BINNING (key fix) ----
        log_x = np.log10(x)
        bins = np.linspace(log_x.min(), log_x.max(), 20)
        bin_ids = np.digitize(log_x, bins)
        
        x_binned = []
        y_binned = []
        
        for b in np.unique(bin_ids):
            xb = x[bin_ids == b]
            yb = y[bin_ids == b]
            if len(xb) == 0:
                continue
            x_binned.append(np.mean(xb))
            y_binned.append(np.mean(yb))
        
        x_binned = np.array(x_binned)
        y_binned = np.array(y_binned)
        
        # sort after binning
        order = np.argsort(x_binned)
        x_binned = x_binned[order]
        y_binned = y_binned[order]
        
        if len(x_binned) < 3:
            continue
        
        # ---- SMOOTH (log interpolation) ----
        log_xb = np.log10(x_binned)
        log_yb = np.log10(y_binned)
        
        f = interp1d(log_xb, log_yb, kind='linear')
        x_smooth_log = np.linspace(log_xb.min(), log_xb.max(), 300)
        y_smooth_log = f(x_smooth_log)
        
        x_smooth = 10 ** x_smooth_log
        y_smooth = 10 ** y_smooth_log
        
        plt.plot(
            x_smooth,
            y_smooth,
            label=label,
            color=PALETTE.get(label),
            linewidth=5,
            linestyle=linestyle
        )
        
    plt.xlabel("#Cells in Target Schema (cols × rows)", fontsize=20, fontweight='bold')
    plt.ylabel("Runtime (s)", fontsize=20, fontweight='bold')
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=20, ncol=2)
    
    plt.xscale("log")
    plt.yscale("log")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_runtime_vs_size()