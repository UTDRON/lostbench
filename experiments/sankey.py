import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch, Patch
from matplotlib.path import Path
from matplotlib import colormaps
import os

DATA_PATH_LOOKUP = "../groundtruth/anes_sankey_categorical_flow.json"

with open(DATA_PATH_LOOKUP, 'r') as f:
    anes_sankey_categorical_data = json.load(f)

def build_year_batches(year_list, batch_size=10):
    return [year_list[i:i + batch_size] for i in range(0, len(year_list), batch_size)]

def build_sankey_inputs(data, start_year=None, end_year=None):
    years_sorted = sorted(data.keys(), key=int)

    if start_year is not None:
        years_sorted = [y for y in years_sorted if int(y) >= int(start_year)]
    if end_year is not None:
        years_sorted = [y for y in years_sorted if int(y) <= int(end_year)]

    set_sizes = [data[y]["total"] for y in years_sorted]

    # Build links as (source_year, target_year, [primary_flow, secondary_flow]).
    # Map [ln_ny, ly_ny] so blue is ln_ny and red is ly_ny.
    links = []
    for i in range(len(years_sorted) - 1):
        src = years_sorted[i]
        tgt = years_sorted[i + 1]
        primary = data[src].get("ln_ny", 0)
        secondary = data[src].get("ly_ny", 0)
        links.append((src, tgt, [primary, secondary]))

    return years_sorted, set_sizes, links


years = sorted(anes_sankey_categorical_data.keys(), key=int)
year_batches_10 = build_year_batches(years, batch_size=10)

batches_inputs = []
for categorical_sankey_years in year_batches_10:
    batch_years, batch_set_sizes, batch_links = build_sankey_inputs(
        anes_sankey_categorical_data,
        start_year=categorical_sankey_years[0],
        end_year=categorical_sankey_years[-1],
    )
    batches_inputs.append((batch_years, batch_set_sizes, batch_links))

# print("year_batches_10:", year_batches_10)
segment_colors = ['#1f78b4', '#e31a1c']

# Curved band (Sankey style)
def draw_curved_band(ax, x0, x1, y0, y1, h, color="gray", alpha=0.6):
    dx = (x1 - x0) * 0.4

    verts = [
        # bottom curve
        (x0, y0),
        (x0 + dx, y0),
        (x1 - dx, y1),
        (x1, y1),

        # top line
        (x1, y1 + h),

        # return curve
        (x1 - dx, y1 + h),
        (x0 + dx, y0 + h),
        (x0, y0 + h),

        (x0, y0)
    ]

    codes = [
        Path.MOVETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CLOSEPOLY
    ]

    path = Path(verts, codes)
    patch = PathPatch(
                path,
                facecolor=color,
                alpha=alpha,
                edgecolor='white',
                linewidth=0.5
            )
    ax.add_patch(patch)


def plot_sankey_batch(years, set_sizes, links, segment_colors, output_dir):
    period = f"{years[0]}-{years[-1]}"
    label_specs = [
        ("ln_ny", "#1f78b4"),
        ("ly_ny", "#e31a1c"),
        ("ln_nn", "#2ca25f"),
        ("ly_nn", "#ff7f00"),
    ]

    norm_sizes = np.array(set_sizes) / max(set_sizes)
    node_cmap = colormaps.get_cmap('Greens')
    node_colors = [node_cmap(0.2 + 0.3 * s) for s in norm_sizes]

    max_size = max(set_sizes)
    scale = 8 / max_size

    node_heights = [s * scale for s in set_sizes]
    flow_values = []
    for _, _, value in links:
        if isinstance(value, (list, tuple, np.ndarray)):
            flow_values.append([v for v in value if v > 0])
        else:
            flow_values.append([value])

    flow_heights = [[v * scale for v in values] for values in flow_values]

    # Layout
    x_positions = list(range(len(years)))
    node_width = 0.4

    fig, ax = plt.subplots(figsize=(16, 6))
    center_y = max(node_heights) / 2
    top_label_gap = max(node_heights) * 0.01
    top_label_line_gap = max(node_heights) * 0.04

    # Draw nodes
    node_tops = []
    for i, (x, h) in enumerate(zip(x_positions, node_heights)):
        bottom = center_y - h / 2
        top = bottom + h
        node_tops.append(top)

        rect = Rectangle(
                        (x - node_width / 2, bottom), node_width, h,
                        facecolor=node_colors[i],
                        linewidth=1.5
                    )
        ax.add_patch(rect)

        ax.text(
            x,
            center_y,
            f"{set_sizes[i]}",
            ha='center',
            va='center',
            fontsize=18,
            fontweight='bold',
            rotation=90,
            color='#1a1a1a'
        )

        year_data = anes_sankey_categorical_data.get(years[i], {})
        for label_index, (label_name, label_color) in enumerate(label_specs):
            ax.text(
                x,
                top + top_label_gap + (label_index * top_label_line_gap),
                f"{label_name}: {year_data.get(label_name, 0)}",
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                color=label_color,
            )

    # Draw flows(top-alignment)
    for i, ((src, tgt, vals), heights) in enumerate(zip(links, flow_heights)):
        x0 = x_positions[i] + node_width / 2
        x1 = x_positions[i + 1] - node_width / 2

        offset = 0
        for j, (val, h) in enumerate(zip(vals, heights)):
            y0 = node_tops[i] - offset - h
            y1 = node_tops[i + 1] - offset - h

            draw_curved_band(
                ax, x0, x1, y0, y1, h,
                color=segment_colors[j % len(segment_colors)],
                alpha=0.85
            )

            offset += h

    ax.set_xlim(-1, len(years))
    ax.set_ylim(0, max(node_heights) * 1.34)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(years, fontsize=18, fontweight='bold')
    ax.set_yticks([])

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    legend_handles = [
        Patch(facecolor=segment_colors[0], edgecolor='none', label='Blue: ln_ny'),
        Patch(facecolor=segment_colors[1], edgecolor='none', label='Red: ly_ny')
    ]
    ax.legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=11)

    ax.set_title(f"Flow of Variables ({period})", fontsize=18, pad=20, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"sankey_{years[0]}_{years[-1]}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    # print("Saved:", output_path)

output_dir = "sankey_outputs"
os.makedirs(output_dir, exist_ok=True)

for batch_years, batch_set_sizes, batch_links in batches_inputs[:3]:
    # print("years:", batch_years)
    # print("set_sizes:", batch_set_sizes)
    # print("links:", batch_links)
    plot_sankey_batch(batch_years, batch_set_sizes, batch_links, segment_colors, output_dir)