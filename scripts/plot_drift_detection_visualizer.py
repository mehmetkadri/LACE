"""Visualize DFD and MDR across the seven synthetic datasets.

The x position encodes Delay for Detection (DFD). Detector identity is
encoded by color. Circle fill encodes Missed Detection Rate (MDR): full,
half, and empty represent 0%, 50%, and 100% MDR, respectively.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
BASELINE_CSV_DIR = ROOT / "outputs" / "csv" / "baseline"
ABLATION_CSV_DIR = ROOT / "outputs" / "csv" / "ablation"

DFD_FILE = "delay_for_detection.csv"
MDR_FILE = "missed_detection_rate.csv"
LACE_SOURCE_METHOD = "C3M_MC_NVI"

METHODS = ["LACE", "LD3", "DDM", "RDDM", "ADWIN"]
METHOD_COLORS = {
    "LACE": "#a30800",
    "LD3": "#2c7bb6",
    "DDM": "#abd9e9",
    "RDDM": "#fbc226",
    "ADWIN": "#fc8d59",
}

SYNTHETIC_DATASETS = [
    "Synthetic_Incremental1",
    "Synthetic_Incremental2",
    "Synthetic_Incremental3",
    "Synthetic_Sudden1",
    "Synthetic_Sudden2",
    "Synthetic_Sudden3",
    "Synthetic_Reoccurring",
]

DATASET_LABELS = {
    "Synthetic_Incremental1": "Incremental1",
    "Synthetic_Incremental2": "Incremental2",
    "Synthetic_Incremental3": "Incremental3",
    "Synthetic_Sudden1": "Sudden1",
    "Synthetic_Sudden2": "Sudden2",
    "Synthetic_Sudden3": "Sudden3",
    "Synthetic_Reoccurring": "Reoccurring",
}

OUTPUT_PATH = ROOT / "synthetic_drift_detection_dfd_mdr.png"
DATA_PATH = ROOT / "synthetic_drift_detection_dfd_mdr.csv"
AVERAGE_OUTPUT_PATH = (
    ROOT / "synthetic_drift_detection_dfd_mdr_avg.png"
)
AVERAGE_DATA_PATH = (
    ROOT / "synthetic_drift_detection_dfd_mdr_avg.csv"
)


def read_metric_table(path):
    """Read a method-by-dataset metric table with numeric values."""
    if not path.exists():
        raise FileNotFoundError("Metric file not found: {}".format(path))

    table = pd.read_csv(path, index_col=0)
    table.index = table.index.astype(str).str.strip()
    table.columns = table.columns.astype(str).str.strip()
    return table.apply(pd.to_numeric, errors="coerce")


def collect_drift_results():
    """Merge baseline results with C3M_MC_NVI represented as LACE."""
    baseline_dfd = read_metric_table(BASELINE_CSV_DIR / DFD_FILE)
    baseline_mdr = read_metric_table(BASELINE_CSV_DIR / MDR_FILE)
    ablation_dfd = read_metric_table(ABLATION_CSV_DIR / DFD_FILE)
    ablation_mdr = read_metric_table(ABLATION_CSV_DIR / MDR_FILE)

    if LACE_SOURCE_METHOD not in ablation_dfd.index:
        raise ValueError(
            "{} is missing from {}".format(
                LACE_SOURCE_METHOD, ABLATION_CSV_DIR / DFD_FILE
            )
        )
    if LACE_SOURCE_METHOD not in ablation_mdr.index:
        raise ValueError(
            "{} is missing from {}".format(
                LACE_SOURCE_METHOD, ABLATION_CSV_DIR / MDR_FILE
            )
        )

    rows = []
    for method in METHODS:
        if method == "LACE":
            source_method = LACE_SOURCE_METHOD
            dfd_table = ablation_dfd
            mdr_table = ablation_mdr
        else:
            source_method = method
            dfd_table = baseline_dfd
            mdr_table = baseline_mdr

        if source_method not in dfd_table.index:
            raise ValueError(
                "{} is missing from the DFD table.".format(source_method)
            )
        if source_method not in mdr_table.index:
            raise ValueError(
                "{} is missing from the MDR table.".format(source_method)
            )

        for dataset in SYNTHETIC_DATASETS:
            rows.append(
                {
                    "Dataset": dataset,
                    "Dataset Label": DATASET_LABELS[dataset],
                    "Method": method,
                    "DFD": dfd_table.loc[source_method, dataset],
                    "MDR": mdr_table.loc[source_method, dataset],
                }
            )

    return pd.DataFrame(rows)


def marker_fillstyle(mdr):
    """Map MDR to the fraction of the circular marker that remains filled."""
    if np.isclose(mdr, 0.0):
        return "full"
    if np.isclose(mdr, 0.5):
        return "bottom"
    if np.isclose(mdr, 1.0):
        return "none"
    raise ValueError(
        "Unsupported MDR value {}. Expected 0.0, 0.5, or 1.0.".format(mdr)
    )


def detector_legend_handles():
    """Create color-only handles so detector identity is separate from MDR."""
    return [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linewidth=4,
            label=method,
        )
        for method in METHODS
    ]


def mdr_legend_handles(include_average=False):
    """Create fill-only handles for the three possible detection outcomes."""
    entries = [
        ("full", "0% MDR (2/2 detected)"),
        ("bottom", "50% MDR (1/2 detected)"),
    ]
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=10,
            fillstyle=fillstyle,
            markerfacecolor="#555555",
            markerfacecoloralt="white",
            markeredgecolor="#555555",
            markeredgewidth=2,
            label=label,
        )
        for fillstyle, label in entries
    ]

    if include_average:
        handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color="#777777",
                    linewidth=1.0,
                    label="",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markersize=11,
                    fillstyle="none",
                    markeredgecolor="#555555",
                    markeredgewidth=2,
                    label="Percentage (%) (only for average)",
                ),
            ]
        )

    return handles


def calculate_detector_averages(results):
    """Average MDR over all datasets and DFD over its finite observations."""
    averages = (
        results.groupby("Method", as_index=False, sort=False)
        .agg(DFD=("DFD", "mean"), MDR=("MDR", "mean"))
        .set_index("Method")
        .reindex(METHODS)
        .reset_index()
    )
    averages.insert(0, "Dataset", "Average")
    averages.insert(1, "Dataset Label", "Average")
    return averages


def average_marker_y_offsets(averages, collision_distance=100.0):
    """Slightly separate average markers whose mean DFD values collide."""
    ordered = averages.sort_values("DFD").reset_index(drop=True)
    clusters = []
    current_cluster = []

    for row in ordered.itertuples(index=False):
        if (
            current_cluster
            and row.DFD - current_cluster[-1].DFD >= collision_distance
        ):
            clusters.append(current_cluster)
            current_cluster = []
        current_cluster.append(row)
    if current_cluster:
        clusters.append(current_cluster)

    offsets = {method: 0.0 for method in METHODS}
    for cluster in clusters:
        if len(cluster) > 1:
            cluster_offsets = np.linspace(-0.08, 0.08, len(cluster))
            for row, offset in zip(cluster, cluster_offsets):
                offsets[row.Method] = float(offset)

    return offsets


def plot_drift_results(results, output_path, include_average=False):
    """Plot DFD by synthetic dataset with marker fill determined by MDR."""
    fig, axis = plt.subplots(figsize=(14, 7))
    plotted_datasets = list(SYNTHETIC_DATASETS)
    if include_average:
        plotted_datasets.append("Average")

    y_positions = {
        dataset: position
        for position, dataset in enumerate(plotted_datasets)
    }

    for row in results.itertuples(index=False):
        if pd.isna(row.DFD) or pd.isna(row.MDR):
            continue

        color = METHOD_COLORS[row.Method]
        fillstyle = marker_fillstyle(row.MDR)
        axis.plot(
            [row.DFD],
            [y_positions[row.Dataset]],
            marker="o",
            markersize=13,
            linestyle="None",
            fillstyle=fillstyle,
            markerfacecolor=color,
            markerfacecoloralt="white",
            markeredgecolor=color,
            markeredgewidth=2.2,
            alpha=0.95,
            zorder=10 if row.Method == "LACE" else 3,
        )

    if include_average:
        detector_averages = calculate_detector_averages(results)
        average_y = y_positions["Average"]
        average_y_offsets = average_marker_y_offsets(detector_averages)

        for row in detector_averages.itertuples(index=False):
            color = METHOD_COLORS[row.Method]
            zorder = 10 if row.Method == "LACE" else 5
            marker_y = average_y + average_y_offsets[row.Method]
            axis.scatter(
                row.DFD,
                marker_y,
                s=400,
                marker="o",
                facecolor="white",
                edgecolor=color,
                linewidth=2.5,
                zorder=zorder,
            )
            axis.text(
                row.DFD,
                marker_y,
                "{:.0f}%".format(100 * row.MDR),
                ha="center",
                va="center",
                fontsize=6.5,
                fontweight="bold",
                color=color,
                zorder=zorder + 1,
            )

    finite_dfd = results["DFD"].dropna()
    x_max = float(finite_dfd.max())
    axis.set_xlim(left=-0.02 * x_max, right=1.05 * x_max)
    axis.set_yticks(range(len(plotted_datasets)))
    axis.set_yticklabels(
        [
            DATASET_LABELS.get(dataset, dataset)
            for dataset in plotted_datasets
        ],
        fontweight="bold",
    )
    axis.invert_yaxis()
    axis.set_xlabel("Delay for Detection (DFD)", fontweight="bold")
    axis.set_ylabel("Synthetic dataset", fontweight="bold")
    axis.grid(axis="x", linestyle="--", alpha=0.35)
    axis.grid(axis="y", linestyle=":", alpha=0.25)
    axis.set_axisbelow(True)

    detector_legend = axis.legend(
        handles=detector_legend_handles(),
        title="Detector",
        loc="upper right",
        bbox_to_anchor=(1.0, 0.805 if include_average else 0.875),
        frameon=True,
        edgecolor="black",
    )
    detector_legend.get_title().set_fontweight("bold")
    for text in detector_legend.get_texts():
        method = text.get_text()
        text.set_color(METHOD_COLORS[method])
        text.set_fontweight("bold")
    axis.add_artist(detector_legend)

    mdr_legend = axis.legend(
        handles=mdr_legend_handles(include_average=include_average),
        title="MDR indicator",
        loc="upper right",
        frameon=True,
        edgecolor="black",
    )
    mdr_legend.get_title().set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    results = collect_drift_results()
    results.to_csv(DATA_PATH, index=False)
    plot_drift_results(results, OUTPUT_PATH)

    average_results = pd.concat(
        [results, calculate_detector_averages(results)],
        ignore_index=True,
    )
    average_results.to_csv(AVERAGE_DATA_PATH, index=False)
    plot_drift_results(
        results,
        AVERAGE_OUTPUT_PATH,
        include_average=True,
    )

    plotted_count = int(results["DFD"].notna().sum())
    print("Saved figure: {}".format(OUTPUT_PATH))
    print("Saved plot data: {}".format(DATA_PATH))
    print("Saved average figure: {}".format(AVERAGE_OUTPUT_PATH))
    print("Saved average plot data: {}".format(AVERAGE_DATA_PATH))
    print("Plotted {} finite DFD observations.".format(plotted_count))


if __name__ == "__main__":
    main()
