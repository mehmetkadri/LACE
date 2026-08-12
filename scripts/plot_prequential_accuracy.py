"""Plot windowed prequential accuracy from raw multilabel predictions.

The stream is divided into 25 consecutive windows. Each point is the mean
example-based Jaccard accuracy over the corresponding 4% of samples.
Predictions are read directly from outputs/predictions.
"""

from pathlib import Path
import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from skmultilearn.dataset import load_from_arff


dataset_specs = {
    'Birds': {
        'type': 'real',
        'n_targets': 19,
        'n_instances': 645
    },
    'Water-qual': {
        'type': 'real',
        'n_targets': 14,
        'n_instances': 1060
    },
    'Gnegative': {
        'type': 'real',
        'n_targets': 8,
        'n_instances': 1392
    },
    'Langlog': {
        'type': 'real',
        'n_targets': 75,
        'n_instances': 1460
    },
    'Human': {
        'type': 'real',
        'n_targets': 14,
        'n_instances': 3106
    },
    'Slashdot': {
        'type': 'real',
        'n_targets': 22,
        'n_instances': 3782
    },
    'Eukaryote': {
        'type': 'real',
        'n_targets': 22,
        'n_instances': 7766
    },
    'OHSUMED-F': {
        'type': 'real',
        'n_targets': 23,
        'n_instances': 13929
    },
    'Yahoo-soc': {
        'type': 'real',
        'n_targets': 27,
        'n_instances': 14512
    },
    'Synthetic_Sudden1': {
        'type': 'synthetic',
        'synth_type': 'sudden1',
        'n_targets': 50,
        'n_instances': 20000
    },
    'Synthetic_Sudden2': {
        'type': 'synthetic',
        'synth_type': 'sudden2',
        'n_targets': 50,
        'n_instances': 20000
    },
    'Synthetic_Sudden3': {
        'type': 'synthetic',
        'synth_type': 'sudden3',
        'n_targets': 50,
        'n_instances': 20000
    },
    'Synthetic_Incremental1': {
        'type': 'synthetic',
        'synth_type': 'gradual1',
        'n_targets': 50,
        'n_instances': 20000
    },
    'Synthetic_Incremental2': {
        'type': 'synthetic',
        'synth_type': 'gradual2',
        'n_targets': 50,
        'n_instances': 20000
    },
    'Synthetic_Incremental3': {
        'type': 'synthetic',
        'synth_type': 'gradual3',
        'n_targets': 50,
        'n_instances': 20000
    },
    'Synthetic_Reoccurring': {
        'type': 'synthetic',
        'synth_type': 'reoccurring',
        'n_targets': 50,
        'n_instances': 20000
    },
    'tmc2007-500': {
        'type': 'real',
        'n_targets': 22,
        'n_instances': 28600
    },
    'Mediamill': {
        'type': 'real',
        'n_targets': 101,
        'n_instances': 43907
    },
    'Imdb': {
        'type': 'real',
        'n_targets': 28,
        'n_instances': 120900
    },
    'Emotions': {
        'type': 'real',
        'n_targets': 6,
        'n_instances': 593
    },
    'VirusGO': {
        'type': 'real',
        'n_targets': 6,
        'n_instances': 207
    },
    'Flags': {
        'type': 'real',
        'n_targets': 7,
        'n_instances': 194
    },
    'Scene': {
        'type': 'real',
        'n_targets': 6,
        'n_instances': 2407
    },
    'Genbase': {
        'type': 'real',
        'n_targets': 27,
        'n_instances': 662
    },
    'Medical': {
        'type': 'real',
        'n_targets': 45,
        'n_instances': 978
    },
    'Corel5k': {
        'type': 'real',
        'n_targets': 374,
        'n_instances': 5000
    },
    'Stackex': {
        'type': 'real',
        'n_targets': 227,
        'n_instances': 1675
    },
    'CHD49': {
        'type': 'real',
        'n_targets': 6,
        'n_instances': 555
    },
    'Corel16k': {
        'type': 'real',
        'n_targets': 153,
        'n_instances': 13766
    },
    'Yahoo-com': {
        'type': 'real',
        'n_targets': 33,
        'n_instances': 12444
    },
    'Bookmarks': {
        'type': 'real',
        'n_targets': 208,
        'n_instances': 87856
    }
}


ROOT = Path(__file__).resolve().parent
PREDICTION_ROOT = ROOT / "outputs" / "predictions"
SELECTED_DATASET = "Corel5k"

DATASET_CONFIG = {
    SELECTED_DATASET: {
        "n_targets": dataset_specs[SELECTED_DATASET]["n_targets"],
        "label_location": "start"
    }
}

# The order follows the current real-dataset average-rank ordering.
METHOD_FILES = {
    "LACE": "C3M_MC_NVI",
    "LD3": "LD3",
    "RDDM": "RDDM",
    "ND": "ND",
    "ADWIN": "ADWIN",
    "FHDDMS_add": "FHDDMS_add",
    "MDDM_G": "MDDM_G",
    "MDDM_E": "MDDM_E",
    "MDDM_A": "MDDM_A",
    "FHDDM": "FHDDM",
    "KSWIN": "KSWIN",
    "FHDDMS": "FHDDMS",
    "SeqDrift2": "SeqDrift2",
    "HDDM_W": "HDDM_W",
    "HDDM_A": "HDDM_A",
    "EDDM": "EDDM",
    "DDM": "DDM",
}

FILTER_METHODS = ["LACE", "LD3", "RDDM", "ND", "ADWIN"]
COLOR_PALETTE = ["#a30800", "#fc8d59", "#fbc226", "#2c7bb6", "#abd9e9"]
MARKER_PALETTE = ["o", "s", "x", "*", "^"]
N_WINDOWS = 25


def find_dataset_file(dataset):
    """Find the raw ARFF file in this project or the adjacent LACE project."""
    candidates = [
        ROOT / "src" / "datasets" / "{}.arff".format(dataset),
        ROOT.parent / "LACE" / "src" / "datasets" / "{}.arff".format(dataset),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join("  - {}".format(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find the raw {} dataset. Searched:\n{}".format(
            dataset, searched
        )
    )


def load_true_labels(dataset):
    """Load the multilabel targets in the same orientation used for evaluation."""
    if dataset not in DATASET_CONFIG:
        raise ValueError(
            "No dataset configuration for {!r}. Add its target count and "
            "label location to DATASET_CONFIG.".format(dataset)
        )

    config = DATASET_CONFIG[dataset]
    dataset_path = find_dataset_file(dataset)
    _, labels = load_from_arff(
        str(dataset_path),
        label_count=config["n_targets"],
        label_location=config["label_location"],
    )
    labels = labels.toarray().astype(np.int8)

    # Prediction generation excludes all-zero target rows.
    labels = labels[np.any(labels, axis=1)]
    return labels


def load_prediction_payload(dataset, file_method):
    """Load one raw prediction matrix and its detected drift points."""
    prediction_path = (
        PREDICTION_ROOT / dataset / "{}.pkl".format(file_method)
    )
    if not prediction_path.exists():
        raise FileNotFoundError(
            "Prediction file not found: {}".format(prediction_path)
        )

    with prediction_path.open("rb") as file:
        payload = pickle.load(file)

    if "predictions" not in payload:
        raise KeyError(
            "Missing 'predictions' in {}".format(prediction_path)
        )

    predictions = np.asarray(payload["predictions"], dtype=np.int8)
    if predictions.ndim != 2:
        raise ValueError(
            "{} has invalid prediction shape {}".format(
                prediction_path, predictions.shape
            )
        )
    return predictions, payload.get("drift_points", [])


def example_accuracy_per_sample(y_true, y_pred):
    """Return example-based Jaccard accuracy for every stream sample."""
    intersection = np.logical_and(y_true, y_pred).sum(axis=1)
    union = np.logical_or(y_true, y_pred).sum(axis=1)
    return np.divide(
        intersection,
        union,
        out=np.ones(len(y_true), dtype=float),
        where=union != 0,
    )


def windowed_prequential_accuracy(sample_accuracy, n_windows=N_WINDOWS):
    """Average sequential sample accuracies in consecutive equal windows."""
    if len(sample_accuracy) < n_windows:
        raise ValueError(
            "{} samples cannot be divided into {} windows.".format(
                len(sample_accuracy), n_windows
            )
        )
    windows = np.array_split(sample_accuracy, n_windows)
    return np.asarray([window.mean() for window in windows], dtype=float)


def compute_curves(dataset, filter_methods):
    """Compute one prequential-accuracy curve for each configured method."""
    y_true = load_true_labels(dataset)
    curves = {}
    drift_points_by_method = {}

    unknown_methods = [
        method for method in filter_methods if method not in METHOD_FILES
    ]
    if unknown_methods:
        raise ValueError(
            "Unknown methods in FILTER_METHODS: {}".format(unknown_methods)
        )

    for display_name in filter_methods:
        file_method = METHOD_FILES[display_name]
        y_pred, drift_points = load_prediction_payload(dataset, file_method)
        if y_pred.shape != y_true.shape:
            raise ValueError(
                "{}: truth shape {} does not match prediction shape {}".format(
                    display_name, y_true.shape, y_pred.shape
                )
            )

        sample_accuracy = example_accuracy_per_sample(y_true, y_pred)
        curves[display_name] = windowed_prequential_accuracy(sample_accuracy)
        drift_points_by_method[display_name] = drift_points

    curves = pd.DataFrame(curves)
    curves.index = np.linspace(100 / N_WINDOWS, 100, N_WINDOWS)
    curves.index.name = "percentage_of_data"
    return curves, drift_points_by_method, len(y_true)


def spread_endpoint_labels(endpoint_values, minimum, maximum, min_gap=0.008):
    """Return non-overlapping y positions while preserving endpoint order."""
    ordered = sorted(endpoint_values.items(), key=lambda item: item[1])
    positions = {}
    previous = None

    for method, value in ordered:
        position = value if previous is None else max(value, previous + min_gap)
        positions[method] = position
        previous = position

    overflow = max(positions.values()) - maximum
    if overflow > 0:
        positions = {
            method: position - overflow
            for method, position in positions.items()
        }

    underflow = minimum - min(positions.values())
    if underflow > 0:
        positions = {
            method: position + underflow
            for method, position in positions.items()
        }

    return positions


def plot_curves(
    curves,
    drift_points_by_method,
    sample_count,
    dataset,
    output_path,
):
    """Plot filtered method curves and their predicted drift points."""
    fig, axis = plt.subplots(figsize=(15, 8))
    percentages = curves.index.to_numpy(dtype=float)
    method_colors = {
        method: COLOR_PALETTE[index]
        for index, method in enumerate(curves.columns)
    }
    method_markers = {
        method: MARKER_PALETTE[index]
        for index, method in enumerate(curves.columns)
    }

    for method in curves.columns:
        color = method_colors[method]
        if method == "LACE":
            linewidth = 3.0
            alpha = 1.0
            zorder = 10
        else:
            linewidth = 2.0
            alpha = 0.9
            zorder = 3

        axis.plot(
            percentages,
            curves[method],
            label=method,
            color=color,
            marker=method_markers[method],
            markersize=7 if method == "LACE" else 6,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )

        for drift_point in drift_points_by_method.get(method, []):
            if isinstance(drift_point, dict):
                drift_position = drift_point.get(
                    "processed_idx", drift_point.get("stream_idx")
                )
            else:
                drift_position = drift_point

            if drift_position is None:
                continue
            if not 1 <= drift_position <= sample_count:
                continue

            axis.axvline(
                x=100.0 * drift_position / sample_count,
                color=color,
                linestyle=":",
                linewidth=2.0,
                alpha=0.9,
                zorder=2,
            )

    data_min = float(curves.min().min())
    data_max = float(curves.max().max())
    y_min = max(0.0, data_min - 0.02)
    y_max = data_max + 0.02

    endpoint_values = {
        method: float(curves[method].iloc[-1])
        for method in curves.columns
    }
    endpoint_label_y = spread_endpoint_labels(
        endpoint_values,
        minimum=y_min + 0.005,
        maximum=y_max - 0.005,
    )
    for method, endpoint in endpoint_values.items():
        axis.annotate(
            method,
            xy=(100, endpoint),
            xytext=(103.2, endpoint_label_y[method]),
            color=method_colors[method],
            fontsize=11,
            fontweight="bold",
            va="center",
            ha="left",
            arrowprops={
                "arrowstyle": "-",
                "color": method_colors[method],
                "linewidth": 1.8,
            },
            annotation_clip=False,
        )

    axis.set_xlabel("Percentage of data")
    axis.set_ylabel("Prequential accuracy")
    axis.set_title(
        "{} — Prequential Example-Based Accuracy "
        "(25 consecutive 4% windows)".format(dataset)
    )
    axis.set_xlim(0, 108)
    axis.set_xticks(np.arange(0, 101, 20))
    axis.set_ylim(y_min, y_max)
    axis.grid(alpha=0.25)
    axis.set_axisbelow(True)

    legend = axis.legend(
        handles=axis.get_legend_handles_labels()[0]
        + [
            Line2D(
                [0],
                [0],
                color="#444444",
                linestyle=":",
                linewidth=2.0,
                label="Predicted drift",
            )
        ],
        loc="upper right",
        borderaxespad=0.8,
        frameon=True,
        ncol=1,
    )
    for text in legend.get_texts():
        if text.get_text() == "LACE":
            text.set_color("#a30800")
            text.set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot prequential accuracy from raw prediction files."
    )
    parser.add_argument(
        "--dataset",
        default=SELECTED_DATASET,
        help="Dataset directory name (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG output path.",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        output_path = ROOT / "{}_prequential_accuracy.png".format(
            args.dataset
        )
    elif not output_path.is_absolute():
        output_path = ROOT / output_path

    curves, drift_points_by_method, sample_count = compute_curves(
        args.dataset, FILTER_METHODS
    )
    plot_curves(
        curves,
        drift_points_by_method,
        sample_count,
        args.dataset,
        output_path,
    )

    curve_path = output_path.with_suffix(".csv")
    curves.to_csv(curve_path)

    print("Saved figure: {}".format(output_path))
    print("Saved curves: {}".format(curve_path))
    print("\nFinal 4%-window prequential accuracy:")
    print(curves.iloc[-1].sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
