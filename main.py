# Updated on 2024-06-21 at 09.40

import argparse
import csv
import os
import pickle
import subprocess
import sys
try:
    import fcntl
except ImportError:
    fcntl = None
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
import warnings

import numpy as np

np.float = float

from tqdm.auto import tqdm
from skmultilearn.dataset import load_from_arff
from skmultiflow.data import DataStream
from skmultiflow.meta import ClassifierChain
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MultiLabelBinarizer

import time
from datetime import datetime

import src.util as util

# Suppressing specific warnings to keep the output clean.
np.seterr(divide="ignore", invalid="ignore", over="ignore")
warnings.filterwarnings(
    "ignore",
    message=r"ks_2samp: Exact calculation unsuccessful. Switching to method=asymp.",
    category=RuntimeWarning,
)

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

ld3_params = {
    'Enron': {
        'window_size': 250,
        'correlation_thresh': 4,
        'len': 0
    },
    'Imdb': {
        'window_size': 500,
        'correlation_thresh': 4,
        'len': 2
    },
    'Eukaryote': {
        'window_size': 50,
        'correlation_thresh': 2,
        'len': 1
    },
    'Plant': {
        'window_size': 20,
        'correlation_thresh': 2,
        'len': 1
    },
    'tmc2007-500': {
        'window_size': 500,
        'correlation_thresh': 4,
        'len': 1
    },
    'OHSUMED-F': {
        'window_size': 500,
        'correlation_thresh': 5,
        'len': 1
    },
    'Birds': {
        'window_size': 10,
        'correlation_thresh': 2,
        'len': 0
    },
    'yeast': {
        'window_size': 50,
        'correlation_thresh': 4,
        'len': 0
    },
    'Synthetic_Sudden2': {
        'window_size': 500,
        'correlation_thresh': 3,
        'len': 0
    },
    'Synthetic_Incremental2': {
        'window_size': 500,
        'correlation_thresh': 4,
        'len': 0
    },
    'Synthetic_Reoccurring': {
        'window_size': 500,
        'correlation_thresh': 6,
        'len': 0
    }, ### Using default parameters from here on for synthetic datasets, as they are not used for tuning
    'Default': {
        'window_size': 500,
        'correlation_thresh': 4,
        'len': 0
    }
}

PAPER_DATASETS = [
    'Synthetic_Sudden2',
    'Synthetic_Sudden1',
    'Synthetic_Sudden3',
    'Synthetic_Incremental1',
    'Synthetic_Incremental2',
    'Synthetic_Incremental3',
    'Synthetic_Reoccurring',
    'Birds',
    #'Enron',
    'Eukaryote',
    'Gnegative',
    'Human',
    'Imdb',
    'Mediamill',
    'OHSUMED-F',
    #'Plant',
    'Slashdot',
    'tmc2007-500',
    #'yeast',
]

ALL_DATASETS = list(dataset_specs.keys())

PAPER_DETECTORS = [
    "LACE",
    "LD3",
    "RDDM",
    "ADWIN",
    "EDDM",
    "DDM",
    "KSWIN",
    "HDDM_A",
    "HDDM_W",
    "FHDDM",
    "FHDDMS",
    "FHDDMS_add",
    "MDDM_A",
    "MDDM_E",
    "MDDM_G",
    "SeqDrift2",
    "ND",
]

BASELINE_DETECTORS = PAPER_DETECTORS[1:]


BASE_DETECTOR = {
    "n_permutations": 1000,
    "seed": 42,
}


CLUSTERING_CONFIGS = {
    "C3M": {
        "clustering_method": "c3m",
        "matrix_type": "transposed_window",
    },
    "Agglo": {
        "clustering_method": "agglomerative",
        "matrix_type": "transposed_window",
        "linkage_method": "average",
        "distance_calculation_strategy": "1-(cosine)",
    },
    "KMeans": {
        "clustering_method": "kmeans",
        "matrix_type": "transposed_window",
    },
}


WINDOW_CONFIGS = {
    "w250": 250,
    "w500": 500,
    "w1000": 1000,
}


THRESHOLD_CONFIGS = {
    "000_050": {
        "similarity_lower_threshold": 0.00,
        "similarity_upper_threshold": 0.50,
    },
    "005_050": {
        "similarity_lower_threshold": 0.05,
        "similarity_upper_threshold": 0.50,
    },
    "010_050": {
        "similarity_lower_threshold": 0.10,
        "similarity_upper_threshold": 0.50,
    },
    "005_060": {
        "similarity_lower_threshold": 0.05,
        "similarity_upper_threshold": 0.60,
    },
    "010_060": {
        "similarity_lower_threshold": 0.10,
        "similarity_upper_threshold": 0.60,
    },
    "010_070": {
        "similarity_lower_threshold": 0.10,
        "similarity_upper_threshold": 0.70,
    },
}


NVI_CONFIGS = {
    "010NVI": {
        "use_nvi": True,
        "distance_threshold": 0.10,
    },
    "025NVI": {
        "use_nvi": True,
        "distance_threshold": 0.25,
    },
    "035NVI": {
        "use_nvi": True,
        "distance_threshold": 0.35,
    },
    "050NVI": {
        "use_nvi": True,
        "distance_threshold": 0.50,
    },
}

NO_MC_THRESHOLD_CONFIGS = {
    "sim010": {"similarity_threshold": 0.10},
    "sim025": {"similarity_threshold": 0.25},
    "sim035": {"similarity_threshold": 0.35},
    "sim050": {"similarity_threshold": 0.50},
}

COMPONENT_ABLATION_CONFIGS = {
    "C3M_MC_NVI": {
        "use_monte_carlo": True,
        "use_nvi": True,
    },
    "C3M_MC_NoNVI": {
        "use_monte_carlo": True,
        "use_nvi": False,
    },
}

for key, threshold_cfg in NO_MC_THRESHOLD_CONFIGS.items():
    COMPONENT_ABLATION_CONFIGS[f"C3M_NoMC_NVI_{key}"] = {
        "use_monte_carlo": False,
        "use_nvi": True,
        "n_permutations": 0,
        **threshold_cfg,
    }

    COMPONENT_ABLATION_CONFIGS[f"C3M_NoMC_NoNVI_{key}"] = {
        "use_monte_carlo": False,
        "use_nvi": False,
        "n_permutations": 0,
        **threshold_cfg,
    }


CLUSTERING_ABLATION_CONFIGS = {
    "LACE-C3M_best": {
        "clustering_method": "c3m",
        "matrix_type": "transposed_window",
    },
    "LACE-Agglo_best": {
        "clustering_method": "agglomerative",
        "matrix_type": "transposed_window",
        "linkage_method": "average",
        "distance_calculation_strategy": "1-(cosine)",
    },
    "LACE-KMeans_best": {
        "clustering_method": "kmeans",
        "matrix_type": "transposed_window",
    },
}


# None keeps the dataset-dependent best window: 250 for streams with fewer than
# 1500 instances, 1000 otherwise. Use "w250", "w500", or "w1000" to force one.
BEST_WINDOW_KEY = "w500"
BEST_THRESHOLD_KEY = "005_050"
BEST_NVI_KEY = "050NVI"


def make_lace_config(
    detector_name,
    clustering_key,
    threshold_key,
    nvi_key,
    window_key=None,
    extra_overrides=None,
):
    cfg = {
        **BASE_DETECTOR,
        **CLUSTERING_CONFIGS[clustering_key],
        **THRESHOLD_CONFIGS[threshold_key],
        **NVI_CONFIGS[nvi_key],
        "similarity_metric": "ari",
        "feed_mode": "unsupervised",
        "debug_name": detector_name,
        "use_monte_carlo": True,
        "window_size": None,
        "step": None,
    }

    if window_key is not None:
        window_size = WINDOW_CONFIGS[window_key]
        cfg.update({
            "window_size": window_size,
            "step": window_size // 2,
        })

    if extra_overrides is not None:
        cfg.update(extra_overrides)

    return cfg


def generate_lace_sensitivity_configs():
    configs = {}

    for window_key in WINDOW_CONFIGS:
        for threshold_key in THRESHOLD_CONFIGS:
            for nvi_key in NVI_CONFIGS:
                detector_name = f"LACE-C3M_{window_key}_{threshold_key}_{nvi_key}"

                configs[detector_name] = make_lace_config(
                    detector_name=detector_name,
                    clustering_key="C3M",
                    threshold_key=threshold_key,
                    nvi_key=nvi_key,
                    window_key=window_key,
                )

    return configs


LACE_SENSITIVITY_CONFIGS = generate_lace_sensitivity_configs()


BEST_LACE_CONFIG_NAME = "LACE-C3M_w500_005_050_050NVI"
BEST_LACE_ALIAS_NAMES = {
    "LACE",
    "C3M_MC_NVI",
    "LACE-C3M_best",
}


def make_best_lace_config(detector_name, extra_overrides=None):
    canonical_debug_name = (
        BEST_LACE_CONFIG_NAME
        if detector_name in BEST_LACE_ALIAS_NAMES
        else detector_name
    )

    return make_lace_config(
        detector_name=canonical_debug_name,
        clustering_key="C3M",
        threshold_key=BEST_THRESHOLD_KEY,
        nvi_key=BEST_NVI_KEY,
        window_key=BEST_WINDOW_KEY,
        extra_overrides=extra_overrides,
    )


COMPONENT_ABLATION_DETECTOR_CONFIGS = {
    detector_name: (
        make_best_lace_config(detector_name)
        if detector_name in BEST_LACE_ALIAS_NAMES
        else make_best_lace_config(
            detector_name=detector_name,
            extra_overrides=overrides,
        )
    )
    for detector_name, overrides in COMPONENT_ABLATION_CONFIGS.items()
}


CLUSTERING_ABLATION_DETECTOR_CONFIGS = {
    detector_name: (
        make_best_lace_config(detector_name)
        if detector_name in BEST_LACE_ALIAS_NAMES
        else make_best_lace_config(
            detector_name=detector_name,
            extra_overrides=overrides,
        )
    )
    for detector_name, overrides in CLUSTERING_ABLATION_CONFIGS.items()
}


ABLATION_DETECTORS = (
    list(COMPONENT_ABLATION_CONFIGS.keys())
    + list(CLUSTERING_ABLATION_CONFIGS.keys())
)


HYPERPARAMETER_SENSITIVITY_DETECTORS = list(LACE_SENSITIVITY_CONFIGS.keys())


STATIC_DETECTOR_INFO = {
    # Main method
    "LACE": make_best_lace_config("LACE"),

    **LACE_SENSITIVITY_CONFIGS,

    **COMPONENT_ABLATION_DETECTOR_CONFIGS,

    **CLUSTERING_ABLATION_DETECTOR_CONFIGS,

    # Baselines
    "LD3": {
        "window_size": None,
        "correlation_thresh": None,
        "len": None,
    },

    "ADWIN": {},
    "EDDM": {},
    "DDM": {},
    "KSWIN": {},
    "HDDM_A": {},
    "HDDM_W": {},
    "FHDDM": {},
    "FHDDMS": {},
    "FHDDMS_add": {},
    "MDDM_A": {},
    "MDDM_E": {},
    "MDDM_G": {},
    "SeqDrift2": {},
    "RDDM": {},
    "ND": {},
}

TORNADO_DETS = [
    "FHDDM",
    "FHDDMS",
    "FHDDMS_add",
    "MDDM_A",
    "MDDM_E",
    "MDDM_G",
    "SeqDrift2",
    "RDDM",
]


def is_lace_detector_config(cfg):
    return "clustering_method" in cfg and "similarity_metric" in cfg


DRIFT_DECISION_COLUMNS = [
    "Indices of Monte Carlo related drifts",
    "Indices of Monte Carlo related no-drifts",
    "Indices of ARI Threshold related drifts",
    "Indices of ARI Threshold related no-drifts",
    "Indices of NVI related drifts",
    "Indices of NVI related no-drifts",
]

DRIFT_DECISION_COLUMN_BY_SOURCE = {
    ("monte_carlo", True): DRIFT_DECISION_COLUMNS[0],
    ("monte_carlo", False): DRIFT_DECISION_COLUMNS[1],
    ("ari_threshold", True): DRIFT_DECISION_COLUMNS[2],
    ("ari_threshold", False): DRIFT_DECISION_COLUMNS[3],
    ("nvi", True): DRIFT_DECISION_COLUMNS[4],
    ("nvi", False): DRIFT_DECISION_COLUMNS[5],
}

DRIFT_DECISION_FIELDNAMES = ["dataset"] + DRIFT_DECISION_COLUMNS

DRIFT_DECISION_COUNT_COLUMN_GROUPS = [
    (
        "NVI",
        DRIFT_DECISION_COLUMNS[4],
        DRIFT_DECISION_COLUMNS[5],
    ),
    (
        "ARI",
        DRIFT_DECISION_COLUMNS[2],
        DRIFT_DECISION_COLUMNS[3],
    ),
    (
        "Monte Carlo",
        DRIFT_DECISION_COLUMNS[0],
        DRIFT_DECISION_COLUMNS[1],
    ),
]


def sort_datasets_by_size(datasets):
    return sorted(
        datasets,
        key=lambda dataset_name: dataset_specs[dataset_name].get(
            "n_instances",
            sys.maxsize,
        ),
    )


def _annotate_new_decisions(detector, previous_decision_count, stream_idx, processed_idx):
    decision_log = getattr(detector, "decision_log", None)
    if decision_log is None:
        return

    for decision in decision_log[previous_decision_count:]:
        decision["stream_idx"] = stream_idx
        decision["processed_idx"] = processed_idx


def _group_drift_decision_indices(decision_log):
    grouped = {column: [] for column in DRIFT_DECISION_COLUMNS}

    for decision in decision_log:
        column = DRIFT_DECISION_COLUMN_BY_SOURCE.get(
            (
                decision.get("decision_source"),
                bool(decision.get("is_drift")),
            )
        )
        if column is None:
            continue

        index = decision.get("stream_idx")
        if index is None:
            index = decision.get("processed_idx")
        if index is None:
            index = decision.get("t")

        grouped[column].append(index)

    return grouped


def _drift_decision_report_path(args, detector_name):
    return Path(args.decision_output_dir) / f"{detector_name}.csv"


def _drift_decision_count_report_path(args, detector_name):
    return Path(args.decision_output_dir) / f"{detector_name}_count.csv"


def _join_indices(indices):
    return ";".join(str(index) for index in indices)


def _count_indices(cell_value):
    if cell_value is None:
        return 0

    text = str(cell_value).strip()
    if not text:
        return 0

    return len([value for value in text.split(";") if value.strip()])


def _make_drift_decision_row(dataset_name, decision_log):
    grouped = _group_drift_decision_indices(decision_log)
    row = {"dataset": dataset_name}

    for column in DRIFT_DECISION_COLUMNS:
        row[column] = _join_indices(grouped[column])

    return row


def _drift_decision_row_sort_key(row):
    dataset_name = row.get("dataset", "")
    return (
        dataset_specs.get(dataset_name, {}).get("n_instances", sys.maxsize),
        dataset_name,
    )


def _read_existing_drift_decision_rows(output_file, dataset_name):
    if not output_file.exists():
        return []

    rows = []
    with open(output_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_dataset = row.get("dataset")
            if not row_dataset or row_dataset == dataset_name:
                continue

            rows.append({
                fieldname: row.get(fieldname, "")
                for fieldname in DRIFT_DECISION_FIELDNAMES
            })

    return rows


def write_drift_decision_count_report(args, detector_name, rows):
    output_file = _drift_decision_count_report_path(args, detector_name)
    temp_file = output_file.with_name(f".{output_file.name}.tmp")

    header_top = ["dataset"]
    header_bottom = [""]

    for group_name, _, _ in DRIFT_DECISION_COUNT_COLUMN_GROUPS:
        header_top.extend([group_name, group_name])
        header_bottom.extend(["Drift", "No Drift"])

    count_rows = []
    for row in rows:
        dataset_name = row.get("dataset", "")
        if not dataset_name:
            continue

        count_row = [dataset_name]

        for _, drift_column, no_drift_column in DRIFT_DECISION_COUNT_COLUMN_GROUPS:
            count_row.extend([
                _count_indices(row.get(drift_column, "")),
                _count_indices(row.get(no_drift_column, "")),
            ])

        count_rows.append(count_row)

    with open(temp_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header_top)
        writer.writerow(header_bottom)
        writer.writerows(count_rows)

    os.replace(temp_file, output_file)
    print(f"[DECISION COUNT CSV] {output_file}")


def write_drift_decision_report(args, dataset_name, detector_name, decision_log):
    output_dir = Path(args.decision_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = _drift_decision_report_path(args, detector_name)
    lock_file = output_dir / f".{detector_name}.csv.lock"
    temp_file = output_dir / f".{detector_name}.csv.tmp"

    new_row = _make_drift_decision_row(dataset_name, decision_log)

    with open(lock_file, "w") as lock_handle:
        if fcntl is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)

        try:
            rows = _read_existing_drift_decision_rows(output_file, dataset_name)
            rows.append(new_row)
            rows.sort(key=_drift_decision_row_sort_key)

            with open(temp_file, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=DRIFT_DECISION_FIELDNAMES,
                )
                writer.writeheader()
                writer.writerows(rows)

            os.replace(temp_file, output_file)
            write_drift_decision_count_report(args, detector_name, rows)
        finally:
            if fcntl is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    print(f"[DECISION CSV] {output_file}")


def build_detector_info(dataset_name, ld3_params, dets_to_run=None):
    if dets_to_run is None:
        dets_to_run = PAPER_DETECTORS

    detector_info = {}

    for detector_name in dets_to_run:
        cfg = deepcopy(STATIC_DETECTOR_INFO[detector_name])

        # Dataset-dependent params for LACE detectors
        if is_lace_detector_config(cfg):
            if cfg.get("window_size") is None:
                # give warning
                raise ValueError(
                    f"LACE detector '{detector_name}' has no window_size set for dataset '{dataset_name}'."
                )

            if cfg.get("step") is None:
                cfg["step"] = cfg["window_size"] // 2

        # Dataset-dependent params for LD3
        elif detector_name == "LD3":
            params = ld3_params.get(dataset_name, ld3_params["Default"])

            cfg["window_size"] = params["window_size"]
            cfg["correlation_thresh"] = params["correlation_thresh"]
            cfg["len"] = params["len"]

        detector_info[detector_name] = cfg

    return detector_info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LACE and baseline drift detectors for multi-label data streams."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="paper",
        help=(
            "Dataset name, 'paper', or 'all'. "
            "Example: Birds, Enron, Synthetic_Sudden2, paper, all."
        ),
    )

    parser.add_argument(
        "--detector",
        type=str,
        default="LACE",
        help=(
            "Detector name, 'paper', 'baselines', 'ablation', 'sensitivity', or 'all-available'. "
            "Example: LACE, LD3, RDDM, ND, paper, baselines, ablation, sensitivity."
        ),
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="src/datasets",
        help="Directory containing the dataset files.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/predictions",
        help="Directory where prediction outputs will be saved.",
    )

    parser.add_argument(
        "--decision-output-dir",
        type=str,
        default="outputs/drift_decisions",
        help="Directory where LACE drift-decision CSV files will be saved.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )

    parser.add_argument(
        "--decision-report-only",
        action="store_true",
        help=(
            "Run detectors and write drift-decision CSV files without creating "
            "or overwriting prediction pickle files."
        ),
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of stream samples to process for quick tests.",
    )

    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=6,
        help="Number of dataset-detector subprocesses to run in parallel.",
    )

    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=8,
        help="Native compute threads allowed per subprocess.",
    )

    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    if args.parallel_workers < 1:
        parser.error("--parallel-workers must be at least 1")

    if args.threads_per_worker < 1:
        parser.error("--threads-per-worker must be at least 1")

    return args


def resolve_datasets(dataset_arg):
    if dataset_arg == "all":
        return sort_datasets_by_size([d for d in ALL_DATASETS if d in dataset_specs])

    if dataset_arg == "paper":
        return sort_datasets_by_size([d for d in PAPER_DATASETS if d in dataset_specs])

    if dataset_arg not in dataset_specs:
        available = ", ".join(["all", "paper"] + list(dataset_specs.keys()))
        raise ValueError(
            f"Unknown dataset '{dataset_arg}'. Available datasets are: {available}"
        )

    return sort_datasets_by_size([dataset_arg])


def resolve_detectors(detector_arg):
    if detector_arg == "paper":
        return PAPER_DETECTORS
    
    if detector_arg == "baselines":
        return BASELINE_DETECTORS

    if detector_arg == "ablation":
        return ABLATION_DETECTORS
    
    if detector_arg == "component-ablation":
        return list(COMPONENT_ABLATION_CONFIGS.keys())
    
    if detector_arg == "clustering-ablation":
        return list(CLUSTERING_ABLATION_CONFIGS.keys())

    if detector_arg in {"sensitivity", "hyperparameter-sensitivity"}:
        return HYPERPARAMETER_SENSITIVITY_DETECTORS

    if detector_arg == "all-available":
        return list(STATIC_DETECTOR_INFO.keys())

    if detector_arg not in STATIC_DETECTOR_INFO:
        available = ", ".join(STATIC_DETECTOR_INFO.keys())
        raise ValueError(
            f"Unknown detector '{detector_arg}'. Available detectors are: {available}"
        )

    return [detector_arg]



def run_experiment(dataset_name, detector_name, args):
    output_dir = Path(args.output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{detector_name}.pkl"

    if output_file.exists() and not args.overwrite and not args.decision_report_only:
        print(f"[SKIP] {output_file} already exists. Use --overwrite to rerun.")
        return

    params = dataset_specs[dataset_name]

    DATASET = dataset_name
    DATA_TYPE = params["type"]

    try:
        if DATA_TYPE == "synthetic":
            SYNTH_TYPE = params["synth_type"]
            print(f"Generating synthetic stream {SYNTH_TYPE}...")

            gen = util.StreamGenerator()
            stream, sample_size, n_features, n_targets = gen.get_stream(SYNTH_TYPE)

        else:
            print(f"Loading dataset {DATASET}...")

            n_targets = params["n_targets"]
            dataset_path = Path(args.dataset_dir) / f"{DATASET}.arff"

            X, y = load_from_arff(
                str(dataset_path),
                label_count=n_targets,
                label_location="start",
            )

            X = X.toarray()
            y = y.toarray().astype(np.int8)

            sample_size = len(X)
            n_features = X.shape[1]

            if len(np.unique(y)) > 2:
                mlb = MultiLabelBinarizer()
                y = mlb.fit_transform(y)

            stream = DataStream(data=X, y=y, n_targets=n_targets)

        if args.max_samples is not None:
            max_samples = min(sample_size, args.max_samples)
        else:
            max_samples = sample_size


        detector_info = build_detector_info(
            dataset_name=DATASET,
            ld3_params=ld3_params,
            dets_to_run=[detector_name],
        )

        detector_cfg = detector_info[detector_name]
        is_our = is_lace_detector_config(detector_cfg)

        print("=" * 80)
        print(f"Dataset       : {DATASET}")
        print(f"Dataset type  : {DATA_TYPE}")
        print(f"Samples       : {sample_size}")
        print(f"Features      : {n_features}")
        print(f"Labels        : {n_targets}")
        print(f"Detector      : {detector_name}")
        print(f"Config        : {detector_cfg}")
        if args.decision_report_only:
            print(f"Output        : {output_file} (preserved)")
        else:
            print(f"Output        : {output_file}")
        if is_our:
            print(
                "Decision CSV  : "
                f"{_drift_decision_report_path(args, detector_name)}"
            )
        print("=" * 80)

        stream.restart()

        detector = util.build_detector(
            detector_name,
            detector_cfg,
            is_our=is_our,
        )

        pre_sample = [np.zeros(n_features), np.zeros(n_features)]
        pre_label = [np.ones(n_targets), np.zeros(n_targets)]

        started_at = datetime.now()
        t0 = time.perf_counter()

        classifier = ClassifierChain(GaussianNB())
        classifier.fit(np.array(pre_sample), np.array(pre_label))

        results_true = []
        preds = []
        drift_points = []

        ld3_drift = False
        tornado_drift = False

        stream_idx = 0
        processed_idx = 0

        p_bar = tqdm(total=max_samples, desc=f"{DATASET} | {detector_name}")

        while stream_idx < max_samples and stream.has_more_samples():
            X_i, y_i = stream.next_sample()

            if DATA_TYPE == "synthetic" and y_i.ndim == 1:
                y_i = np.array([y_i])

            if np.sum(y_i) == 0:
                stream_idx += 1
                p_bar.update(1)
                continue

            results_true.append(y_i[0].copy())

            drift_detected = False

            if detector_name in TORNADO_DETS and tornado_drift:
                drift_detected = True
                print(f"Drift@ {stream_idx} ({detector_name})")

                detector.reset()
                classifier.reset()
                classifier.fit(np.array(pre_sample), np.array(pre_label))

            elif (
                detector is not None
                and detector_name not in TORNADO_DETS
                and (
                    detector.detected_change()
                    if detector_name != "LD3"
                    else ld3_drift
                )
            ):
                drift_detected = True
                print(f"Drift@ {stream_idx} ({detector_name})")

                if not (is_our or detector_name == "LD3"):
                    detector.reset()

                classifier.reset()
                classifier.fit(np.array(pre_sample), np.array(pre_label))

            if drift_detected:
                drift_points.append(
                    {
                        "stream_idx": stream_idx,
                        "processed_idx": processed_idx,
                    }
                )

            pred = classifier.predict(X_i)
            preds.append(pred[0].copy())

            classifier.partial_fit(X_i, y_i)

            if detector_name == "LD3":
                ld3_drift, warning, correlation, score = detector.update(
                    pred.astype(np.int32)
                )

            elif detector_name in TORNADO_DETS:
                _, tornado_drift = detector.run(
                    pred.astype(np.int32).flatten().tolist()
                    == y_i.astype(np.int32).flatten().tolist()
                )

            elif is_our and detector is not None:
                feed_mode = getattr(detector.cfg, "feed_mode", "supervised")
                detector_input = y_i if feed_mode == "supervised" else pred
                previous_decision_count = len(getattr(detector, "decision_log", []))

                detector.add_element(
                    np.asarray(detector_input).astype(np.int32)
                )
                _annotate_new_decisions(
                    detector,
                    previous_decision_count,
                    stream_idx,
                    processed_idx,
                )

            elif detector is not None:
                detector.add_element(
                    pred.astype(np.int32).flatten().tolist()
                    == y_i.astype(np.int32).flatten().tolist()
                )

            stream_idx += 1
            processed_idx += 1
            p_bar.update(1)

        p_bar.close()

        elapsed_seconds = time.perf_counter() - t0
        ended_at = datetime.now()

        y_true_arr = np.array(results_true)
        preds_arr = np.array(preds)
        drift_decision_log = (
            list(getattr(detector, "decision_log", []))
            if is_our and detector is not None
            else []
        )

        result = {
            "dataset": DATASET,
            "dataset_type": DATA_TYPE,
            "detector": detector_name,
            "detector_params": detector_cfg,
            "drift_points": drift_points,
            "effectiveness_metrics": {
                "accuracy": util.accuracy_example(y_true_arr, preds_arr),
                "hamming_score": util.hamming_score(y_true_arr, preds_arr),
                "f1_example": util.f1_example(y_true_arr, preds_arr),
                "f1_micro": util.f1_micro(y_true_arr, preds_arr),
                "f1_macro": util.f1_macro(y_true_arr, preds_arr),
            },
            "drift_metrics": {
                "n_drifts_detected": len(drift_points),
                "delay_for_detection": util.DFD(drift_points, [4000, 10000] if "Synthetic" in DATASET else None),
                "missed_detection_rate": util.MDR(drift_points, [4000, 10000] if "Synthetic" in DATASET else None),
            },
            "time": {
                "seconds": elapsed_seconds,
                "started_at": started_at.isoformat(),
                "ended_at": ended_at.isoformat(),
                "processed_samples": processed_idx,
                "stream_samples_seen": stream_idx,
            },
            "predictions": preds_arr,
        }

        if is_our and detector is not None:
            write_drift_decision_report(
                args,
                DATASET,
                detector_name,
                drift_decision_log,
            )

        if args.decision_report_only:
            print(f"[PRESERVED] Prediction output was not written: {output_file}")
        else:
            with open(output_file, "wb") as f:
                pickle.dump(result, f)
            print(f"[SAVED] {output_file}")

        print("Metrics:")
        for metric_name, metric_value in result["effectiveness_metrics"].items():
            print(f"  {metric_name}: {metric_value}")
        print("Drift Metrics:")
        for metric_name, metric_value in result["drift_metrics"].items():
            print(f"  {metric_name}: {metric_value}")

        stream.restart()

    except Exception as e:
        print(f"[ERROR] An error occurred while processing {DATASET} with {detector_name}: {e}")
        raise


def _spawn_worker(dataset_name, detector_name, args):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--dataset", dataset_name,
        "--detector", detector_name,
        "--dataset-dir", args.dataset_dir,
        "--output-dir", args.output_dir,
        "--decision-output-dir", args.decision_output_dir,
    ]

    if args.overwrite:
        cmd.append("--overwrite")

    if args.decision_report_only:
        cmd.append("--decision-report-only")

    if args.max_samples is not None:
        cmd += ["--max-samples", str(args.max_samples)]

    threads = str(args.threads_per_worker)
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "VECLIB_MAXIMUM_THREADS": threads,
        "NUMEXPR_NUM_THREADS": threads,
    })

    print(f"\n>>> {dataset_name} / {detector_name}")
    subprocess.run(cmd, check=True, env=env)


def _load_result(output_dir, dataset_name, detector_name):
    pkl_path = Path(output_dir) / dataset_name / f"{detector_name}.pkl"
    if not pkl_path.exists():
        return None

    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _round_metric(value):
    if value is None:
        return None
    return round(float(value), 6)


def write_summary(args, datasets, detectors):
    summary_path = Path(args.output_dir) / "effectiveness_time_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset_name in datasets:
        for detector_name in detectors:
            result = _load_result(args.output_dir, dataset_name, detector_name)
            if result is None:
                continue

            effectiveness = result.get("effectiveness_metrics", {})
            timing = result.get("time", {})

            rows.append({
                "dataset": dataset_name,
                "detector": detector_name,
                "accuracy": _round_metric(effectiveness.get("accuracy")),
                "f1_micro": _round_metric(effectiveness.get("f1_micro")),
                "f1_macro": _round_metric(effectiveness.get("f1_macro")),
                "seconds": _round_metric(timing.get("seconds")),
                "processed_samples": timing.get("processed_samples"),
                "stream_samples_seen": timing.get("stream_samples_seen"),
                "started_at": timing.get("started_at"),
                "ended_at": timing.get("ended_at"),
            })

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "detector",
                "accuracy",
                "f1_micro",
                "f1_macro",
                "seconds",
                "processed_samples",
                "stream_samples_seen",
                "started_at",
                "ended_at",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[SUMMARY] {summary_path}")


def main():
    args = parse_args()

    selected_datasets = resolve_datasets(args.dataset)
    selected_detectors = resolve_detectors(args.detector)

    if args.worker:
        run_experiment(selected_datasets[0], selected_detectors[0], args)
        return

    print("Selected datasets:", selected_datasets)
    print("Selected detectors:", selected_detectors)
    print(
        "Parallel execution: "
        f"{args.parallel_workers} workers x {args.threads_per_worker} threads"
    )

    jobs = [
        (dataset_name, detector_name)
        for dataset_name in selected_datasets
        for detector_name in selected_detectors
    ]

    with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor:
        futures = {
            executor.submit(_spawn_worker, dataset_name, detector_name, args): (
                dataset_name,
                detector_name,
            )
            for dataset_name, detector_name in jobs
        }

        for future in as_completed(futures):
            dataset_name, detector_name = futures[future]
            try:
                future.result()
            except Exception:
                print(f"[FAILED] {dataset_name} / {detector_name}")
                raise

    write_summary(args, selected_datasets, selected_detectors)


if __name__ == "__main__":
    main()
