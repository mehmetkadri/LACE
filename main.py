import argparse
import pickle
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
    'Plant': {
        'type': 'real',
        'n_targets': 12,
        'n_instances': 978
    },
    'Gnegative': {
        'type': 'real',
        'n_targets': 8,
        'n_instances': 1392
    },
    'Enron': {
        'type': 'real',
        'n_targets': 53,
        'n_instances': 1702
    },
    'yeast': {
        'type': 'real',
        'n_targets': 14,
        'n_instances': 2417
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
    'EUKARYOTEPseAAC': {
        'window_size': 50,
        'correlation_thresh': 2,
        'len': 1
    },
    'PLANTPseAAC': {
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
    'Enron',
    'Eukaryote',
    'Gnegative',
    'Human',
    'Imdb',
    'Mediamill',
    'OHSUMED-F',
    'Plant',
    'Slashdot',
    'tmc2007-500',
    'yeast',
]

dataset_specs = {
    k: v for k, v in dataset_specs.items()
    if k in PAPER_DATASETS
}

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

BASE_DETECTOR = {
    "n_permutations": 1000,
    "seed": 42,
}


STATIC_DETECTOR_INFO = {
    # Main method
    "LACE": {
        **BASE_DETECTOR,
        "clustering_method": "c3m",
        "similarity_metric": "ari",
        "matrix_type": "transposed_window",
        "feed_mode": "unsupervised",
        "debug_name": "LACE",
        "window_size": None,
        "step": None,
    },

    "LACE-Agglomerative_Variant": {
        **BASE_DETECTOR,
        "clustering_method": "agglomerative",
        "similarity_metric": "ari",
        "linkage_method": "average",
        "distance_calculation_strategy": "1-(cosine)",
        "feed_mode": "unsupervised",
        "debug_name": "LACE-Agglomerative_Variant",
        "window_size": None,
        "step": None,
    },

    "LACE-KMeans_Variant": {
        **BASE_DETECTOR,
        "clustering_method": "kmeans",
        "similarity_metric": "ari",
        "matrix_type": "transposed_window",
        "feed_mode": "unsupervised",
        "debug_name": "LACE-KMeans_Variant",
        "window_size": None,
        "step": None,
    },

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

def build_detector_info(dataset_name, ld3_params, dets_to_run=None):
    if dets_to_run is None:
        dets_to_run = PAPER_DETECTORS

    detector_info = {}

    lace_ws = 250 if dataset_specs[dataset_name]["n_instances"] < 1500 else 1000

    for detector_name in dets_to_run:
        cfg = deepcopy(STATIC_DETECTOR_INFO[detector_name])

        # Dataset-dependent params for LACE detectors
        if detector_name in {
            "LACE",
            "LACE-Agglomerative_Variant",
            "LACE-KMeans_Variant",
        }:
            cfg["window_size"] = lace_ws
            cfg["step"] = lace_ws // 2

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
        default="all",
        help="Dataset name or 'all'. Example: Birds, Enron, Synthetic_Sudden2, all.",
    )

    parser.add_argument(
        "--detector",
        type=str,
        default="LACE",
        help=(
            "Detector name, 'paper', or 'all-available' (including ablations). "
            "Example: LACE, LD3, RDDM, ND, paper, all-available."
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
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of stream samples to process for quick tests.",
    )

    return parser.parse_args()


def resolve_datasets(dataset_arg):
    if dataset_arg == "all":
        return list(dataset_specs.keys())

    if dataset_arg not in dataset_specs:
        available = ", ".join(dataset_specs.keys())
        raise ValueError(
            f"Unknown dataset '{dataset_arg}'. Available datasets are: {available}"
        )

    return [dataset_arg]


def resolve_detectors(detector_arg):
    if detector_arg == "paper":
        return PAPER_DETECTORS

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

    if output_file.exists() and not args.overwrite:
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

        print("=" * 80)
        print(f"Dataset       : {DATASET}")
        print(f"Dataset type  : {DATA_TYPE}")
        print(f"Samples       : {sample_size}")
        print(f"Features      : {n_features}")
        print(f"Labels        : {n_targets}")
        print(f"Detector      : {detector_name}")
        print(f"Config        : {detector_cfg}")
        print(f"Output        : {output_file}")
        print("=" * 80)

        stream.restart()

        is_our = "LACE" in detector_name

        detector = util.build_detector(
            detector_name,
            detector_cfg,
            is_our=is_our,
        )

        pre_sample = [np.zeros(n_features), np.zeros(n_features)]
        pre_label = [np.ones(n_targets), np.zeros(n_targets)]

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

                detector.add_element(
                    np.asarray(detector_input).astype(np.int32)
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


        y_true_arr = np.array(results_true)
        preds_arr = np.array(preds)

        result = {
            "dataset": DATASET,
            "dataset_type": DATA_TYPE,
            "detector": detector_name,
            "detector_params": detector_cfg,
            "drift_points": drift_points,
            "metrics": {
                "accuracy": util.accuracy_example(y_true_arr, preds_arr),
                "f1_micro": util.f1_micro(y_true_arr, preds_arr),
                "f1_macro": util.f1_macro(y_true_arr, preds_arr),
            },
            "predictions": preds_arr,
        }

        with open(output_file, "wb") as f:
            pickle.dump(result, f)

        print(f"[SAVED] {output_file}")
        print("Metrics:")
        for metric_name, metric_value in result["metrics"].items():
            print(f"  {metric_name}: {metric_value}")

        stream.restart()

    except Exception as e:
        print(f"[ERROR] An error occurred while processing {DATASET} with {detector_name}: {e}")
        raise

def main():
    args = parse_args()

    selected_datasets = resolve_datasets(args.dataset)
    selected_detectors = resolve_detectors(args.detector)

    print("Selected datasets:", selected_datasets)
    print("Selected detectors:", selected_detectors)

    for dataset_name in selected_datasets:
        for detector_name in selected_detectors:
            run_experiment(dataset_name, detector_name, args)


if __name__ == "__main__":
    main()