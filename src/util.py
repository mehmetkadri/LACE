import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

np.float = float

from sklearn.metrics import f1_score

from skmultilearn.dataset import load_from_arff
from sklearn.preprocessing import MultiLabelBinarizer
from skmultiflow.data import MultilabelGenerator, ConceptDriftStream

import pickle

from baselines.ld3 import LD3
from skmultiflow.drift_detection import ADWIN, EDDM, KSWIN, HDDM_W, HDDM_A, DDM
from baselines.tornados.drift_detection import fhddm, fhddms, fhddms_add, mddm_a, mddm_e, mddm_g, rddm, seq_drift2
from src.drift_detector import DriftDetector, DetectorConfig


def accuracy_example(y_true, y_pred):
    N = len(y_true)
    if N == 0:
        return 0.0
    sum_ = 0
    for i in range(N):
        nom = np.logical_and(y_true[i], y_pred[i]).sum()
        denom = np.logical_or(y_true[i], y_pred[i]).sum()
        if denom > 0:
            sum_ += nom / denom
        else:
            sum_ += 1.0
    return sum_ / N

def f1_micro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro', zero_division=0)

def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

def build_detector(det_name, params, is_our):
    if is_our:
        return DriftDetector(config=DetectorConfig(**params))
    if det_name == 'LD3':
        return LD3(**params)
    if det_name == 'ADWIN':
        return ADWIN()
    if det_name == 'EDDM':
        return EDDM()
    if det_name == 'DDM':
        return DDM()
    if det_name == 'KSWIN':
        return KSWIN()
    if det_name == 'HDDM_A':
        return HDDM_A()
    if det_name == 'HDDM_W':
        return HDDM_W()
    if det_name == 'FHDDM':
        return fhddm.FHDDM()
    if det_name == 'FHDDMS':
        return fhddms.FHDDMS()
    if det_name == 'FHDDMS_add':
        return fhddms_add.FHDDMS_add()
    if det_name == 'MDDM_A':
        return mddm_a.MDDM_A()
    if det_name == 'MDDM_E':
        return mddm_e.MDDM_E()
    if det_name == 'MDDM_G':
        return mddm_g.MDDM_G()
    if det_name == 'SeqDrift2':
        return seq_drift2.SeqDrift2ChangeDetector()
    if det_name == 'RDDM':
        return rddm.RDDM()
    if det_name == 'ND':
        return None
    raise ValueError(f'Unknown detector: {det_name}')

def plot_detector_metrics(
    results_by_name,
    results_true,
    dataset_name,
    metrics_to_plot=None,
    save_plots=False,
    output_dir='outputs/plots',
    drift_points_by_name=None,
    top_method=None,
    processed_stream_indices=None,
    legend_position='top',
    cumulative=False
):
    """
    Args:
        results_by_name: dict mapping detector names to prediction lists
        results_true: list of true labels
        dataset_name: name of dataset for title/logging
        metrics_to_plot: list of metric names to plot (e.g., ['accuracy', 'f1_micro', 'f1_macro']).
                        If None, plots all metrics: ['accuracy', 'f1_micro', 'f1_macro']
        save_plots: whether to save the plots
        output_dir: directory to save the plots
        drift_points_by_name: optional dict mapping detector names to drift sample indices
        legend_position: position of the legend ('top', 'bottom')
        cumulative: if True, each point is computed from the start of the stream up to the current bin end.
    """
    # Default to all metrics if not specified
    if metrics_to_plot is None:
        metrics_to_plot = ['accuracy', 'f1_micro', 'f1_macro']
    
    # Set up plotting (for non-pdf saves)
    mpl.rcParams['figure.dpi'] = 600
    
    detector_order = list(results_by_name.keys())
    if len(detector_order) == 0:
        raise ValueError("results_by_name is empty. No detector results to plot.")
    
    color_palette = ["#a30800",'#fc8d59',"#fbc226",'#2c7bb6','#abd9e9']
    marker_palette = ('o', '^', 'x', 's', '*', 'D', 'P', 'v', '<', '>')
    
    def pick_color(idx):
        return color_palette[int(idx) % len(color_palette)]
    
    def pick_marker(idx):
        return marker_palette[int(idx) % len(marker_palette)]
    
    def stream_idx_to_processed_x(stream_idx):
        if processed_stream_indices is None:
            return None

        stream_indices = np.asarray(processed_stream_indices)

        if len(stream_indices) == 0:
            return None

        pos = np.searchsorted(stream_indices, stream_idx, side='left')

        if pos >= len(stream_indices):
            pos = len(stream_indices) - 1

        processed_position = pos + 1

        return (100.0 * float(processed_position)) / float(len(results_true))
    
    n_chunks = min(25, len(results_true))
    if n_chunks <= 0:
        raise ValueError("results_true is empty.")
    
    metrics_by_detector = {
        'accuracy': {},
        'f1_micro': {},
        'f1_macro': {}
    }
    
    for det_name in detector_order:
        preds = results_by_name[det_name]
        usable_len = min(len(preds), len(results_true))
        preds = preds[:usable_len]
        true = results_true[:usable_len]
        chunk_size = usable_len // n_chunks
        if chunk_size == 0:
            raise ValueError(f"Not enough predictions for detector {det_name} to build chunks.")
    
        accs = []
        f1mics = []
        f1macs = []
    
        for j in range(n_chunks):
            if j == n_chunks - 1:
                end = min(len(preds), len(results_true))
            else:
                end = (j + 1) * chunk_size
            if cumulative:
                start = 0
            else:
                start = j * chunk_size
            y_true_chunk = np.array(true[start:end])
            y_pred_chunk = np.array(preds[start:end])
    
            accs.append(np.round(accuracy_example(np.array([y_true_chunk]), np.array([y_pred_chunk])), decimals=4))
            f1mics.append(np.round(f1_micro(y_true_chunk, y_pred_chunk), decimals=4))
            f1macs.append(np.round(f1_macro(y_true_chunk, y_pred_chunk), decimals=4))
    
        metrics_by_detector['accuracy'][det_name] = accs
        metrics_by_detector['f1_micro'][det_name] = f1mics
        metrics_by_detector['f1_macro'][det_name] = f1macs
    
    x_vals = np.linspace(100 / n_chunks, 100, n_chunks)

    legend_locations = {
        'top': 'upper left',
        'bottom': 'lower left'
    }

    if legend_position not in legend_locations:
        raise ValueError("legend_position must be either 'top' or 'bottom'.")
    
    def plot_metric(metric_key, ylabel):
        fig, ax = plt.subplots(figsize=(12, 5))
        endpoint_info = []
        for idx, det_name in enumerate(detector_order):
            detector_color = pick_color(idx)
            y_values = metrics_by_detector[metric_key][det_name]

            line_zorder = 10 if det_name == top_method else 3
            marker_zorder = 11 if det_name == top_method else 4
            alpha = 1.0 if det_name == top_method else 0.85
            linewidth = 2.2 if det_name == top_method else 1.8

            ax.plot(
                x_vals,
                y_values,
                color=detector_color,
                marker=pick_marker(idx),
                linestyle='-',
                linewidth=linewidth,
                alpha=alpha,
                zorder=line_zorder,
                label=det_name,
            )

            endpoint_info.append((det_name, detector_color, float(y_values[-1])))

            if drift_points_by_name is not None and len(results_true) > 0:
                drift_points = drift_points_by_name.get(det_name, [])
                for drift_point in drift_points:
                    drift_point = drift_point.get('processed_idx', None) if isinstance(drift_point, dict) else drift_point
                    if drift_point is None:
                        continue

                    drift_x = (100.0 * float(drift_point)) / float(len(results_true))

                    ax.axvline(
                        x=drift_x,
                        color=detector_color,
                        linestyle='--',
                        linewidth=1.5,
                        alpha=0.5,
                        zorder=1 if det_name != top_method else 2,
                    )

        if "Synth" in dataset_name:
            for real_drift in [4000, 10000]:
                drift_x = stream_idx_to_processed_x(real_drift)

                if drift_x is None:
                    continue

                ax.axvline(
                    x=drift_x,
                    color='black',
                    linestyle='--',
                    linewidth=1.8,
                    alpha=0.8,
                    zorder=0,
                )

                ax.text(
                    drift_x,
                    ax.get_ylim()[0],
                    str(real_drift),
                    rotation=90,
                    va='bottom',
                    ha='right',
                    fontsize=8,
                    color='black',
                )

        ax.set_xlabel('Percentage of data', labelpad=10, fontsize=13, fontweight='bold')
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_ylabel(ylabel, labelpad=10, fontsize=13, fontweight='bold')
        legend = ax.legend(
            loc=legend_locations[legend_position],
            fontsize=11,
            frameon=True,
            framealpha=0.85,
            borderpad=0.45,
            labelspacing=0.40,
            handlelength=1.8
        )

        for text in legend.get_texts():
            text.set_fontweight('bold')

        # Replace legend with endpoint labels and connector lines on the right.
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        if y_range == 0:
            y_range = 1.0
        min_gap = 0.06 * y_range

        # Sort by endpoint value and spread labels to reduce overlap.
        endpoint_info.sort(key=lambda item: item[2])
        placed = []
        for det_name, detector_color, y_end in endpoint_info:
            y_label = y_end
            if placed:
                y_label = max(y_label, placed[-1] + min_gap)
            placed.append(y_label)

        # If top labels overflow, shift all labels down while preserving spacing.
        overflow = placed[-1] - y_max if placed else 0.0
        if overflow > 0:
            placed = [y - overflow for y in placed]
            if placed[0] < y_min:
                underflow = y_min - placed[0]
                placed = [y + underflow for y in placed]

        x_connector_start = 100.0
        x_connector_end = 103.0
        x_text = 103.2
        ax.set_xlim(0, 103)

        for (det_name, detector_color, y_end), y_label in zip(endpoint_info, placed):
            ax.plot(
                [x_connector_start, x_connector_end],
                [y_end, y_label],
                color=detector_color,
                linewidth=1.5,
                alpha=0.9,
            )
            ax.text(
                x_text,
                y_label,
                det_name,
                color=detector_color,
                fontsize=11,
                fontweight='bold',
                va='center',
                ha='left',
            )

        fig.subplots_adjust(right=0.86)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(12)
        
            
        return fig
    
    
    print(f'Plotting results for {dataset_name}...')
    if save_plots:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f'  Plot(s) will be saved to {output_path.resolve()}')
    
    metric_labels = {
        'accuracy': 'Example-based accuracy',
        'f1_micro': 'Micro-averaged F1 score',
        'f1_macro': 'Macro-averaged F1 score'
    }
    
    # Plot only requested metrics
    for metric_key in metrics_to_plot:
        if metric_key not in metric_labels:
            print(f'Warning: Unknown metric "{metric_key}", skipping...')
            continue
        plot_suffix = 'cumulative' if cumulative else 'binned'
        fig = plot_metric(metric_key, metric_labels[metric_key])
        if save_plots:
            fig.savefig(
                output_path / f'{dataset_name}_{metric_key}_{plot_suffix}.png',
                dpi=600,
                bbox_inches='tight',
                pad_inches=0.05
            )
            fig.savefig(
                output_path / f'{dataset_name}_{metric_key}_{plot_suffix}.pdf',
                bbox_inches='tight',
                pad_inches=0.05
            )
        plt.show()


def load_predictions_and_labels(dataset_name, dataset_specs, predictions_dir='outputs/predictions', filter_methods=None, label_location='end'):
    """
    Load predictions from file and true labels from dataset (from scratch).
    
    Args:
        dataset_name: Name of the dataset (e.g., 'Enron', 'Synthetic_Sudden2')
        dataset_specs: Dictionary with dataset specifications
        predictions_dir: Directory where prediction pickle files are stored
        filter_methods: List of methods to include in the analysis
        label_location: Location of labels ('start' or 'end')
    Returns:
        Tuple of (results_by_name, results_true, data_type, drift_points_by_name, processed_stream_indices)
        - results_by_name: dict mapping detector names to prediction lists
        - results_true: list of true labels
        - data_type: 'synthetic' or 'real'
        - drift_points_by_name: dict mapping detector names to drift sample indices
        - processed_stream_indices: original stream/sample index for each kept true label
    """    
    # Load predictions from directory (one file per method)
    dataset_predictions_dir = Path(predictions_dir) / dataset_name
    if not dataset_predictions_dir.exists() or not dataset_predictions_dir.is_dir():
        raise FileNotFoundError(
            f"Prediction directory not found: {dataset_predictions_dir}. "
            f"Expected structure: outputs/predictions/<dataset>/<method>.pkl"
        )

    if filter_methods is not None:
        prediction_files = [
            f for f in dataset_predictions_dir.glob('*.pkl') if f.stem in filter_methods
        ]
    else:
        prediction_files = sorted(dataset_predictions_dir.glob('*.pkl'))

    if len(prediction_files) == 0:
        raise FileNotFoundError(
            f"No prediction files found in {dataset_predictions_dir}. "
            f"Expected at least one <method>.pkl file."
        )

    print(f'Loading predictions from {dataset_predictions_dir}')
    results_by_name = {}
    drift_points_by_name = {}
    data_type = None

    for prediction_file in prediction_files:
        with open(prediction_file, 'rb') as f:
            prediction_data = pickle.load(f)

        detector_name = prediction_data.get('detector_name', prediction_file.stem)
        predictions = prediction_data.get('predictions')

        if predictions is None:
            raise KeyError(f"Missing 'predictions' key in {prediction_file}")

        if hasattr(predictions, 'tolist'):
            predictions = predictions.tolist()

        results_by_name[detector_name] = predictions
        drift_points_by_name[detector_name] = prediction_data.get('drift_points', [])

        file_data_type = prediction_data.get('data_type')
        if data_type is None:
            data_type = file_data_type
        elif file_data_type is not None and file_data_type != data_type:
            raise ValueError(
                f"Inconsistent data_type in prediction files. "
                f"Found both '{data_type}' and '{file_data_type}'."
            )

    if data_type is None:
        data_type = dataset_specs[dataset_name]['type']
    
    # Load true labels from dataset
    if data_type == 'synthetic':
        synth_type = dataset_specs[dataset_name]['synth_type']
        print(f'Generating synthetic stream {synth_type} to get true labels...')

        gen = StreamGenerator()
        stream, sample_size, n_features, n_targets = gen.get_stream(synth_type)

        results_true = []
        original_stream_indices = []

        stream.restart()
        
        sample_count = 0
        while stream.has_more_samples() and sample_count < sample_size:
            X, y = stream.next_sample()

            if y.ndim == 1:
                y = np.array([y])

            results_true.extend(y)
            original_stream_indices.extend([sample_count] * len(y))

            sample_count += 1

    else:
        print(f'Loading dataset {dataset_name} to get true labels...')

        n_targets = dataset_specs[dataset_name]['n_targets']
        X, y = load_from_arff(
            './src/datasets/{}'.format(dataset_name + '.arff'),
            label_count=n_targets,
            label_location=label_location
        )

        X = X.toarray()
        y = y.toarray().astype(np.int8)
        
        if len(np.unique(y)) > 2:
            mlb = MultiLabelBinarizer()
            y = mlb.fit_transform(y)
        
        results_true = y.tolist()
        original_stream_indices = list(range(len(results_true)))
    
    print(f'Loaded {len(results_true)} true labels before zero-label filtering')
    print(f'Loaded predictions for {len(results_by_name)} detectors')

    # Remove all-zero label instances from true labels, while keeping original stream indices.
    def filter_zero_labels_with_indices(true_labels, stream_indices):
        filtered_true = []
        filtered_stream_indices = []

        for true_label, stream_idx in zip(true_labels, stream_indices):
            if np.any(true_label):
                filtered_true.append(true_label)
                filtered_stream_indices.append(stream_idx)

        return filtered_true, filtered_stream_indices
    
    results_true, processed_stream_indices = filter_zero_labels_with_indices(
        results_true,
        original_stream_indices
    )

    print(f'Loaded {len(results_true)} true labels after zero-label filtering')
    
    return results_by_name, results_true, data_type, drift_points_by_name, processed_stream_indices

class StreamGenerator():
    def __init__(self):
        pass

    def get_stream(self, type):
        sample_size = 0
        stream = None
        n_features = 0
        n_targets = 0
        if type == 'sudden1':
            n_features = 200
            n_targets = 50
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=100) 
            s2 = MultilabelGenerator(n_samples=6009, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=10009, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=25, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=25, random_state=0)
            sample_size=20000
        elif type == 'sudden2':
            n_features = 200
            n_targets = 50
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=100) 
            s2 = MultilabelGenerator(n_samples=6009, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=10009, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=1, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=1, random_state=0)
            sample_size=20000
        elif type == 'sudden3':
            n_features = 200
            n_targets = 50
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=100) 
            s2 = MultilabelGenerator(n_samples=6009, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=10009, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=50, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=50, random_state=0)
            sample_size=20000
        elif type == 'gradual1':
            n_features = 200
            n_targets = 50
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=100) 
            s2 = MultilabelGenerator(n_samples=6500, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=15500, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=250, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=250, random_state=0)
            sample_size=20000
        elif type == 'gradual2':
            n_features = 200
            n_targets = 50
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=100) 
            s2 = MultilabelGenerator(n_samples=6500, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=15500, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=500, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=500, random_state=0)
            sample_size=20000
        elif type == 'gradual3':
            n_features = 200
            n_targets = 50
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=100) 
            s2 = MultilabelGenerator(n_samples=6500, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=15500, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=1000, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=1000, random_state=0)
            sample_size=20000
        elif type == 'reoccurring':
            n_features = 200
            n_targets = 50
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0) 
            s2 = MultilabelGenerator(n_samples=6500, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=15500, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=500, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=500, random_state=0)
            sample_size=20000
        return stream, sample_size, n_features, n_targets