from __future__ import annotations
import random
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Optional


from .window import SlidingWindow
from .clustering import cluster
from .partition_comparison import adjusted_rand_index as ari
from .partition_comparison import variation_of_information
from .partition_randomization import monte_carlo_simulation

import numpy as np


@dataclass(frozen=True)
class DetectorConfig:
    """Configuration for the minimal drift detector."""

    window_size: int = 1000
    step: int = window_size // 2

    ratio_threshold: float = 0.50 # likelihood threshold for drift detection (monte carlo)
    similarity_upper_threshold: float = 0.50 # upper similarity threshold: above this, assume no drift
    similarity_lower_threshold: float = 0.05   # lower similarity threshold: below this, assume drift detected
    distance_threshold: float = 0.35  # distance threshold for VI: below this, assume no drift (used when similarity_metric='vi')
    similarity_metric: str = 'ari'  # metric for similarity calculation (e.g., 'ari', 'vi)
    clustering_method: str = 'c3m'  # clustering method to use (e.g., 'omiecinski')
    n_permutations: int = 1000
    seed: int = 42

    feed_mode: str = 'unsupervised'  # 'unsupervised' or 'supervised' for any custom detector
    matrix_type: str = 'cooccurrence'  # matrix type for C3M clustering: 'cooccurrence' or 'transposed_window'

    cutoff_point: Optional[float] = None  # Optional cutoff point for clustering (e.g., distance threshold for agglomerative clustering)
    no_clusters: Optional[int] = None  # Optional fixed number of clusters for clustering algorithms that support it
    linkage_method: Optional[str] = None  # Linkage method for agglomerative clustering (e.g., 'ward', 'complete', 'average')
    distance_calculation_strategy: str = 'normalized'  # 'normalized' | '1-(cosine)' | '1-(jaccardian)'

    cluster_debug: bool = False  # print detailed merge traces from clustering

    debug_name: Optional[str] = None  # Optional name to include in debug prints for easier tracing of configurations

@dataclass(frozen=True)
class DriftEvent:
    """A drift event emitted by the detector."""

    t: int
    statistic: float
    ratio: Optional[float] = None


class DriftDetector:
    """Minimal streaming drift detector.

    This detector:
      1) Maintains a sliding window of recent instances (active label sets).
      2) Periodically splits the window into two halves and clusters each half.
      3) Compares the clusterings using a similarity metric (e.g., ARI).
      4) Uses shortcut thresholds to quickly rule out no-drift or drift cases.
      5) For intermediate similarity values, runs a Monte Carlo simulation to estimate 
      the likelihood of the observed similarity under the null hypothesis of no drift.
      6) Emits a DriftEvent if the likelihood is below the configured threshold.

    """

    def __init__(self, config: DetectorConfig):
        self.cfg = config
        self.window = SlidingWindow(config.window_size)
        self.t = 0
        self._since_last_test = 0
        self._rng = random.Random(config.seed)
        self._drift_detected = False  # Flag for detected_change() interface
        self.label_count = None  # Track number of labels for add_element()

    def add_element(self, predictions: np.ndarray):
        """
        Add a prediction vector (numpy array of 0s and 1s) to the detector.
        
        Converts [0, 1, 1, 0, ...] to ['l1', 'l2'] (active label names).
        Compatible with standard drift detector interface.
        """
        # Ensure predictions is 1D
        predictions = np.asarray(predictions).flatten()
        
        # Store label count on first call
        if self.label_count is None:
            self.label_count = len(predictions)
        
        # Convert binary predictions to active label names
        active_indices = np.flatnonzero(predictions)
        label_names = [f"l{j}" for j in active_indices]
        
        # Update the detector
        event = self.update(label_names)
        
        # Store drift flag for next detected_change() call
        self._drift_detected = event is not None
    
    def detected_change(self) -> bool:
        """Check if drift was detected on the last update."""
        drift = self._drift_detected
        self._drift_detected = False  # Reset flag after reading
        return drift

    def update(self, labels: Iterable[str]) -> Optional[DriftEvent]:
        """Ingest one instance (its active label set) and possibly detect drift."""
        self.t += 1
        self.window.push(labels)
        self._since_last_test += 1

        # Warm-up: wait until window is full
        if not self.window.is_full():
            return None

        # Only test periodically
        if self._since_last_test < self.cfg.step:
            return None

        self._since_last_test = 0

        snapshot = self.window.snapshot()
        mid = len(snapshot) // 2
        w_old = snapshot[:mid]
        w_new = snapshot[mid:]


        labels_old = set()
        for inst in w_old:
            labels_old.update(inst)
        labels_new = set()
        for inst in w_new:
            labels_new.update(inst)
        
        common_labels = labels_old & labels_new

        if len(common_labels) == 0:
            return None  # no common labels to compare
        
        config_id = self.cfg.debug_name if self.cfg.debug_name is not None else f"{self.cfg.window_size}_{self.cfg.clustering_method}"
        

        cluster_params = {
            'common_labels': common_labels,
            'clustering': self.cfg.clustering_method,
            'cluster_debug': self.cfg.cluster_debug,
            'cutoff': self.cfg.cutoff_point,
            'cluster_no': self.cfg.no_clusters,
            'linkage_method': self.cfg.linkage_method,
            'distance_calculation_strategy': self.cfg.distance_calculation_strategy,
            'matrix_type': self.cfg.matrix_type
        }

        p_old = cluster(
            w_old,
            debug_label=f"{config_id}_OLD",
            **cluster_params,
        )

        p_new = cluster(
            w_new,
            debug_label=f"{config_id}_NEW",
            **cluster_params,
        )

        # check if either partition is a single cluster (all labels in one cluster) or all singletons (each label in its own cluster). 
        # In these cases, ARI is not meaningful, so we can skip directly to checking the VI.
        if (len(set(p_old.values())) == 1) or (
            len(set(p_new.values())) == 1) or (
                len(set(p_old.values())) == len(set(p_old.keys()))) or (
                len(set(p_new.values())) == len(set(p_new.keys()))):
            vi = variation_of_information(p_old, p_new)
            if vi['normalized_vi'] < self.cfg.distance_threshold:
                return None
            else:
                return DriftEvent(t=self.t, statistic=vi['normalized_vi'])



        # Calculate similarity between partitions
        sim = ari(p_old, p_new)


        ### Shortcut checks based on similarity thresholds to avoid expensive Monte Carlo simulation 
        # when similarity is clearly high (no drift) or low (drift detected).

        ## Check if similarity is above upper threshold (no drift)
        if sim > self.cfg.similarity_upper_threshold:
            return None
        
        ## Check if similarity is below lower threshold (immediate drift)
        if sim < self.cfg.similarity_lower_threshold:
            return DriftEvent(t=self.t, statistic=sim)



        # Intermediate range: run Monte Carlo simulation to calculate the ratio of random permutations 
        # that are less similar than the observed similarity.
        random_less_similar_ratio = monte_carlo_simulation(
            p_0=p_old,
            p_1=p_new,
            observed_sim=sim,
            metric_fn=ari,  # Using ARI metric function
            n_permutations=self.cfg.n_permutations,
            rng=self._rng,
        )
        
        if random_less_similar_ratio <= self.cfg.ratio_threshold:
            return DriftEvent(t=self.t, statistic=sim, ratio=random_less_similar_ratio)
            

        return None