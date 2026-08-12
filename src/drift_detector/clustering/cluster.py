from __future__ import annotations

from typing import Dict, List, Optional, Set

from .kmeans import k_means_clustering
from .agglomerative import agglomerative_clustering
from .c3m import (
    calculate_cluster_count,
    calculate_matrices,
    c3m_clustering,
    sort_labels,
)

Partition = Dict[str, int]

TESTING_RUN = False  # Set to True to enable detailed debug prints for clustering steps


def cluster(window: List[List[str]], common_labels: Optional[Set[str]] = None,
           clustering: str = "c3m", debug_label: str = "", cluster_debug: bool = False,
           **algo_kwargs) -> Partition:
    """Process a window through filtering, elimination, and clustering.

    Handles clustering

    Args:
        window: List of label instances
        common_labels: Set of common labels between windows (for elimination)
        clustering: Which clustering algorithm to use ("c3m" | "agglomerative")
        filter_strategy: "before", "after", or "none"
        eliminate_strategy: "before", "after", or "none"
        debug: General debug flag
        debug_label: Label for debug output
        cluster_debug: Detailed clustering debug output
        **algo_kwargs: Algorithm-specific keyword arguments forwarded to the
            chosen clustering function.
    Returns:
        Partition dict mapping labels to cluster IDs
    """

    cluster_debug = cluster_debug or TESTING_RUN

    window = [[lab for lab in inst if lab in common_labels] for inst in window]
    
    # Remove empty instances after filtering/elimination, as they don't contribute to clustering and may cause issues
    window = [inst for inst in window if inst]


    # Calculate the cluster count
    c_matrix = calculate_matrices(window)
    cluster_count = calculate_cluster_count(c_matrix)
    labels = sort_labels({lab for inst in window for lab in inst})

    # Cluster the window with method-specific kwargs only.
    # This prevents passing incompatible params (e.g., frequency_type to agglomerative).
    if clustering == "agglomerative":
        agglomerative_keys = {"cutoff", "cluster_no", "linkage_method", "distance_calculation_strategy"}
        selected_kwargs = {k: v for k, v in algo_kwargs.items() if k in agglomerative_keys and v is not None}
        partition = agglomerative_clustering(
            window,
            labels=labels,
            cluster_count=cluster_count,
            debug=cluster_debug,
            debug_label=debug_label,
            **selected_kwargs,
        )
    elif clustering == "c3m":
        partition, centers = c3m_clustering(c_matrix, cluster_count, labels)

        if cluster_debug:
            print(f"[{debug_label}] Cluster centers (by label index): {centers}")
            print(f"[{debug_label}] Final partition (label index -> cluster ID): {partition}")
            
    elif clustering == "kmeans":
        partition, centers = k_means_clustering(window, cluster_count, labels=labels, debug=cluster_debug, debug_label=debug_label)
    else:
        raise ValueError(f"Unknown clustering method: {clustering!r}")
    
    return partition
