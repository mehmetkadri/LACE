from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

Partition = Dict[str, int]

TESTING_RUN = False  # Set to True to enable detailed debug prints for clustering steps


def agglomerative_clustering(
    window: List[List[str]],
    labels: List[str],
    cluster_count: int,
    linkage_method: str = 'average',
    distance_calculation_strategy: str = 'normalized',
    debug: bool = False,
    debug_label: str = "",
) -> Partition:
    """Agglomerative clustering of labels based on co-occurrence in the window.

    This is a more standard hierarchical clustering approach that does not follow
    the SPLIT+MERGE logic of Omiecinski. It simply clusters labels based on their
    co-occurrence patterns in the window, using a distance metric and linkage
    method, and cuts the dendrogram at the number of clusters.

    Args:
        window: list of instances, each instance is list of labels (active label set)
        labels: list of all possible labels
        cluster_count: number of clusters to form
        debug: whether to print debug information
        debug_label: optional label to identify debug output (e.g., "detector1_old")

    Returns:
        Partition mapping label -> cluster_id
    """

    prefix = f"[{debug_label}] " if debug_label else ""

    if not labels:
        return {}

    if len(labels) == 1:
        return {labels[0]: 0}

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)

    # Build co-occurrence matrix: cooc[i][j] = number of instances containing both label i and label j
    cooc = np.zeros((n, n), dtype=float)
    for inst in window:
        idxs = [label_to_idx[l] for l in inst if l in label_to_idx]
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                cooc[idxs[a], idxs[b]] += 1
                cooc[idxs[b], idxs[a]] += 1

    if debug:
        print(f"{prefix}[AGGLO] Labels: {labels} ({n} labels)")
        print(f"{prefix}[AGGLO] Co-occurrence matrix (rows/cols = labels):")
        for i, lab in enumerate(labels):
            print(f"{prefix}[AGGLO]   {lab}: {cooc[i].tolist()}")

    # Convert co-occurrence to pairwise distance using the selected strategy.
    strategy = distance_calculation_strategy.strip().lower()
    if strategy == 'normalized':
        # distance = 1 - cooc / max_cooc, clamped to [0, 1]
        max_cooc = cooc.max()
        if max_cooc == 0:
            # No label ever co-occurs with any other → fully uniform distance of 1
            dist_matrix = np.ones((n, n), dtype=float)
        else:
            dist_matrix = 1.0 - (cooc / max_cooc)
        np.fill_diagonal(dist_matrix, 0.0)
    elif strategy in {'1-(cosine)', 'cosine'}:
        dist_matrix = np.ones((n, n), dtype=float)
        np.fill_diagonal(dist_matrix, 0.0)
        norms = np.linalg.norm(cooc, axis=1)
        for i in range(n):
            for j in range(i + 1, n):
                denom = norms[i] * norms[j]
                if denom == 0.0:
                    dist = 1.0
                else:
                    sim = float(np.dot(cooc[i], cooc[j]) / denom)
                    sim = max(0.0, min(1.0, sim))
                    dist = 1.0 - sim
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
    elif strategy in {'1-(jaccardian)', 'jaccardian', '1-(jaccard)', 'jaccard'}:
        dist_matrix = np.ones((n, n), dtype=float)
        np.fill_diagonal(dist_matrix, 0.0)
        binary = cooc > 0
        for i in range(n):
            for j in range(i + 1, n):
                intersection = float(np.logical_and(binary[i], binary[j]).sum())
                union = float(np.logical_or(binary[i], binary[j]).sum())
                if union == 0.0:
                    dist = 1.0
                else:
                    dist = 1.0 - (intersection / union)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
    else:
        raise ValueError(
            "distance_calculation_strategy must be one of "
            "{'normalized', '1-(cosine)', '1-(jaccardian)'}, "
            f"got {distance_calculation_strategy!r}"
        )

    if debug:
        print(f"{prefix}[AGGLO] Distance strategy: {distance_calculation_strategy}")
        print(f"{prefix}[AGGLO] Distance matrix:")
        for i, lab in enumerate(labels):
            print(f"{prefix}[AGGLO]   {lab}: {[round(x,3) for x in dist_matrix[i].tolist()]}")

    # Agglomerative clustering via scipy (average linkage)
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method=linkage_method)


    # Fixed cluster count (capped at n to avoid scipy errors)
    n_clusters = min(cluster_count, n)
    labels_arr = fcluster(Z, t=n_clusters, criterion='maxclust')
    if debug:
        print(f"{prefix}[AGGLO] Mode=cluster_count({cluster_count}, capped={n_clusters}): {len(set(labels_arr))} clusters formed")

    # fcluster returns 1-indexed IDs; remap to 0-indexed for consistency with the rest of the codebase
    unique_ids = sorted(set(labels_arr.tolist()))
    remap = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    partition: Partition = {lab: remap[int(cid)] for lab, cid in zip(labels, labels_arr)}

    if debug:
        clusters_repr: Dict[int, List[str]] = {}
        for lab, cid in partition.items():
            clusters_repr.setdefault(cid, []).append(lab)
        print(f"{prefix}[AGGLO] Final partition:")
        for cid, labs in sorted(clusters_repr.items()):
            print(f"{prefix}[AGGLO]   Cluster {cid}: {sorted(labs)}")

    return partition

