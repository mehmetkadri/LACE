from __future__ import annotations

from typing import Dict, List

import numpy as np

Partition = Dict[str, int]

TESTING_RUN = False  # Set to True to enable detailed debug prints for clustering steps


def window_to_matrix_transpose(window: List[List[str]], debug: bool = False, debug_label: str = "") -> np.ndarray:
    """Create a transposed window matrix.

    Args:
        window: List of instances, each instance is a list of labels
        debug: Whether to print debug information
        debug_label: Label for debug output

    Returns:
        transposed_matrix: 2D numpy array where entry (i, j) indicates if label i is present in instance j
    """
    # Build label index mapping
    universe: List[str] = sorted({lab for inst in window for lab in inst}, key=lambda s: int(s[1:]))
    label_to_idx = {lab: i for i, lab in enumerate(universe)}
    n_labels = len(universe)
    n_instances = len(window)

    # Initialize transposed matrix
    transposed_matrix = np.zeros((n_labels, n_instances), dtype=int)

    # Fill transposed matrix
    for j, inst in enumerate(window):
        for lab in inst:
            i = label_to_idx[lab]
            transposed_matrix[i, j] = 1

    if debug:
        print(f"[C3M] Original window with {n_instances} instances:")
        for i, inst in enumerate(window):
            print(f"[C3M]   Instance {i}: {inst}")
        print(f"[C3M] Creating transposed matrix for window with {n_instances} instances")
        for i, lab in enumerate(universe):
            print(f"[C3M]   {lab}: {transposed_matrix[i].tolist()}")
    return transposed_matrix


def calculate_s_matrix(d_matrix_param: np.ndarray) -> np.ndarray:
    """Calculate the S matrix from the D (co-occurrence) matrix.

    S[i][j] = D[i][j] / sum_k D[i][k] 

    Args:
        d_matrix: Co-occurrence matrix where D[i][j] counts how many instances contain both label i and label j

    Returns:
        s_matrix: Similarity matrix based on co-occurrence
    """
    # Calculate row sums for normalization
    row_sums = d_matrix_param.sum(axis=1, keepdims=True)
    s_matrix = d_matrix_param / row_sums
    return s_matrix

def calculate_sp_matrix(d_matrix_param: np.ndarray) -> np.ndarray:
    """Calculate the S' matrix from the D (co-occurrence) matrix.

    S'[i][j] = D[i][j] / sum_k D[k][j] 

    Args:
        d_matrix: Co-occurrence matrix where D[i][j] counts how many instances contain both label i and label j

    Returns:
        sp_matrix: Similarity prime matrix based on co-occurrence
    """
    # Calculate column sums for normalization
    col_sums = d_matrix_param.sum(axis=0, keepdims=True)
    sp_matrix = d_matrix_param / col_sums
    return sp_matrix

def calculate_c_matrix(s_matrix_param: np.ndarray, spt_matrix_param: np.ndarray) -> np.ndarray:
    """Calculate the C matrix from the S matrix.

    C = S dot S

    Args:
        s_matrix: Similarity matrix based on co-occurrence
        sp_matrix: Similarity prime matrix based on co-occurrence

    Returns:
        c_matrix: Matrix used for clustering in C3M
    """
    c_matrix = s_matrix_param @ spt_matrix_param
    return c_matrix

def calculate_matrices(window: List[List[str]]) -> np.ndarray:
    d_matrix = window_to_matrix_transpose(window)
    s_matrix = calculate_s_matrix(d_matrix)
    sp_matrix = calculate_sp_matrix(d_matrix)
    spt_matrix = sp_matrix.T
    c_matrix = calculate_c_matrix(s_matrix, spt_matrix)

    return c_matrix

def calculate_cluster_count(
    c_matrix: np.ndarray
    ) -> int:
    # returns the C matrix and the estimated cluster count

    n_c_est = np.trace(c_matrix)
    n_clusters = max(1, int(round(n_c_est)))

    return n_clusters

def c3m_clustering(
    c_matrix: np.ndarray,
    cluster_count: int,
    labels: List[str]
) -> Partition:
    """ Perform C3M clustering on labels in a window.

    Args:
        c_matrix: The C matrix used for clustering in C3M
        cluster_count: The number of clusters to form
        labels: A list of all possible labels
    Returns:
        partition: Dict mapping labels to cluster IDs based on C3M clustering
    """

    # sort labels by diagonal descending
    diag = np.diag(c_matrix)
    label_indices = list(range(len(diag)))
    label_indices.sort(key=lambda i: diag[i], reverse=True)

    # choose centers, skipping false-seed duplicates
    centers = []
    for idx in label_indices:
        if len(centers) >= cluster_count:
            break

        is_duplicate = False
        for c in centers:
            same_seed = (
                np.isclose(c_matrix[idx, idx], c_matrix[idx, c]) and
                np.isclose(c_matrix[idx, idx], c_matrix[c, idx]) and
                np.isclose(c_matrix[idx, idx], c_matrix[c, c])
            )
            if same_seed:
                is_duplicate = True
                break

        if not is_duplicate:
            centers.append(idx)


    partition: Partition = {}

    center_set = set(centers)

    # Force each center to map to its own cluster ID
    for k, center_idx in enumerate(centers):
        label = labels[center_idx] if labels is not None else str(center_idx)
        partition[label] = k

    # Assign only non-center labels
    for i in range(c_matrix.shape[0]):
        if i in center_set:
            continue

        best_center_pos = max(
            range(len(centers)),
            key=lambda k: c_matrix[i, centers[k]]
        )
        label = labels[i] if labels is not None else str(i)
        partition[label] = best_center_pos

    # change centers from indices to labels in the debug output
    center_labels = [labels[idx] if labels is not None else str(idx) for idx in centers]
    return partition, center_labels
