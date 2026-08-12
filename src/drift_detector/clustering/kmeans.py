from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans

Partition = Dict[str, int]

TESTING_RUN = False  # Set to True to enable detailed debug prints for clustering steps



def k_means_clustering(
    window: List[List[str]],
    cluster_count: int,
    labels: List[str],
    debug: bool = False,
    debug_label: str = "",
) -> Partition:
    """Perform k-means clustering on labels in a window.

    Each label is represented as a binary occurrence vector over the window:
        row = label
        column = instance
        value = 1 if label appears in that instance, else 0

    Then k-means clusters the label vectors.
    """

    if debug:
        print(f"{debug_label} [K-MEANS] Applying k-means clustering with k={cluster_count}")
        print("params: window size:", len(window), "cluster_count:", cluster_count, "labels:", labels)

    # ------------------------------------------------------------
    # 1. Decide which labels should be clustered
    # ------------------------------------------------------------
    if labels is not None:
        universe: List[str] = list(labels)
    else:
        universe: List[str] = sorted({lab for inst in window for lab in inst})

    n_labels = len(universe)
    n_instances = len(window)

    if n_labels == 0:
        if debug:
            print(f"{debug_label} [K-MEANS] Empty label universe. Returning empty partition.")
        return {}, []

    # k-means cannot have more clusters than samples
    k = min(cluster_count, n_labels)

    if k <= 0:
        raise ValueError(f"cluster_count must be positive, got {cluster_count}")

    # ------------------------------------------------------------
    # 2. Build label-instance matrix
    #
    # Shape:
    #   rows    = labels
    #   columns = instances in the window
    #
    # Example:
    #   X[i, t] = 1 if label i appears in instance t
    # ------------------------------------------------------------
    label_to_row = {lab: i for i, lab in enumerate(universe)}

    X = np.zeros((n_labels, n_instances), dtype=np.float32)

    for t, instance_labels in enumerate(window):
        for lab in instance_labels:
            if lab in label_to_row:
                X[label_to_row[lab], t] = 1.0

    if debug:
        print(f"{debug_label} [K-MEANS] universe:", universe)
        print(f"{debug_label} [K-MEANS] label-instance matrix shape:", X.shape)
        print(f"{debug_label} [K-MEANS] label frequencies:", {
            lab: int(X[i].sum()) for i, lab in enumerate(universe)
        })

    # ------------------------------------------------------------
    # 3. Handle degenerate cases
    # ------------------------------------------------------------

    # If there is only one label, it must be one cluster.
    if n_labels == 1:
        partition = {universe[0]: 0}
        centers = [universe[0]]

        if debug:
            print(f"{debug_label} [K-MEANS] Only one label. Partition:", partition)

        return partition, centers

    # If all vectors are identical, sklearn may warn because it cannot
    # create the requested number of distinct clusters.
    unique_vectors = np.unique(X, axis=0)

    if len(unique_vectors) < k:
        if debug:
            print(
                f"{debug_label} [K-MEANS] Only {len(unique_vectors)} distinct label vectors "
                f"exist, reducing k from {k} to {len(unique_vectors)}."
            )

        k = len(unique_vectors)

    if k == 1:
        partition = {lab: 0 for lab in universe}

        # Pick the most frequent label as representative center.
        center_label = universe[int(np.argmax(X.sum(axis=1)))]
        centers = [center_label]

        if debug:
            print(f"{debug_label} [K-MEANS] Degenerate one-cluster partition:", partition)
            print(f"{debug_label} [K-MEANS] center:", centers)

        return partition, centers

    # ------------------------------------------------------------
    # 4. Run k-means
    # ------------------------------------------------------------
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    assigned_clusters = kmeans.fit_predict(X)

    # ------------------------------------------------------------
    # 5. Convert assignments into partition dict
    # ------------------------------------------------------------
    partition: Partition = {
        lab: int(cluster_id)
        for lab, cluster_id in zip(universe, assigned_clusters)
    }

    # ------------------------------------------------------------
    # 6. Choose real labels closest to each centroid as centers
    #
    # KMeans centroids are numeric vectors, not real labels.
    # For interpretability, we choose the closest actual label vector
    # to each centroid.
    # ------------------------------------------------------------
    centers = []

    for cluster_id in range(k):
        member_indices = np.where(assigned_clusters == cluster_id)[0]

        if len(member_indices) == 0:
            continue

        centroid = kmeans.cluster_centers_[cluster_id]

        distances = np.linalg.norm(X[member_indices] - centroid, axis=1)
        closest_member_index = member_indices[int(np.argmin(distances))]

        centers.append(universe[closest_member_index])

    if debug:
        print(f"{debug_label} [K-MEANS] partition:", partition)
        print(f"{debug_label} [K-MEANS] centers:", centers)

        grouped = {}
        for lab, cid in partition.items():
            grouped.setdefault(cid, []).append(lab)

        print(f"{debug_label} [K-MEANS] grouped clusters:")
        for cid, labs in sorted(grouped.items()):
            print(f"{debug_label} [K-MEANS]   cluster {cid}: {labs}")

    return partition, centers
