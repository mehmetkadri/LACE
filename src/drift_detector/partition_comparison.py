from typing import Dict
from sklearn.metrics import adjusted_rand_score
import math
from collections import defaultdict

Partition = Dict[str, int]

def adjusted_rand_index(p1: Partition, p2: Partition) -> float:

    labels1 = []
    labels2 = []
    common_labels = set(p1.keys()) & set(p2.keys())
    for label in common_labels:
        labels1.append(p1[label])
        labels2.append(p2[label])


    if not labels1 or not labels2:
        return 0.0

    return adjusted_rand_score(labels1, labels2)
    

def variation_of_information(p_old: dict, p_new: dict, log_base: float = 2.0):
    """
    Compute VI and normalized VI from two partitions given as:
        {element: cluster_id}

    Example:
        p_old = {'l1': 0, 'l17': 0, 'l18': 0, 'l19': 0, 'l22': 0, 'l5': 0}
        p_new = {'l1': 0, 'l17': 1, 'l18': 0, 'l19': 0, 'l22': 0, 'l5': 0}

    Returns:
        {
            'vi': ...,
            'normalized_vi': ...,
            'h_old': ...,
            'h_new': ...,
            'mutual_information': ...,
            'n': ...
        }
    """
    # --- basic checks ---
    if set(p_old.keys()) != set(p_new.keys()):
        raise ValueError("p_old and p_new must contain exactly the same elements.")

    elements = list(p_old.keys())
    n = len(elements)

    if n == 0:
        raise ValueError("Partitions must contain at least one element.")

    if log_base <= 0 or log_base == 1:
        raise ValueError("log_base must be positive and not equal to 1.")

    def log_fn(x):
        return math.log(x, log_base)

    # --- build clusters ---
    old_clusters = defaultdict(list)
    new_clusters = defaultdict(list)

    for elem in elements:
        old_clusters[p_old[elem]].append(elem)
        new_clusters[p_new[elem]].append(elem)

    old_cluster_sets = {cid: set(members) for cid, members in old_clusters.items()}
    new_cluster_sets = {cid: set(members) for cid, members in new_clusters.items()}

    # --- entropy H(U) and H(V) ---
    def entropy(cluster_dict):
        h = 0.0
        for members in cluster_dict.values():
            prob = len(members) / n
            h -= prob * log_fn(prob)
        return h

    h_old = entropy(old_clusters)
    h_new = entropy(new_clusters)

    # --- mutual information I(U, V) ---
    mi = 0.0
    for old_id, old_members in old_cluster_sets.items():
        a_i = len(old_members)
        for new_id, new_members in new_cluster_sets.items():
            b_j = len(new_members)
            n_ij = len(old_members & new_members)

            if n_ij == 0:
                continue

            p_ij = n_ij / n
            mi += p_ij * log_fn((n * n_ij) / (a_i * b_j))

    # --- VI ---
    vi = h_old + h_new - 2.0 * mi

    # Numeric cleanup for tiny negative values like -1e-16
    if abs(vi) < 1e-12:
        vi = 0.0

    # --- normalized VI ---
    # Using log(n) normalization, which is convenient for fixed element count
    max_scale = log_fn(n)
    normalized_vi = vi / max_scale if max_scale > 0 else 0.0

    return {
        "vi": vi,
        "normalized_vi": normalized_vi,
        "h_old": h_old,
        "h_new": h_new,
        "mutual_information": mi,
        "n": n,
    }