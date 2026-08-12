import random
from typing import Callable, Dict, List


def permute_partition_fixed_sizes(
    p: Dict[str, int],
    rng: random.Random,
) -> Dict[str, int]:
    """
    Permute labels across clusters while keeping:
      - same labels
      - same number of clusters
      - same cluster sizes
    """
    clusters: Dict[int, List[str]] = {}
    for label, cid in p.items():
        clusters.setdefault(cid, []).append(label)

    # Sort cluster_ids to ensure deterministic order across runs
    cluster_ids = sorted(clusters.keys())
    cluster_sizes = [len(clusters[cid]) for cid in cluster_ids]

    # Sort labels to ensure deterministic order before shuffling
    all_labels = sorted(p.keys())
    rng.shuffle(all_labels)

    new_partition: Dict[str, int] = {}
    idx = 0
    for cid, size in zip(cluster_ids, cluster_sizes):
        for label in all_labels[idx : idx + size]:
            new_partition[label] = cid
        idx += size

    return new_partition


def monte_carlo_simulation(
    p_0: Dict[str, int],
    p_1: Dict[str, int],
    observed_sim: float,
    metric_fn: Callable,
    n_permutations: int,
    rng: random.Random,
) -> float:
    """Run Monte Carlo simulation to estimate the ratio of random permutations that are less similar than the observed similarity.
    
    Args:
        p_0: Reference partition
        p_1: New partition
        observed_sim: Observed similarity between p_0 and p_1
        metric_fn: Metric function for computing similarity (e.g., ari)
        n_permutations: Number of permutations to test
        rng: Random number generator
    
    Returns:
        ratio (float between 0 and 1)
    """
    count = 0
    for _ in range(n_permutations):
        p_perm = permute_partition_fixed_sizes(p_1, rng)
        simulated_sim = metric_fn(p_0, p_perm)
        if simulated_sim <= observed_sim:
            count += 1

    return (count + 1) / (n_permutations + 1)