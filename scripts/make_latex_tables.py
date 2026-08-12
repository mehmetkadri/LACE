from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

CSV_ROOT = Path("outputs/csv")
LATEX_ROOT = Path("outputs/latex")
PER_DATA_LATEX_ROOT = LATEX_ROOT / "per_data"
LATEX_ROOT.mkdir(parents=True, exist_ok=True)

FAMILIES = {
    "baseline": CSV_ROOT / "baseline",
    "hyperparameter_sensitivity": CSV_ROOT / "hyperparameter_sensitivity",
    "ablation": CSV_ROOT / "ablation",
}

# Main effectiveness metrics used for mean-rank ordering.
# Current recommended version: exclude macro-F1.
EFFECTIVENESS_RANK_METRICS = [
    "accuracy",
    "hamming_score",
    "f1_example",
    "f1_micro",
    # "f1_macro",  # uncomment if you want macro-F1 included again
]

# Metrics to show in the effectiveness table.
EFFECTIVENESS_DISPLAY_METRICS = [
    "accuracy",
    "hamming_score",
    "f1_example",
    "f1_micro",
    # "f1_macro",
]

DRIFT_DISPLAY_METRICS = [
    "missed_detection_rate",
    "delay_for_detection",
]

TIME_DISPLAY_METRICS = [
    "time",
    "time_per_item_cs",
]

SUMMARY_PREDICTIVE_RANK_METRICS = [
    "accuracy",
    "hamming_score",
    "f1_example",
    "f1_micro",
]

SUMMARY_DISPLAY_METRICS = [
    "effectiveness_mean_rank",
    *SUMMARY_PREDICTIVE_RANK_METRICS,
    *DRIFT_DISPLAY_METRICS,
    *TIME_DISPLAY_METRICS,
]

SUMMARY_METRIC_LABELS = {
    "effectiveness_mean_rank": "Overall",
    "accuracy": r"$Accuracy_{ex}$",
    "hamming_score": "Hamming",
    "f1_example": r"$F1_{ex}$",
    "f1_micro": r"$F1_{micro}$",
    "missed_detection_rate": "MDR",
    "delay_for_detection": "DFD",
    "time": "Time (s)",
    "time_per_item_cs": "Time/Item (cs)",
}

# Datasets excluded from final baseline effectiveness comparison.
# Adjust this list depending on the selected setup.
EXCLUDED_EFFECTIVENESS_DATASETS = {
    "20NG",
    "Bibtex",
    "Eurlex",
    "Yelp",
    "Plant",
    "yeast",
    "Enron",
    "CAL500",
}

# Synthetic datasets are used for drift-detection metrics.
SYNTHETIC_PREFIX = "Synthetic"

# Number of hyperparameter variants to show in the paper table.
TOP_K_HYPERPARAMETER_VARIANTS = 10

# Keep this aligned with BEST_WINDOW_KEY/BEST_THRESHOLD_KEY/BEST_NVI_KEY in
# /Users/mehmetkadrigofralilar/Desktop/LACE/main.py.
CONFIGURED_BEST_LACE_VARIANT = "LACE-C3M_w500_005_050_050NVI"

# Lower is better for these metrics.
LOWER_IS_BETTER = {
    "n_drifts_detected": True,
    "delay_for_detection": True,
    "missed_detection_rate": True,
    "time": True,
    "time_per_sample": True,
    "time_per_item_cs": True,
    "avg_rank": True,
    "effectiveness_mean_rank": True,
}

EFFECTIVENESS_METRIC_SET = set(EFFECTIVENESS_RANK_METRICS) | set(EFFECTIVENESS_DISPLAY_METRICS)
DRIFT_METRIC_SET = set(DRIFT_DISPLAY_METRICS) | {"n_drifts_detected"}
TIME_METRIC_SET = set(TIME_DISPLAY_METRICS) | {"time_per_sample"}

# Higher is better for all remaining effectiveness metrics.
METRIC_LABELS = {
    "accuracy": r"$Accuracy_{ex}$",
    "hamming_score": "Hamming",
    "f1_example": r"$F1_{ex}$",
    "f1_micro": r"$F1_{micro}$",
    "f1_macro": "Macro-F1",
    "n_drifts_detected": "\\#Drifts",
    "delay_for_detection": "DFD",
    "missed_detection_rate": "MDR",
    "time": "Time",
    "time_per_sample": "Time/Sample",
    "time_per_item_cs": "Time/Item (cs)",
    "avg_rank": "Avg. Rank",
    "effectiveness_mean_rank": "Effectiveness",
}

DERIVED_METRIC_SOURCES = {
    "time_per_item_cs": "time_per_sample",
}

CSV_TABLES_TO_SKIP = {
    "all_metrics_long",
}

COMPONENT_ABLATION_METHOD_PATTERN = re.compile(r"^C3M_.*$")
COMPONENT_ABLATION_PARSE_PATTERN = re.compile(
    r"^C3M_(?P<mc>MC|NoMC)_(?P<nvi>NVI|NoNVI)(?:_sim(?P<tau>\d{3}))?$"
)
CLUSTERING_ABLATION_METHOD_PATTERN = re.compile(r"^LACE[-_].*_best$")

CHECK_MARK = r"$\checkmark$"
X_MARK = r"$\times$"

HIGHLIGHT_RTOL = 1e-10
HIGHLIGHT_ATOL = 1e-12


# ============================================================
# Basic helpers
# ============================================================

def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "$": r"\$",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def pretty_method_name(method: str) -> str:
    if method == "LACE":
        return r"\textbf{LACE}"
    return latex_escape(method)


def check_or_x(value: bool) -> str:
    return CHECK_MARK if value else X_MARK


def parse_component_ablation_method(method: str) -> Dict[str, str]:
    match = COMPONENT_ABLATION_PARSE_PATTERN.fullmatch(method)

    if not match:
        return {
            "Monte Carlo": "--",
            "NVI": "--",
            "Tau": "--",
        }

    tau = match.group("tau")

    return {
        "Monte Carlo": check_or_x(match.group("mc") == "MC"),
        "NVI": check_or_x(match.group("nvi") == "NVI"),
        "Tau": X_MARK if tau is None else f"{int(tau) / 100:.2f}",
    }


def parse_clustering_ablation_method(method: str) -> Dict[str, str]:
    labels = {
        "LACE-KMeans_best": "KMeans",
        "LACE-C3M_best": r"$C^3$M",
        "LACE-Agglo_best": "Agglomerative",
    }

    return {"Clustering": labels.get(method, latex_escape(method))}


def build_method_descriptor_table(
    methods: Sequence[str],
    descriptor_kind: Optional[str],
    default_column_name: str,
) -> pd.DataFrame:
    display = pd.DataFrame(index=list(methods))

    if descriptor_kind == "component_ablation":
        parsed = [
            parse_component_ablation_method(method)
            for method in methods
        ]
        return pd.DataFrame(parsed, index=list(methods))

    if descriptor_kind == "clustering_ablation":
        parsed = [
            parse_clustering_ablation_method(method)
            for method in methods
        ]
        return pd.DataFrame(parsed, index=list(methods))

    display[default_column_name] = [pretty_method_name(method) for method in methods]
    return display


def direction_arrow(lower_is_better: bool) -> str:
    return ""


def pretty_metric_name(
    metric: str,
    *,
    include_arrow: bool = False,
    lower_is_better: Optional[bool] = None,
) -> str:
    label = METRIC_LABELS.get(metric, latex_escape(metric))

    if not include_arrow:
        return label

    if lower_is_better is None:
        lower_is_better = is_lower_better(metric)

    return f"{label} {direction_arrow(lower_is_better)}".strip()


def safe_latex_identifier(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9:-]+", "-", str(text)).strip("-").lower()


def is_lower_better(metric: str) -> bool:
    if "rank" in metric.lower():
        return True

    return LOWER_IS_BETTER.get(metric, False)


def metric_dataset_scope(metric: str) -> str:
    if metric in EFFECTIVENESS_METRIC_SET:
        return "effectiveness"

    if metric in DRIFT_METRIC_SET:
        return "drift"

    if metric in TIME_METRIC_SET:
        return "time"

    return "all"


def filter_metric_columns(metric: str, table: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce the dataset split used by the paper tables:
    effectiveness -> retained real datasets,
    drift detection -> synthetic datasets,
    efficiency/time -> all available datasets.
    """

    scope = metric_dataset_scope(metric)

    if scope == "effectiveness":
        columns = [
            dataset
            for dataset in table.columns
            if (
                not dataset.startswith(SYNTHETIC_PREFIX)
                and dataset not in EXCLUDED_EFFECTIVENESS_DATASETS
            )
        ]
        return table.loc[:, columns]

    if scope == "drift":
        columns = [
            dataset
            for dataset in table.columns
            if dataset.startswith(SYNTHETIC_PREFIX)
        ]
        return table.loc[:, columns]

    return table


def format_number(value: float, metric: str) -> str:
    if pd.isna(value):
        return "--"

    if metric in {"delay_for_detection", "time"}:
        return f"{value:.2f}"

    if metric == "time_per_sample":
        return f"{value:.6f}"

    if metric == "time_per_item_cs":
        return f"{value:.2f}"

    if metric in {"avg_rank", "effectiveness_mean_rank"}:
        return f"{value:.4f}"

    return f"{value:.4f}"


def read_metric_table(family_dir: Path, metric: str) -> Optional[pd.DataFrame]:
    path = family_dir / f"{metric}.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df.apply(pd.to_numeric, errors="coerce")


def is_component_ablation_method(method: str) -> bool:
    return COMPONENT_ABLATION_METHOD_PATTERN.fullmatch(method) is not None


def is_clustering_ablation_method(method: str) -> bool:
    return CLUSTERING_ABLATION_METHOD_PATTERN.fullmatch(method) is not None


def is_ablation_method(method: str) -> bool:
    return (
        is_component_ablation_method(method)
        or is_clustering_ablation_method(method)
    )


def filter_tables_by_method(
    tables: Dict[str, pd.DataFrame],
    method_predicate: Callable[[str], bool],
) -> Dict[str, pd.DataFrame]:
    filtered = {}

    for metric, table in tables.items():
        methods = [
            method
            for method in table.index
            if method_predicate(method)
        ]

        if methods:
            filtered[metric] = table.loc[methods]

    return filtered


def split_ablation_tables(
    ablation_tables: Dict[str, pd.DataFrame],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    return {
        "component_ablation": filter_tables_by_method(
            ablation_tables,
            is_component_ablation_method,
        ),
        "clustering_ablation": filter_tables_by_method(
            ablation_tables,
            is_clustering_ablation_method,
        ),
    }


def filter_family_table_methods(
    family: str,
    table: pd.DataFrame,
) -> pd.DataFrame:
    if family != "ablation":
        return table

    allowed_methods = [
        method
        for method in table.index
        if is_ablation_method(method)
    ]

    return table.loc[allowed_methods]


def read_family_tables(
    family: str,
    metrics: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    family_dir = FAMILIES[family]
    tables = {}
    requested_metrics = set(metrics)
    metrics_to_read = set(metrics)

    for metric in requested_metrics:
        source_metric = DERIVED_METRIC_SOURCES.get(metric)
        if source_metric is not None:
            metrics_to_read.add(source_metric)

    for metric in sorted(metrics_to_read):
        table = read_metric_table(family_dir, metric)
        if table is not None:
            table = filter_family_table_methods(family, table)
            tables[metric] = filter_metric_columns(metric, table)

    if "time_per_item_cs" in requested_metrics and "time_per_sample" in tables:
        tables["time_per_item_cs"] = tables["time_per_sample"] * 100.0

    for metric in list(tables):
        if metric not in requested_metrics:
            del tables[metric]

    return tables


def discover_metric_names(family: str) -> List[str]:
    family_dir = FAMILIES[family]

    if not family_dir.exists():
        return []

    metric_names = [
        path.stem
        for path in family_dir.glob("*.csv")
        if path.stem not in CSV_TABLES_TO_SKIP
    ]

    for derived_metric, source_metric in DERIVED_METRIC_SOURCES.items():
        if source_metric in metric_names:
            metric_names.append(derived_metric)

    return sorted(set(metric_names))


def common_datasets(
    tables: Dict[str, pd.DataFrame],
    metrics: Sequence[str],
    exclude: Iterable[str] = (),
    synthetic_only: bool = False,
    real_only: bool = False,
) -> List[str]:
    available = None

    for metric in metrics:
        if metric not in tables:
            continue

        cols = set(tables[metric].columns)
        available = cols if available is None else available & cols

    if available is None:
        return []

    excluded = set(exclude)
    datasets = sorted(d for d in available if d not in excluded)

    if synthetic_only:
        datasets = [d for d in datasets if d.startswith(SYNTHETIC_PREFIX)]

    if real_only:
        datasets = [d for d in datasets if not d.startswith(SYNTHETIC_PREFIX)]

    return datasets


def common_methods(
    tables: Dict[str, pd.DataFrame],
    metrics: Sequence[str],
) -> List[str]:
    available = None

    for metric in metrics:
        if metric not in tables:
            continue

        index = set(tables[metric].index)
        available = index if available is None else available & index

    if available is None:
        return []

    return sorted(available)


# ============================================================
# Ranking and aggregation
# ============================================================

def compute_metric_mean_ranks(
    tables: Dict[str, pd.DataFrame],
    metrics: Sequence[str],
    methods: Sequence[str],
    datasets: Sequence[str],
) -> pd.DataFrame:
    """
    For each metric:
        rank methods within each dataset, then average ranks over datasets.

    Returns:
        rows = methods
        columns = per-metric mean ranks + effectiveness_mean_rank
    """

    result = pd.DataFrame(index=list(methods))

    for metric in metrics:
        if metric not in tables:
            continue

        table = tables[metric].reindex(index=methods, columns=datasets)
        ascending = is_lower_better(metric)

        ranks = table.rank(
            axis=0,
            method="average",
            ascending=ascending,
        )

        result[metric] = ranks.mean(axis=1, skipna=True)

    result["effectiveness_mean_rank"] = result[
        [m for m in metrics if m in result.columns]
    ].mean(axis=1, skipna=True)

    return result.sort_values("effectiveness_mean_rank", ascending=True)


def compute_raw_means(
    tables: Dict[str, pd.DataFrame],
    metrics: Sequence[str],
    methods: Sequence[str],
    datasets: Sequence[str],
) -> pd.DataFrame:
    """
    Returns raw metric averages over datasets.
    """

    result = pd.DataFrame(index=list(methods))

    for metric in metrics:
        if metric not in tables:
            continue

        table = tables[metric].reindex(index=methods, columns=datasets)
        result[metric] = table.mean(axis=1, skipna=True)

    return result


def select_best_lace_variant(
    hyper_tables: Dict[str, pd.DataFrame],
    rank_metrics: Sequence[str],
    excluded_datasets: Iterable[str],
) -> Tuple[str, pd.DataFrame]:
    """
    Selects the configured LACE variant used as the baseline LACE row.
    Falls back to the lowest effectiveness mean-rank variant only if the
    configured variant is absent from the hyperparameter tables.

    Ranking is computed only among hyperparameter variants.
    """

    datasets = common_datasets(
        hyper_tables,
        rank_metrics,
        exclude=excluded_datasets,
        real_only=True,
    )

    if not datasets:
        raise RuntimeError("No real datasets found for hyperparameter selection.")

    methods = sorted(
        set.intersection(
            *[
                set(hyper_tables[m].index)
                for m in rank_metrics
                if m in hyper_tables
            ]
        )
    )

    if not methods:
        raise RuntimeError("No hyperparameter methods found.")

    rank_table = compute_metric_mean_ranks(
        tables=hyper_tables,
        metrics=rank_metrics,
        methods=methods,
        datasets=datasets,
    )

    if CONFIGURED_BEST_LACE_VARIANT in rank_table.index:
        best_variant = CONFIGURED_BEST_LACE_VARIANT
    else:
        best_variant = rank_table.index[0]

    return best_variant, rank_table


def combine_baseline_with_best_lace(
    baseline_tables: Dict[str, pd.DataFrame],
    hyper_tables: Dict[str, pd.DataFrame],
    best_variant: str,
    metrics: Sequence[str],
    lace_display_name: str = "LACE",
) -> Dict[str, pd.DataFrame]:
    """
    Adds the selected LACE variant as a row named 'LACE' to baseline tables.
    """

    combined = {}

    for metric in metrics:
        if metric not in baseline_tables:
            continue

        baseline_table = baseline_tables[metric].copy()

        if metric not in hyper_tables:
            combined[metric] = baseline_table
            continue

        hyper_table = hyper_tables[metric]

        if best_variant not in hyper_table.index:
            combined[metric] = baseline_table
            continue

        lace_row = hyper_table.loc[[best_variant]].copy()
        lace_row.index = [lace_display_name]

        if lace_display_name in baseline_table.index:
            baseline_table = baseline_table.drop(index=lace_display_name)

        combined[metric] = pd.concat([lace_row, baseline_table], axis=0)

    return combined


# ============================================================
# Highlighting and LaTeX writing
# ============================================================

def apply_best_second_highlighting(
    display_df: pd.DataFrame,
    numeric_df: pd.DataFrame,
    metric_columns: Sequence[str],
    lower_is_better_overrides: Optional[Mapping[str, bool]] = None,
) -> pd.DataFrame:
    """
    Bold best value and underline second-best value for each numeric column.

    For lower-better metrics, smaller is better.
    For higher-better metrics, larger is better.
    """

    highlighted = display_df.copy()
    lower_is_better_overrides = lower_is_better_overrides or {}

    for metric in metric_columns:
        if metric not in numeric_df.columns or metric not in highlighted.columns:
            continue

        values = numeric_df[metric].dropna().astype(float)

        if values.empty:
            continue

        lower = lower_is_better_overrides.get(metric, is_lower_better(metric))
        sorted_values = values.sort_values(ascending=lower)
        best_group: Optional[Tuple[float, str]] = None
        second_group: Optional[Tuple[float, str]] = None

        for idx, value in sorted_values.items():
            text = str(highlighted.loc[idx, metric])

            if best_group is None:
                best_group = (value, text)
                continue

            if _same_highlight_group(value, text, best_group):
                continue

            if second_group is None:
                second_group = (value, text)
                break

        if best_group is None:
            continue

        for idx, value in numeric_df[metric].items():
            if pd.isna(value):
                continue

            text = str(highlighted.loc[idx, metric])

            if _same_highlight_group(float(value), text, best_group):
                highlighted.loc[idx, metric] = rf"\textbf{{{text}}}"
            elif second_group is not None and _same_highlight_group(
                float(value),
                text,
                second_group,
            ):
                highlighted.loc[idx, metric] = rf"\underline{{{text}}}"

    return highlighted


def _same_highlight_group(
    value: float,
    text: str,
    group: Tuple[float, str],
) -> bool:
    group_value, group_text = group
    return text == group_text or np.isclose(
        value,
        group_value,
        rtol=HIGHLIGHT_RTOL,
        atol=HIGHLIGHT_ATOL,
    )


def write_latex_table(
    df: pd.DataFrame,
    output_path: Path,
    caption: str,
    label: str,
    column_format: Optional[str] = None,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if column_format is None:
        column_format = "l" * len(df.columns)

    lines = [
        r"\begin{table}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
        " & ".join(str(column) for column in df.columns) + r" \\",
        r"\midrule",
    ]

    for row in df.itertuples(index=False, name=None):
        lines.append(" & ".join(str(value) for value in row) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )

    latex = "\n".join(lines)

    output_path.write_text(latex, encoding="utf-8")


def write_latex_table_with_grouped_header(
    df: pd.DataFrame,
    output_path: Path,
    caption: str,
    label: str,
    grouped_header: Sequence[str],
    cmidrules: Sequence[Tuple[int, int]],
    column_format: str,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{table}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
        " & ".join(grouped_header) + r" \\",
        " ".join(
            rf"\cmidrule(lr){{{start}-{end}}}"
            for start, end in cmidrules
        ),
        " & ".join(str(column) for column in df.columns) + r" \\",
        r"\midrule",
    ]

    for row in df.itertuples(index=False, name=None):
        lines.append(" & ".join(str(value) for value in row) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )

    latex = "\n".join(lines)
    output_path.write_text(latex, encoding="utf-8")


def build_display_table(
    numeric_df: pd.DataFrame,
    metric_columns: Sequence[str],
    method_column_name: str = "Method",
    highlight: bool = True,
    lower_is_better_overrides: Optional[Mapping[str, bool]] = None,
) -> pd.DataFrame:
    leading_columns = build_method_descriptor_table(
        methods=list(numeric_df.index),
        descriptor_kind=None,
        default_column_name=method_column_name,
    )

    return build_display_table_with_leading_columns(
        numeric_df=numeric_df,
        metric_columns=metric_columns,
        leading_columns=leading_columns,
        highlight=highlight,
        lower_is_better_overrides=lower_is_better_overrides,
    )


def build_display_table_with_leading_columns(
    numeric_df: pd.DataFrame,
    metric_columns: Sequence[str],
    leading_columns: pd.DataFrame,
    highlight: bool = True,
    lower_is_better_overrides: Optional[Mapping[str, bool]] = None,
) -> pd.DataFrame:
    lower_is_better_overrides = lower_is_better_overrides or {}
    display = leading_columns.reindex(numeric_df.index).copy()

    for metric in metric_columns:
        if metric not in numeric_df.columns:
            continue

        display[metric] = [
            format_number(v, metric)
            for v in numeric_df[metric]
        ]

    if highlight:
        display = apply_best_second_highlighting(
            display_df=display,
            numeric_df=numeric_df,
            metric_columns=metric_columns,
            lower_is_better_overrides=lower_is_better_overrides,
        )

    rename_cols = {
        metric: pretty_metric_name(
            metric,
            include_arrow=True,
            lower_is_better=lower_is_better_overrides.get(
                metric,
                is_lower_better(metric),
            ),
        )
        for metric in metric_columns
        if metric in display.columns
    }

    display = display.rename(columns=rename_cols)
    return display.reset_index(drop=True)


def summary_metric_header(metric: str, lower_is_better: bool) -> str:
    label = SUMMARY_METRIC_LABELS.get(metric, pretty_metric_name(metric))
    return f"{label} {direction_arrow(lower_is_better)}".strip()


def build_summary_numeric_table(
    tables: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rank_metrics = [
        metric
        for metric in SUMMARY_PREDICTIVE_RANK_METRICS
        if metric in tables
    ]

    if not rank_metrics:
        return pd.DataFrame()

    methods = common_methods(tables, rank_metrics)

    if not methods:
        return pd.DataFrame()

    eff_datasets = common_datasets(
        tables,
        rank_metrics,
        exclude=EXCLUDED_EFFECTIVENESS_DATASETS,
        real_only=True,
    )

    rank_table = compute_metric_mean_ranks(
        tables=tables,
        metrics=rank_metrics,
        methods=methods,
        datasets=eff_datasets,
    )

    predictive_columns = [
        metric
        for metric in [
            "effectiveness_mean_rank",
            *SUMMARY_PREDICTIVE_RANK_METRICS,
        ]
        if metric in rank_table.columns
    ]

    summary_parts = [rank_table[predictive_columns]]

    drift_metrics = [
        metric
        for metric in DRIFT_DISPLAY_METRICS
        if metric in tables
    ]

    if drift_metrics:
        drift_datasets = common_datasets(
            tables,
            drift_metrics,
            synthetic_only=True,
        )
        summary_parts.append(
            compute_raw_means(
                tables=tables,
                metrics=drift_metrics,
                methods=methods,
                datasets=drift_datasets,
            )
        )

    time_metrics = [
        metric
        for metric in TIME_DISPLAY_METRICS
        if metric in tables
    ]

    if time_metrics:
        time_datasets = common_datasets(
            tables,
            time_metrics,
        )
        summary_parts.append(
            compute_raw_means(
                tables=tables,
                metrics=time_metrics,
                methods=methods,
                datasets=time_datasets,
            )
        )

    summary = pd.concat(summary_parts, axis=1)

    summary_columns = [
        metric
        for metric in SUMMARY_DISPLAY_METRICS
        if metric in summary.columns
    ]

    return summary[summary_columns].sort_values(
        "effectiveness_mean_rank",
        ascending=True,
    )


def build_summary_display_table(
    numeric_df: pd.DataFrame,
    use_hyperparameter_columns: bool = False,
    descriptor_kind: Optional[str] = None,
) -> pd.DataFrame:
    metric_columns = [
        metric
        for metric in SUMMARY_DISPLAY_METRICS
        if metric in numeric_df.columns
    ]

    display = pd.DataFrame(index=numeric_df.index)

    if use_hyperparameter_columns:
        parsed = pd.DataFrame(
            [parse_lace_variant(method) for method in numeric_df.index],
            index=numeric_df.index,
        )
        display["$w$"] = parsed["w"].map(
            lambda value: "--" if pd.isna(value) else str(int(value))
        )
        display["$\\tau_l$"] = parsed["tau_l"].map(
            lambda value: "--" if pd.isna(value) else f"{value:.2f}"
        )
        display["$\\tau_u$"] = parsed["tau_u"].map(
            lambda value: "--" if pd.isna(value) else f"{value:.2f}"
        )
        display["$\\tau_{NVI}$"] = parsed["tau_NVI"].map(
            lambda value: "--" if pd.isna(value) else f"{value:.2f}"
        )
    else:
        display = build_method_descriptor_table(
            methods=list(numeric_df.index),
            descriptor_kind=descriptor_kind,
            default_column_name="Variant",
        )

    for metric in metric_columns:
        display[metric] = [
            format_number(value, metric)
            for value in numeric_df[metric]
        ]

    lower_is_better_overrides = {
        metric: True
        for metric in [
            "effectiveness_mean_rank",
            *SUMMARY_PREDICTIVE_RANK_METRICS,
        ]
        if metric in metric_columns
    }

    display = apply_best_second_highlighting(
        display_df=display,
        numeric_df=numeric_df,
        metric_columns=metric_columns,
        lower_is_better_overrides=lower_is_better_overrides,
    )

    display = display.rename(
        columns={
            metric: summary_metric_header(
                metric,
                lower_is_better_overrides.get(metric, is_lower_better(metric)),
            )
            for metric in metric_columns
        }
    )

    return display.reset_index(drop=True)


def make_study_summary_table(
    study: str,
    tables: Dict[str, pd.DataFrame],
    top_k: Optional[int] = None,
    required_method: Optional[str] = None,
):
    if not tables:
        return

    numeric = build_summary_numeric_table(tables)

    if numeric.empty:
        return

    if top_k is not None:
        top_methods = list(numeric.head(top_k).index)

        if (
            required_method is not None
            and required_method in numeric.index
            and required_method not in top_methods
        ):
            top_methods = [required_method] + top_methods

        numeric = numeric.loc[top_methods].copy()

    use_hyperparameter_columns = study == "sensitivity"
    descriptor_kind = (
        study
        if study in {"component_ablation", "clustering_ablation"}
        else None
    )
    display = build_summary_display_table(
        numeric,
        use_hyperparameter_columns=use_hyperparameter_columns,
        descriptor_kind=descriptor_kind,
    )

    captions = {
        "sensitivity": (
            "Top LACE hyperparameter-sensitivity summary. "
            "Predictive-performance columns report mean ranks over retained "
            "real datasets, with Pred. averaging the displayed predictive "
            "mean ranks. MDR and DFD are averaged over synthetic datasets; "
            "Time is reported in seconds and Time/Item in centiseconds over "
            "all available datasets."
        ),
        "baseline": (
            "Baseline summary. Predictive-performance columns report mean "
            "ranks over retained real datasets, with Pred. averaging the "
            "displayed predictive mean ranks. MDR and DFD are averaged over "
            "synthetic datasets; Time is reported in seconds and Time/Item in "
            "centiseconds over all available datasets."
        ),
        "ablation": (
            "Ablation summary. Predictive-performance columns report mean "
            "ranks over retained real datasets, with Pred. averaging the "
            "displayed predictive mean ranks. MDR and DFD are averaged over "
            "synthetic datasets; Time is reported in seconds and Time/Item in "
            "centiseconds over all available datasets."
        ),
        "component_ablation": (
            "Component ablation summary for C3M\\_* variants. "
            "Predictive-performance columns report mean ranks over retained "
            "real datasets, with Pred. averaging the displayed predictive "
            "mean ranks. MDR and DFD are averaged over synthetic datasets; "
            "Time is reported in seconds and Time/Item in centiseconds over "
            "all available datasets."
        ),
        "clustering_ablation": (
            "Clustering ablation summary for LACE-*\\_best variants. "
            "Predictive-performance columns report mean ranks over retained "
            "real datasets, with Pred. averaging the displayed predictive "
            "mean ranks. MDR and DFD are averaged over synthetic datasets; "
            "Time is reported in seconds and Time/Item in centiseconds over "
            "all available datasets."
        ),
    }
    labels = {
        "component_ablation": "tab:component_ablation_summary",
        "clustering_ablation": "tab:clustering_ablation_summary",
    }

    grouped_header = [
        "",
        r"\multicolumn{5}{c}{Predictive Performance Mean Ranks}",
        r"\multicolumn{2}{c}{Drift Detection}",
        r"\multicolumn{2}{c}{Efficiency}",
    ]
    cmidrules = [
        (2, 6),
        (7, 8),
        (9, 10),
    ]
    column_format = "l" + ("r" * (len(display.columns) - 1))

    if use_hyperparameter_columns:
        grouped_header = [
            "",
            "",
            "",
            "",
            r"\multicolumn{5}{c}{Predictive Performance Mean Ranks}",
            r"\multicolumn{2}{c}{Drift Detection}",
            r"\multicolumn{2}{c}{Efficiency}",
        ]
        cmidrules = [
            (5, 9),
            (10, 11),
            (12, 13),
        ]
        column_format = "r" * len(display.columns)
    elif descriptor_kind == "component_ablation":
        grouped_header = [
            "",
            "",
            "",
            r"\multicolumn{5}{c}{Predictive Performance Mean Ranks}",
            r"\multicolumn{2}{c}{Drift Detection}",
            r"\multicolumn{2}{c}{Efficiency}",
        ]
        cmidrules = [
            (4, 8),
            (9, 10),
            (11, 12),
        ]
        column_format = "ccr" + ("r" * (len(display.columns) - 3))

    write_latex_table_with_grouped_header(
        df=display,
        output_path=LATEX_ROOT / f"{study}_summary.tex",
        caption=captions.get(study, f"{latex_escape(study)} summary."),
        label=labels.get(study, f"tab:{safe_latex_identifier(study)}_summary"),
        grouped_header=grouped_header,
        cmidrules=cmidrules,
        column_format=column_format,
    )


# ============================================================
# Hyperparameter table
# ============================================================

LACE_PATTERN = re.compile(
    r"LACE-C3M_w(?P<w>\d+)_(?P<tl>\d{3})_(?P<tu>\d{3})_(?P<nvi>\d{3})NVI"
)


def parse_lace_variant(method: str) -> Dict[str, object]:
    match = LACE_PATTERN.match(method)

    if not match:
        return {
            "w": np.nan,
            "tau_l": np.nan,
            "tau_u": np.nan,
            "tau_NVI": np.nan,
        }

    return {
        "w": int(match.group("w")),
        "tau_l": int(match.group("tl")) / 100.0,
        "tau_u": int(match.group("tu")) / 100.0,
        "tau_NVI": int(match.group("nvi")) / 100.0,
    }


def make_hyperparameter_selection_table(
    hyper_tables: Dict[str, pd.DataFrame],
    best_variant: str,
):
    metrics_needed = (
        EFFECTIVENESS_RANK_METRICS
        + DRIFT_DISPLAY_METRICS
        + TIME_DISPLAY_METRICS
    )

    datasets_real = common_datasets(
        hyper_tables,
        EFFECTIVENESS_RANK_METRICS,
        exclude=EXCLUDED_EFFECTIVENESS_DATASETS,
        real_only=True,
    )

    datasets_synth = common_datasets(
        hyper_tables,
        DRIFT_DISPLAY_METRICS,
        synthetic_only=True,
    )

    datasets_time = common_datasets(
        hyper_tables,
        TIME_DISPLAY_METRICS,
    )

    methods = sorted(hyper_tables[EFFECTIVENESS_RANK_METRICS[0]].index)

    rank_table = compute_metric_mean_ranks(
        tables=hyper_tables,
        metrics=EFFECTIVENESS_RANK_METRICS,
        methods=methods,
        datasets=datasets_real,
    )

    raw_effectiveness = compute_raw_means(
        tables=hyper_tables,
        metrics=EFFECTIVENESS_DISPLAY_METRICS,
        methods=methods,
        datasets=datasets_real,
    )

    raw_drift = compute_raw_means(
        tables=hyper_tables,
        metrics=DRIFT_DISPLAY_METRICS,
        methods=methods,
        datasets=datasets_synth,
    )

    raw_time = compute_raw_means(
        tables=hyper_tables,
        metrics=TIME_DISPLAY_METRICS,
        methods=methods,
        datasets=datasets_time,
    )

    summary = pd.concat(
        [
            rank_table[["effectiveness_mean_rank"]],
            raw_effectiveness,
            raw_drift,
            raw_time,
        ],
        axis=1,
    )

    summary = summary.sort_values("effectiveness_mean_rank", ascending=True)

    # Always include the selected best variant, plus top-K.
    top_methods = list(summary.head(TOP_K_HYPERPARAMETER_VARIANTS).index)

    if best_variant not in top_methods:
        top_methods = [best_variant] + top_methods

    summary = summary.loc[top_methods].copy()

    parsed = pd.DataFrame(
        [parse_lace_variant(m) for m in summary.index],
        index=summary.index,
    )

    numeric = pd.concat([parsed, summary], axis=1)

    # Use compact columns for paper table.
    keep_cols = [
        "w",
        "tau_l",
        "tau_u",
        "tau_NVI",
        "effectiveness_mean_rank",
        "accuracy",
        "hamming_score",
        "f1_example",
        "f1_micro",
        "missed_detection_rate",
        "delay_for_detection",
        "time",
        "time_per_item_cs",
    ]

    keep_cols = [c for c in keep_cols if c in numeric.columns]
    numeric = numeric[keep_cols]

    # Do not bold/underline hyperparameters, only metrics.
    display = pd.DataFrame(index=numeric.index)
    display["Variant"] = [latex_escape(m) for m in numeric.index]

    for col in keep_cols:
        if col in {"w"}:
            display[col] = numeric[col].map(lambda x: "--" if pd.isna(x) else str(int(x)))
        elif col in {"tau_l", "tau_u", "tau_NVI"}:
            display[col] = numeric[col].map(lambda x: "--" if pd.isna(x) else f"{x:.2f}")
        else:
            display[col] = [
                format_number(v, col)
                for v in numeric[col]
            ]

    metric_cols_to_highlight = [
        c for c in keep_cols
        if c not in {"w", "tau_l", "tau_u", "tau_NVI"}
    ]

    display = apply_best_second_highlighting(
        display_df=display,
        numeric_df=numeric,
        metric_columns=metric_cols_to_highlight,
    )

    metric_renames = {
        c: pretty_metric_name(c, include_arrow=True)
        for c in keep_cols
        if c not in {"w", "tau_l", "tau_u", "tau_NVI"}
    }

    display = display.rename(
        columns={
            **metric_renames,
            "w": "$w$",
            "tau_l": "$\\tau_l$",
            "tau_u": "$\\tau_u$",
            "tau_NVI": "$\\tau_{NVI}$",
        }
    )

    write_latex_table(
        df=display.reset_index(drop=True),
        output_path=LATEX_ROOT / "hyperparameter_selection_topk.tex",
        caption=(
            "Top LACE hyperparameter settings. Effectiveness mean rank is "
            "computed over retained real datasets; MDR and DFD are computed "
            "over synthetic datasets."
        ),
        label="tab:hyperparameter_selection",
    )


# ============================================================
# Baseline comparison tables
# ============================================================

def make_baseline_tables(
    combined_tables: Dict[str, pd.DataFrame],
):
    # -------------------------
    # Effectiveness mean-rank table
    # -------------------------

    eff_datasets = common_datasets(
        combined_tables,
        EFFECTIVENESS_RANK_METRICS,
        exclude=EXCLUDED_EFFECTIVENESS_DATASETS,
        real_only=True,
    )

    methods = list(combined_tables[EFFECTIVENESS_RANK_METRICS[0]].index)

    rank_table = compute_metric_mean_ranks(
        tables=combined_tables,
        metrics=EFFECTIVENESS_RANK_METRICS,
        methods=methods,
        datasets=eff_datasets,
    )

    # LACE should always be on top in the final table.
    if "LACE" in rank_table.index:
        ordered = ["LACE"] + [m for m in rank_table.index if m != "LACE"]
        rank_table = rank_table.loc[ordered]

    metric_cols = ["effectiveness_mean_rank"] + [
        m for m in EFFECTIVENESS_RANK_METRICS
        if m in rank_table.columns
    ]

    display = build_display_table(
        numeric_df=rank_table,
        metric_columns=metric_cols,
        highlight=True,
        lower_is_better_overrides={metric: True for metric in metric_cols},
    )

    write_latex_table(
        df=display,
        output_path=LATEX_ROOT / "baseline_effectiveness_mean_ranks.tex",
        caption=(
            "Predictive effectiveness comparison using mean ranks over "
            "retained real datasets. Lower ranks are better."
        ),
        label="tab:baseline_effectiveness_mean_ranks",
    )

    # -------------------------
    # Raw effectiveness averages
    # -------------------------

    raw_eff = compute_raw_means(
        tables=combined_tables,
        metrics=EFFECTIVENESS_DISPLAY_METRICS,
        methods=methods,
        datasets=eff_datasets,
    )

    raw_eff = raw_eff.reindex(rank_table.index)

    display = build_display_table(
        numeric_df=raw_eff,
        metric_columns=[m for m in EFFECTIVENESS_DISPLAY_METRICS if m in raw_eff.columns],
        highlight=True,
    )

    write_latex_table(
        df=display,
        output_path=LATEX_ROOT / "baseline_effectiveness_raw_means.tex",
        caption=(
            "Raw predictive effectiveness averages over retained real datasets. "
            "Higher values are better."
        ),
        label="tab:baseline_effectiveness_raw",
    )

    # -------------------------
    # Drift-detection table
    # -------------------------

    drift_datasets = common_datasets(
        combined_tables,
        DRIFT_DISPLAY_METRICS,
        synthetic_only=True,
    )

    raw_drift = compute_raw_means(
        tables=combined_tables,
        metrics=DRIFT_DISPLAY_METRICS,
        methods=methods,
        datasets=drift_datasets,
    )

    raw_drift = raw_drift.reindex(rank_table.index)

    display = build_display_table(
        numeric_df=raw_drift,
        metric_columns=[m for m in DRIFT_DISPLAY_METRICS if m in raw_drift.columns],
        highlight=True,
    )

    write_latex_table(
        df=display,
        output_path=LATEX_ROOT / "baseline_drift_detection.tex",
        caption=(
            "Drift-detection performance on synthetic datasets. "
            "Lower MDR and DFD are better."
        ),
        label="tab:baseline_drift_detection",
    )

    # -------------------------
    # Runtime table
    # -------------------------

    time_datasets = common_datasets(
        combined_tables,
        TIME_DISPLAY_METRICS,
    )

    raw_time = compute_raw_means(
        tables=combined_tables,
        metrics=TIME_DISPLAY_METRICS,
        methods=methods,
        datasets=time_datasets,
    )

    raw_time = raw_time.reindex(rank_table.index)

    display = build_display_table(
        numeric_df=raw_time,
        metric_columns=[m for m in TIME_DISPLAY_METRICS if m in raw_time.columns],
        highlight=True,
    )

    write_latex_table(
        df=display,
        output_path=LATEX_ROOT / "baseline_runtime.tex",
        caption=(
            "Runtime comparison over retained real datasets and synthetic datasets. "
            "Time is reported in seconds; time per item is reported in "
            "centiseconds. Lower values are better."
        ),
        label="tab:baseline_runtime",
    )

    # -------------------------
    # Compact final table
    # -------------------------

    compact = pd.DataFrame(index=rank_table.index)
    compact["effectiveness_mean_rank"] = rank_table["effectiveness_mean_rank"]

    for metric in ["accuracy", "hamming_score", "f1_example", "f1_micro"]:
        if metric in rank_table.columns:
            compact[metric] = rank_table[metric]

    for metric in DRIFT_DISPLAY_METRICS:
        if metric in raw_drift.columns:
            compact[metric] = raw_drift[metric]

    for metric in TIME_DISPLAY_METRICS:
        if metric in raw_time.columns:
            compact[metric] = raw_time[metric]

    display = build_display_table(
        numeric_df=compact,
        metric_columns=list(compact.columns),
        highlight=True,
        lower_is_better_overrides={
            metric: True
            for metric in ["effectiveness_mean_rank", *EFFECTIVENESS_RANK_METRICS]
            if metric in compact.columns
        },
    )

    write_latex_table(
        df=display,
        output_path=LATEX_ROOT / "baseline_compact_final.tex",
        caption=(
            "Compact baseline comparison. Effectiveness columns are mean ranks "
            "over retained real datasets; drift and runtime columns are raw "
            "averages. Runtime is reported as seconds and centiseconds per item."
        ),
        label="tab:baseline_compact_final",
    )


# ============================================================
# Ablation table
# ============================================================

def make_ablation_table(
    ablation_tables: Dict[str, pd.DataFrame],
    *,
    output_stem: str = "ablation_table",
    caption: Optional[str] = None,
    label: Optional[str] = None,
    descriptor_kind: Optional[str] = None,
):
    if not ablation_tables:
        return

    metrics_for_table = (
        EFFECTIVENESS_DISPLAY_METRICS
        + DRIFT_DISPLAY_METRICS
        + TIME_DISPLAY_METRICS
    )

    metrics_available = [m for m in metrics_for_table if m in ablation_tables]

    if not metrics_available:
        return

    methods = sorted(ablation_tables[metrics_available[0]].index)

    eff_datasets = common_datasets(
        ablation_tables,
        [m for m in EFFECTIVENESS_RANK_METRICS if m in ablation_tables],
        exclude=EXCLUDED_EFFECTIVENESS_DATASETS,
        real_only=True,
    )

    drift_datasets = common_datasets(
        ablation_tables,
        [m for m in DRIFT_DISPLAY_METRICS if m in ablation_tables],
        synthetic_only=True,
    )

    rank_table = compute_metric_mean_ranks(
        tables=ablation_tables,
        metrics=[m for m in EFFECTIVENESS_RANK_METRICS if m in ablation_tables],
        methods=methods,
        datasets=eff_datasets,
    )

    raw_eff = compute_raw_means(
        tables=ablation_tables,
        metrics=[m for m in EFFECTIVENESS_DISPLAY_METRICS if m in ablation_tables],
        methods=methods,
        datasets=eff_datasets,
    )

    raw_drift = compute_raw_means(
        tables=ablation_tables,
        metrics=[m for m in DRIFT_DISPLAY_METRICS if m in ablation_tables],
        methods=methods,
        datasets=drift_datasets,
    )

    raw_time = compute_raw_means(
        tables=ablation_tables,
        metrics=[m for m in TIME_DISPLAY_METRICS if m in ablation_tables],
        methods=methods,
        datasets=common_datasets(
            ablation_tables,
            [m for m in TIME_DISPLAY_METRICS if m in ablation_tables],
        ),
    )

    summary = pd.concat(
        [
            rank_table[["effectiveness_mean_rank"]],
            raw_eff,
            raw_drift,
            raw_time,
        ],
        axis=1,
    ).sort_values("effectiveness_mean_rank")

    metric_cols = list(summary.columns)

    if descriptor_kind is None:
        display = build_display_table(
            numeric_df=summary,
            metric_columns=metric_cols,
            highlight=True,
        )
    else:
        leading_columns = build_method_descriptor_table(
            methods=list(summary.index),
            descriptor_kind=descriptor_kind,
            default_column_name="Method",
        )
        display = build_display_table_with_leading_columns(
            numeric_df=summary,
            metric_columns=metric_cols,
            leading_columns=leading_columns,
            highlight=True,
        )

    write_latex_table(
        df=display,
        output_path=LATEX_ROOT / f"{output_stem}.tex",
        caption=caption or (
            "Ablation results. Effectiveness mean rank is computed over "
            "retained real datasets; raw effectiveness, drift, and runtime "
            "metrics are averaged over their corresponding datasets."
        ),
        label=label or f"tab:{safe_latex_identifier(output_stem)}",
    )


# ============================================================
# Per-dataset metric tables
# ============================================================

def build_per_data_display_table(
    metric: str,
    metric_table: pd.DataFrame,
    descriptor_kind: Optional[str] = None,
    highlight: bool = True,
) -> pd.DataFrame:
    numeric = metric_table.copy()
    dataset_columns = list(numeric.columns)

    display = build_method_descriptor_table(
        methods=list(numeric.index),
        descriptor_kind=descriptor_kind,
        default_column_name="Method",
    )

    for dataset in dataset_columns:
        display[dataset] = [
            format_number(value, metric)
            for value in numeric[dataset]
        ]

    if highlight:
        lower = is_lower_better(metric)
        display = apply_best_second_highlighting(
            display_df=display,
            numeric_df=numeric,
            metric_columns=dataset_columns,
            lower_is_better_overrides={
                dataset: lower
                for dataset in dataset_columns
            },
        )

    display = display.rename(
        columns={
            dataset: (
                f"{latex_escape(dataset)} "
                f"{direction_arrow(is_lower_better(metric))}"
            ).strip()
            for dataset in dataset_columns
        }
    )

    return display.reset_index(drop=True)


def make_per_data_metric_table(
    setting: str,
    metric: str,
    metric_table: pd.DataFrame,
):
    if metric_table.empty:
        return

    descriptor_kind = (
        setting
        if setting in {"component_ablation", "clustering_ablation"}
        else None
    )

    display = build_per_data_display_table(
        metric=metric,
        metric_table=metric_table,
        descriptor_kind=descriptor_kind,
        highlight=True,
    )

    setting_label = {
        "ablation": "Ablation",
        "baseline": "Baseline",
        "clustering_ablation": "Clustering ablation",
        "component_ablation": "Component ablation",
        "sensitivity": "Sensitivity",
    }.get(setting, latex_escape(setting))

    better_text = "Lower" if is_lower_better(metric) else "Higher"
    leading_column_format = "l"

    if descriptor_kind == "component_ablation":
        leading_column_format = "ccr"

    write_latex_table(
        df=display,
        output_path=PER_DATA_LATEX_ROOT / setting / f"{metric}.tex",
        caption=(
            f"{setting_label} per-dataset {pretty_metric_name(metric)} values. "
            f"{better_text} values are better."
        ),
        label=(
            f"tab:per-data-{safe_latex_identifier(setting)}-"
            f"{safe_latex_identifier(metric)}"
        ),
        column_format=leading_column_format + ("r" * len(metric_table.columns)),
    )


def make_per_data_tables(
    table_sets: Mapping[str, Dict[str, pd.DataFrame]],
):
    for setting, tables in table_sets.items():
        if not tables:
            continue

        for metric in sorted(tables):
            make_per_data_metric_table(
                setting=setting,
                metric=metric,
                metric_table=tables[metric],
            )


# ============================================================
# Main
# ============================================================

def main():
    per_data_metrics = sorted(
        set().union(
            *[
                discover_metric_names(family)
                for family in FAMILIES
            ]
        )
    )

    all_needed_metrics = sorted(
        set(
            EFFECTIVENESS_RANK_METRICS
            + EFFECTIVENESS_DISPLAY_METRICS
            + DRIFT_DISPLAY_METRICS
            + TIME_DISPLAY_METRICS
            + per_data_metrics
        )
    )

    baseline_tables = read_family_tables("baseline", all_needed_metrics)
    hyper_tables = read_family_tables("hyperparameter_sensitivity", all_needed_metrics)
    ablation_tables = read_family_tables("ablation", all_needed_metrics)

    if not hyper_tables:
        raise RuntimeError("No hyperparameter-sensitivity CSVs found.")

    best_variant, hyper_rank_table = select_best_lace_variant(
        hyper_tables=hyper_tables,
        rank_metrics=EFFECTIVENESS_RANK_METRICS,
        excluded_datasets=EXCLUDED_EFFECTIVENESS_DATASETS,
    )

    print(f"Configured LACE variant used for baseline row: {best_variant}")
    print(hyper_rank_table.head(10))

    make_hyperparameter_selection_table(
        hyper_tables=hyper_tables,
        best_variant=best_variant,
    )

    if not baseline_tables:
        raise RuntimeError("No baseline CSVs found.")

    combined_baseline_tables = combine_baseline_with_best_lace(
        baseline_tables=baseline_tables,
        hyper_tables=hyper_tables,
        best_variant=best_variant,
        metrics=all_needed_metrics,
        lace_display_name="LACE",
    )

    make_study_summary_table(
        "sensitivity",
        hyper_tables,
        top_k=TOP_K_HYPERPARAMETER_VARIANTS,
        required_method=best_variant,
    )
    make_study_summary_table("baseline", combined_baseline_tables)

    make_baseline_tables(combined_baseline_tables)

    split_ablation_table_sets = {}

    if ablation_tables:
        split_ablation_table_sets = split_ablation_tables(ablation_tables)

        component_ablation_tables = split_ablation_table_sets["component_ablation"]
        clustering_ablation_tables = split_ablation_table_sets["clustering_ablation"]

        make_study_summary_table("component_ablation", component_ablation_tables)
        make_study_summary_table("clustering_ablation", clustering_ablation_tables)

        make_ablation_table(
            component_ablation_tables,
            output_stem="component_ablation_table",
            descriptor_kind="component_ablation",
            caption=(
                "Component ablation results for C3M\\_* variants. "
                "Effectiveness mean rank is computed over retained real "
                "datasets; raw effectiveness, drift, and runtime metrics are "
                "averaged over their corresponding datasets."
            ),
            label="tab:component_ablation_results",
        )
        make_ablation_table(
            clustering_ablation_tables,
            output_stem="clustering_ablation_table",
            descriptor_kind="clustering_ablation",
            caption=(
                "Clustering ablation results for LACE-*\\_best variants. "
                "Effectiveness mean rank is computed over retained real "
                "datasets; raw effectiveness, drift, and runtime metrics are "
                "averaged over their corresponding datasets."
            ),
            label="tab:clustering_ablation_results",
        )

    make_per_data_tables(
        {
            "baseline": combined_baseline_tables,
            "sensitivity": hyper_tables,
            **split_ablation_table_sets,
        }
    )

    print(f"LaTeX tables written to: {LATEX_ROOT}")


if __name__ == "__main__":
    main()
