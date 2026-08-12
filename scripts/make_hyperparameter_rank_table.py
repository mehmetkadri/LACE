"""Create a mixed rank/value hyperparameter-sensitivity table.

The predictive metrics are average ranks: configurations are ranked separately
within each real dataset (average ranks are assigned to ties), and those ranks
are averaged by configuration.  MDR, DFD, and runtime per item remain averages
of their original values rather than ranks.  Lower is better in every column.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("outputs/csv/hyperparameter_sensitivity/all_metrics_long.csv")
DEFAULT_CSV_OUTPUT = Path(
    "outputs/csv/hyperparameter_sensitivity/all_metrics_average_ranks.csv"
)
DEFAULT_LATEX_OUTPUT = Path(
    "outputs/latex/hyperparameter_average_ranks_top10.tex"
)

TOP_K = 10
SYNTHETIC_PREFIX = "synthetic"
EXCLUDED_REAL_DATASETS = {
    "20NG",
    "Bibtex",
    "Eurlex",
    "Yelp",
    "Plant",
    "yeast",
    "Enron",
    "CAL500",
}

PREDICTIVE_METRICS = [
    "accuracy",
    "hamming_score",
    "f1_example",
    "f1_micro",
]
DRIFT_METRICS = ["missed_detection_rate", "delay_for_detection"]
RUNTIME_METRIC = "runtime_per_item_cs"
RAW_VALUE_METRICS = DRIFT_METRICS + [RUNTIME_METRIC]

# True means that a smaller raw value receives the better rank.
LOWER_IS_BETTER = {
    "accuracy": False,
    "hamming_score": False,
    "f1_example": False,
    "f1_micro": False,
    "missed_detection_rate": True,
    "delay_for_detection": True,
    RUNTIME_METRIC: True,
}

METHOD_PATTERN = re.compile(
    r"^LACE-C3M_w(?P<w>\d+)_(?P<tau_l>\d{3})_(?P<tau_u>\d{3})_"
    r"(?P<tau_nvi>\d{3})NVI$"
)

LATEX_HEADERS = {
    "w": r"$w$",
    "tau_l": r"$\tau_l$",
    "tau_u": r"$\tau_u$",
    "tau_nvi": r"$\tau_{NVI}$",
    "effectiveness_rank": "Effectiveness",
    "accuracy": r"$Accuracy_{ex}$",
    "hamming_score": "Hamming",
    "f1_example": r"$F1_{ex}$",
    "f1_micro": r"$F1_{micro}$",
    "missed_detection_rate": "MDR",
    "delay_for_detection": "DFD",
    RUNTIME_METRIC: "Runtime per Item (cs)",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--latex-output", type=Path, default=DEFAULT_LATEX_OUTPUT)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    return parser.parse_args()


def parse_method(method: str) -> dict[str, float]:
    match = METHOD_PATTERN.fullmatch(method)
    if match is None:
        raise ValueError("Cannot parse hyperparameters from method: {}".format(method))
    return {
        "w": int(match.group("w")),
        "tau_l": int(match.group("tau_l")) / 100.0,
        "tau_u": int(match.group("tau_u")) / 100.0,
        "tau_nvi": int(match.group("tau_nvi")) / 100.0,
    }


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("Input CSV not found: {}".format(path))

    data = pd.read_csv(path)
    required = {"dataset", "method", "time_per_sample", *PREDICTIVE_METRICS, *DRIFT_METRICS}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError("Input CSV is missing columns: {}".format(missing))
    if data.duplicated(["dataset", "method"]).any():
        duplicates = data.loc[
            data.duplicated(["dataset", "method"], keep=False),
            ["dataset", "method"],
        ]
        raise ValueError("Duplicate dataset/method rows:\n{}".format(duplicates.head(10)))

    numeric_columns = PREDICTIVE_METRICS + DRIFT_METRICS + ["time_per_sample"]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    data[RUNTIME_METRIC] = data["time_per_sample"] * 100.0
    return data


def rows_for_metric(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    is_synthetic = data["dataset"].str.lower().str.startswith(SYNTHETIC_PREFIX)
    if metric in PREDICTIVE_METRICS:
        return data.loc[
            ~is_synthetic & ~data["dataset"].isin(EXCLUDED_REAL_DATASETS)
        ]
    if metric in DRIFT_METRICS:
        return data.loc[is_synthetic]
    return data


def mean_rank_for_metric(data: pd.DataFrame, metric: str) -> pd.Series:
    selected = rows_for_metric(data, metric)
    available = selected.dropna(subset=[metric]).copy()
    if available.empty:
        raise ValueError("Metric {} has no applicable numeric values.".format(metric))

    available["dataset_rank"] = available.groupby("dataset")[metric].rank(
        method="average",
        ascending=LOWER_IS_BETTER[metric],
    )
    return available.groupby("method")["dataset_rank"].mean()


def raw_mean_for_metric(data: pd.DataFrame, metric: str) -> pd.Series:
    selected = rows_for_metric(data, metric)
    return selected.groupby("method")[metric].mean()


def compute_summary(data: pd.DataFrame) -> pd.DataFrame:
    methods = sorted(data["method"].unique())
    summary = pd.DataFrame(index=pd.Index(methods, name="method"))
    for metric in PREDICTIVE_METRICS:
        summary[metric] = mean_rank_for_metric(data, metric).reindex(methods)
    for metric in RAW_VALUE_METRICS:
        summary[metric] = raw_mean_for_metric(data, metric).reindex(methods)

    summary["effectiveness_rank"] = summary[PREDICTIVE_METRICS].mean(axis=1)
    parameters = pd.DataFrame(
        [parse_method(method) for method in summary.index],
        index=summary.index,
    )
    output = pd.concat([parameters, summary], axis=1)
    column_order = [
        "w",
        "tau_l",
        "tau_u",
        "tau_nvi",
        "effectiveness_rank",
        *PREDICTIVE_METRICS,
        *RAW_VALUE_METRICS,
    ]
    output = output[column_order].assign(_method=output.index)
    return output.sort_values(
        ["effectiveness_rank", "accuracy", "_method"],
        kind="stable",
    ).drop(columns="_method")


def highlighted_values(values: pd.Series, decimals: int = 2) -> list[str]:
    numeric = values.dropna()
    unique = np.sort(numeric.unique())
    best = unique[0] if len(unique) else np.nan
    second = unique[1] if len(unique) > 1 else np.nan
    formatted = []
    for value in values:
        if pd.isna(value):
            formatted.append("--")
        elif np.isclose(value, best):
            formatted.append(r"\textbf{{{:.{decimals}f}}}".format(value, decimals=decimals))
        elif np.isclose(value, second):
            formatted.append(r"\underline{{{:.{decimals}f}}}".format(value, decimals=decimals))
        else:
            formatted.append("{:.{decimals}f}".format(value, decimals=decimals))
    return formatted


def make_latex_table(summary: pd.DataFrame, top_k: int) -> str:
    table = summary.head(top_k).copy()
    display = pd.DataFrame(index=table.index)
    display["w"] = table["w"].astype(int).astype(str)
    for column in ["tau_l", "tau_u", "tau_nvi"]:
        display[column] = table[column].map("{:.2f}".format)
    for column in ["effectiveness_rank", *PREDICTIVE_METRICS]:
        display[column] = highlighted_values(table[column])
    display["missed_detection_rate"] = highlighted_values(
        table["missed_detection_rate"], decimals=4
    )
    display["delay_for_detection"] = highlighted_values(
        table["delay_for_detection"], decimals=2
    )
    display[RUNTIME_METRIC] = highlighted_values(table[RUNTIME_METRIC], decimals=2)

    headers = [LATEX_HEADERS[column] for column in display.columns]
    alignment = "r" * len(headers)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Top hyperparameter settings by average predictive rank. "
        r"Predictive metrics are average ranks; MDR, DFD, and runtime per item "
        r"are average values. Lower is better.}",
        r"\label{tab:hyperparameter_average_ranks}",
        r"\begin{tabular}{" + alignment + "}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for _, row in display.iterrows():
        lines.append(" & ".join(str(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_arguments()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    data = load_results(args.input)
    summary = compute_summary(data)

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    summary.reset_index().to_csv(args.csv_output, index=False)

    latex = make_latex_table(summary, args.top_k)
    args.latex_output.parent.mkdir(parents=True, exist_ok=True)
    args.latex_output.write_text(latex, encoding="utf-8")

    print("Wrote {} configuration summaries to {}".format(len(summary), args.csv_output))
    print("Wrote top-{} LaTeX table to {}".format(args.top_k, args.latex_output))
    print("\nTop configurations:\n{}".format(summary.head(args.top_k).to_string()))


if __name__ == "__main__":
    main()
