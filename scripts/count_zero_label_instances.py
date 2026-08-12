from __future__ import annotations

import argparse
import csv
from pathlib import Path


DATASETS = [
    "Birds",
    "Water-qual",
    "Gnegative",
    "Langlog",
    "Human",
    "Slashdot",
    "Eukaryote",
    "OHSUMED-F",
    "tmc2007-500",
    "Mediamill",
    "Imdb",
    "Emotions",
    "VirusGO",
    "Flags",
    "Scene",
    "Genbase",
    "Medical",
    "Corel5k",
    "Stackex",
    "CHD49",
    "Corel16k",
    "Yahoo-com",
]

N_TARGETS = {
    "Birds": 19,
    "Water-qual": 14,
    "Gnegative": 8,
    "Langlog": 75,
    "Human": 14,
    "Slashdot": 22,
    "Eukaryote": 22,
    "OHSUMED-F": 23,
    "tmc2007-500": 22,
    "Mediamill": 101,
    "Imdb": 28,
    "Emotions": 6,
    "VirusGO": 6,
    "Flags": 7,
    "Scene": 6,
    "Genbase": 27,
    "Medical": 45,
    "Corel5k": 374,
    "Stackex": 227,
    "CHD49": 6,
    "Corel16k": 153,
    "Yahoo-com": 33,
}


def is_nonzero(value: str) -> bool:
    value = value.strip().strip("'\"")
    if not value or value == "?":
        return False

    return float(value) != 0.0


def dense_row_has_label(line: str, n_targets: int) -> bool:
    values = next(csv.reader([line], skipinitialspace=True))
    if len(values) < n_targets:
        raise ValueError(
            f"Expected at least {n_targets} values in dense ARFF row, found {len(values)}"
        )

    return any(is_nonzero(value) for value in values[:n_targets])


def sparse_row_has_label(line: str, n_targets: int) -> bool:
    body = line.strip()[1:-1].strip()
    if not body:
        return False

    for entry in body.split(","):
        parts = entry.strip().split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"Malformed sparse ARFF entry: {entry!r}")

        index = int(parts[0])
        if index < n_targets and is_nonzero(parts[1]):
            return True

    return False


def count_all_zero_labels(dataset_dir: Path, dataset_name: str) -> dict[str, object]:
    dataset_path = dataset_dir / f"{dataset_name}.arff"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    n_targets = N_TARGETS[dataset_name]
    n_instances = 0
    all_zero_count = 0
    in_data = False

    with dataset_path.open() as arff_file:
        for raw_line in arff_file:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue

            if not in_data:
                if line.lower() == "@data":
                    in_data = True
                continue

            has_label = (
                sparse_row_has_label(line, n_targets)
                if line.startswith("{")
                else dense_row_has_label(line, n_targets)
            )
            n_instances += 1
            if not has_label:
                all_zero_count += 1

    non_zero_count = n_instances - all_zero_count

    return {
        "dataset": dataset_name,
        "n_instances": n_instances,
        "n_targets": n_targets,
        "all_zero_label_instances": all_zero_count,
        "non_zero_label_instances": non_zero_count,
        "all_zero_label_fraction": all_zero_count / n_instances if n_instances else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count instances whose multilabel target vector is entirely zero "
            "for the selected real datasets."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("src/datasets"),
        help="Directory containing the ARFF dataset files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/csv/all_zero_label_counts.csv"),
        help="CSV path for the all-zero-label count report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [count_all_zero_labels(args.dataset_dir, dataset) for dataset in DATASETS]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} dataset counts to {args.output}")


if __name__ == "__main__":
    main()
