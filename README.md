# LACE: Label Cluster Evolution for Multi-Label Concept Drift Detection

This repository contains the anonymized replication package for the submitted CIKM 2026 short paper:

**LACE: Unsupervised Concept Drift Detection in Multi-Label Data Streams Through Label Cluster Evolution**

The package includes the implementation of LACE, baseline drift detectors, dataset-loading utilities, experiment scripts, a Docker configuration, and saved prediction outputs used for reproducing the reported results.

---

## Overview

LACE is an unsupervised concept drift detection method for multi-label data streams. It monitors the evolution of predicted label associations over time. The method maintains a sliding window over predicted label vectors, divides the window into two consecutive subwindows, clusters labels according to their co-occurrence patterns, and compares the resulting label partitions using Adjusted Rand Index (ARI) and a Monte Carlo-based decision procedure.

The repository supports:

- Running LACE on real and synthetic multi-label data streams.
- Running baseline drift detectors.
- Saving prediction outputs and detected drift points.
- Generating final predictive-performance tables from saved prediction files.
- Generating synthetic drift-detection performance tables.
- Running the package either locally or through Docker.
- Inspecting results through a lightweight walkthrough notebook.

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── main.py
├── main.ipynb
├── baselines/
│   ├── ld3.py
│   └── tornados/
├── src/
│   ├── util.py
│   ├── datasets/
│   │   └── *.arff
│   └── drift_detector/
│       ├── detector.py
│       ├── partition_comparison.py
│       ├── partition_randomization.py
│       ├── window.py
│       └── clustering/
│           ├── agglomerative.py
│           ├── c3m.py
│           ├── cluster.py
│           ├── kmeans.py
│           └── omiecinski.py
└── outputs/
    └── predictions/
        └── <dataset>/
            └── <detector>.pkl
```

The primary entry point is:

```bash
python main.py
```

The notebook `main.ipynb` is provided as a lightweight walkthrough for running the script, and inspecting saved outputs. The notebook does not contain a separate experiment implementation; it calls `main.py` to avoid maintaining two different execution paths.

---

## Environment

The recommended environment is:

```text
Python 3.8
```

The code was prepared and tested with a Python 3.8-compatible dependency stack. This is important because some dependencies, especially `scikit-multiflow`, may not install reliably with newer Python versions.

---

## Installation Without Docker

Create and activate a virtual environment:

```bash
python3.8 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, the activation command is:

```powershell
.venv\Scripts\Activate.ps1
```

Upgrade packaging tools and install dependencies:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If local installation fails because of old scientific Python dependencies, use the Docker workflow below.

---

## Docker Usage

Docker provides a self-contained Linux environment with the required Python version and system-level build tools. This is useful because there remains no need to manually install Python 3.8, compilers, `scikit-multiflow`, or other fragile dependencies.

Only Docker should be installed. The image can be built from the provided `Dockerfile`.

### Build the Docker image

From the repository root, run:

```bash
docker build -t lace-replication .
```

Here, `lace-replication` is just the local name assigned to the Docker image. It is not a file in the repository.

### Check that the command-line interface works

```bash
docker run --rm lace-replication python main.py --help
```

### Run a quick smoke test

```bash
docker run --rm lace-replication python main.py --dataset Birds --detector LACE --max-samples 100 --overwrite
```

This only verifies that the container, dependencies, and code entry point work.

### Save generated outputs locally

By default, files created inside a container disappear when the container exits. To keep generated prediction files in the repository directory, mount the local `outputs/` directory into the container.

For macOS, Linux, Git Bash, or WSL:

```bash
docker run --rm \
  -v "$(pwd)/outputs:/app/outputs" \
  lace-replication \
  python main.py --dataset Birds --detector LACE --overwrite
```

For Windows PowerShell:

```powershell
docker run --rm `
  -v "${PWD}\outputs:/app/outputs" `
  lace-replication `
  python main.py --dataset Birds --detector LACE --overwrite
```

For Windows Command Prompt:

```cmd
docker run --rm -v "%cd%\outputs:/app/outputs" lace-replication python main.py --dataset Birds --detector LACE --overwrite
```

### Reproduce the paper detector set with Docker

```bash
docker run --rm \
  -v "$(pwd)/outputs:/app/outputs" \
  lace-replication \
  python main.py --dataset all --detector paper --overwrite
```

### Run all available detectors, including LACE variants, with Docker

```bash
docker run --rm \
  -v "$(pwd)/outputs:/app/outputs" \
  lace-replication \
  python main.py --dataset all --detector all-available --overwrite
```

---

## Running Experiments Without Docker

### Run LACE on one dataset

```bash
python main.py --dataset Birds --detector LACE --overwrite
```

### Run one baseline on one dataset

```bash
python main.py --dataset Birds --detector LD3 --overwrite
```

Example detectors:

```text
LACE
LD3
RDDM
ADWIN
EDDM
DDM
KSWIN
HDDM_A
HDDM_W
FHDDM
FHDDMS
FHDDMS_add
MDDM_A
MDDM_E
MDDM_G
SeqDrift2
ND
```

### Run all paper detectors on one dataset

```bash
python main.py --dataset Birds --detector paper --overwrite
```

### Run LACE on all paper datasets

```bash
python main.py --dataset all --detector LACE --overwrite
```

### Run all paper detectors on all paper datasets

```bash
python main.py --dataset all --detector paper --overwrite
```

This command may take a long time depending on the machine.

### Run all available detectors, including LACE variants

```bash
python main.py --dataset Birds --detector all-available --overwrite
```

The `all-available` option includes the LACE ablation variants:

```text
LACE-Agglomerative_Variant
LACE-KMeans_Variant
```

---

## Command-Line Arguments

### `--dataset`

Dataset name or `all`.

Example:

```bash
python main.py --dataset Birds
```

### `--detector`

Detector name, `paper`, or `all-available`.

Example:

```bash
python main.py --detector LACE
python main.py --detector paper
python main.py --detector all-available
```

### `--dataset-dir`

Directory containing real dataset files in ARFF format.

Default:

```text
src/datasets
```

### `--output-dir`

Directory where prediction outputs are saved.

Default:

```text
outputs/predictions
```

### `--overwrite`

Overwrite existing prediction files.

### `--max-samples`

Optional limit on the number of processed stream samples. This is useful for quick testing.

Example:

```bash
python main.py --dataset Birds --detector LACE --max-samples 100 --overwrite
```

---

## Datasets

The script includes the dataset specifications used in the experiments.

Real datasets:

```text
Birds
Plant
Gnegative
Enron
yeast
Human
Slashdot
Eukaryote
OHSUMED-F
tmc2007-500
Mediamill
Imdb
```

Synthetic datasets:

```text
Synthetic_Sudden1
Synthetic_Sudden2
Synthetic_Sudden3
Synthetic_Incremental1
Synthetic_Incremental2
Synthetic_Incremental3
Synthetic_Reoccurring
```

Real datasets are expected as ARFF files under:

```text
src/datasets/
```

For example:

```text
src/datasets/Birds.arff
src/datasets/Enron.arff
```

Synthetic streams are generated directly by the code.

---

## Output Format

Each run saves one pickle file per dataset-detector pair:

```text
outputs/predictions/<dataset>/<detector>.pkl
```

Example:

```text
outputs/predictions/Birds/LACE.pkl
```

Each pickle file contains a dictionary with the following fields:

```python
{
    "dataset": ...,
    "dataset_type": ...,
    "detector": ...,
    "detector_params": ...,
    "drift_points": ...,
    "metrics": {
        "accuracy": ...,
        "f1_micro": ...,
        "f1_macro": ...
    },
    "predictions": ...
}
```

The reported predictive metrics are:

```text
accuracy
f1_micro
f1_macro
```

Detected drift points are stored as a list of dictionaries:

```python
[
    {
        "stream_idx": ...,
        "processed_idx": ...
    }
]
```

`stream_idx` refers to the original stream index, while `processed_idx` refers to the index after skipping all-zero label instances.

---

## Using the Walkthrough Notebook

The notebook:

```text
main.ipynb
```

is intended as a compact walkthrough. It demonstrates how to:

1. Run `main.py` on a small dataset.
2. Load the saved prediction file.
3. Inspect detector parameters.
4. Inspect predictive metrics.
5. Inspect detected drift points.
6. Compute synthetic drift-detection performance.

The notebook does not contain a separate experiment implementation. It calls `main.py` to avoid maintaining two different execution paths.

---
## Synthetic Drift-Detection Performance

The notebook also computes drift-detection performance on synthetic datasets using two metrics:

- **Missed Detection Rate (MDR)**
- **Delay for Detection (DFD)**

These metrics are computed only for synthetic datasets, where the real drift points are known:

```text
4000
10000
```

For each real drift point, the first detected drift after that point is counted as the true detection. For every real drift except the last one, only detections before the next real drift are considered. This prevents a detection after the second real drift from being matched to the first real drift.

A real drift is counted as missed if no detection occurs after it within its valid interval.

The metrics are defined as:

```text
MDR = number of missed real drifts / number of real drifts
DFD = detected drift index - real drift index
```

The notebook generates the following files:

```text
outputs/final_tables/drift_performance/synthetic_mdr_dfd_raw.csv
outputs/final_tables/drift_performance/synthetic_mdr_results.csv
outputs/final_tables/drift_performance/synthetic_dfd_results.csv
```

The raw file includes the true drift points, detected drift points, matched detections, per-drift delays, number of missed drifts, MDR, and DFD for each detector-dataset pair.

---

## Reproducing the Main Paper Results

To reproduce the main paper detector set on all paper datasets:

```bash
python main.py --dataset all --detector paper --overwrite
```

With Docker:

```bash
docker run --rm \
  -v "$(pwd)/outputs:/app/outputs" \
  lace-replication \
  python main.py --dataset all --detector paper --overwrite
```

Then open `main.ipynb` and run the result-table cells to generate the final tables.

---

## Reproducing LACE Variant Results

To run all available detectors, including LACE variants:

```bash
python main.py --dataset all --detector all-available --overwrite
```

With Docker:

```bash
docker run --rm \
  -v "$(pwd)/outputs:/app/outputs" \
  lace-replication \
  python main.py --dataset all --detector all-available --overwrite
```

This includes:

```text
LACE
LACE-Agglomerative_Variant
LACE-KMeans_Variant
```

along with all baseline detectors.

---

## Citation

This repository is associated with an anonymized CIKM 2026 submission. Citation information will be added after the review process.

---

## License

This repository is provided for anonymous review and reproducibility purposes.
