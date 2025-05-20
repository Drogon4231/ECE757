# ML-Guided Runtime Prediction for TDG Partitioning  
ECE 757 – Advanced Computer Architecture II  
University of Wisconsin–Madison, Spring 2025

---

## Project Overview

This project applies machine learning to optimize partition size selection for matrix workloads on circuit-based task dependency graphs (TDGs). Using benchmark data from runtime measurements and extracted graph features, we train a regression model to predict the best partition configuration for improved performance.

---

## Repository Structure

```
ECE757/
├── data/
│   ├── graphs/                # TDG .txt files
│   └── ml_data/              # runtime_data.hpp, training_data.csv
├── scripts/
│   ├── extract_features.py   # Parses graph + runtime data
│   ├── train_model.py        # Trains XGBoost model
│   ├── predict.py            # Predicts best partition size for workloads
│   ├── diagnostics.py        # Compares predicted vs actual bests
│   ├── benchmark.py          # Collects actual runtime data for various matrix sizes
│   └── models/               # model.pkl, scaler.pkl
├── results/                  # Diagnostic output
├── docs/
│   └── A_CA2___Final_Report.pdf
└── README.md
```

---

## Methodology

- Feature engineering using graph characteristics and matrix metadata
- XGBoost regression model trained to predict runtime
- Inference pipeline to select the partition size with lowest predicted runtime
- Evaluation by comparing against ground-truth best runtimes
- Benchmarking for actual workload executions

---

## Usage Instructions

### 1. Extract Features

```bash
python scripts/extract_features.py
```

Parses TDG graph files and C++ runtime header to generate a CSV for training.

### 2. Train the Model

```bash
python scripts/train_model.py
```

Trains a regression model and saves `model.pkl` and `scaler.pkl` under `scripts/models/`.

### 3. Predict Best Partition Sizes

```bash
python scripts/predict.py
```

Uses the trained model to predict optimal partition sizes for multiple matrix values and TDG inputs.

### 4. Run Diagnostics

```bash
python scripts/diagnostics.py
```

Generates a summary comparing predicted vs. actual best partitions. Saves output to `results/diagnostics_summary.csv`.

### 5. Optional: Run Benchmark

```bash
python scripts/benchmark.py
```

Executes the test binary for different matrix sizes to capture raw runtime data.

---

## Final Report

[docs/A_CA2___Final_Report.pdf](docs/A_CA2___Final_Report.pdf)

---

## Author

**Harshith Kantamneni**  
MS in Electrical & Computer Engineering  
University of Wisconsin–Madison

---

## License

MIT License – see `LICENSE`.
