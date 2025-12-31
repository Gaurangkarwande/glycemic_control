# 📊 Glycemic Control

Modeling causal relationships between glucose levels and clinical interventions in ICU settings using graph-based deep learning.

## Overview

Standard predictive models in critical care capture correlations without explaining *why* changes occur. This project uses **causal modeling** with Graph Neural Networks to identify variables influencing glucose regulation.

**Key approaches:**
- Causal discovery to build structural models connecting clinical variables to glucose outcomes
- Graph Neural Networks (GNNs) with DAG representations for learning relationships
- Interpretable insights from high-dimensional ICU time series data

## Project Structure

```
glycemic_control/
├── src/
│   ├── constants.py           # Clinical variable definitions
│   ├── dataset.py             # PyTorch datasets
│   ├── dataloader.py          # GNN data loading
│   ├── utils.py               # Utilities (early stopping, LR scheduler)
│   ├── train_gnn.py           # GNN training pipeline
│   ├── train_gnn_set.py       # Batch training across DAGs
│   └── models/
│       ├── temporal_gnn.py    # DCRNN-based models
│       └── transformer.py     # Time series transformer
├── notebooks/                 # Exploration notebooks
├── dataset_scripts.py         # Data preprocessing
├── clinical_dag_scripts.py    # Clinical DAG processing
├── aggregate_dags_script.py   # DAG aggregation via majority voting
└── requirements.txt
```

## Installation

```bash
conda create -n glycemic python=3.8
conda activate glycemic
pip install -r requirements.txt
```

**Key dependencies:** PyTorch, PyTorch Geometric, PyTorch Geometric Temporal, pandas, scikit-learn

## Clinical Variables

| Category | Variables |
|----------|-----------|
| Organ Scores | SOFA, liver, CNS, cardiovascular, coagulation, respiration, renal |
| Patient Metrics | weight, height, BMI, age, Charlson index |
| Interventions | tube feeding, dextrose, insulin |
| Target | glucose (discretized into 5 classes) |

## Usage

### 1. Prepare Dataset
```bash
python dataset_scripts.py --fpath_raw_csv /path/to/cohort.csv --dirpath_out /path/to/output/
```

### 2. Process Clinical DAGs
```bash
python clinical_dag_scripts.py --fpath_excel /path/to/dags.xlsx --dag_type diabetic --dirpath_save /path/to/dags/
```

### 3. Train GNN Model
```bash
python -m src.train_gnn \
    --fpath_train_data train.csv \
    --fpath_valid_data valid.csv \
    --fpath_test_data test.csv \
    --fpath_dag dag.csv \
    --dirpath_results /path/to/results/
```

## Models

- **RecurrentGCN**: DCRNN-based graph neural network for temporal classification
- **TimeSeriesTransformer**: Encoder-decoder transformer for sequence forecasting

## References

- Wu et al. (2020). *Deep Transformer Models for Time Series Forecasting*
- Li et al. (2018). *Diffusion Convolutional Recurrent Neural Network*

---
