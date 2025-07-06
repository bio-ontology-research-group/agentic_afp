# Multi-agent protein function prediction

## Requirements

- Python 3.10
- PyTorch
- CAMEL-AI

## Setup

```bash
conda env create -f environment.yml
conda activate agenticfp
```

## Run tests

```bash
pytest
```
## Usage

Initially, we introduce a ProteinCentricAgent, which takes initial GO
function predictions and looks for potential overlooked terms to
refine its predictions. It relies on sources such as InterPro and
Diamond. It uses text descriptions of proteins and GO terms.

To run:

```bash
python run_protein_centric.py
python propagate_annotations.py
python evaluate_all.py
```

# Details:

* Model used: `google/gemini-2.0-flash-001`
* Protein agent is at: `agents/protein_centric_agent.py`

# Preliminary results:

## Molecular Function (MF) 
| Prediction Type | Fmax  | Smin  | AUPR  | AUC   |
|----------------|-------|-------|-------|------- |
| Initial        | 0.642 | 7.364 | 0.642 | 0.957  |
| Refined        | 0.660 | 7.240 | 0.658 | 0.959  |
| Improvement    | +0.018| -0.124| +0.016| +0.002 |

## Cellular Component (CC) 

| Prediction Type | Fmax  | Smin  | AUPR  | AUC   |
|----------------|-------|-------|-------|-------|
| Initial        | 0.693 | 7.530 | 0.723 | 0.936 |
| Refined        | 0.702 | 7.331 | 0.730 | 0.938 |
| Improvement    | +0.009| -0.199| +0.007| +0.002|

## Biological Process (BP)
| Prediction Type | Fmax  | Smin   | AUPR  | AUC   |
|----------------|-------|--------|-------|-------|
| Initial        | 0.414 | 27.440 | 0.354 | 0.868 |
| Refined        | 0.425 | 27.433 | 0.361 | 0.870 |
| Improvement    | +0.011| -0.007 | +0.007| +0.002|


# Data

We extend the `test.pkl` file with diamond predictions and uniprot text information run:

```
python diamond_preds.py
python get_protein_uniprot_info.py
```
