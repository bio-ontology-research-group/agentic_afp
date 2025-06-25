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

Initially, we introduce a ProteinAgent, which takes initial GO
function predictions and looks for potential overlooked terms in
InterPro and Diamond. InterPro annotations and Diamond scores are used
to increase initial prediction scores.

```bash
python predict.py
python evaluate.py
```

Example output of `predict.py`:
```bash
GO term: GO:0016020, Score: 0.7, Explanation: InterPro domain + Diamond 0.63 similarity
```

Example output of `evaluate.py`:
```bash
# INITIAL PREDICTIONS
Computing Fmax
agentic_go cc
Fmax: 0.673, Smin: 8.654, threshold: 0.41, spec: 311
Precision: 0.692, Recall: 0.655
WFmax: 0.571, threshold: 0.3
AUC: 0.932
AUPR: 0.667
AVGIC: 7.368

# REFINED PREDICTIONS
Number of prop annotations: 22
Computing Fmax
agentic_go cc
Fmax: 0.688, Smin: 8.169, threshold: 0.49, spec: 318
Precision: 0.729, Recall: 0.652
WFmax: 0.588, threshold: 0.3
AUC: 0.935
AUPR: 0.680
AVGIC: 7.259

```

# Details:

* Model used: `deepseek/deepseek-chat-v3-0324`
* Protein agent is at: `agents/protein.py`



# Data

To extend the `test.pkl` file with diamond predictions and uniprot text information run:

```
python diamond_preds.py
python get_protein_uniprot_info.py

```
