# Multi-agent protein function prediction

## Requirements

- Python 3.10
- PyTorch
- CAMEL-AI
- BLAST

## Setup

```bash
conda env create -f environment.yml
```

## Run tests

```bash
pytest
```
## Usage

```bash
python agents.py
```

Example output:

```bash
# Testing for GO:0110165
Agent's interpretation: All of the top 10 similar proteins have the GO function GO:0110165. Therefore, the hypothesis is strongly supported by the data.

# Testing for GO:0000000 (a dummy example)
Agent's interpretation: Based on the UniProt search results, there is little evidence to support the hypothesized function GO:0000000 for the provided protein sequence. None of the top 10 similar proteins in UniProt have this GO function. Therefore, the analysis suggests that the hypothesized function is unlikely to be correct.

```
