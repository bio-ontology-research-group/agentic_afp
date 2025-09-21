# LLM agent-based protein function prediction

<div align="center">
<img src="figs/agent_workflow.png" alt="Alt text" width="500">
</div>

## Requirements

- Python 3.10
- PyTorch
- CAMEL-AI

## Setup

```bash
conda env create -f environment.yml
conda activate agenticfp
```


## Usage

We used OpenRouter. Before running the scripts, set up the OpenRouter API key as follows:

```
export OPENROUTER_API_KEY=<your openrouter api key>
```

Now, you can run the agent with the following command:

```bash
python function_agent.py --run_number 0 --model_name gemini
```

- We used the parameter `run_number` to analyze performance variance. You can safely ignore it since the default is 0.
- Models options are: `gemini` and `gpt`.


After running, we need to propagate annotations from more specific classes to more general ones. And then evaluate :smile:.

```bash
python propagate_annotations.py 0 gemini
python evaluate_all.py 0 gemini
```

- Make sure you use the same `run_number` and `model_name` parameters you used for the `function_agent.py` script.

## Example

We show and example of our how our method refines the predictions for
a protein:

<details>
<summary>Metadata retrieval</summary>
Before creating an agent, we retrieve information from UniProtKB and
PubMed. For example: for the protein with ID 11H\_STRNX we retrieve
the following information:

<div align="center">
<img src="figs/metadata.png" alt="Alt text" width="500">
</div>
</details>

<details>
<summary>Agent creation</summary>
    
Using the retrieved information from UniProtKB (**[uniprot_info]**) and PubMed (**[abstracts]**), we create
and agent and its role and context:

<div align="center">
<img src="figs/agent_creation.png" alt="Alt text" width="500">
</div>
</details>

<details>
<summary>Reasoning process</summary>

To start the reasoning process, we select initial GO term predictions
with scores $\geq 0.1$, and retrieve the following information:
* Initial MLP prediction score
* Diamond Score
* Definition and labels
* Taxonomical constraints

We denote this information as **[go_terms_info]**.
Then, we instruct the agent to analyze the GO terms and suggest refinements of predictions:

<div align="center">
<img src="figs/reasoning.png" alt="Alt text" width="500">
</div>
</details>

