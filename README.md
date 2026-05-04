# How Context Ordering and Structure Shape Instruction Following in Large Language Models

**Scott Ellis**  
EN.705.743 – ChatGPT from Scratch
Johns Hopkins University

---

## Overview

This project studies whether prompt topology—the ordering, grouping, and formatting of prompt elements—changes instruction-following behavior in large language models, even when the underlying semantic content stays the same. Instead of modifying model weights at inference time, the project adds a prompt-compilation layer that rewrites the same task into several structured prompt variants before generation.

The repository includes:

* a main script (`main.py`) for running the full pipeline end-to-end,
    * an optional notebook (`Ellis-FinalProject.ipynb`) for interactive execution and exploration,
* scripts for converting the IFEval and IFEval-FC datasets into a shared task format,
* prompt topology utilities for building alternate prompt structures,
* experiment code for running generations through Ollama,
* scoring code for JSON validity, schema adherence, and format adherence,
* and analysis utilities for aggregating results into CSV summaries.


The overall evaluation pipeline is consistent with the project goal and course expectations: convert tasks, run experiments, score outputs, and analyze results.

---

Here’s a cleaner, more accurate version:

---

## Entry Point

The primary entry point for this project is:

```text
main.py
```

This script is the recommended way to run the full pipeline, including data preparation, experiment execution, evaluation, and visualization.

An interactive alternative is also provided:

```text
Ellis-FinalProject.ipynb
```

The notebook mirrors the same workflow in a step-by-step format and can be used for exploration, debugging, or incremental execution.

## Repository Structure

```text
main.py
Ellis-FinalProject.ipynb
data_processed/
    tasks_ifeval.jsonl
    tasks_ifeval_fc.jsonl
data_raw/
    ifeval-fc_input_data.parquet
    ifeval-fc_README.md
    ifeval_input_data.jsonl
    ifeval_README.md
hftokenizer/
    merges.txt
    special_tokens_map.json
    tokenizer.json
    tokenizer_config.json
    vocab.json
runs/
    combined_model_summary_ifeval_and_fc.csv
    combined_model_topology_ifeval_and_fc.csv
    combined_scores_ifeval_and_fc.csv
    combined_topology_summary_ifeval_and_fc.csv
src/
    analysis_tools.py
    analyze.py
    convert_ifeval.py
    convert_ifeval_fc.py
    experiment.py
    gpt.py
    hftokenizer.py
    prompt_topologies.py
    run_ollama.py
    sampler.py
    score.py
    score_fc_checkers.py
    train_model.py
```

---

## What This Project Does

At a high level, the project runs the following workflow:

1. **Prepare datasets** from IFEval and IFEval-FC into a common JSONL task format.
2. **Compile prompts** into multiple topology variants while preserving the same task meaning.
3. **Generate model outputs** through Ollama.
4. **Score outputs** for instruction-following compliance, JSON validity, schema adherence, and format constraints.
5. **Aggregate results** into summary CSV files for later analysis and plotting.

This matches the implemented code structure:

- `convert_ifeval.py` converts raw IFEval examples into standardized tasks.
- `convert_ifeval_fc.py` converts IFEval-FC parquet data into schema-based JSON tasks.
- `prompt_topologies.py` defines the six prompt topologies used in the experiments.
- `run_ollama.py` and `experiment.py` handle generation runs, resume logic, and writing raw/scored outputs.
- `score.py` and `score_fc_checkers.py` implement automated scoring.
- `analyze.py` and `analysis_tools.py` generate summary CSVs from `scores.csv`.

---

## Requirements

This project requires the following:

- **Python 3**
- **Jupyter Notebook** or JupyterLab
- **Ollama** installed locally and running
- Common Python packages used in the code, including:
  - `pandas`
  - `numpy`
  - `requests`
  - `torch`
  - `json`
  - `seaborn`


## Ollama Setup

This repository uses Ollama as the inference backend. The generation scripts send requests to:

```text
http://localhost:11434/api/generate
```

That endpoint is hard-coded in both `src/run_ollama.py` and `src/experiment.py`, so Ollama must be installed and running locally before you execute any generation cells or scripts.

## How to Run the Project

### Recommended workflow

Open and run:

```text
python main.py
```

That notebook should be used as the top-level driver for the project.

A typical run sequence is:

1. Set up the Python environment.
2. Start Ollama locally.
3. Run `main.py` to execute the full pipeline.
5. Inspect outputs written under `runs/` including CSV summaries and saved visualizations.

## Prompt Topologies Implemented

The project evaluates six topology variants:

- `C_E_T` — Constraints > Examples > Task
- `E_C_T` — Examples > Constraints > Task
- `T_C_E` — Task > Constraints > Examples
- `Cdelim_E_T` — Delimited-bullets Constraints > Examples > Task
- `Cinterleave_T` — Constraints interleaved with Examples > Task
- `C_E_T_C` — Constraints > Examples > Task > Constraints repeated

These are defined in `src/prompt_topologies.py`. For IFEval-FC tasks, the prompt builder also appends a strong JSON-only reminder at the end of the prompt.

---

## Output Files

Running `main.py` generates several outputs:

### Processed task files

Located in `data_processed/`:

* `tasks_ifeval.jsonl`
* `tasks_ifeval_fc.jsonl`

These are the normalized task inputs used for experimentation.

### Per-run outputs

Located in `runs/<run_id>/`:

* `raw.jsonl` — raw prompts, model responses, parameters, and metadata
* `scores.csv` — scored evaluation results for each task/topology/repeat/model combination

These are created automatically during experiment execution.

### Summary outputs

Generated during the analysis stage of `main.py`:

* `summary_model_topology.csv`
* `summary_topology.csv`
* `summary_dataset_tasktype.csv`

The script also produces combined summaries across datasets:

* `combined_model_summary_ifeval_and_fc.csv`
* `combined_model_topology_ifeval_and_fc.csv`
* `combined_scores_ifeval_and_fc.csv`
* `combined_topology_summary_ifeval_and_fc.csv`

All summary files are written to the `runs/` directory.

### Visualizations

Saved to `runs/` as image files:

* `ifeval_pass_rate_heatmap.png`
* `ifeval_fc_pass_rate_heatmap.png`

These plots visualize model performance across prompt topologies for both datasets.

---


---

## Description of Key Source Files

### `src/train_model.py`
Training-related helper script for the GPT-from-scratch portion of the project.

### `src/gpt.py`
Core GPT model implementation used by the project.

### `src/sampler.py`
Sampling and decoding utilities for generation.

### `src/hftokenizer.py`
Tokenizer helper code associated with the tokenizer files stored in `hftokenizer/`.

### `src/convert_ifeval.py`
Converts raw IFEval JSONL into the project’s common task format.

### `src/convert_ifeval_fc.py`
Converts raw IFEval-FC parquet data into schema-constrained JSON tasks.

### `src/prompt_topologies.py`
Defines the topology variants and prompt-construction logic.

### `src/run_ollama.py`
Command-line driver for running evaluations through Ollama and saving raw/scored outputs.

### `src/experiment.py`
Notebook-friendly experiment runner with similar functionality and progress logging.

### `src/score.py`
Main automated scoring logic for instruction-following and schema-format checks.

### `src/score_fc_checkers.py`
Specialized deterministic format checkers for IFEval-FC tasks.

### `src/analyze.py`
Creates per-run summary CSVs from `scores.csv`.

### `src/analysis_tools.py`
Notebook-friendly analysis helpers for loading and summarizing runs.

---

## Acknowledgments / Data Sources

This project uses the following benchmark datasets:

* **[IFEval](https://huggingface.co/datasets/google/IFEval)** for general instruction-following evaluation
* **[IFEval-FC](https://huggingface.co/datasets/NikolaiSkripko/IFEval-FC)** for stricter function-calling / format-constrained evaluation

The following models were used:

* **[Qwen3:8B](https://huggingface.co/Qwen/Qwen3-8B)**
* **[Llama 3.2:3B](https://huggingface.co/meta-llama/Llama-3.2-3B)**
* **[Gemma 3:12B](https://huggingface.co/google/gemma-3-12b-it)**
* **[GPT-OSS:20B](https://huggingface.co/openai/gpt-oss-20b)**

---