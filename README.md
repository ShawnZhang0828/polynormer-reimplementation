# Polynormer-Reimplementation

A project to reimplement Polynormer, a graph machine learning model introduced by Chenhui Deng, Zichao Yue, and Zhiru Zhang. The goal is to reproduce the results reported in the original paper ([Polynormer: Polynomial-Expressive Graph Transformer in Linear Time](https://arxiv.org/abs/2403.01232)).

## Overview

This repository provides:

- a command-line training tool in `train.py`
- model code in (local and global layers, and Polynormer) `models/`
- dataset and helpers in `utils/`
- a notebook experiment runner in `run_experiments.ipynb`

The current code trains a `Polynormer` model with:

- a local warm-up stage
- a local-to-global stage
- optional ReLU via `--use_relu true|false`
- accuracy or ROC-AUC evaluation depending on dataset

## Requirements

Install the project dependencies:

```bash
pip install -r requirements.txt
```

## Supported Datasets

The datasets currently supported by `utils/data_loaders.py` are:

- `computer`
- `photo`
- `cs`
- `physics`
- `wikics`
- `roman-empire`
- `amazon-ratings`
- `minesweeper`
- `tolokers`
- `questions`
- `ogbn-arxiv`

## Train Using the Command Line

The main training script is:

```bash
python train.py [arguments]
```

### Common arguments

- `--dataset`
- `--hidden_dim`
- `--n_local_layers`
- `--n_global_layers`
- `--n_local_heads`
- `--n_global_heads`
- `--warm_up_epochs`
- `--local_to_global_epochs`
- `--use_relu`
- `--use_local_attention_network`
- `--lr`
- `--dropout`
- `--metric`

### Example: `computer` without ReLU

```bash
python train.py --dataset computer --hidden_dim 512 --n_local_layers 5 --n_global_layers 1 --n_local_heads 8 --n_global_heads 8 --warm_up_epochs 200 --local_to_global_epochs 1000 --use_relu false --use_local_attention_network true --lr 0.001 --dropout 0.7 --metric accuracy
```

### Example: `roman-empire` with ReLU

```bash
python train.py --dataset roman-empire --hidden_dim 512 --n_local_layers 10 --n_global_layers 2 --n_local_heads 8 --n_global_heads 8 --warm_up_epochs 100 --local_to_global_epochs 2500 --use_relu true --use_local_attention_network true --lr 0.001 --dropout 0.3 --metric accuracy
```

## Checkpoints

`train.py` saves checkpoints under `checkpoints/`.

The notebook and training script expect checkpoint names in the following format:

```text
checkpoints/checkpoint_<dataset>_<plain|relu>
```

Examples:

- `checkpoints/checkpoint_computer_plain`
- `checkpoints/checkpoint_computer_relu`
- `checkpoints/checkpoint_roman-empire_plain`

## Run the Notebook

The notebook `run_experiments.ipynb` runs the experiment designed in the original paper:

### Google Colab

To execute the notebook in Google Colab:

1. Copy the repo to local Colab storage under `/content/`.
2. Mount Drive.
3. `cd` into the repository folder.
4. Install `requirements.txt`.
5. Run the experiment cells.

The notebook takes around 2hours to finish execution.

## Metrics

The repo currently supports:

- `accuracy`
- `roc_auc`

Use:

- `accuracy` for `computer`, `photo`, `cs`, `physics`, `wikics`, `roman-empire`, `amazon-ratings`, `ogbn-arxiv`
- `roc_auc` for `minesweeper`, `tolokers`, `questions`

## Project Structure

```text
models/
  global_attention.py
  local_attention.py
  polynormer.py
utils/
  data_loaders.py
  io.py
  metrics.py
  seed.py
train.py
run_experiments.ipynb
```
