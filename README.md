# AutoAugment: implementation and experiments

This repository contains a notebook-based implementation of AutoAugment, including image augmentation operations, a recurrent controller that samples policies, and a WideResNet child model trained to evaluate candidate policies.

## Start here

Open [AutoAugment.ipynb](AutoAugment.ipynb) in Jupyter or Google Colab. The notebook is the canonical implementation and should be run from top to bottom.

### Local setup

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter notebook AutoAugment.ipynb
```

A CUDA-capable GPU is strongly recommended. The policy search repeatedly trains child networks and is computationally expensive.

## Repository layout

- `AutoAugment.ipynb` — full, ordered experiment and implementation.
- `Theory.pdf` — accompanying theory notes.
- `docs/REPRODUCIBILITY.md` — environment, data, and experiment notes.
- `requirements.txt` — Python package dependencies.

## Experiment scope

This is a lightweight reproduction rather than a compute-matched reproduction of the original AutoAugment study. The notebook describes reductions to policy-search iterations, child-model training, and dataset subset size. Reported observations are recorded in the notebook; they should be interpreted in light of those reduced compute settings.

## Data and outputs

The notebook uses CIFAR-10 and asks torchvision to download it under `./data`. 
