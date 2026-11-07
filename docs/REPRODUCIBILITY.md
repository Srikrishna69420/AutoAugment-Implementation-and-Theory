# Reproducibility notes

## Canonical run

Run `AutoAugment.ipynb` from its first cell through the final cell. The notebook contains the augmentation search space, policy class, controller, child WideResNet, CIFAR-10 pipeline, training loop, and observations.

## Dataset

The experiment uses CIFAR-10 through `torchvision.datasets.CIFAR10` with `download=True`. The notebook stores the dataset under `./data` relative to the notebook kernel's working directory. Ensure that this location is writable and has enough disk space.

## Environment

Install the packages listed in the repository's `requirements.txt`. The notebook imports PyTorch, torchvision, NumPy, and Pillow. A compatible CUDA/PyTorch installation is recommended for GPU execution; install the PyTorch build appropriate for the machine rather than assuming every CUDA version is interchangeable.

## Compute and interpretation

This project intentionally uses reduced search/training settings compared with the full paper. The notebook itself records the settings and reported observations. Do not present the resulting numbers as a full reproduction of the paper's compute budget or results.

## Repeatability

The notebook sets Python, NumPy, and PyTorch seeds in its setup cell. GPU kernels and data-loader behavior may still introduce nondeterminism depending on the installed software and hardware. Record the Python, PyTorch, torchvision, CUDA, GPU, and notebook parameter versions for a formal experiment.

## Outputs

The notebook prints policy/reward information and experiment observations. Save any newly generated checkpoints, logs, or plots outside version control unless they are intentionally curated for publication.
