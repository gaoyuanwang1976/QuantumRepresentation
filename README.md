# QuantumRepresentation
Quantum representation learning. This repository contains the code accompanying our manuscript https://arxiv.org/abs/2610.07196

### Environment Setup

The Conda environment and required dependencies are specified in qvae.yml. To create and activate the environment, run:

conda env create -f qvae.yml
conda activate <environment_name>

### Running the Experiments

The Python command provided in run_command.sh can be used to run the QVAE training and obtain the corresponding quantum    representations. 

### Example Analysis

The folder analysis_MNIST_raw01234/ contains example analysis files for investigating the learned latent dimensions and their properties.

### Dataset

The dataset/ folder includes an example MNIST dataset consisting of digits 0-4 at reduced image resolution. These samples do not include rotated images or other variations.

This dataset corresponds to the first experiment presented in the manuscript.
