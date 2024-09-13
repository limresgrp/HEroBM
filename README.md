# HEroBM

![Alt text](logo.svg?raw=true "HEroBM")

## HEroBM: Reconstructing Atomistic Structures from Coarse-Grained Molecular Simulations
This code implements HEroBM, a method for reconstructing atomistic structures from coarse-grained (CG) molecular simulations. HEroBM is based on deep equivariant graph neural networks and a hierarchical approach. It is accurate, versatile, and efficient, and can handle any type of CG mapping and is transferable to systems of varying sizes.

## Paper Reference
[HEroBM on arXiv](https://arxiv.org/abs/2404.16911)

## Instructions for setting up virtual environment and installing all required packages:

1 - Create a new virtual environment with python>=3.8
    Suggested version is 3.10
    Using conda, the script to run is `conda create -n "herobm" python=3.10`
    Using python, the script to run is `python -m venv ./herobm-venv`

2 - Activate the newly created virtual environment: `conda activate herobm`
    or `source herobm-venv/bin/activate`

3 - Clone the GEqTrain repository using `git clone https://github.com/limresgrp/GEqTrain.git` and go inside GEqTrain folder

4 - Run `./install.sh` to install Pytorch, selecting your CUDA version (or using cpu only), and GEqTrain dependencies

5 - Clone the CGMap repository using `git clone https://github.com/limresgrp/CGmap.git` and go inside the CGMap folder

6 - Run `pip install -e .` to install cgmap and all its required packages inside your virtual environment

7 - Return inside the HEroBM folder and run `pip install -e .` to install herobm and all its required packages inside your virtual environment

--- Soon GEqTrain and CGMap repositories will be provided as a Conda package, greatly simplifying the installation process ---