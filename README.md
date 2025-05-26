# HEroBM

![Alt text](logo.svg?raw=true "HEroBM")

## HEroBM: Reconstructing Atomistic Structures from Coarse-Grained Molecular Simulations

This repository contains the code for HEroBM, a method designed to reconstruct atomistic structures from coarse-grained (CG) molecular simulations. HEroBM leverages deep equivariant graph neural networks and a hierarchical approach to achieve high accuracy, versatility, and efficiency. It is capable of handling diverse CG mappings and demonstrates transferability across systems of varying sizes.

## Paper Reference

For a detailed description of HEroBM and its underlying methodology, please refer to our paper:

[HEroBM on arXiv](https://arxiv.org/abs/2404.16911)

---

## Installation Guide

To set up your environment and install all necessary packages, please follow these steps:

1.  **Create a Virtual Environment**:
    Create a new virtual environment with Python version 3.8 or higher. Python 3.10 is suggested.
    * Using `conda`:
        ```bash
        conda create -n "herobm" python=3.10
        ```
    * Using `venv`:
        ```bash
        python -m venv ./herobm-venv
        ```

2.  **Activate the Virtual Environment**:
    Activate the newly created virtual environment.
    * Using `conda`:
        ```bash
        conda activate herobm
        ```
    * Using `venv`:
        ```bash
        source herobm-venv/bin/activate
        ```

3.  **Clone and Install GEqTrain**:
    Clone the GEqTrain repository and navigate into its directory.
    ```bash
    git clone [https://github.com/limresgrp/GEqTrain.git](https://github.com/limresgrp/GEqTrain.git)
    cd GEqTrain
    ```
    Then, run the installation script. This will install PyTorch (allowing you to select your CUDA version or opt for CPU-only), along with all GEqTrain dependencies.
    ```bash
    ./install.sh
    ```
    Return to the parent directory after installation: `cd ..`

4.  **Clone and Install CGMap**:
    Clone the CGMap repository and navigate into its directory.
    ```bash
    git clone [https://github.com/limresgrp/CGmap.git](https://github.com/limresgrp/CGmap.git)
    cd CGmap
    ```
    Install CGMap and its required packages within your virtual environment.
    ```bash
    pip install -e .
    ```
    Return to the parent directory after installation: `cd ..`

5.  **Install HEroBM**:
    Navigate back into the HEroBM folder and install HEroBM and its required packages.
    ```bash
    pip install -e .
    ```

---

**Note:** GEqTrain and CGMap repositories will soon be available as Conda packages, which will significantly streamline the installation process.

---

## Running Backmapping using Deployed Models

The `herobm-backmap` command-line tool allows you to reconstruct atomistic structures from coarse-grained simulations using pre-trained HEroBM models. Below is a detailed explanation of its usage and available options.

### Usage

```bash
herobm-backmap [-h] [-m MAPPING] [-i INPUT] [-it INPUTTRAJ] [-o OUTPUT] [-s SELECTION] [-ts TRAJSLICE] [--cg] [-mo MODEL] [-b BEAD_TYPES_FILENAME] [-d DEVICE] [-bs BEAD_STATS] [-t TOLERANCE]
```

### Options

* `-h`, `--help`: Show the help message and exit.

* `-m MAPPING`, `--mapping MAPPING`:
    Specifies the coarse-grained mapping scheme to be used (e.g., `martini3`, `martini2`).
    * **Default:** `martini3`

* `-i INPUT`, `--input INPUT`:
    Path to the input coarse-grained structure file (e.g., `.gro`, `.pdb`). This file defines the initial coarse-grained coordinates from which the atomistic structure will be reconstructed.

* `-it INPUTTRAJ`, `--inputtraj INPUTTRAJ`:
    Path to the input coarse-grained trajectory file. Use this option when you want to backmap multiple frames from a simulation.

* `-o OUTPUT`, `--output OUTPUT`:
    Directory where the backmapped atomistic files will be saved.

* `-s SELECTION`, `--selection SELECTION`:
    Allows you to specify a subset of the system (e.g., `protein`, `lipid`) for backmapping. This is useful for focusing reconstruction efforts on specific molecules or regions.
    * **Default:** `all` (backmaps the entire system)

* `-ts TRAJSLICE`, `--trajslice TRAJSLICE`:
    Defines a slice of the input trajectory to be processed. This is a string in the format `'start:stop:step'`, similar to Python list slicing. For example, `'900:1000:10'` would process frames from 900 to 990 (inclusive), taking every 10th frame.

* `--cg`:
    This flag **must be set** when your input file (`-i` or `-it`) is an actual coarse-grained structure or trajectory. It tells HEroBM to treat the input as coarse-grained data rather than an atomistic reference.

* `-mo MODEL`, `--model MODEL`:
    Path to the pre-trained HEroBM model file (`.pth`). This specifies which trained neural network model to use for the backmapping process. Deployed models are typically located in the `deployed/` directory within the HEroBM installation.

* `-b BEAD_TYPES_FILENAME`, `--bead-types-filename BEAD_TYPES_FILENAME`:
    Path to a YAML file that defines the bead types assigned to each coarse-grained bead. This file is crucial for correctly mapping coarse-grained beads to their corresponding atomistic structures, especially when deployed models were trained with specific bead type assignments (e.g., assigning the same bead type to all backbone beads, or differentiating bead types based on their chemical environment). By default, HEroBM will use the `bead_types.yaml`, which assigns a unique bead type to each backbone bead of different residues. This can lead to incorrect backmapping if the model expects a different assignment.
    * **Default:** `bead_types.yaml`
    * **Important Note for Martini Models:** For Martini models (e.g., `martini2`, `martini3`), it is crucial to specify `bead_types.bbcommon.yaml`. This file ensures that the bead types used during backmapping align with those used during model training, where all backbone (BB) beads were assigned the same bead type. This file is located within the `CGMap` repository, specifically in the data folder for each Martini version (e.g., `cgmap/data/martini2/bead_types.bbcommon.yaml` or `cgmap/data/martini3/bead_types.bbcommon.yaml`).

* `-d DEVICE`, `--device DEVICE`:
    Specifies the PyTorch device to use for computation. You can choose between `cuda:N` (for a specific GPU, e.g., `cuda:0`, `cuda:1`) or `cpu` for CPU-only computation.
    * **Default:** `cuda:0`

* `-bs BEAD_STATS`, `--bead-stats BEAD_STATS`:
    Path to a CSV file containing bead-to-bead distance statistics. Providing this file enables an initial minimization of bead positions before the full backmapping procedure, which can improve the quality of the reconstructed structures.

* `-t TOLERANCE`, `--tolerance TOLERANCE`:
    Energy tolerance for the minimization step, specified in kJ/(mol nm). This parameter controls how strictly the system is minimized to relax steric clashes after initial reconstruction.
    * **Default:** `500.0`

### Examples

**1. Backmapping Martini3 Proteins from a Single CG File:**

This command will backmap the protein part of a Martini3 coarse-grained system from a single `.gro` or `.pdb` file. The output will be saved in the `backmapped/test` directory.

```bash
herobm-backmap -m martini3 -i /path/to/your/cgfile.gro -o backmapped/test -s protein --cg -mo deployed/martini3.protein.v2.pth -d cuda:0 -b bead_types.bbcommon.yaml -bs cgdist.martini3.protein.csv
```

(Replace `/path/to/your/cgfile.gro` with the actual path to your coarse-grained input file. You can also specify `cuda:0` for a specific GPU or cpu for CPU-only computation. `cgdist.martini3.protein.csv` contains pairs of equilibrium distances that respect the expected CG distribution, and is used to minimize the CG structure before backmapping, thus improving the overall backmapping quality.)

**2. Backmapping Martini2 Proteins from a CG Trajectory:**

This example demonstrates how to backmap coarse-grained trajectories using the Martini2 force field. It will process all frames in the input trajectory and save the backmapped atomistic structures.

```bash
herobm-backmap -m martini2 -i /path/to/your/cgtraj.gro -it /path/to/your/cgtraj.xtc -ts ::100 -o backmapped/test -s protein --cg -mo deployed/martini2.protein.v2.pth -d cuda:0 -b bead_types.bbcommon.yaml -bs cgdist.martini2.protein.csv
```

(Replace `/path/to/your/cgtraj.gro` with the actual path to your coarse-grained gro file and `/path/to/your/cgtraj.xtc` with the actual path to your coarse-grained trajectory file. the `-ts ::100` samples all the frames from the trajectory with a stride of 100. The `bead_types.bbcommon.yaml` file for Martini2 is the one present in the folder of CGMap repo.)