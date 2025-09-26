![Alt text](logo.svg?raw=true "HEroBM")

## HEroBM: Reconstructing Atomistic Structures from Coarse-Grained Molecular Simulations


This repository contains the code for **HEroBM**, a method designed to reconstruct atomistic structures from coarse-grained (CG) molecular simulations.

HEroBM leverages **deep equivariant graph neural networks** and a **hierarchical approach** to achieve high accuracy, versatility, and efficiency. It is capable of handling diverse CG mappings and demonstrates transferability across systems of varying sizes.

-----

## 📖 Paper Reference

For a detailed description of HEroBM and its underlying methodology, please refer to our paper:
👉 [HEroBM: A deep equivariant graph neural network for high-fidelity backmapping from coarse-grained to all-atom structures](https://doi.org/10.1063/5.0280330)

-----

## 🚀 Installation Guide

Follow these steps to set up your environment for HEroBM.

### Step 1: Create and Activate a Virtual Environment

It is highly recommended to use a dedicated virtual environment. **Python \>= 3.10** is suggested.

Using **conda**:

```bash
conda create -n "herobm"
conda activate herobm
```

Using **venv**:

```bash
python -m venv ./herobm-venv
source herobm-venv/bin/activate
```

### Step 2: Install Dependencies (GEqTrain & CGMap)

HEroBM depends on two other packages: **GEqTrain** and **CGMap**.

**Install GEqTrain:**

```bash
git clone --branch herobm https://github.com/limresgrp/GEqTrain.git
cd GEqTrain
# !!! Follow instructions in the GEqTrain README to install PyTorch for your hardware 
pip install -e .
cd ..
```

**Install CGMap:**

```bash
git clone https://github.com/limresgrp/CGmap.git
cd CGmap
pip install -e .
cd ..
```

### Step 3: Install HEroBM

```bash
pip install -e .
```

-----

## ▶️ Running Backmapping with Pre-trained Models

The `herobm-backmap` command-line tool reconstructs atomistic structures from coarse-grained simulations.

> **💡 Note on Deployed Models**
> When you use a pre-trained, deployed model (`-mo path/to/deployed_model.pth`), key metadata is already embedded within the model file. **This means you often do not need to specify the `--mapping`, `--bead-types-filename`, or `--bead-stats` options**, as the model knows which settings it was trained with. You only need to provide these flags if you want to explicitly override the embedded metadata.

### Usage

```bash
herobm-backmap [-h] [-i INPUT] [-it INPUTTRAJ] [-o OUTPUT]
               [-s SELECTION] [-ts TRAJSLICE] [-mo MODEL]
               [-m MAPPING] [-b BEAD_TYPES_FILENAME]
               [-a] [-d DEVICE] [-c CHUNKING]
               [-bs BEAD_STATS] [-ns NUM_STEPS] [-t TOLERANCE]
```

### Key Options

  * `-i, --input INPUT` → Input coarse-grained **or atomistic** structure file (`.gro`, `.pdb`).
  * `-it, --inputtraj INPUTTRAJ` → Input trajectory file (`.xtc`, `.trr`).
  * `-o, --output OUTPUT` → Output directory (default: `./output`).
  * `-mo, --model MODEL` → Path to the trained/deployed HEroBM model (`.pth`).
  * `-s, --selection SELECTION` → Atom selection (default: `all`).
  * `-ts, --trajslice TRAJSLICE` → Slice trajectory (`start:stop:step`).
  * `-a, --atomistic` → Flag if inputs are **atomistic** (for MAP→BACKMAP validation).
  * `-d, --device DEVICE` → Torch device (default: `cuda:0`).
  * `-c, --chunking N` → Process trajectory in chunks (max atoms per batch).
  * `-ns, --num-steps INT` → Steps for CG minimization (default: `1000`).
  * `-t, --tolerance FLOAT` → Energy tolerance for atomistic minimisation (default: `500.0`).

### Advanced Options (Metadata Overrides)

  * `-m, --mapping MAPPING` → Specifies the CG mapping. It corresponds to a folder name containing the mapping's `.yaml` files. By default, the tool looks inside the `CGmap/cgmap/data/` directory, but you can also provide an **absolute path** to a custom mapping folder.
  * `-b, --bead-types-filename FILE` → The YAML file inside the mapping folder that assigns a unique integer to each bead type. These integers are used as node features in the graph model. If a `bead_types.yaml` file is not found, one is created automatically. You can create custom files for different type assignments.
    > **Example**: For Martini protein models, the deployed models use a custom `bead_types.bbcommon.yaml`. It assigns the **same integer** to all backbone (BB) beads, regardless of the residue type (e.g., `ALA_BB`, `ARG_BB`, and `GLY_BB` all get type `13`). This improves model transferability across different proteins. In contrast, the default `bead_types.yaml` would assign a different integer to each (`ALA_BB: 18`, `ARG_BB: 20`, etc.).
  * `-bs, --bead-stats FILE` → Path to a CSV file containing soft distance restraints for connected CG beads. Providing this file enables an initial energy minimization of the input CG structure **before** backmapping. This significantly improves the quality of the final atomistic structure, especially when backmapping real CG trajectories. See the section below on how to generate this file.

-----

### Generating Bead Statistics (`--bead-stats`)

The `--bead-stats` CSV file can be generated using the `cg_analysis.py` script included with HEroBM. This script takes one or more atomistic structure/trajectory files as input, maps them to the coarse-grained representation on-the-fly, and calculates distance statistics (mean, std, min, max) for connected beads.

**Usage:**

Run the script from the HEroBM root directory, pointing it to your atomistic data.

```bash
python herobm/scripts/cg_analysis.py \
    -i /path/to/atomistic/structures/struct.pdb \
    -m martini3 \
    -o cg_stats.martini3.csv
```

  * `-i`: Path to a structure input file containing atomistic system (usually format is `.pdb` or `.gro`). Can also specify a folder if having multiple pdbs and no trajectory.
  * `-t`: Path to an input trajectory file to load in the structure input file.
  * `-m`: The name of the CG mapping to use for the analysis.
  * `-o`: The name of the output CSV file to be created.

-----

### Examples

**1. Backmapping a Single Martini3 Protein Structure (Metadata-Driven)**

```bash
herobm-backmap \
    -i /path/to/cgfile.gro \
    -o backmapped/ \
    -s protein \
    -mo deployed/martini3/protein.Sep.2025.pt \
    -d cuda:0
```

👉 Mapping, bead types, and bead stats will be loaded directly from the model's embedded metadata.

**2. Backmapping a Martini2 Protein Trajectory with Overrides**

Here, we explicitly provide the mapping, bead types, and bead stats, which will override any metadata in `best_model.pth`.

```bash
herobm-backmap \
    -m martini2 \
    -i /path/to/cgtraj.gro \
    -it /path/to/cgtraj.xtc \
    -ts ::100 \
    -o backmapped/ \
    -s protein \
    -mo training/myrun/best_model.pth \
    -b <path_to_cgmap>/cgmap/data/martini2/bead_types.bbcommon.yaml \
    -bs cg_stats.martini2.protein.csv \
    -ns 1500 \
    -d cuda:0
```

**3. Using Chunked Processing for Large Systems**

```bash
herobm-backmap \
    -i huge_system.gro \
    -it huge_system.xtc \
    -o backmapped_large/ \
    -mo deployed/martini3/large.pt \
    -c 50000 \
    -d cuda:0
```

---

### 🧠 Creating a Training Dataset

Before you can train your own HEroBM model, you need to create a dataset. This is done using the `herobm-dataset` script, which processes atomistic structures and/or trajectories, maps them to a coarse-grained representation, and saves the required data in `.npz` format for training.

The core of this process is the `HierarchicalMapper`, which not only performs the coarse-graining but also calculates the hierarchical relationships and relative vectors between atoms and beads. This information is crucial for the model to learn how to reconstruct the fine-grained details from the coarse-grained input.

#### Usage

The script takes various arguments to specify the input files, the mapping scheme, and the output directory.

```bash
herobm-dataset [-h] -m MAPPING -i INPUT [-if INPUTFORMAT] [-t INPUTTRAJ] 
               [-tf TRAJFORMAT] [-f FILTER] [-o OUTPUT] [-s SELECTION] 
               [-ts TRAJSLICE] [-b BEAD_TYPES_FILENAME] [-c CUTOFF]
```

#### Key Options

  * `-m, --mapping MAPPING`: **(Required)** The name of the CG mapping to use (e.g., `martini3`). This corresponds to a folder in the `cgmap/data` directory or an absolute path to a custom mapping folder.
  * `-i, --input INPUT`: **(Required)** Path to the input folder containing atomistic structures (e.g., `.pdb`, `.gro`) or a single structure file.
  * `-o, --output OUTPUT`: The output folder where the generated `.npz` files will be saved.
  * `-s, --selection SELECTION`: An MDAnalysis selection string to specify which part of the system to process (e.g., `"protein"`). Default is `"all"`.
  * `-it, --inputtraj INPUTTRAJ`: Path to an input trajectory file or a folder of trajectories.
  * `-b, --bead-types-filename FILE`: The name of the bead types YAML file to use or create within the mapping directory. Default is `bead_types.yaml`.
  * `-c, --cutoff CUTOFF`: (Recommended) A cutoff distance in Angstroms to pre-compute the graph edges (bead neighbors). This also includes information about preceding and following beads in a sequence, which is beneficial for training.

#### Example

This command processes all `.pdb` files in the `/path/to/atomistic/pdbs/` directory, filters them according to `targets.train.pdb`, applies the `martini3` mapping with common backbone bead types, and saves the output `.npz` files to `/path/to/output/npz/train`.

```bash
herobm-dataset \
    -m martini3 \
    -i /path/to/atomistic/pdbs/ \
    -f /path/to/filters/targets.train.pdb \
    -if pdb \
    -s protein \
    -o /path/to/output/npz/train \
    -b bead_types.bbcommon.yaml \
    -c 10.0
```

-----

## 🏢 Deployment Guide for Shared Systems

This guide outlines the **one-time setup** to make HEroBM available to all users on a shared system (e.g., `/apps/herobm`) via a self-contained Conda environment.

### Part 1: One-Time Administrator Setup

1. **Create the Installation Directory**

```bash
export HEROBM_ROOT=/apps/herobm
sudo mkdir -p $HEROBM_ROOT
sudo chown $USER $HEROBM_ROOT
```

2. **Set Up the Conda Environment**

```bash
cd $HEROBM_ROOT
conda create --prefix ./env python=3.10 -y
conda activate ./env
```

3. **Install Dependencies**

```bash
# From conda-forge
conda install -c conda-forge pdbfixer -y

# Install PyTorch (adjust CUDA version if needed)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

4. **Install HEroBM and its Dependencies**

```bash
mkdir -p $HEROBM_ROOT/src && cd $HEROBM_ROOT/src

git clone --branch herobm https://github.com/limresgrp/GEqTrain.git
git clone https://github.com/limresgrp/CGmap.git
git clone https://github.com/your-user/HEroBM.git   # Replace with correct URL

pip install -e ./GEqTrain
pip install -e ./CGmap
pip install -e ./HEroBM

conda deactivate
```

5. **Deploy the Activation Script**

```bash
chmod +x $HEROBM_ROOT/sourceme.sh
```

6. **(Optional) Deploy Models**

```bash
mkdir -p $HEROBM_ROOT/models
# scp your-model.pth user@host:$HEROBM_ROOT/models/
```

7. **Finalize Permissions**

```bash
sudo chown -R root:root $HEROBM_ROOT
sudo find $HEROBM_ROOT -type d -exec chmod 755 {} \;
sudo find $HEROBM_ROOT -type f -exec chmod 644 {} \;
sudo chmod +x $HEROBM_ROOT/sourceme.sh
sudo chmod +x $HEROBM_ROOT/env/bin/*
sudo chmod -R 777 "$HEROBM_ROOT/src/CGmap/cgmap/data/"
```

---

### Part 2: End-User Instructions

To use the centrally installed HEroBM:

```bash
# Load the environment
source /apps/herobm/sourceme.sh

# Check usage
herobm-backmap --help

# Example with deployed model
herobm-backmap \
    -i input.pdb \
    -it trajectory.xtc \
    -mo $HEROBM_MODELS_DIR/your-model.pth \
    -o ./output_structures \
    -s 'protein'

# Deactivate when done
conda deactivate
```

-----

## 🔧 Activation Script (`sourceme.sh`)

This script should be placed at `/apps/herobm/sourceme.sh`.

```bash
#!/bin/bash
# ==============================================================================
#  sourceme.sh for HEroBM
# ==============================================================================
#
#  Purpose:
#  Configures the shell environment to use HEroBM. It activates the Conda
#  environment and sets variables for deployed models.
#
#  Usage:
#  source /apps/herobm/sourceme.sh
# ==============================================================================

# Root directory
export HEROBM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check environment
if [ ! -d "$HEROBM_ROOT/env" ]; then
    echo "ERROR: HEroBM environment not found at $HEROBM_ROOT/env" >&2
    return 1
fi

# Activate environment
echo "Activating HEroBM environment..."
conda activate "$HEROBM_ROOT/env"

# Model directory
export HEROBM_MODELS_DIR="$HEROBM_ROOT/models"

echo "=========================================================="
echo " HEroBM Environment is now active."
echo ""
echo "  - Root Directory:    $HEROBM_ROOT"
echo "  - Deployed Models:   $HEROBM_MODELS_DIR"
echo ""
echo "  Use 'herobm-backmap' to run backmapping."
echo "  To deactivate: conda deactivate"
echo "=========================================================="
```