![HEroBM logo](logo.svg?raw=true "HEroBM")

## HEroBM: Backmapping Coarse-Grained Simulations to Atomistic Resolution

HEroBM reconstructs atomistic structures from coarse-grained (CG) structures/trajectories using equivariant graph neural networks and hierarchical reconstruction.

Paper: [HEroBM: A deep equivariant graph neural network for high-fidelity backmapping from coarse-grained to all-atom structures](https://doi.org/10.1063/5.0280330)

This README focuses on CLI workflows (no `webapp/` here).

## Repository Inputs Provided

This repository includes:
- `config/` example configs for Goten-based training experiments.
- `config/experiment/protein_CA.yaml` as the A2A CA training entry config.
- `mappings/tutorial/protein.ca/` with CA-only tutorial mapping YAMLs (to be completed with hierarchy labels).

Note: for now, the documented runnable example here is the A2A CA setup.

## 1) Environment Setup (`uv` + dependencies)

```bash
./venv_setup.sh
source .venv-herobm/bin/activate
```

The setup script installs HEroBM + GEqTrain + CGmap in one environment.

## 2) End-to-End Tutorial (A2A, CA-only)

Use the pipeline script (interactive console):

```bash
./mapping_dataset_pipeline.sh
```

Or run commands non-interactively using `option1..4`.

### Step A: Complete CA Mapping Files (`option1`)

This assigns hierarchical labels in `mappings/tutorial/protein.ca/*.yaml` and writes a completed mapping folder.

```bash
./mapping_dataset_pipeline.sh option1 \
  --mapping-input-dir "$PWD/mappings/tutorial/protein.ca" \
  --atomistic-dir /path/to/residue_structures \
  --mapping-output-dir "$PWD/mappings/tutorial/protein.ca.hierarchy"
```

Set mapping path for next steps:

```bash
export MAPPING_DIR="$PWD/mappings/tutorial/protein.ca.hierarchy"
```

### Step B: Build NPZ Dataset (`option2`)

Folder mode (typical train/valid split):

```bash
./mapping_dataset_pipeline.sh option2 \
  --mapping-dir "$MAPPING_DIR" \
  --input-mode folder \
  --pdb-dir /path/to/a2a/train_pdbs \
  --output-dir /path/to/a2a/npz/train \
  --input-format pdb \
  --selection protein \
  --bead-types-filename bead_types.yaml \
  --cutoff 10.0

./mapping_dataset_pipeline.sh option2 \
  --mapping-dir "$MAPPING_DIR" \
  --input-mode folder \
  --pdb-dir /path/to/a2a/valid_pdbs \
  --output-dir /path/to/a2a/npz/valid \
  --input-format pdb \
  --selection protein \
  --bead-types-filename bead_types.yaml \
  --cutoff 10.0
```

Single structure + trajectory mode:

```bash
./mapping_dataset_pipeline.sh option2 \
  --mapping-dir "$MAPPING_DIR" \
  --input-mode single \
  --pdb-file /path/to/system.pdb \
  --traj-file /path/to/system.xtc \
  --output-dir /path/to/a2a/npz/out \
  --selection protein \
  --bead-types-filename bead_types.yaml \
  --cutoff 10.0
```

Important: at the end of dataset building, `herobm-dataset` prints a config snippet with:
- `out_irreps`
- `num_types`
- `avg_num_neighbors`
- `type_names`

Use these values to update your training config.

### Step C: Train (`config/experiment/protein_CA.yaml`)

Use `config/experiment/protein_CA.yaml` as the training entry point for the A2A CA example.

Before training, ensure dataset paths/parameters in config files are consistent with your generated NPZ directories.

```bash
geqtrain-train config/experiment/protein_CA.yaml -d cuda:0
```

### Step D: Compute CG Distance Statistics (`option3`, optional)

This step uses `herobm/scripts/cg_analysis.py` (via `herobm-cgstats`) to build a CSV of bead distance statistics for optional CG pre-minimization before backmapping.

```bash
./mapping_dataset_pipeline.sh option3 \
  --mapping-dir "$MAPPING_DIR" \
  --input /path/to/atomistic/reference_or_folder \
  --output cgdist.protein_ca.csv \
  --workers 4
```

What this computes:
- Intra-residue distances for bead pairs within the same residue.
- Inter-residue distances only for consecutive residues and only for `BB`-`BB` bead pairs.

Output columns include `count`, `mean_distance`, `std_distance`, `min_distance`, `max_distance`.

How it is used:
- Pass this CSV at inference time (`--bead-stats`) to enable an optional CG-space pre-minimization step before atomistic reconstruction.
- This step is optional, but often improves robustness on real/noisy CG trajectories.

### Step E: Deploy Model with Metadata (`option4`)

Deploy a trained checkpoint and embed mapping metadata (plus optional bead stats CSV):

```bash
./mapping_dataset_pipeline.sh option4 \
  --model /path/to/training_run/best_model.pth \
  --mapping-dir "$MAPPING_DIR" \
  --output deployed/protein_ca.pt \
  --bead-types bead_types.yaml \
  --bead-stats cgdist.protein_ca.csv
```

This wraps `herobm-deploy` and stores metadata needed for inference in the deployed model.

### Step F: Inference with Deployed Model

Basic inference (metadata-driven):

```bash
herobm-backmap \
  -i /path/to/cg_structure.gro \
  -it /path/to/cg_trajectory.xtc \
  -mo deployed/protein_ca.pt \
  -s protein \
  -o backmapped_out \
  -d cuda:0
```

Explicit override mode (if needed):

```bash
herobm-backmap \
  -i /path/to/cg_structure.gro \
  -it /path/to/cg_trajectory.xtc \
  -mo deployed/protein_ca.pt \
  -m "$MAPPING_DIR" \
  -b "$MAPPING_DIR/bead_types.yaml" \
  -bs cgdist.protein_ca.csv \
  -o backmapped_out \
  -d cuda:0
```

## 3) CA Mapping Notes

For CA-only protein CG mapping:
- base mapping templates: `mappings/tutorial/protein.ca/`
- completed mappings with hierarchy labels: `mappings/tutorial/protein.ca.hierarchy/`

Typical CA mapping characteristics:
- one backbone-like bead (`BB`) per residue
- terminal caps as `RE`
- `CA` used as main anchor in mapping definitions (`P0A CM`)

## 4) Command Reference

```bash
./mapping_dataset_pipeline.sh --help
./mapping_dataset_pipeline.sh option1 --help
./mapping_dataset_pipeline.sh option2 --help
./mapping_dataset_pipeline.sh option3 --help
./mapping_dataset_pipeline.sh option4 --help

herobm-dataset --help
herobm-cgstats --help
geqtrain-train --help
herobm-deploy --help
herobm-backmap --help
```
