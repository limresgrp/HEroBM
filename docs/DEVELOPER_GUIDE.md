# Developer Guide: CLI and Script Arguments

This guide documents the CLI workflows and arguments used in this repository.

## 1) Environment

```bash
./venv_setup.sh
source .venv-herobm/bin/activate
```

## 2) Unified Workflow Script

Main entrypoint:

```bash
./interactive_console.sh
```

Non-interactive usage:

```bash
./interactive_console.sh optionX [args]
```

Available options:

- `complete-mapping`: complete mapping YAML files by assigning hierarchical labels
- `build-dataset`: build NPZ dataset
- `train`: run training (`geqtrain-train`)
- `cg-stats`: compute bead-distance statistics CSV (`herobm-cgstats`)
- `deploy`: deploy model with metadata (`herobm-deploy`)
- `backmap`: run inference/backmapping (`herobm-backmap`)

Legacy aliases are still accepted: `option1..option6` and `1..6`.

## 3) `interactive_console.sh` Option Arguments

### Command `complete-mapping`

Complete mapping YAML files:

- `--mapping-input-dir DIR` (required)
- `--atomistic-dir DIR` (required)
- `--mapping-output-dir DIR` (required)
- `--distance-threshold FLOAT` (default: `0.25`)
- `--overwrite-existing`
- `--python PYTHON_EXE` (default: `python`)

### Command `build-dataset`

Build NPZ dataset:

- `--mapping-dir DIR` (required)
- `--output-dir DIR` (required)
- `--input-mode folder|single`
- `--pdb-dir DIR` (folder mode)
- `--pdb-file FILE` + `--traj-file FILE` (single mode)
- `--input-format EXT` (default: `pdb`)
- `--selection SEL` (default: `protein`)
- `--bead-types-filename FILE` (default: `bead_types.yaml`)
- `--ignore-hydrogens`
- `--cutoff FLOAT`
- `--workers INT`
- `--traj-dir PATH` (optional trajectory file/folder in folder mode)
- `--traj-format EXT`
- `--trajslice SLICE`
- `--filter FILE`

### Command `train`

Training wrapper:

- `--config FILE` (required)
- `--device DEVICE`
- `--ddp`
- `--master-addr HOST`
- `--master-port PORT`
- `--find-unused-parameters`

### Command `cg-stats`

CG distance statistics:

- `--mapping-dir DIR` (required)
- `--input PATH` (required)
- `--inputtraj PATH`
- `--output FILE` (default: `cgdist.csv`)
- `--workers INT` (default: `1`)

### Command `deploy`

Deploy model:

- `--model FILE` (required)
- `--mapping-dir DIR` (required)
- `--output FILE` (default: `deployed_model.pt`)
- `--bead-types FILENAME_OR_PATH` (default: `bead_types.yaml`)
- `--bead-stats FILE`
- `--ignore-hydrogens` or `--predict-hydrogens`
- `--verbose LEVEL` (default: `INFO`)
- `--extra-metadata KEY=VALUE ...`

### Command `backmap`

Inference/backmapping:

- `--model FILE` (required)
- `--input FILE` (required)
- `--inputtraj FILE`
- `--output DIR` (default: `./output`)
- `--selection SEL` (default: `protein`)
- `--device DEVICE` (default: `cuda:0`)
- `--mapping DIR`
- `--bead-types-filename FILE`
- `--bead-stats FILE`
- `--ignore-hydrogens` or `--predict-hydrogens`
- `--trajslice SLICE`
- `--chunking INT`
- `--num-steps INT`
- `--tolerance FLOAT`
- `--atomistic`

## 4) Direct Script Entry Points

You can bypass `interactive_console.sh` and call scripts directly.

### 4.1 Mapping Label Completion (`cgmap.scripts.assign_hierarchical_labels`)

```bash
python -m cgmap.scripts.assign_hierarchical_labels [args]
```

Arguments:

- `-m, --mapping PATH` (required): mapping YAML file
- `-a, --atomistic PATH` (required): atomistic PDB
- `-c, --cg PATH`: optional aligned CG structure
- `-o, --output PATH`: output YAML (default: overwrite input)
- `--mapping-folder PATH`: mapping folder used when CG is auto-generated
- `--distance-threshold FLOAT` (default: `0.25`)
- `--overwrite-existing`
- `--plot-hierarchy PATH`
- `--plot-resid INT`
- `--plotly-hierarchy PATH`

### 4.2 Dataset Build (`herobm-dataset`)

```bash
herobm-dataset [args]
```

Arguments:

- `-m, --mapping PATH` (required)
- `-i, --input PATH` (required): file or folder
- `-if, --inputformat EXT` (default: `*`)
- `-t, --inputtraj PATH`
- `-tf, --trajformat EXT`
- `-f, --filter FILE`
- `-o, --output DIR`
- `-s, --selection SEL` (default: `all`)
- `-ts, --trajslice SLICE`
- `-b, --bead-types-filename FILE` (default: `bead_types.yaml`)
- `-c, --cutoff FLOAT`
- `--r-max FLOAT`: include `r_max` hint in printed config snippet
- `--isatomistic` (default true)
- `-cg` (set coarse-grained input mode)
- `-w, --workers INT`
- `--ignore-hydrogens`

### 4.3 Training (`geqtrain-train`)

```bash
geqtrain-train CONFIG.yaml [args]
```

Arguments:

- `config` positional (required)
- `-d, --device DEVICE`
- `--ddp`
- `-ma, --master-addr HOST`
- `-mp, --master-port PORT`
- `-u, --find-unused-parameters`
- `-o, --override KEY=VALUE` (repeatable)

### 4.4 CG Distance Statistics (`herobm-cgstats`)

```bash
herobm-cgstats [args]
```

Arguments:

- `-i, --input PATH` (required): PDB/GRO file or folder
- `-m, --mapping PATH` (required)
- `-t, --inputtraj PATH`
- `-o, --output FILE` (default: `out.csv`)
- `-w, --workers INT` (default: `1`)

### 4.5 Deployment (`herobm-deploy`)

```bash
herobm-deploy [args]
```

Base arguments (from GEqTrain deploy parser):

- `-m, --model PATH` (required)
- `-o, --out-file PATH` (default: `deployed.pth`)
- `-e, --extra-metadata KEY=VALUE ...`
- `--interactive-metadata`

HEroBM-specific arguments:

- `--verbose LEVEL` (default: `INFO`)
- `--mapping PATH` (required)
- `--bead-types-filename PATH` (default: `bead_types.yaml`)
- `--bead-stats PATH`
- `--ignore-hydrogens` or `--predict-hydrogens`

### 4.6 Backmapping Inference (`herobm-backmap`)

```bash
herobm-backmap [args]
```

Arguments:

- `-mo, --model PATH` (required)
- `-i, --input PATH`
- `-it, --inputtraj PATH`
- `-o, --output DIR` (default: `./output`)
- `-m, --mapping PATH`
- `-b, --bead-types-filename PATH`
- `--ignore-hydrogens` or `--predict-hydrogens`
- `-a, --atomistic`
- `-s, --selection SEL` (default: `all`)
- `-ts, --trajslice SLICE`
- `-d, --device DEVICE` (default: `cuda:0`)
- `-c, --chunking INT`
- `-bs, --bead-stats PATH`
- `-ns, --num-steps INT` (default: `1000`)
- `-t, --tolerance FLOAT` (default: `500.0`)

## 5) Config Placeholders to Replace

Before training, update:

- `config/experiment/protein_CA.yaml`
  - `root: /path/to/your/results`
- `config/data/protein_CA.yaml`
  - `dataset_npz: /path/to/your/a2a.npz`
