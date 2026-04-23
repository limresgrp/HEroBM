![HEroBM logo](logo.svg?raw=true "HEroBM")

## HEroBM

Backmapping coarse-grained structures/trajectories to atomistic resolution using equivariant GNNs and hierarchical reconstruction.

Paper: [HEroBM: A deep equivariant graph neural network for high-fidelity backmapping from coarse-grained to all-atom structures](https://doi.org/10.1063/5.0280330)

## Quick Start

### 1) Setup environment

```bash
./venv_setup.sh
source .venv-herobm/bin/activate
```

This installs HEroBM, GEqTrain, and CGmap in one `uv` virtual environment.

### 2) Run inference with an existing deployed model (first example)

Use the interactive console script in non-interactive mode (`backmap`):

```bash
./interactive_console.sh backmap \
  --model deployed/protein.CA/Mar2026.pt \
  --input data/A2A/md/a2a.pdb \
  --inputtraj data/A2A/md/a2a.xtc \
  --output output/readme_inference \
  --device cuda:0 \
  --atomistic
```

If the CG PDB uses a different residue name than the mapping expects, keep your
selection as usual and remap the selected residue name for backmapping:

```bash
./interactive_console.sh backmap \
  --model deployed/protein.CA/Mar2026.pt \
  --input data/A2A/md/a2a.pdb \
  --selection "resname COMP13" \
  --resname-map COMP13=CH1 \
  --output output/readme_inference \
  --device cuda:0 \
  --atomistic
```

You can pass `--resname-map` multiple times to remap several residue names.

If you prefer prompts:

```bash
./interactive_console.sh
```

then choose command `backmap` (or `6`).

## Interactive Console Workflow

`interactive_console.sh` exposes the full pipeline:

- `complete-mapping`: complete raw mapping YAML files with hierarchy labels
- `build-dataset`: build NPZ dataset
- `train`: train (`geqtrain-train`)
- `cg-stats`: compute CG distance statistics CSV (`herobm-cgstats`, optional)
- `deploy`: deploy model with metadata (`herobm-deploy`)
- `backmap`: run inference/backmapping (`herobm-backmap`)

The backmapping command accepts `--resname-map OLD=NEW` to rename selected CG
residues before the mapping step. This is useful when a structure file uses a
different ligand residue name than the corresponding mapping YAML. The option
can be repeated multiple times.

## Documentation

- End-to-end CA tutorial: [`docs/TUTORIAL_CA_ONLY.md`](docs/TUTORIAL_CA_ONLY.md)
- Detailed CLI/dev guide (all script arguments): [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md)

If you want direct script usage instead of `interactive_console.sh`, use the developer guide above.
