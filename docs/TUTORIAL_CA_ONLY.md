# Tutorial: CA-Only Workflow (Mapping -> Dataset -> Train -> Deploy -> Backmap)

This tutorial uses the CA-only example mappings in:

- `mappings/raw/protein.ca`

and the A2A example structure/trajectory in:

- `data/A2A/md/a2a.pdb`
- `data/A2A/md/a2a.xtc`

## 1) Environment Setup

```bash
./venv_setup.sh
source .venv-herobm/bin/activate
```

## 2) Complete Raw CA Mapping Files

Generate hierarchy-completed mappings from `mappings/raw/protein.ca`.
This is necessary to tell HEroBM which is the hierarchy for reconstructing atoms from beads.
The definition of hierarchies can be performed manually, but we offer an automated script which usually finds a good hierarchy definition:

```bash
./interactive_console.sh complete-mapping \
  --mapping-input-dir mappings/raw/protein.ca \
  --atomistic-dir data/A2A/md \
  --mapping-output-dir mappings/complete/protein.ca.hierarchy \
  --overwrite-existing
```

Result:

- completed mapping folder: `mappings/complete/protein.ca.hierarchy`

## 3) Build NPZ Dataset

Create a dataset from a single structure + trajectory:

```bash
./interactive_console.sh build-dataset \
  --mapping-dir mappings/complete/protein.ca.hierarchy \
  --input-mode single \
  --pdb-file data/A2A/md/a2a.pdb \
  --traj-file data/A2A/md/a2a.xtc \
  --output-dir data/A2A/npz/protein.CA.tutorial \
  --selection protein \
  --bead-types-filename bead_types.yaml \
  --ignore-hydrogens \
  --cutoff 12.0
```

At the end, `herobm-dataset` prints a configuration snippet with:

- `out_irreps`
- `num_types`
- `avg_num_neighbors`
- `type_names`
- optional `r_max` (if provided with `--r-max`)

Use these values to verify/update your `config/data/protein_CA.yaml` and `config/model/protein_CA_goten.yaml`.


## 4) Update Config Placeholders

Before training, edit:

- `config/data/protein_CA.yaml`
  - set `dataset_npz` to your new NPZ path, for example:
    - `dataset_npz: /ABS/PATH/TO/HEroBM/data/A2A/npz/protein.CA.tutorial/a2a.npz`
- `config/experiment/protein_CA.yaml`
  - set `root` to your training output directory, for example:
    - `root: /ABS/PATH/TO/results`

## 5) Train with Default Experiment Config

```bash
./interactive_console.sh train \
  --config config/experiment/protein_CA.yaml \
  --device cuda:0
```

Training outputs are written under:

- `<root>/<run_name>/`

Main files:

- `best_model.pth`
- `last_model.pth`
- `trainer.pth`
- `metrics_epoch.csv`

## 6) Deploy the Trained Model

```bash
./interactive_console.sh deploy \
  --model /ABS/PATH/TO/results/protein_CA_goten/best_model.pth \
  --mapping-dir mappings/complete/protein.ca.hierarchy \
  --bead-types bead_types.yaml \
  --output deployed/protein.CA/tutorial.pt
```

## 7) Test Backmapping on Atomistic Input (Accuracy Check)

Run atomistic->CG->backmapping with the deployed model:

```bash
./interactive_console.sh backmap \
  --model deployed/protein.CA/tutorial.pt \
  --input data/A2A/md/a2a.pdb \
  --inputtraj data/A2A/md/a2a.xtc \
  --output output/tutorial_eval \
  --device cuda:0 \
  --atomistic
```

You should get paired files such as:

- `output/tutorial_eval/a2a.pdb.true_0.pdb`
- `output/tutorial_eval/a2a.pdb.backmapped_0.pdb`

## 8) Compute RMSD Between True and Backmapped Structures

Example quick check (raw RMSD, no fitting):

```bash
python - <<'PY'
import MDAnalysis as mda
import numpy as np

u_true = mda.Universe("output/tutorial_eval/a2a.pdb.true_0.pdb")
u_pred = mda.Universe("output/tutorial_eval/a2a.pdb.backmapped_0.pdb")

ref = u_true.select_atoms("all").positions
prd = u_pred.select_atoms("all").positions
rmsd = np.sqrt(np.mean(np.sum((prd - ref) ** 2, axis=1)))
print(f"RMSD (all atoms, raw): {rmsd:.4f} A")
PY
```

To compute heavy-atom RMSD, change selection to `not name H*`.

## 9) Optional: CG Distance Statistics

Generate optional CG statistics (used for optional CG pre-minimization during inference):

```bash
./interactive_console.sh cg-stats \
  --mapping-dir mappings/complete/protein.ca.hierarchy \
  --input data/A2A/md/a2a.pdb \
  --inputtraj data/A2A/md/a2a.xtc \
  --output cgdist/tutorial.csv \
  --workers 1
```

Then pass it at inference time with `backmap`:

```bash
./interactive_console.sh backmap \
  --model deployed/protein.CA/tutorial.pt \
  --input data/A2A/md/a2a.pdb \
  --inputtraj data/A2A/md/a2a.xtc \
  --output output/tutorial_eval_with_cgmin \
  --device cuda:0 \
  --atomistic \
  --bead-stats cgdist/tutorial.csv
```
