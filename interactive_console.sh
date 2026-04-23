#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"

print_main_help() {
  cat <<USAGE
Usage:
  ${SCRIPT_NAME} complete-mapping [args] # Complete mapping YAML files with hierarchical labels
  ${SCRIPT_NAME} build-dataset [args]    # Build NPZ dataset using herobm-dataset
  ${SCRIPT_NAME} train [args]            # Train model (geqtrain-train)
  ${SCRIPT_NAME} cg-stats [args]         # Compute CG bead distance statistics (herobm-cgstats)
  ${SCRIPT_NAME} deploy [args]           # Deploy model with HEroBM metadata (herobm-deploy)
  ${SCRIPT_NAME} backmap [args]          # Run inference/backmapping (herobm-backmap)
  ${SCRIPT_NAME} test [args]             # Evaluate RMSD on NPZ or atomistic inputs (herobm-test)

Interactive mode:
  ${SCRIPT_NAME}
  (you will be prompted to choose a command and enter required paths)

Options:
  1 / complete-mapping : Complete mapping YAML files by assigning hierarchical labels
  2 / build-dataset    : Build NPZ dataset from atomistic structures using a completed mapping folder
  3 / train            : Run training using a selected config (geqtrain-train)
  4 / cg-stats         : Compute bead-distance statistics CSV for optional CG pre-minimization
  5 / deploy           : Deploy a trained checkpoint with mapping/bead metadata embedded
  6 / backmap          : Run backmapping inference with trained/deployed model
  7 / test             : Map to CG, backmap, and compute RMSD against references
  h / --help  : Show help and option descriptions
  q           : Quit interactive mode

Legacy aliases are still accepted: option1..option7 and 1..7.
Run '${SCRIPT_NAME} <command> --help' for details.
USAGE
}

print_option1_help() {
  cat <<USAGE
Command: complete-mapping
Complete mapping YAML files using cgmap.scripts.assign_hierarchical_labels

Usage:
  ${SCRIPT_NAME} complete-mapping \
    --mapping-input-dir DIR \
    --atomistic-dir DIR \
    --mapping-output-dir DIR \
    [--distance-threshold FLOAT] \
    [--skip-missing-structures] \
    [--overwrite-existing] \
    [--python PYTHON_EXE]

Required arguments:
  --mapping-input-dir DIR    Folder containing input mapping YAML files (*.yaml)
  --atomistic-dir DIR        Folder containing atomistic residue structures (*.pdb or *.gro)
  --mapping-output-dir DIR   Folder where completed mapping YAML files are written

Optional arguments:
  --distance-threshold FLOAT Passed to assign_hierarchical_labels (default: 0.25)
  --skip-missing-structures Continue even if some mappings have no matching residue in atomistic input
  --overwrite-existing       Recompute labels even if already present
  --python PYTHON_EXE        Python executable to use (default: python)

Notes:
- The script scans every `.pdb`/`.gro` file in `--atomistic-dir`, reads the residue names present
  in each structure, and completes any still-incomplete mapping YAML whose `molecule` matches.
- Mapping YAMLs that are already complete are copied/skipped unless `--overwrite-existing` is used.
- File names in `--atomistic-dir` do not matter; only the structure format and contained residue names matter.
- By default the command fails if any incomplete mapping has no matching residue in the atomistic input.
  Use `--skip-missing-structures` to continue and manually complete those copied YAMLs later.
USAGE
}

print_option2_help() {
  cat <<USAGE
Command: build-dataset
Build NPZ dataset using herobm-dataset

Usage:
  ${SCRIPT_NAME} build-dataset \
    --mapping-dir DIR \
    [--input-mode folder|single] \
    [--pdb-dir DIR | --pdb-file FILE --traj-file FILE] \
    --output-dir DIR \
    [--input-format EXT] \
    [--selection SEL] \
    [--bead-types-filename FILE] \
    [--ignore-hydrogens] \
    [--cutoff FLOAT] \
    [--workers INT] \
    [--traj-dir PATH] \
    [--traj-format EXT] \
    [--trajslice SLICE] \
    [--filter FILE]

Required arguments:
  --mapping-dir DIR          Completed mapping folder (from complete-mapping)
  --output-dir DIR           Output folder for .npz dataset

Input mode options:
  --input-mode folder        Use a folder of structures (default when --pdb-dir is set)
  --input-mode single        Use a single structure + trajectory pair
  --pdb-dir DIR              Folder with atomistic structures (folder mode)
  --pdb-file FILE            Single structure file, e.g. *.pdb/*.gro (single mode)
  --traj-file FILE           Single trajectory file, e.g. *.xtc/*.trr (single mode)

Optional arguments:
  --input-format EXT         Structure extension for --pdb-dir scan (default: pdb)
  --selection SEL            MDAnalysis selection (default: protein)
  --bead-types-filename FILE Bead types YAML filename in mapping dir (default: bead_types.yaml)
  --ignore-hydrogens         Ignore hierarchy labels for atoms with names starting with H
  --cutoff FLOAT             Precompute neighbor edges cutoff (recommended)
  --workers INT              Number of workers for herobm-dataset
  --traj-dir PATH            Optional trajectory file/folder
  --traj-format EXT          Trajectory extension when --traj-dir is a folder
  --trajslice SLICE          Trajectory slice, e.g. ::10
  --filter FILE              Text file listing basenames to include
USAGE
}

print_option3_help() {
  cat <<USAGE
Command: train
Train model using geqtrain-train

Usage:
  ${SCRIPT_NAME} train \
    --config FILE \
    [--device DEVICE] \
    [--ddp] \
    [--master-addr HOST] \
    [--master-port PORT] \
    [--find-unused-parameters]

Required arguments:
  --config FILE              Training config file for geqtrain-train

Optional arguments:
  --device DEVICE            Training device, e.g. cuda:0 or cpu
  --ddp                      Enable distributed training mode
  --master-addr HOST         DDP master address
  --master-port PORT         DDP master port
  --find-unused-parameters   Enable DDP find_unused_parameters
USAGE
}

print_option4_help() {
  cat <<USAGE
Command: cg-stats
Compute CG bead-distance statistics using herobm-cgstats

Usage:
  ${SCRIPT_NAME} cg-stats \
    --mapping-dir DIR \
    --input PATH \
    [--inputtraj PATH] \
    [--output FILE] \
    [--workers INT]

Required arguments:
  --mapping-dir DIR          Mapping folder used for atomistic->CG conversion
  --input PATH               Atomistic input file or folder (*.pdb/*.gro)

Optional arguments:
  --inputtraj PATH           Trajectory file/folder corresponding to input
  --output FILE              Output CSV path (default: cgdist.csv)
  --workers INT              Number of workers (default: 1)
USAGE
}

print_option5_help() {
  cat <<USAGE
Command: deploy
Deploy a trained model with HEroBM metadata using herobm-deploy

Usage:
  ${SCRIPT_NAME} deploy \
    --model FILE \
    --mapping-dir DIR \
    [--output FILE] \
    [--bead-types FILENAME_OR_PATH] \
    [--bead-stats FILE] \
    [--ignore-hydrogens | --predict-hydrogens] \
    [--verbose LEVEL] \
    [--extra-metadata KEY=VALUE ...]

Required arguments:
  --model FILE               Trained model checkpoint (e.g. best_model.pth)
  --mapping-dir DIR          Completed mapping folder

Optional arguments:
  --output FILE              Deployed model path (default: deployed_model.pt)
  --bead-types NAME_OR_PATH  Bead types filename/path (default: bead_types.yaml)
  --bead-stats FILE          CSV from cg-stats (optional)
  --ignore-hydrogens         Store in metadata that H* hierarchy labels are ignored
  --predict-hydrogens        Store in metadata that H* hierarchy labels are used
  --verbose LEVEL            Logging level for deploy (default: INFO)
  --extra-metadata ...       Extra key=value pairs embedded in metadata
USAGE
}

print_option6_help() {
  cat <<USAGE
Command: backmap
Run inference/backmapping using herobm-backmap

Usage:
  ${SCRIPT_NAME} backmap \
    --model FILE \
    --input FILE \
    [--inputtraj FILE] \
    [--output DIR] \
    [--selection SEL] \
    [--device DEVICE] \
    [--mapping DIR] \
    [--bead-types-filename FILE] \
    [--bead-stats FILE] \
    [--ignore-hydrogens | --predict-hydrogens] \
    [--trajslice SLICE] \
    [--chunking INT] \
    [--num-steps INT] \
    [--tolerance FLOAT] \
    [--atomistic]

Required arguments:
  --model FILE               Trained/deployed model file
  --input FILE               Input structure file (.pdb/.gro)

Optional arguments:
  --inputtraj FILE           Input trajectory (.xtc/.trr)
  --output DIR               Output directory (default: ./output)
  --selection SEL            Selection string (default: protein)
  --device DEVICE            Inference device (default: cuda:0)
  --mapping DIR              Mapping folder (override model metadata)
  --bead-types-filename FILE Bead types file (override metadata)
  --bead-stats FILE          Bead stats CSV (override metadata)
  --ignore-hydrogens         Ignore H* hierarchy labels (override metadata)
  --predict-hydrogens        Use H* hierarchy labels (override metadata)
  --trajslice SLICE          Trajectory slice, e.g. ::10
  --chunking INT             Max atoms per chunk
  --num-steps INT            CG minimization steps
  --tolerance FLOAT          Atomistic minimization tolerance
  --atomistic                Mark input as atomistic
USAGE
}

print_option7_help() {
  cat <<USAGE
Command: test
Evaluate a model by mapping atomistic input to CG, backmapping, and computing RMSD using herobm-test

Usage:
  ${SCRIPT_NAME} test \
    --model FILE \
    --output DIR \
    [--npz PATH | --pdb-dir DIR | --pdb-file FILE [--traj-file FILE]] \
    [--traj-dir PATH] \
    [--input-format EXT] \
    [--traj-format EXT] \
    [--filter FILE] \
    [--input-selection SEL] \
    [--mapping DIR] \
    [--bead-types-filename FILE] \
    [--bead-stats FILE] \
    [--ignore-hydrogens | --predict-hydrogens] \
    [--base-rmsd-selection NAME ...] \
    [--rmsd-selection LABEL=SELECTION ...] \
    [--trajslice SLICE] \
    [--device DEVICE] \
    [--chunking INT] \
    [--tolerance FLOAT]

Required arguments:
  --model FILE               Trained/deployed model file
  --output DIR               Output directory for generated structures and RMSD CSVs

Input arguments:
  --npz PATH                 NPZ file or folder of NPZ files to evaluate
  --pdb-dir DIR              Folder of atomistic structures to map on the fly
  --pdb-file FILE            Single atomistic structure to map on the fly
  --traj-file FILE           Trajectory paired with --pdb-file
  --traj-dir PATH            Trajectory file or folder paired with --pdb-dir
  --input-format EXT         Structure extension for --pdb-dir scan (default: pdb)
  --traj-format EXT          Trajectory extension when --traj-dir is a folder
  --filter FILE              Text file listing basenames to include from --pdb-dir
  --input-selection SEL      Selection used when generating NPZs from raw atomistic input

Model/evaluation arguments:
  --mapping DIR              Mapping folder (override model metadata; required for raw input unless embedded)
  --bead-types-filename FILE Bead types file (override metadata)
  --bead-stats FILE          Bead stats CSV (override metadata)
  --ignore-hydrogens         Ignore H* hierarchy labels (override metadata)
  --predict-hydrogens        Use H* hierarchy labels (override metadata)
  --base-rmsd-selection NAME Built-in RMSD selection alias. Available: protein-backbone, protein-sidechains
  --rmsd-selection ...       Custom RMSD selection in the form label=MDAnalysis_selection
  --trajslice SLICE          Trajectory slice, e.g. ::10
  --device DEVICE            Inference device (default: cuda:0)
  --chunking INT             Max atoms per chunk
  --tolerance FLOAT          Atomistic minimization tolerance when writing structures (default: 0)

Notes:
- RMSD for all atoms is always computed.
- Built-in RMSD aliases are optional extras on top of the all-atoms metric.
- Output includes generated CG/backmapped/true structures plus `rmsd_per_frame.csv` and `rmsd_summary.csv`.
USAGE
}

require_dir() {
  local d="$1"
  local label="$2"
  if [[ ! -d "$d" ]]; then
    echo "Error: ${label} does not exist or is not a directory: $d" >&2
    exit 1
  fi
}

require_file() {
  local f="$1"
  local label="$2"
  if [[ ! -f "$f" ]]; then
    echo "Error: ${label} does not exist or is not a file: $f" >&2
    exit 1
  fi
}

prompt_with_default() {
  local label="$1"
  local default="$2"
  local value
  read -r -p "${label} [${default}]: " value
  if [[ -z "$value" ]]; then
    echo "$default"
  else
    echo "$value"
  fi
}

prompt_optional() {
  local label="$1"
  local value
  read -r -p "${label} (leave blank to skip): " value
  echo "$value"
}

prompt_option_choice() {
  local choice
  while true; do
    read -r -p $'Choose a command:\n  1) complete-mapping  - Complete mapping YAML files by assigning hierarchical labels\n  2) build-dataset     - Build NPZ dataset from atomistic structures using completed mappings\n  3) train             - Run training (geqtrain-train)\n  4) cg-stats          - Compute bead-distance statistics CSV (herobm-cgstats)\n  5) deploy            - Deploy trained model with HEroBM metadata (herobm-deploy)\n  6) backmap           - Run inference/backmapping (herobm-backmap)\n  7) test              - Evaluate RMSD via map->backmap (herobm-test)\n  h) Show help and option descriptions\n  q) Quit\nSelection: ' choice
    case "$choice" in
      1|complete-mapping) echo "complete-mapping"; return 0 ;;
      2|build-dataset) echo "build-dataset"; return 0 ;;
      3|train) echo "train"; return 0 ;;
      4|cg-stats) echo "cg-stats"; return 0 ;;
      5|deploy) echo "deploy"; return 0 ;;
      6|backmap) echo "backmap"; return 0 ;;
      7|test) echo "test"; return 0 ;;
      h|H) echo "help"; return 0 ;;
      q|Q) echo "quit"; return 0 ;;
      *) echo "Invalid choice: $choice" >&2 ;;
    esac
  done
}

extract_molecule_name() {
  local mapping_file="$1"
  awk -F':' '/^molecule:/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; exit}' "$mapping_file"
}

mapping_has_complete_labels() {
  local py_bin="$1"
  local mapping_file="$2"

  "$py_bin" - "$mapping_file" <<'PY'
import re
import sys
from pathlib import Path

import yaml

label_re = re.compile(r"^P\d+[A-Z]+$")
mapping_file = Path(sys.argv[1])
conf = yaml.safe_load(mapping_file.read_text()) or {}
atoms = conf.get("atoms") or {}

if not atoms:
    print("0")
    raise SystemExit(0)

for raw_value in atoms.values():
    tokens = str(raw_value).split()
    if not tokens:
        print("0")
        raise SystemExit(0)
    bead_count = len(tokens[0].split(","))
    label_columns = []
    for token in tokens[1:]:
        values = [value.strip() for value in token.split(",")]
        if len(values) < bead_count:
            values += [""] * (bead_count - len(values))
        elif len(values) > bead_count:
            values = values[:bead_count]
        if any(label_re.match(value) for value in values):
            label_columns.append(values)
    if not label_columns:
        print("0")
        raise SystemExit(0)
    merged = [""] * bead_count
    for values in label_columns:
        for i, value in enumerate(values):
            if label_re.match(value):
                merged[i] = value
    if any(not label_re.match(value) for value in merged):
        print("0")
        raise SystemExit(0)

print("1")
PY
}

list_structure_resnames() {
  local py_bin="$1"
  local structure_file="$2"

  "$py_bin" - "$structure_file" <<'PY'
import sys

import MDAnalysis as mda

u = mda.Universe(sys.argv[1])
resnames = sorted({str(res.resname).strip().upper() for res in u.residues if str(res.resname).strip()})
for resname in resnames:
    print(resname)
PY
}

run_option1() {
  local mapping_input_dir=""
  local atomistic_dir=""
  local mapping_output_dir=""
  local distance_threshold="0.25"
  local skip_missing_structures=0
  local overwrite_existing=0
  local py_bin="python"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --mapping-input-dir)
        mapping_input_dir="$2"
        shift 2
        ;;
      --atomistic-dir)
        atomistic_dir="$2"
        shift 2
        ;;
      --mapping-output-dir)
        mapping_output_dir="$2"
        shift 2
        ;;
      --distance-threshold)
        distance_threshold="$2"
        shift 2
        ;;
      --skip-missing-structures)
        skip_missing_structures=1
        shift
        ;;
      --overwrite-existing)
        overwrite_existing=1
        shift
        ;;
      --python)
        py_bin="$2"
        shift 2
        ;;
      -h|--help)
        print_option1_help
        exit 0
        ;;
      *)
        echo "Unknown argument for complete-mapping: $1" >&2
        print_option1_help
        exit 1
        ;;
    esac
  done

  if [[ -z "$mapping_input_dir" || -z "$atomistic_dir" || -z "$mapping_output_dir" ]]; then
    echo "Error: complete-mapping requires --mapping-input-dir, --atomistic-dir, and --mapping-output-dir" >&2
    print_option1_help
    exit 1
  fi

  require_dir "$mapping_input_dir" "mapping input dir"
  require_dir "$atomistic_dir" "atomistic dir"

  if ! command -v "$py_bin" >/dev/null 2>&1; then
    echo "Error: Python executable not found: $py_bin" >&2
    exit 1
  fi

  mkdir -p "$mapping_output_dir"

  mapfile -t mapping_files < <(find "$mapping_input_dir" -maxdepth 1 -type f -name '*.yaml' ! -name 'bead_types*.yaml' | sort)
  if [[ ${#mapping_files[@]} -eq 0 ]]; then
    echo "Error: no YAML mapping files found in $mapping_input_dir" >&2
    exit 1
  fi

  mapfile -t atomistic_files < <(find "$atomistic_dir" -maxdepth 1 -type f \( -iname '*.pdb' -o -iname '*.gro' \) | sort)
  if [[ ${#atomistic_files[@]} -eq 0 ]]; then
    echo "Error: no atomistic structure files (*.pdb, *.gro) found in $atomistic_dir" >&2
    exit 1
  fi

  declare -A pending_mapping_by_resname=()
  declare -A attempted_mapping_by_resname=()
  local mapping_file out_file molecule source_file completed_flag
  local already_complete=0

  for mapping_file in "${mapping_files[@]}"; do
    molecule="$(extract_molecule_name "$mapping_file")"
    if [[ -z "$molecule" ]]; then
      echo "ERROR: missing molecule field in $(basename "$mapping_file")" >&2
      exit 1
    fi
    molecule="${molecule^^}"
    out_file="$mapping_output_dir/$(basename "$mapping_file")"
    source_file="$mapping_file"

    if [[ "$overwrite_existing" -eq 0 && -f "$out_file" ]]; then
      completed_flag="$(mapping_has_complete_labels "$py_bin" "$out_file")"
      if [[ "$completed_flag" == "1" ]]; then
        ((already_complete+=1))
        continue
      fi
    fi

    completed_flag="$(mapping_has_complete_labels "$py_bin" "$mapping_file")"
    if [[ "$completed_flag" == "1" ]]; then
      if [[ ! -f "$out_file" ]] || ! cmp -s "$mapping_file" "$out_file"; then
        cp "$mapping_file" "$out_file"
      fi
      ((already_complete+=1))
      continue
    fi

    if [[ -n "${pending_mapping_by_resname[$molecule]:-}" ]]; then
      echo "ERROR: duplicate molecule entry '${molecule}' in $(basename "${pending_mapping_by_resname[$molecule]}") and $(basename "$mapping_file")" >&2
      exit 1
    fi

    pending_mapping_by_resname["$molecule"]="$mapping_file"
  done

  local total_mappings="${#mapping_files[@]}"
  local completed_count="$already_complete"
  local failures=0
  local missing_structures=0
  local structure_file resname
  local -a structure_resnames=()
  local -a remaining_resnames=()
  local -a failed_attempts=()
  local -a skipped_missing_mappings=()
  local cmd

  if [[ "${#pending_mapping_by_resname[@]}" -gt 0 ]]; then
    for structure_file in "${atomistic_files[@]}"; do
      if [[ "${#pending_mapping_by_resname[@]}" -eq 0 ]]; then
        break
      fi

      mapfile -t structure_resnames < <(list_structure_resnames "$py_bin" "$structure_file")
      if [[ ${#structure_resnames[@]} -eq 0 ]]; then
        echo "Skipping $(basename "$structure_file"): no residue names found"
        continue
      fi

      remaining_resnames=()
      for resname in "${structure_resnames[@]}"; do
        if [[ -n "${pending_mapping_by_resname[$resname]:-}" ]]; then
          remaining_resnames+=("$resname")
        fi
      done

      if [[ ${#remaining_resnames[@]} -eq 0 ]]; then
        continue
      fi

      echo "Scanning $(basename "$structure_file") for: ${remaining_resnames[*]}"

      for resname in "${remaining_resnames[@]}"; do
        mapping_file="${pending_mapping_by_resname[$resname]}"
        out_file="$mapping_output_dir/$(basename "$mapping_file")"
        attempted_mapping_by_resname["$resname"]=1
        echo "[$((completed_count + 1))/$total_mappings] $(basename "$mapping_file") <- $(basename "$structure_file") [$resname]"

        cmd=(
          "$py_bin" -m cgmap.scripts.assign_hierarchical_labels
          -m "$mapping_file"
          -a "$structure_file"
          -o "$out_file"
          --mapping-folder "$mapping_input_dir"
          --distance-threshold "$distance_threshold"
        )
        if [[ "$overwrite_existing" -eq 1 ]]; then
          cmd+=(--overwrite-existing)
        fi

        if "${cmd[@]}"; then
          unset 'pending_mapping_by_resname[$resname]'
          ((completed_count+=1))
        else
          echo "WARNING: failed to complete $(basename "$mapping_file") using $(basename "$structure_file"); will keep searching" >&2
          failed_attempts+=("$(basename "$mapping_file") <- $(basename "$structure_file") [$resname]")
        fi
      done
    done
  fi

  if [[ -f "$mapping_input_dir/bead_types.yaml" && ! -f "$mapping_output_dir/bead_types.yaml" ]]; then
    cp "$mapping_input_dir/bead_types.yaml" "$mapping_output_dir/bead_types.yaml"
    echo "Copied bead_types.yaml to output mapping folder"
  fi

  if [[ "${#pending_mapping_by_resname[@]}" -gt 0 ]]; then
    local unresolved
    for unresolved in "${!pending_mapping_by_resname[@]}"; do
      if [[ -n "${attempted_mapping_by_resname[$unresolved]:-}" ]]; then
        echo "ERROR: found residue ${unresolved} in structure files, but all completion attempts failed for $(basename "${pending_mapping_by_resname[$unresolved]}")" >&2
        ((failures+=1))
      else
        mapping_file="${pending_mapping_by_resname[$unresolved]}"
        out_file="$mapping_output_dir/$(basename "$mapping_file")"
        if [[ "$skip_missing_structures" -eq 1 ]]; then
          if [[ ! -f "$out_file" ]] || ! cmp -s "$mapping_file" "$out_file"; then
            cp "$mapping_file" "$out_file"
          fi
          echo "WARNING: no structure file contained residue ${unresolved} for $(basename "$mapping_file"); copied raw mapping to output for manual completion" >&2
          skipped_missing_mappings+=("$(basename "$mapping_file") [$unresolved]")
          ((missing_structures+=1))
        else
          echo "ERROR: no structure file contained residue ${unresolved} for $(basename "$mapping_file"). Re-run with --skip-missing-structures to continue and manually complete that YAML." >&2
          ((failures+=1))
        fi
      fi
    done
  fi

  if [[ "$failures" -eq 0 && "${#failed_attempts[@]}" -gt 0 ]]; then
    echo "Resolved after retrying with later structure files:"
    printf '  %s\n' "${failed_attempts[@]}"
  fi

  if [[ "$missing_structures" -gt 0 ]]; then
    echo "Skipped $missing_structures mapping(s) with no matching atomistic residue. Complete these manually in $mapping_output_dir:"
    printf '  %s\n' "${skipped_missing_mappings[@]}"
  fi

  if [[ "$failures" -gt 0 ]]; then
    echo "Completed with $failures failures." >&2
    exit 1
  fi

  echo "Complete-mapping completed successfully. Output folder: $mapping_output_dir"
}

run_option2() {
  local input_mode=""
  local mapping_dir=""
  local pdb_dir=""
  local pdb_file=""
  local traj_file=""
  local output_dir=""
  local input_format="pdb"
  local selection="protein"
  local bead_types_filename="bead_types.yaml"
  local ignore_hydrogens=0
  local cutoff=""
  local workers=""
  local traj_dir=""
  local traj_format=""
  local trajslice=""
  local filter_file=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --mapping-dir)
        mapping_dir="$2"
        shift 2
        ;;
      --input-mode)
        input_mode="$2"
        shift 2
        ;;
      --pdb-dir)
        pdb_dir="$2"
        shift 2
        ;;
      --pdb-file)
        pdb_file="$2"
        shift 2
        ;;
      --traj-file)
        traj_file="$2"
        shift 2
        ;;
      --output-dir)
        output_dir="$2"
        shift 2
        ;;
      --input-format)
        input_format="$2"
        shift 2
        ;;
      --selection)
        selection="$2"
        shift 2
        ;;
      --bead-types-filename)
        bead_types_filename="$2"
        shift 2
        ;;
      --ignore-hydrogens)
        ignore_hydrogens=1
        shift
        ;;
      --cutoff)
        cutoff="$2"
        shift 2
        ;;
      --workers)
        workers="$2"
        shift 2
        ;;
      --traj-dir)
        traj_dir="$2"
        shift 2
        ;;
      --traj-format)
        traj_format="$2"
        shift 2
        ;;
      --trajslice)
        trajslice="$2"
        shift 2
        ;;
      --filter)
        filter_file="$2"
        shift 2
        ;;
      -h|--help)
        print_option2_help
        exit 0
        ;;
      *)
        echo "Unknown argument for build-dataset: $1" >&2
        print_option2_help
        exit 1
        ;;
    esac
  done

  if [[ -z "$mapping_dir" || -z "$output_dir" ]]; then
    echo "Error: build-dataset requires --mapping-dir and --output-dir" >&2
    print_option2_help
    exit 1
  fi

  require_dir "$mapping_dir" "mapping dir"

  if [[ -z "$input_mode" ]]; then
    if [[ -n "$pdb_file" || -n "$traj_file" ]]; then
      input_mode="single"
    elif [[ -n "$pdb_dir" ]]; then
      input_mode="folder"
    fi
  fi

  if [[ -z "$input_mode" ]]; then
    echo "Error: could not infer input mode. Provide --input-mode or input paths." >&2
    print_option2_help
    exit 1
  fi

  local input_path=""
  local traj_input=""
  case "$input_mode" in
    folder)
      if [[ -z "$pdb_dir" ]]; then
        echo "Error: folder mode requires --pdb-dir" >&2
        exit 1
      fi
      require_dir "$pdb_dir" "pdb dir"
      input_path="$pdb_dir"
      traj_input="$traj_dir"
      ;;
    single)
      if [[ -z "$pdb_file" || -z "$traj_file" ]]; then
        echo "Error: single mode requires --pdb-file and --traj-file" >&2
        exit 1
      fi
      require_file "$pdb_file" "pdb file"
      require_file "$traj_file" "trajectory file"
      input_path="$pdb_file"
      traj_input="$traj_file"
      ;;
    *)
      echo "Error: invalid --input-mode '$input_mode' (expected: folder|single)" >&2
      exit 1
      ;;
  esac

  mkdir -p "$output_dir"

  local cmd=(
    herobm-dataset
    -m "$mapping_dir"
    -i "$input_path"
    -if "$input_format"
    -o "$output_dir"
    -s "$selection"
    -b "$bead_types_filename"
  )
  if [[ "$ignore_hydrogens" -eq 1 ]]; then
    cmd+=(--ignore-hydrogens)
  fi

  if [[ -n "$cutoff" ]]; then
    cmd+=(-c "$cutoff")
  fi
  if [[ -n "$workers" ]]; then
    cmd+=(-w "$workers")
  fi
  if [[ -n "$traj_input" ]]; then
    cmd+=(-t "$traj_input")
  fi
  if [[ "$input_mode" == "folder" && -n "$traj_format" ]]; then
    cmd+=(-tf "$traj_format")
  fi
  if [[ -n "$trajslice" ]]; then
    cmd+=(-ts "$trajslice")
  fi
  if [[ -n "$filter_file" ]]; then
    cmd+=(-f "$filter_file")
  fi

  echo "Running: ${cmd[*]}"
  "${cmd[@]}"

  echo "Build-dataset completed successfully. NPZ output folder: $output_dir"
}

run_option3() {
  local config_file=""
  local device=""
  local ddp=0
  local master_addr=""
  local master_port=""
  local find_unused_parameters=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config)
        config_file="$2"
        shift 2
        ;;
      --device)
        device="$2"
        shift 2
        ;;
      --ddp)
        ddp=1
        shift
        ;;
      --master-addr)
        master_addr="$2"
        shift 2
        ;;
      --master-port)
        master_port="$2"
        shift 2
        ;;
      --find-unused-parameters)
        find_unused_parameters=1
        shift
        ;;
      -h|--help)
        print_option3_help
        exit 0
        ;;
      *)
        echo "Unknown argument for train: $1" >&2
        print_option3_help
        exit 1
        ;;
    esac
  done

  if [[ -z "$config_file" ]]; then
    echo "Error: train requires --config" >&2
    print_option3_help
    exit 1
  fi
  require_file "$config_file" "training config"

  local cmd=(geqtrain-train "$config_file")
  if [[ -n "$device" ]]; then
    cmd+=(-d "$device")
  fi
  if [[ "$ddp" -eq 1 ]]; then
    cmd+=(--ddp)
  fi
  if [[ -n "$master_addr" ]]; then
    cmd+=(--master-addr "$master_addr")
  fi
  if [[ -n "$master_port" ]]; then
    cmd+=(--master-port "$master_port")
  fi
  if [[ "$find_unused_parameters" -eq 1 ]]; then
    cmd+=(--find-unused-parameters)
  fi

  echo "Running: ${cmd[*]}"
  "${cmd[@]}"
  echo "Training completed successfully."
}

run_option4() {
  local mapping_dir=""
  local input_path=""
  local inputtraj=""
  local output_csv="cgdist.csv"
  local workers="1"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --mapping-dir)
        mapping_dir="$2"
        shift 2
        ;;
      --input)
        input_path="$2"
        shift 2
        ;;
      --inputtraj)
        inputtraj="$2"
        shift 2
        ;;
      --output)
        output_csv="$2"
        shift 2
        ;;
      --workers)
        workers="$2"
        shift 2
        ;;
      -h|--help)
        print_option4_help
        exit 0
        ;;
      *)
        echo "Unknown argument for cg-stats: $1" >&2
        print_option4_help
        exit 1
        ;;
    esac
  done

  if [[ -z "$mapping_dir" || -z "$input_path" ]]; then
    echo "Error: cg-stats requires --mapping-dir and --input" >&2
    print_option4_help
    exit 1
  fi

  require_dir "$mapping_dir" "mapping dir"
  if [[ ! -e "$input_path" ]]; then
    echo "Error: input does not exist: $input_path" >&2
    exit 1
  fi
  if [[ -n "$inputtraj" && ! -e "$inputtraj" ]]; then
    echo "Error: inputtraj does not exist: $inputtraj" >&2
    exit 1
  fi

  local cmd=(
    herobm-cgstats
    -i "$input_path"
    -m "$mapping_dir"
    -o "$output_csv"
    -w "$workers"
  )
  if [[ -n "$inputtraj" ]]; then
    cmd+=(-t "$inputtraj")
  fi

  echo "Running: ${cmd[*]}"
  "${cmd[@]}"
  echo "Cg-stats completed successfully. Stats CSV: $output_csv"
}

run_option5() {
  local model_path=""
  local mapping_dir=""
  local output_model="deployed_model.pt"
  local bead_types="bead_types.yaml"
  local bead_stats=""
  local ignore_hydrogens=""
  local verbose="INFO"
  local extra_metadata=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model)
        model_path="$2"
        shift 2
        ;;
      --mapping-dir)
        mapping_dir="$2"
        shift 2
        ;;
      --output)
        output_model="$2"
        shift 2
        ;;
      --bead-types)
        bead_types="$2"
        shift 2
        ;;
      --bead-stats)
        bead_stats="$2"
        shift 2
        ;;
      --ignore-hydrogens)
        ignore_hydrogens="true"
        shift
        ;;
      --predict-hydrogens)
        ignore_hydrogens="false"
        shift
        ;;
      --verbose)
        verbose="$2"
        shift 2
        ;;
      --extra-metadata)
        shift
        while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
          extra_metadata+=("$1")
          shift
        done
        ;;
      -h|--help)
        print_option5_help
        exit 0
        ;;
      *)
        echo "Unknown argument for deploy: $1" >&2
        print_option5_help
        exit 1
        ;;
    esac
  done

  if [[ -z "$model_path" || -z "$mapping_dir" ]]; then
    echo "Error: deploy requires --model and --mapping-dir" >&2
    print_option5_help
    exit 1
  fi

  require_file "$model_path" "model checkpoint"
  require_dir "$mapping_dir" "mapping dir"

  if [[ -n "$bead_stats" ]]; then
    require_file "$bead_stats" "bead stats csv"
  fi

  local cmd=(
    herobm-deploy
    -m "$model_path"
    -o "$output_model"
    --mapping "$mapping_dir"
    --bead-types-filename "$bead_types"
    --verbose "$verbose"
  )
  if [[ -n "$bead_stats" ]]; then
    cmd+=(--bead-stats "$bead_stats")
  fi
  if [[ "$ignore_hydrogens" == "true" ]]; then
    cmd+=(--ignore-hydrogens)
  elif [[ "$ignore_hydrogens" == "false" ]]; then
    cmd+=(--predict-hydrogens)
  fi
  if [[ ${#extra_metadata[@]} -gt 0 ]]; then
    cmd+=(--extra-metadata "${extra_metadata[@]}")
  fi

  echo "Running: ${cmd[*]}"
  "${cmd[@]}"
  echo "Deploy completed successfully. Deployed model: $output_model"
}

run_option6() {
  local model_path=""
  local input_file=""
  local inputtraj=""
  local output_dir="./output"
  local selection="protein"
  local device="cuda:0"
  local mapping_dir=""
  local bead_types_filename=""
  local bead_stats=""
  local ignore_hydrogens=""
  local trajslice=""
  local chunking=""
  local num_steps=""
  local tolerance=""
  local is_atomistic=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model)
        model_path="$2"
        shift 2
        ;;
      --input)
        input_file="$2"
        shift 2
        ;;
      --inputtraj)
        inputtraj="$2"
        shift 2
        ;;
      --output)
        output_dir="$2"
        shift 2
        ;;
      --selection)
        selection="$2"
        shift 2
        ;;
      --device)
        device="$2"
        shift 2
        ;;
      --mapping)
        mapping_dir="$2"
        shift 2
        ;;
      --bead-types-filename)
        bead_types_filename="$2"
        shift 2
        ;;
      --bead-stats)
        bead_stats="$2"
        shift 2
        ;;
      --ignore-hydrogens)
        ignore_hydrogens="true"
        shift
        ;;
      --predict-hydrogens)
        ignore_hydrogens="false"
        shift
        ;;
      --trajslice)
        trajslice="$2"
        shift 2
        ;;
      --chunking)
        chunking="$2"
        shift 2
        ;;
      --num-steps)
        num_steps="$2"
        shift 2
        ;;
      --tolerance)
        tolerance="$2"
        shift 2
        ;;
      --atomistic)
        is_atomistic=1
        shift
        ;;
      -h|--help)
        print_option6_help
        exit 0
        ;;
      *)
        echo "Unknown argument for backmap: $1" >&2
        print_option6_help
        exit 1
        ;;
    esac
  done

  if [[ -z "$model_path" || -z "$input_file" ]]; then
    echo "Error: backmap requires --model and --input" >&2
    print_option6_help
    exit 1
  fi
  require_file "$model_path" "model"
  require_file "$input_file" "input structure"
  if [[ -n "$inputtraj" ]]; then
    require_file "$inputtraj" "input trajectory"
  fi
  if [[ -n "$mapping_dir" ]]; then
    require_dir "$mapping_dir" "mapping dir"
  fi
  if [[ -n "$bead_stats" ]]; then
    require_file "$bead_stats" "bead stats csv"
  fi

  mkdir -p "$output_dir"

  local cmd=(
    herobm-backmap
    -i "$input_file"
    -mo "$model_path"
    -o "$output_dir"
    -s "$selection"
    -d "$device"
  )
  if [[ -n "$inputtraj" ]]; then
    cmd+=(-it "$inputtraj")
  fi
  if [[ -n "$mapping_dir" ]]; then
    cmd+=(-m "$mapping_dir")
  fi
  if [[ -n "$bead_types_filename" ]]; then
    cmd+=(-b "$bead_types_filename")
  fi
  if [[ -n "$bead_stats" ]]; then
    cmd+=(-bs "$bead_stats")
  fi
  if [[ "$ignore_hydrogens" == "true" ]]; then
    cmd+=(--ignore-hydrogens)
  elif [[ "$ignore_hydrogens" == "false" ]]; then
    cmd+=(--predict-hydrogens)
  fi
  if [[ -n "$trajslice" ]]; then
    cmd+=(-ts "$trajslice")
  fi
  if [[ -n "$chunking" ]]; then
    cmd+=(-c "$chunking")
  fi
  if [[ -n "$num_steps" ]]; then
    cmd+=(-ns "$num_steps")
  fi
  if [[ -n "$tolerance" ]]; then
    cmd+=(-t "$tolerance")
  fi
  if [[ "$is_atomistic" -eq 1 ]]; then
    cmd+=(-a)
  fi

  echo "Running: ${cmd[*]}"
  "${cmd[@]}"
  echo "Backmapping completed successfully. Output folder: $output_dir"
}

run_option7() {
  local model_path=""
  local output_dir="./test_output"
  local npz_path=""
  local pdb_dir=""
  local pdb_file=""
  local traj_file=""
  local traj_dir=""
  local input_format="pdb"
  local traj_format=""
  local filter_file=""
  local input_selection="all"
  local mapping_dir=""
  local bead_types_filename=""
  local bead_stats=""
  local ignore_hydrogens=""
  local trajslice=""
  local device="cuda:0"
  local chunking=""
  local tolerance="0"
  local -a base_rmsd_selections=()
  local -a rmsd_selections=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model)
        model_path="$2"
        shift 2
        ;;
      --output)
        output_dir="$2"
        shift 2
        ;;
      --npz)
        npz_path="$2"
        shift 2
        ;;
      --pdb-dir)
        pdb_dir="$2"
        shift 2
        ;;
      --pdb-file)
        pdb_file="$2"
        shift 2
        ;;
      --traj-file)
        traj_file="$2"
        shift 2
        ;;
      --traj-dir)
        traj_dir="$2"
        shift 2
        ;;
      --input-format)
        input_format="$2"
        shift 2
        ;;
      --traj-format)
        traj_format="$2"
        shift 2
        ;;
      --filter)
        filter_file="$2"
        shift 2
        ;;
      --input-selection)
        input_selection="$2"
        shift 2
        ;;
      --mapping)
        mapping_dir="$2"
        shift 2
        ;;
      --bead-types-filename)
        bead_types_filename="$2"
        shift 2
        ;;
      --bead-stats)
        bead_stats="$2"
        shift 2
        ;;
      --ignore-hydrogens)
        ignore_hydrogens="true"
        shift
        ;;
      --predict-hydrogens)
        ignore_hydrogens="false"
        shift
        ;;
      --base-rmsd-selection)
        base_rmsd_selections+=("$2")
        shift 2
        ;;
      --rmsd-selection)
        rmsd_selections+=("$2")
        shift 2
        ;;
      --trajslice)
        trajslice="$2"
        shift 2
        ;;
      --device)
        device="$2"
        shift 2
        ;;
      --chunking)
        chunking="$2"
        shift 2
        ;;
      --tolerance)
        tolerance="$2"
        shift 2
        ;;
      -h|--help)
        print_option7_help
        exit 0
        ;;
      *)
        echo "Unknown argument for test: $1" >&2
        print_option7_help
        exit 1
        ;;
    esac
  done

  if [[ -z "$model_path" ]]; then
    echo "Error: test requires --model" >&2
    print_option7_help
    exit 1
  fi

  local input_modes=0
  [[ -n "$npz_path" ]] && ((input_modes+=1))
  [[ -n "$pdb_dir" ]] && ((input_modes+=1))
  [[ -n "$pdb_file" ]] && ((input_modes+=1))
  if [[ "$input_modes" -ne 1 ]]; then
    echo "Error: test requires exactly one of --npz, --pdb-dir, or --pdb-file" >&2
    print_option7_help
    exit 1
  fi

  require_file "$model_path" "model"
  if [[ -n "$npz_path" ]]; then
    if [[ ! -e "$npz_path" ]]; then
      echo "Error: npz input does not exist: $npz_path" >&2
      exit 1
    fi
  fi
  if [[ -n "$pdb_dir" ]]; then
    require_dir "$pdb_dir" "pdb dir"
  fi
  if [[ -n "$pdb_file" ]]; then
    require_file "$pdb_file" "pdb file"
  fi
  if [[ -n "$traj_file" ]]; then
    require_file "$traj_file" "traj file"
  fi
  if [[ -n "$traj_dir" && ! -e "$traj_dir" ]]; then
    echo "Error: traj dir/file does not exist: $traj_dir" >&2
    exit 1
  fi
  if [[ -n "$mapping_dir" ]]; then
    require_dir "$mapping_dir" "mapping dir"
  fi
  if [[ -n "$bead_stats" ]]; then
    require_file "$bead_stats" "bead stats csv"
  fi
  if [[ -n "$filter_file" ]]; then
    require_file "$filter_file" "filter file"
  fi

  mkdir -p "$output_dir"

  local cmd=(
    herobm-test
    -mo "$model_path"
    -o "$output_dir"
    --input-selection "$input_selection"
    -d "$device"
    -t "$tolerance"
  )
  if [[ -n "$npz_path" ]]; then
    cmd+=(--npz "$npz_path")
  fi
  if [[ -n "$pdb_dir" ]]; then
    cmd+=(--pdb-dir "$pdb_dir" --input-format "$input_format")
  fi
  if [[ -n "$pdb_file" ]]; then
    cmd+=(--pdb-file "$pdb_file")
  fi
  if [[ -n "$traj_file" ]]; then
    cmd+=(--traj-file "$traj_file")
  fi
  if [[ -n "$traj_dir" ]]; then
    cmd+=(--traj-dir "$traj_dir")
  fi
  if [[ -n "$traj_format" ]]; then
    cmd+=(--traj-format "$traj_format")
  fi
  if [[ -n "$filter_file" ]]; then
    cmd+=(--filter "$filter_file")
  fi
  if [[ -n "$mapping_dir" ]]; then
    cmd+=(-m "$mapping_dir")
  fi
  if [[ -n "$bead_types_filename" ]]; then
    cmd+=(-b "$bead_types_filename")
  fi
  if [[ -n "$bead_stats" ]]; then
    cmd+=(-bs "$bead_stats")
  fi
  if [[ "$ignore_hydrogens" == "true" ]]; then
    cmd+=(--ignore-hydrogens)
  elif [[ "$ignore_hydrogens" == "false" ]]; then
    cmd+=(--predict-hydrogens)
  fi
  if [[ -n "$trajslice" ]]; then
    cmd+=(--trajslice "$trajslice")
  fi
  if [[ -n "$chunking" ]]; then
    cmd+=(-c "$chunking")
  fi
  if [[ ${#base_rmsd_selections[@]} -gt 0 ]]; then
    local base_sel
    for base_sel in "${base_rmsd_selections[@]}"; do
      cmd+=(--base-rmsd-selection "$base_sel")
    done
  fi
  if [[ ${#rmsd_selections[@]} -gt 0 ]]; then
    local rmsd_sel
    for rmsd_sel in "${rmsd_selections[@]}"; do
      cmd+=(--rmsd-selection "$rmsd_sel")
    done
  fi

  echo "Running: ${cmd[*]}"
  "${cmd[@]}"
  echo "Test completed successfully. Output folder: $output_dir"
}

run_option1_interactive() {
  local mapping_input_dir atomistic_dir mapping_output_dir distance_threshold py_bin overwrite skip_missing

  echo
  echo "Command selected: complete-mapping (complete mapping YAML files)"
  read -r -p "Mapping input dir: " mapping_input_dir
  read -r -p "Atomistic dir: " atomistic_dir
  read -r -p "Mapping output dir: " mapping_output_dir
  distance_threshold="$(prompt_with_default "Distance threshold" "0.25")"
  py_bin="$(prompt_with_default "Python executable" "python")"
  overwrite="$(prompt_with_default "Overwrite existing labels? (y/N)" "N")"
  skip_missing="$(prompt_with_default "Skip mappings missing atomistic residues? (y/N)" "N")"

  local cmd=(
    --mapping-input-dir "$mapping_input_dir"
    --atomistic-dir "$atomistic_dir"
    --mapping-output-dir "$mapping_output_dir"
    --distance-threshold "$distance_threshold"
    --python "$py_bin"
  )
  if [[ "$overwrite" =~ ^[Yy]$ ]]; then
    cmd+=(--overwrite-existing)
  fi
  if [[ "$skip_missing" =~ ^[Yy]$ ]]; then
    cmd+=(--skip-missing-structures)
  fi

  run_option1 "${cmd[@]}"
}

run_option2_interactive() {
  local mapping_dir output_dir input_mode
  local pdb_dir pdb_file traj_file input_format selection bead_types_filename
  local cutoff workers traj_dir traj_format trajslice filter_file ignore_h

  echo
  echo "Command selected: build-dataset (build NPZ dataset)"
  read -r -p "Completed mapping dir: " mapping_dir
  read -r -p "Output NPZ dir: " output_dir

  while true; do
    read -r -p $'Select input mode:\n  1) Folder with PDB structures\n  2) Single PDB + trajectory file\nMode: ' input_mode
    case "$input_mode" in
      1|folder|FOLDER)
        input_mode="folder"
        read -r -p "PDB folder: " pdb_dir
        traj_dir="$(prompt_optional "Trajectory folder/file (optional)")"
        traj_format="$(prompt_optional "Trajectory format (only if trajectory input is a folder)")"
        break
        ;;
      2|single|SINGLE)
        input_mode="single"
        read -r -p "PDB file: " pdb_file
        read -r -p "Trajectory file: " traj_file
        break
        ;;
      *)
        echo "Invalid mode: $input_mode" >&2
        ;;
    esac
  done

  input_format="$(prompt_with_default "Input format" "pdb")"
  selection="$(prompt_with_default "Selection" "protein")"
  bead_types_filename="$(prompt_with_default "Bead types filename" "bead_types.yaml")"
  ignore_h="$(prompt_with_default "Ignore hydrogen hierarchy labels? (y/N)" "N")"
  cutoff="$(prompt_optional "Cutoff")"
  workers="$(prompt_optional "Workers")"
  trajslice="$(prompt_optional "Trajectory slice")"
  filter_file="$(prompt_optional "Filter file")"

  local cmd=(
    --mapping-dir "$mapping_dir"
    --input-mode "$input_mode"
    --output-dir "$output_dir"
    --input-format "$input_format"
    --selection "$selection"
    --bead-types-filename "$bead_types_filename"
  )
  if [[ "$ignore_h" =~ ^[Yy]$ ]]; then
    cmd+=(--ignore-hydrogens)
  fi
  if [[ "$input_mode" == "folder" ]]; then
    cmd+=(--pdb-dir "$pdb_dir")
    if [[ -n "$traj_dir" ]]; then
      cmd+=(--traj-dir "$traj_dir")
    fi
    if [[ -n "$traj_format" ]]; then
      cmd+=(--traj-format "$traj_format")
    fi
  else
    cmd+=(--pdb-file "$pdb_file" --traj-file "$traj_file")
  fi
  if [[ -n "$cutoff" ]]; then
    cmd+=(--cutoff "$cutoff")
  fi
  if [[ -n "$workers" ]]; then
    cmd+=(--workers "$workers")
  fi
  if [[ -n "$trajslice" ]]; then
    cmd+=(--trajslice "$trajslice")
  fi
  if [[ -n "$filter_file" ]]; then
    cmd+=(--filter "$filter_file")
  fi

  run_option2 "${cmd[@]}"
}

run_option3_interactive() {
  local config_file device ddp master_addr master_port find_unused

  echo
  echo "Command selected: train (run training)"
  read -r -p "Training config file: " config_file
  device="$(prompt_optional "Device (e.g. cuda:0, cpu)")"
  ddp="$(prompt_with_default "Enable DDP? (y/N)" "N")"
  master_addr=""
  master_port=""
  find_unused="N"

  if [[ "$ddp" =~ ^[Yy]$ ]]; then
    master_addr="$(prompt_optional "DDP master address (optional)")"
    master_port="$(prompt_optional "DDP master port (optional)")"
    find_unused="$(prompt_with_default "DDP find_unused_parameters? (y/N)" "N")"
  fi

  local cmd=(--config "$config_file")
  if [[ -n "$device" ]]; then
    cmd+=(--device "$device")
  fi
  if [[ "$ddp" =~ ^[Yy]$ ]]; then
    cmd+=(--ddp)
    if [[ -n "$master_addr" ]]; then
      cmd+=(--master-addr "$master_addr")
    fi
    if [[ -n "$master_port" ]]; then
      cmd+=(--master-port "$master_port")
    fi
    if [[ "$find_unused" =~ ^[Yy]$ ]]; then
      cmd+=(--find-unused-parameters)
    fi
  fi

  run_option3 "${cmd[@]}"
}

run_option4_interactive() {
  local mapping_dir input_path inputtraj output_csv workers

  echo
  echo "Command selected: cg-stats (compute bead-distance statistics CSV)"
  read -r -p "Mapping dir: " mapping_dir
  read -r -p "Input atomistic file/folder: " input_path
  inputtraj="$(prompt_optional "Input trajectory file/folder")"
  output_csv="$(prompt_with_default "Output CSV" "cgdist.csv")"
  workers="$(prompt_with_default "Workers" "1")"

  local cmd=(
    --mapping-dir "$mapping_dir"
    --input "$input_path"
    --output "$output_csv"
    --workers "$workers"
  )
  if [[ -n "$inputtraj" ]]; then
    cmd+=(--inputtraj "$inputtraj")
  fi

  run_option4 "${cmd[@]}"
}

run_option5_interactive() {
  local model_path mapping_dir output_model bead_types bead_stats verbose extra_meta ignore_h
  local -a cmd extra_arr

  echo
  echo "Command selected: deploy (deploy trained model)"
  read -r -p "Model checkpoint path: " model_path
  read -r -p "Mapping dir: " mapping_dir
  output_model="$(prompt_with_default "Output deployed model path" "deployed_model.pt")"
  bead_types="$(prompt_with_default "Bead types filename/path" "bead_types.yaml")"
  bead_stats="$(prompt_optional "Bead stats CSV (optional)")"
  ignore_h="$(prompt_optional "Ignore hydrogens policy [y=yes, n=no, blank=auto]")"
  verbose="$(prompt_with_default "Verbose level" "INFO")"
  extra_meta="$(prompt_optional "Extra metadata key=value (space separated)")"

  cmd=(
    --model "$model_path"
    --mapping-dir "$mapping_dir"
    --output "$output_model"
    --bead-types "$bead_types"
    --verbose "$verbose"
  )
  if [[ -n "$bead_stats" ]]; then
    cmd+=(--bead-stats "$bead_stats")
  fi
  if [[ "$ignore_h" =~ ^[Yy]$ ]]; then
    cmd+=(--ignore-hydrogens)
  elif [[ "$ignore_h" =~ ^[Nn]$ ]]; then
    cmd+=(--predict-hydrogens)
  fi
  if [[ -n "$extra_meta" ]]; then
    # shellcheck disable=SC2206
    extra_arr=($extra_meta)
    cmd+=(--extra-metadata "${extra_arr[@]}")
  fi

  run_option5 "${cmd[@]}"
}

run_option6_interactive() {
  local model_path input_file inputtraj output_dir selection device
  local mapping_dir bead_types_filename bead_stats ignore_h trajslice chunking num_steps tolerance atomistic

  echo
  echo "Command selected: backmap (run inference/backmapping)"
  read -r -p "Model file: " model_path
  read -r -p "Input structure file (.pdb/.gro): " input_file
  inputtraj="$(prompt_optional "Input trajectory file (.xtc/.trr)")"
  output_dir="$(prompt_with_default "Output directory" "./output")"
  selection="$(prompt_with_default "Selection" "protein")"
  device="$(prompt_with_default "Device" "cuda:0")"
  mapping_dir="$(prompt_optional "Mapping directory (optional, overrides model metadata)")"
  bead_types_filename="$(prompt_optional "Bead types filename/path (optional)")"
  bead_stats="$(prompt_optional "Bead stats CSV (optional)")"
  ignore_h="$(prompt_optional "Ignore hydrogens [y=yes, n=no, blank=use metadata/default]")"
  trajslice="$(prompt_optional "Trajectory slice (optional)")"
  chunking="$(prompt_optional "Chunking (optional)")"
  num_steps="$(prompt_optional "CG minimization num-steps (optional)")"
  tolerance="$(prompt_optional "Atomistic minimization tolerance (optional)")"
  atomistic="$(prompt_with_default "Input is atomistic? (y/N)" "N")"

  local cmd=(
    --model "$model_path"
    --input "$input_file"
    --output "$output_dir"
    --selection "$selection"
    --device "$device"
  )
  if [[ -n "$inputtraj" ]]; then
    cmd+=(--inputtraj "$inputtraj")
  fi
  if [[ -n "$mapping_dir" ]]; then
    cmd+=(--mapping "$mapping_dir")
  fi
  if [[ -n "$bead_types_filename" ]]; then
    cmd+=(--bead-types-filename "$bead_types_filename")
  fi
  if [[ -n "$bead_stats" ]]; then
    cmd+=(--bead-stats "$bead_stats")
  fi
  if [[ "$ignore_h" =~ ^[Yy]$ ]]; then
    cmd+=(--ignore-hydrogens)
  elif [[ "$ignore_h" =~ ^[Nn]$ ]]; then
    cmd+=(--predict-hydrogens)
  fi
  if [[ -n "$trajslice" ]]; then
    cmd+=(--trajslice "$trajslice")
  fi
  if [[ -n "$chunking" ]]; then
    cmd+=(--chunking "$chunking")
  fi
  if [[ -n "$num_steps" ]]; then
    cmd+=(--num-steps "$num_steps")
  fi
  if [[ -n "$tolerance" ]]; then
    cmd+=(--tolerance "$tolerance")
  fi
  if [[ "$atomistic" =~ ^[Yy]$ ]]; then
    cmd+=(--atomistic)
  fi

  run_option6 "${cmd[@]}"
}

run_option7_interactive() {
  local model_path output_dir input_mode npz_path pdb_dir pdb_file traj_file traj_dir
  local input_format traj_format filter_file input_selection mapping_dir bead_types_filename bead_stats
  local ignore_h trajslice device chunking tolerance base_selections custom_selections
  local -a cmd base_arr custom_arr

  echo
  echo "Command selected: test (map to CG, backmap, compute RMSD)"
  read -r -p "Model file: " model_path
  output_dir="$(prompt_with_default "Output directory" "./test_output")"

  while true; do
    read -r -p $'Select input mode:\n  1) NPZ file or folder\n  2) Folder with atomistic structures\n  3) Single structure (+ optional trajectory)\nMode: ' input_mode
    case "$input_mode" in
      1|npz|NPZ)
        input_mode="npz"
        read -r -p "NPZ file or folder: " npz_path
        break
        ;;
      2|folder|FOLDER)
        input_mode="folder"
        read -r -p "Structure folder: " pdb_dir
        traj_dir="$(prompt_optional "Trajectory file/folder paired with structure folder")"
        input_format="$(prompt_with_default "Input format" "pdb")"
        traj_format="$(prompt_optional "Trajectory format (required if trajectory input is a folder)")"
        filter_file="$(prompt_optional "Filter file (optional)")"
        break
        ;;
      3|single|SINGLE)
        input_mode="single"
        read -r -p "Structure file (.pdb/.gro): " pdb_file
        traj_file="$(prompt_optional "Trajectory file (.xtc/.trr)")"
        break
        ;;
      *)
        echo "Invalid input mode: $input_mode" >&2
        ;;
    esac
  done

  input_selection="$(prompt_with_default "Input selection for raw atomistic mapping" "all")"
  mapping_dir="$(prompt_optional "Mapping directory (optional, required for raw input unless embedded in model)")"
  bead_types_filename="$(prompt_optional "Bead types filename/path (optional)")"
  bead_stats="$(prompt_optional "Bead stats CSV (optional)")"
  ignore_h="$(prompt_optional "Ignore hydrogens [y=yes, n=no, blank=use metadata/default]")"
  trajslice="$(prompt_optional "Trajectory slice (optional)")"
  device="$(prompt_with_default "Device" "cuda:0")"
  chunking="$(prompt_optional "Chunking (optional)")"
  tolerance="$(prompt_with_default "Atomistic minimization tolerance" "0")"
  base_selections="$(prompt_optional "Base RMSD selections (comma separated: protein-backbone,protein-sidechains)")"
  custom_selections="$(prompt_optional "Custom RMSD selections (semicolon separated label=selection)")"

  cmd=(
    --model "$model_path"
    --output "$output_dir"
    --input-selection "$input_selection"
    --device "$device"
    --tolerance "$tolerance"
  )
  case "$input_mode" in
    npz)
      cmd+=(--npz "$npz_path")
      ;;
    folder)
      cmd+=(--pdb-dir "$pdb_dir" --input-format "$input_format")
      if [[ -n "$traj_dir" ]]; then
        cmd+=(--traj-dir "$traj_dir")
      fi
      if [[ -n "$traj_format" ]]; then
        cmd+=(--traj-format "$traj_format")
      fi
      if [[ -n "$filter_file" ]]; then
        cmd+=(--filter "$filter_file")
      fi
      ;;
    single)
      cmd+=(--pdb-file "$pdb_file")
      if [[ -n "$traj_file" ]]; then
        cmd+=(--traj-file "$traj_file")
      fi
      ;;
  esac
  if [[ -n "$mapping_dir" ]]; then
    cmd+=(--mapping "$mapping_dir")
  fi
  if [[ -n "$bead_types_filename" ]]; then
    cmd+=(--bead-types-filename "$bead_types_filename")
  fi
  if [[ -n "$bead_stats" ]]; then
    cmd+=(--bead-stats "$bead_stats")
  fi
  if [[ "$ignore_h" =~ ^[Yy]$ ]]; then
    cmd+=(--ignore-hydrogens)
  elif [[ "$ignore_h" =~ ^[Nn]$ ]]; then
    cmd+=(--predict-hydrogens)
  fi
  if [[ -n "$trajslice" ]]; then
    cmd+=(--trajslice "$trajslice")
  fi
  if [[ -n "$chunking" ]]; then
    cmd+=(--chunking "$chunking")
  fi
  if [[ -n "$base_selections" ]]; then
    IFS=',' read -r -a base_arr <<< "$base_selections"
    local base_sel
    for base_sel in "${base_arr[@]}"; do
      base_sel="${base_sel//[[:space:]]/}"
      if [[ -n "$base_sel" ]]; then
        cmd+=(--base-rmsd-selection "$base_sel")
      fi
    done
  fi
  if [[ -n "$custom_selections" ]]; then
    IFS=';' read -r -a custom_arr <<< "$custom_selections"
    local custom_sel
    for custom_sel in "${custom_arr[@]}"; do
      custom_sel="${custom_sel#"${custom_sel%%[![:space:]]*}"}"
      custom_sel="${custom_sel%"${custom_sel##*[![:space:]]}"}"
      if [[ -n "$custom_sel" ]]; then
        cmd+=(--rmsd-selection "$custom_sel")
      fi
    done
  fi

  run_option7 "${cmd[@]}"
}

if [[ $# -lt 1 ]]; then
  choice="$(prompt_option_choice)"
  case "$choice" in
    complete-mapping)
      run_option1_interactive
      ;;
    build-dataset)
      run_option2_interactive
      ;;
    train)
      run_option3_interactive
      ;;
    cg-stats)
      run_option4_interactive
      ;;
    deploy)
      run_option5_interactive
      ;;
    backmap)
      run_option6_interactive
      ;;
    test)
      run_option7_interactive
      ;;
    help)
      print_main_help
      ;;
    quit)
      exit 0
      ;;
  esac
  exit 0
fi

case "$1" in
  complete-mapping|option1|1)
    shift
    run_option1 "$@"
    ;;
  build-dataset|option2|2)
    shift
    run_option2 "$@"
    ;;
  train|option3|3)
    shift
    run_option3 "$@"
    ;;
  cg-stats|option4|4)
    shift
    run_option4 "$@"
    ;;
  deploy|option5|5)
    shift
    run_option5 "$@"
    ;;
  backmap|option6|6)
    shift
    run_option6 "$@"
    ;;
  test|option7|7)
    shift
    run_option7 "$@"
    ;;
  -h|--help)
    print_main_help
    ;;
  *)
    echo "Unknown command: $1" >&2
    print_main_help
    exit 1
    ;;
esac
