import argparse
import csv
import logging
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import MDAnalysis as mda
import numpy as np
import yaml
from MDAnalysis.analysis.rms import rmsd

from geqtrain.data import AtomicDataDict
from geqtrain.data._build import dataset_from_config
from geqtrain.data.dataloader import DataLoader
from geqtrain.train.utils import evaluate_end_chunking_condition
from geqtrain.utils import Config
from geqtrain.utils._global_options import register_all_fields

from herobm.backmapping.hierarchical_backmapping import (
    build_CG,
    build_universe,
    load_model,
)
from herobm.mapper import HierarchicalMapper
from herobm.scripts.run_backmapping import (
    _extract_model_output_width,
    _infer_ignore_hydrogens_from_mapping,
    _parse_bool_str,
)
from herobm.utils import DataDict
from herobm.utils.io import replace_words_in_file


BASE_RMSD_SELECTIONS: Dict[str, str] = {
    "protein-backbone": "protein and backbone",
    "protein-sidechains": "protein and not backbone",
}


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a HEroBM model by mapping atomistic input to CG, backmapping, and computing RMSD."
    )

    parser.add_argument("-mo", "--model", type=Path, required=True, help="Trained or deployed model file.")
    parser.add_argument("-o", "--output", type=Path, default=Path("./test_output"), help="Output directory.")

    input_group = parser.add_argument_group("Input")
    input_group.add_argument("--npz", type=Path, help="NPZ file or folder of NPZ files to evaluate.")
    input_group.add_argument("--pdb-dir", type=Path, help="Folder with input structures for on-the-fly mapping.")
    input_group.add_argument("--pdb-file", type=Path, help="Single input structure file for on-the-fly mapping.")
    input_group.add_argument("--traj-file", type=Path, help="Trajectory file paired with --pdb-file.")
    input_group.add_argument("--traj-dir", type=Path, help="Trajectory file or folder paired with --pdb-dir.")
    input_group.add_argument("--input-format", default="pdb", help="Structure extension for --pdb-dir scan.")
    input_group.add_argument("--traj-format", help="Trajectory extension when --traj-dir is a folder.")
    input_group.add_argument("--filter", type=Path, help="Text file listing basenames to include.")
    input_group.add_argument(
        "--input-selection",
        default="all",
        help="MDAnalysis atom selection used when generating NPZs from raw atomistic input.",
    )
    input_group.add_argument("--trajslice", type=str, help="Trajectory slice, e.g. ::10.")

    model_group = parser.add_argument_group("Model Overrides")
    model_group.add_argument("-m", "--mapping", type=str, help="Mapping directory. Overrides model metadata.")
    model_group.add_argument("-b", "--bead-types-filename", type=Path, help="Bead types YAML file. Overrides model metadata.")
    model_group.add_argument("-bs", "--bead-stats", type=Path, help="Optional bead stats CSV. Overrides model metadata.")
    model_group.add_argument(
        "--ignore-hydrogens",
        action="store_true",
        dest="ignore_hydrogens",
        default=None,
        help="Ignore hierarchy labels for H* atoms. Overrides model metadata.",
    )
    model_group.add_argument(
        "--predict-hydrogens",
        action="store_false",
        dest="ignore_hydrogens",
        help="Use hydrogen hierarchy labels. Overrides model metadata.",
    )

    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument("-d", "--device", default="cuda:0", help="Torch device.")
    eval_group.add_argument("-c", "--chunking", type=int, default=0, help="Max atoms per chunk.")
    eval_group.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=0.0,
        help="Atomistic minimization tolerance used when writing structures. Use 0 to disable minimization.",
    )
    eval_group.add_argument(
        "--base-rmsd-selection",
        action="append",
        choices=sorted(BASE_RMSD_SELECTIONS.keys()),
        default=[],
        help="Predefined RMSD selections to include.",
    )
    eval_group.add_argument(
        "--rmsd-selection",
        action="append",
        default=[],
        help="Custom RMSD selection in the form label=MDAnalysis_selection. Can be passed multiple times.",
    )

    parsed = parser.parse_args(args=args)
    validate_input_mode(parsed)
    return parsed


def validate_input_mode(args) -> None:
    has_npz = args.npz is not None
    has_pdb_dir = args.pdb_dir is not None
    has_pdb_file = args.pdb_file is not None
    modes = sum(int(x) for x in (has_npz, has_pdb_dir, has_pdb_file))
    if modes != 1:
        raise ValueError("Provide exactly one of --npz, --pdb-dir, or --pdb-file.")
    if has_pdb_file and args.traj_dir is not None:
        raise ValueError("--traj-dir can only be used with --pdb-dir.")


def resolve_runtime_args(args) -> Tuple[object, Config, Dict]:
    print(f"Loading model from: {args.model}")
    model, model_config, metadata = load_model(str(args.model), device=args.device)
    register_all_fields(model_config)

    if args.mapping is None:
        args.mapping = metadata.pop(DataDict.MAPPING_KEY, None)
        if args.mapping:
            print(f"Using mapping from model metadata: {args.mapping}")

    if args.bead_types_filename is None:
        bead_types_path = metadata.pop(DataDict.BEAD_TYPES_KEY, "bead_types.yaml")
        if bead_types_path:
            args.bead_types_filename = Path(bead_types_path)
            print(f"Using bead types file from model metadata: {args.bead_types_filename}")

    if args.ignore_hydrogens is None:
        ignore_h_from_meta = _parse_bool_str(metadata.pop(DataDict.IGNORE_HYDROGENS_KEY, None))
        if ignore_h_from_meta is not None:
            args.ignore_hydrogens = ignore_h_from_meta
            print(f"Using ignore_hydrogens from model metadata: {args.ignore_hydrogens}")

    if args.ignore_hydrogens is None and args.mapping:
        inferred_ignore_h = _infer_ignore_hydrogens_from_mapping(
            mapping=args.mapping,
            model_output_width=_extract_model_output_width(model_config),
        )
        if inferred_ignore_h is not None:
            args.ignore_hydrogens = inferred_ignore_h
            print(f"Auto-inferred ignore_hydrogens from model width + mapping: {args.ignore_hydrogens}")

    if args.ignore_hydrogens is None:
        args.ignore_hydrogens = False

    if args.bead_stats is None and DataDict.BEAD_STATS_KEY in metadata:
        args.bead_stats = metadata.pop(DataDict.BEAD_STATS_KEY)

    return model, model_config, metadata


def parse_rmsd_selections(base_names: Sequence[str], custom_specs: Sequence[str]) -> List[Tuple[str, str]]:
    selections: List[Tuple[str, str]] = [("all", "all")]
    seen = {"all"}

    for name in base_names:
        if name in seen:
            continue
        selections.append((name, BASE_RMSD_SELECTIONS[name]))
        seen.add(name)

    for spec in custom_specs:
        if "=" in spec:
            label, selection = spec.split("=", 1)
            label = label.strip()
            selection = selection.strip()
        else:
            label = spec.strip()
            selection = spec.strip()
        if not label or not selection:
            raise ValueError(f"Invalid --rmsd-selection value: {spec}")
        if label in seen:
            raise ValueError(f"Duplicate RMSD selection label: {label}")
        selections.append((label, selection))
        seen.add(label)

    return selections


def discover_npz_inputs(npz_path: Path) -> List[Path]:
    if npz_path.is_file():
        if npz_path.suffix.lower() != ".npz":
            raise ValueError(f"Expected a .npz file: {npz_path}")
        return [npz_path]
    if npz_path.is_dir():
        files = sorted(npz_path.glob("*.npz"))
        if not files:
            raise ValueError(f"No .npz files found in {npz_path}")
        return files
    raise FileNotFoundError(f"NPZ input not found: {npz_path}")


def read_filter_basenames(filter_path: Optional[Path]) -> Optional[set]:
    if filter_path is None:
        return None
    return {
        line.strip()
        for line in filter_path.read_text().splitlines()
        if line.strip()
    }


def build_npz_from_raw_input(
    input_file: Path,
    output_dir: Path,
    mapping: str,
    inputtraj: Optional[Path],
    input_selection: str,
    trajslice: Optional[str],
    ignore_hydrogens: bool,
    cutoff: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_npz = output_dir / f"{input_file.stem}.npz"

    args_dict = {
        "mapping": mapping,
        "input": str(input_file),
        "inputtraj": str(inputtraj) if inputtraj else None,
        "output": str(output_dir),
        "selection": input_selection,
        "trajslice": trajslice,
        "ignore_hydrogens": ignore_hydrogens,
        "cutoff": cutoff,
        "isatomistic": True,
    }

    mapper = HierarchicalMapper(args_dict=args_dict)
    mapped = mapper.map()
    if mapped is None:
        raise RuntimeError(f"Failed to map {input_file}")
    mapped.save_npz(filename=str(output_npz), from_pos_unit="Angstrom", to_pos_unit="Angstrom")
    return output_npz


def discover_raw_inputs(args, model_config: Config, tmpdir: Path) -> List[Path]:
    if not args.mapping:
        raise ValueError(
            "Raw atomistic input requires a mapping. Provide --mapping or use a deployed model with embedded mapping metadata."
        )

    cutoff = model_config.get("r_max")
    filter_basenames = read_filter_basenames(args.filter)
    output_npz_dir = tmpdir / "npz"
    npz_files: List[Path] = []

    if args.pdb_file is not None:
        npz_files.append(
            build_npz_from_raw_input(
                input_file=args.pdb_file,
                output_dir=output_npz_dir,
                mapping=args.mapping,
                inputtraj=args.traj_file,
                input_selection=args.input_selection,
                trajslice=args.trajslice,
                ignore_hydrogens=bool(args.ignore_hydrogens),
                cutoff=cutoff,
            )
        )
        return npz_files

    input_files = sorted(args.pdb_dir.glob(f"*.{args.input_format}"))
    if filter_basenames is not None:
        input_files = [path for path in input_files if path.stem in filter_basenames]
    if not input_files:
        raise ValueError(f"No input structures found in {args.pdb_dir} matching *.{args.input_format}")

    for input_file in input_files:
        traj_file = None
        if args.traj_dir is not None:
            if args.traj_dir.is_dir():
                if not args.traj_format:
                    raise ValueError("--traj-format is required when --traj-dir is a directory.")
                candidate = args.traj_dir / f"{input_file.stem}.{args.traj_format}"
                if candidate.exists():
                    traj_file = candidate
            else:
                traj_file = args.traj_dir

        npz_files.append(
            build_npz_from_raw_input(
                input_file=input_file,
                output_dir=output_npz_dir,
                mapping=args.mapping,
                inputtraj=traj_file,
                input_selection=args.input_selection,
                trajslice=args.trajslice,
                ignore_hydrogens=bool(args.ignore_hydrogens),
                cutoff=cutoff,
            )
        )

    return npz_files


def load_npz_dataset(npz_path: Path) -> Dict:
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def make_test_config(root: Path, npz_path: Path) -> Config:
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = Path(tmp) / "test.yaml"
        template_path = Path(__file__).resolve().parent.parent / "backmapping" / "template.test.yaml"
        yaml_path.write_text(template_path.read_text())
        replace_words_in_file(
            str(yaml_path),
            {
                "{ROOT}": str(root),
                "{TEST_DATASET_INPUT}": str(npz_path),
            },
        )
        return Config.from_file(str(yaml_path), defaults={})


def save_frame_structures(
    frame_dataset: Dict,
    prefix: str,
    output_dir: Path,
    frame_index: int,
    n_frames: int,
    tolerance: float,
    box_dimensions,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    positions_pred = frame_dataset[DataDict.ATOM_POSITION_PRED][0]

    cg_u = build_CG(frame_dataset, n_frames, box_dimensions)
    cg_u.trajectory[frame_index]
    cg_sel = cg_u.select_atoms("all")
    cg_sel.positions = np.nan_to_num(frame_dataset[DataDict.BEAD_POSITION])
    cg_filename = output_dir / f"{prefix}.CG_{frame_index}.pdb"
    with mda.Writer(str(cg_filename), n_atoms=cg_sel.n_atoms) as writer:
        writer.write(cg_sel)

    backmapped_u = build_universe(frame_dataset, n_frames, box_dimensions)
    backmapped_u.trajectory[frame_index]

    true_positions = frame_dataset[DataDict.ATOM_POSITION][frame_index]
    valid_mask = ~np.any(np.isnan(positions_pred), axis=-1)

    true_sel = backmapped_u.select_atoms("all")
    true_sel.positions = true_positions[valid_mask]
    true_filename = output_dir / f"{prefix}.true_{frame_index}.pdb"
    with mda.Writer(str(true_filename), n_atoms=true_sel.n_atoms) as writer:
        writer.write(true_sel)

    backmapped_sel = backmapped_u.select_atoms("all")
    backmapped_sel.positions = positions_pred[valid_mask]
    backmapped_filename = output_dir / f"{prefix}.backmapped_{frame_index}.pdb"
    with mda.Writer(str(backmapped_filename), n_atoms=backmapped_sel.n_atoms) as writer:
        writer.write(backmapped_sel)

    backmapped_minimised_filename = None
    if tolerance is not None and tolerance > 0:
        try:
            from herobm.utils.minimisation import minimise_impl
            from herobm.utils.pdbFixer import fixPDB

            topology, positions = fixPDB(str(backmapped_filename), addHydrogens=True)
            backmapped_minimised_filename = output_dir / f"{prefix}.backmapped_min_{frame_index}.pdb"
            minimise_impl(
                topology,
                positions,
                str(backmapped_minimised_filename),
                restrain_atoms=[],
                tolerance=tolerance,
            )
        except Exception as exc:
            logging.warning("Failed to minimise %s: %s", backmapped_filename, exc)

    return true_filename, backmapped_filename, backmapped_minimised_filename, cg_filename


def compute_rmsd_records(
    frame_dataset: Dict,
    frame_index: int,
    positions_pred: np.ndarray,
    selections: Sequence[Tuple[str, str]],
    source_name: str,
) -> List[Dict]:
    valid_mask = ~np.any(np.isnan(positions_pred), axis=-1)
    if not np.any(valid_mask):
        raise RuntimeError(f"No valid predicted atom positions available for {source_name} frame {frame_index}")

    eval_dataset = dict(frame_dataset)
    eval_dataset[DataDict.ATOM_POSITION_PRED] = np.expand_dims(positions_pred, axis=0)
    eval_u = build_universe(eval_dataset, 1, None)
    ref_positions = frame_dataset[DataDict.ATOM_POSITION][frame_index][valid_mask]
    pred_positions = positions_pred[valid_mask]

    records: List[Dict] = []
    for label, selection in selections:
        atomgroup = eval_u.select_atoms(selection)
        if atomgroup.n_atoms == 0:
            rmsd_value = np.nan
        else:
            indices = atomgroup.indices
            rmsd_value = float(
                rmsd(
                    pred_positions[indices],
                    ref_positions[indices],
                    center=True,
                    superposition=True,
                )
            )
        records.append(
            {
                "source": source_name,
                "frame": frame_index,
                "selection_label": label,
                "selection": selection,
                "n_atoms": int(atomgroup.n_atoms),
                "rmsd_angstrom": rmsd_value,
            }
        )
    return records


def run_inference_on_npz(
    model,
    model_config: Config,
    npz_path: Path,
    output_dir: Path,
    selections: Sequence[Tuple[str, str]],
    device: str,
    chunking: int,
    tolerance: float,
) -> List[Dict]:
    base_config = Config(dict(model_config))
    test_config = make_test_config(output_dir, npz_path)
    base_config.update({k: v for k, v in test_config.items() if k not in base_config})
    base_config.update(test_config)
    base_config["chunking"] = chunking > 0
    base_config["batch_max_atoms"] = chunking
    register_all_fields(base_config)

    dataset = dataset_from_config(base_config, prefix="test")
    dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=1)
    npz_dataset = load_npz_dataset(npz_path)

    prefix = npz_path.stem
    n_frames = len(dataset)
    records: List[Dict] = []

    from geqtrain.train.components.inference import run_inference

    for frame_index, data in enumerate(dataloader):
        results = {
            DataDict.BEAD_POSITION: data[AtomicDataDict.POSITIONS_KEY].cpu().numpy(),
            DataDict.BEAD2ATOM_RELATIVE_VECTORS_PRED: [],
            DataDict.ATOM_POSITION_PRED: [],
        }

        already_computed_nodes = None
        num_batch_center_nodes = len(data[AtomicDataDict.EDGE_INDEX_KEY][0].unique())
        while True:
            out, _, center_nodes, _ = run_inference(
                model=model,
                data=data,
                device=device,
                config=base_config,
                already_computed_nodes=already_computed_nodes,
            )
            if out is not None:
                if AtomicDataDict.NODE_FEATURES_KEY in out:
                    results[DataDict.BEAD2ATOM_RELATIVE_VECTORS_PRED].append(
                        out[AtomicDataDict.NODE_FEATURES_KEY].detach().cpu().numpy()
                    )
                if DataDict.ATOM_POSITION in out:
                    results[DataDict.ATOM_POSITION_PRED].append(
                        np.expand_dims(out[DataDict.ATOM_POSITION].detach().cpu().numpy(), axis=0)
                    )

            already_computed_nodes = evaluate_end_chunking_condition(
                already_computed_nodes,
                center_nodes,
                num_batch_center_nodes,
            )
            if already_computed_nodes is None:
                break

        aggregated_rvp = np.concatenate(results[DataDict.BEAD2ATOM_RELATIVE_VECTORS_PRED], axis=0)
        aggregated_app = np.nanmean(
            np.concatenate(results[DataDict.ATOM_POSITION_PRED], axis=0),
            axis=0,
            keepdims=True,
        )

        frame_dataset = dict(npz_dataset)
        frame_dataset.update(
            {
                DataDict.BEAD_POSITION: results[DataDict.BEAD_POSITION],
                DataDict.BEAD2ATOM_RELATIVE_VECTORS_PRED: aggregated_rvp,
                DataDict.ATOM_POSITION_PRED: aggregated_app,
            }
        )

        save_frame_structures(
            frame_dataset=frame_dataset,
            prefix=prefix,
            output_dir=output_dir,
            frame_index=frame_index,
            n_frames=n_frames,
            tolerance=tolerance,
            box_dimensions=None,
        )

        records.extend(
            compute_rmsd_records(
                frame_dataset=frame_dataset,
                frame_index=frame_index,
                positions_pred=aggregated_app[0],
                selections=selections,
                source_name=npz_path.name,
            )
        )

    return records


def write_csv(path: Path, rows: Iterable[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_records(records: Sequence[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, str], List[float]] = {}
    atoms_by_group: Dict[Tuple[str, str], int] = {}
    selection_text: Dict[Tuple[str, str], str] = {}

    for row in records:
        key = (row["selection_label"], row["source"])
        grouped.setdefault(key, [])
        if not np.isnan(row["rmsd_angstrom"]):
            grouped[key].append(float(row["rmsd_angstrom"]))
        atoms_by_group[key] = int(row["n_atoms"])
        selection_text[key] = row["selection"]

    summary_rows: List[Dict] = []
    for (label, source), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=float)
        summary_rows.append(
            {
                "selection_label": label,
                "selection": selection_text[(label, source)],
                "source": source,
                "n_atoms": atoms_by_group[(label, source)],
                "n_frames": int(arr.size),
                "mean_rmsd_angstrom": float(np.nanmean(arr)) if arr.size else np.nan,
                "std_rmsd_angstrom": float(np.nanstd(arr)) if arr.size else np.nan,
            }
        )
    return summary_rows


def main(args=None):
    parsed = parse_args(args=args)
    model, model_config, _ = resolve_runtime_args(parsed)
    selections = parse_rmsd_selections(parsed.base_rmsd_selection, parsed.rmsd_selection)

    parsed.output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        if parsed.npz is not None:
            npz_inputs = discover_npz_inputs(parsed.npz)
        else:
            npz_inputs = discover_raw_inputs(parsed, model_config, tmpdir)

        print(f"Evaluating {len(npz_inputs)} input dataset(s)")
        all_records: List[Dict] = []
        for npz_path in npz_inputs:
            print(f"Testing {npz_path}")
            all_records.extend(
                run_inference_on_npz(
                    model=model,
                    model_config=model_config,
                    npz_path=npz_path,
                    output_dir=parsed.output,
                    selections=selections,
                    device=parsed.device,
                    chunking=parsed.chunking,
                    tolerance=parsed.tolerance,
                )
            )

    per_frame_csv = parsed.output / "rmsd_per_frame.csv"
    summary_csv = parsed.output / "rmsd_summary.csv"
    write_csv(
        per_frame_csv,
        all_records,
        fieldnames=["source", "frame", "selection_label", "selection", "n_atoms", "rmsd_angstrom"],
    )
    summary_rows = summarize_records(all_records)
    write_csv(
        summary_csv,
        summary_rows,
        fieldnames=["selection_label", "selection", "source", "n_atoms", "n_frames", "mean_rmsd_angstrom", "std_rmsd_angstrom"],
    )

    print(f"Per-frame RMSD written to: {per_frame_csv}")
    print(f"Summary RMSD written to: {summary_csv}")
    print("RMSD summary:")
    for row in summary_rows:
        print(
            f"  {row['source']} | {row['selection_label']}: "
            f"mean={row['mean_rmsd_angstrom']:.4f} A, std={row['std_rmsd_angstrom']:.4f} A, n_frames={row['n_frames']}"
        )


if __name__ == "__main__":
    main()
