# herobm/scripts/deploy.py

import argparse
import logging
import re
import yaml
from pathlib import Path

# IMPORTANT: Import the reusable components from GEqTrain
from geqtrain.utils.deploy import build_deployment, get_base_deploy_parser
from geqtrain.train.components.checkpointing import Config

# Import HEroBM specific keys
from herobm.utils.DataDict import MAPPING_KEY, BEAD_TYPES_KEY, BEAD_STATS_KEY, IGNORE_HYDROGENS_KEY


def _as_optional_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    norm = str(value).strip().lower()
    if norm in {"1", "true", "yes", "y", "on"}:
        return True
    if norm in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _extract_model_output_width(config) -> int:
    model_conf = config.get("model", {})
    stack = model_conf.get("stack", [])
    if not isinstance(stack, list):
        return None
    for layer in stack:
        if not isinstance(layer, dict):
            continue
        if layer.get("out_field") != "node_output":
            continue
        out_irreps = layer.get("out_irreps")
        if out_irreps is None:
            continue
        m = re.match(r"^\s*(\d+)x", str(out_irreps))
        if m:
            return int(m.group(1))
    return None


def _infer_ignore_hydrogens_from_mapping(mapping_dir: Path, model_output_width: int):
    if mapping_dir is None or model_output_width is None:
        return None
    if not mapping_dir.exists() or not mapping_dir.is_dir():
        return None

    hierarchy_pattern = re.compile(r"^P\d+[A-Z]+$")

    def is_hydrogen_entry(atom_key: str) -> bool:
        names = [n.strip().upper() for n in str(atom_key).split(",")]
        return len(names) > 0 and all(n.startswith("H") for n in names)

    max_all = 0
    max_noh = 0
    for yaml_file in mapping_dir.glob("*.yaml"):
        if yaml_file.name.startswith("bead_types"):
            continue
        try:
            conf = yaml.safe_load(yaml_file.read_text())
        except Exception:
            continue
        atoms = conf.get("atoms", {}) if isinstance(conf, dict) else {}
        if not isinstance(atoms, dict):
            continue
        cnt_all = 0
        cnt_noh = 0
        for atom_key, atom_spec in atoms.items():
            tokens = str(atom_spec).split()
            has_hier = any(hierarchy_pattern.match(t) for t in tokens)
            if has_hier:
                cnt_all += 1
                if not is_hydrogen_entry(atom_key):
                    cnt_noh += 1
        max_all = max(max_all, cnt_all)
        max_noh = max(max_noh, cnt_noh)

    if model_output_width == max_noh and model_output_width != max_all:
        return True
    if model_output_width == max_all and model_output_width != max_noh:
        return False
    return None

def main():
    parser = argparse.ArgumentParser(description="Deploy a HEroBM backmapping model.")
    parser.add_argument("--verbose", default="INFO", type=str)
    
    # Reuse the base parser from GEqTrain
    parser = get_base_deploy_parser(parser)
    
    # Add HEroBM-specific arguments
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--bead-types-filename", type=Path, default="bead_types.yaml")
    parser.add_argument("--bead-stats", type=Path, required=False)
    parser.add_argument(
        "--ignore-hydrogens",
        action="store_true",
        dest="ignore_hydrogens",
        default=None,
        help="Mark deployed model as trained/inferred with hydrogen hierarchy ignored.",
    )
    parser.add_argument(
        "--predict-hydrogens",
        action="store_false",
        dest="ignore_hydrogens",
        help="Mark deployed model as including hydrogens in hierarchy reconstruction.",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.verbose.upper()))
    if args.model is None:
        parser.error("Missing required argument: --model/-m")

    model_path = args.model
    config_filename = str(model_path.parent / "config.yaml")
    config = Config.from_file(config_filename)

    # --- HEroBM Specific Logic ---
    herobm_metadata = {}
    logging.info(f"Embedding mapping file: {args.mapping}")
    herobm_metadata[MAPPING_KEY] = str(args.mapping)

    logging.info(f"Embedding bead types file: {args.bead_types_filename}")
    herobm_metadata[BEAD_TYPES_KEY] = str(args.bead_types_filename)
    
    if args.bead_stats is not None:
        logging.info(f"Embedding bead stats file: {args.bead_stats}")
        with open(args.bead_stats, 'r') as f:
            herobm_metadata[BEAD_STATS_KEY] = f.read()
    else:
        logging.info("No bead stats file provided; BEAD_STATS metadata will be omitted.")

    # Determine hydrogen hierarchy policy for inference.
    # Priority: CLI explicit flag > config value (if present).
    ignore_hydrogens = args.ignore_hydrogens
    if ignore_hydrogens is None:
        config_val = config.get("ignore_hydrogens", None)
        ignore_hydrogens = _as_optional_bool(config_val)
    if ignore_hydrogens is None:
        ignore_hydrogens = _infer_ignore_hydrogens_from_mapping(
            mapping_dir=args.mapping,
            model_output_width=_extract_model_output_width(config),
        )
        if ignore_hydrogens is not None:
            logging.info("Auto-inferred hydrogen hierarchy policy from model width + mapping.")
    if ignore_hydrogens is not None:
        herobm_metadata[IGNORE_HYDROGENS_KEY] = "true" if ignore_hydrogens else "false"
        logging.info(
            "Embedding hydrogen hierarchy policy: %s=%s",
            IGNORE_HYDROGENS_KEY,
            herobm_metadata[IGNORE_HYDROGENS_KEY],
        )

    # Also include any generic key-value pairs from the command line
    cli_metadata = {}
    for item in args.extra_metadata:
        if "=" not in item:
            raise ValueError(f"Invalid --extra-metadata entry: '{item}'. Expected key=value.")
        k, v = item.split("=", 1)
        cli_metadata[k] = v
    herobm_metadata.update(cli_metadata)

    # --- Call the Core GEqTrain Function ---
    build_deployment(
        model_path=model_path,
        out_file=args.out_file,
        config=config,
        extra_metadata=herobm_metadata  # Pass the combined HEroBM metadata
    )

if __name__ == "__main__":
    main()
