# herobm/scripts/deploy.py

import argparse
import logging
from pathlib import Path

# IMPORTANT: Import the reusable components from GEqTrain
from geqtrain.utils.deploy import build_deployment, get_base_deploy_parser
from geqtrain.utils import Config

# Import HEroBM specific keys
from herobm.utils.DataDict import MAPPING_KEY, BEAD_TYPES_KEY, BEAD_STATS_KEY

def main():
    parser = argparse.ArgumentParser(description="Deploy a HEroBM backmapping model.")
    parser.add_argument("--verbose", default="INFO", type=str)
    
    # Reuse the base parser from GEqTrain
    parser = get_base_deploy_parser(parser)
    
    # Add HEroBM-specific arguments
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--bead-types-filename", type=Path, default="bead_types.yaml")
    parser.add_argument("--bead-stats", type=Path, required=False)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.verbose.upper()))
    if args.model is None:
        parser.error("Missing required argument: --model/-m")

    model_path = args.model
    config_path = model_path.parent / "config.yaml"
    config = Config.from_file(str(config_path))

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
