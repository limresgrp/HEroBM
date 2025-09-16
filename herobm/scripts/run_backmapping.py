import argparse
import logging
from typing import Dict, Union
from pathlib import Path
from herobm.backmapping.hierarchical_backmapping import HierarchicalBackmapping, load_model
from geqtrain.scripts.deploy import MAPPING_KEY, BEAD_TYPES_KEY, BEAD_STATS_KEY
from geqtrain.utils._global_options import register_all_fields

def run_backmapping(
    model, 
    model_config, 
    args_dict: Dict, 
    bead_stats: Union[Path, str] = None, 
    num_steps: int = 1000, 
    tolerance: float = 500.0,
    chunking: int = 0,
):
    """
    Initializes and runs the hierarchical backmapping process.

    Args:
        model: The loaded torch model.
        model_config: The model's configuration dictionary.
        args_dict (Dict): A dictionary of arguments for HierarchicalBackmapping.
        bead_stats (Union[Path, str], optional): Path to bead distance stats or the CSV content as a string. Defaults to None.
        num_steps (int, optional): Number of steps for bead position minimization. Defaults to 1000.
        tolerance (float, optional): Energy tolerance for minimization. Defaults to 500.0.
    """
    preprocess_func = None
    if bead_stats:
        from herobm.backmapping.minimize_cg import minimize_bead_distances
        from functools import partial
        
        if isinstance(bead_stats, Path):
            print(f"Bead minimization enabled using stats from file: {bead_stats}")
            # Create a preprocessing function that will be called on the data
            preprocess_func = partial(minimize_bead_distances, csv_filepath=str(bead_stats), num_steps=num_steps)
        elif isinstance(bead_stats, str):
            print("Bead minimization enabled using stats from model metadata.")
            # Pass the CSV content directly
            preprocess_func = partial(minimize_bead_distances, csv_content=bead_stats, num_steps=num_steps)

    backmapping = HierarchicalBackmapping(
        model=model,
        model_config=model_config,
        args_dict=args_dict, 
        preprocess_npz_func=preprocess_func
    )

    print("Starting backmapping process...")
    backmapped_filenames, backmapped_minimised_filenames, true_filenames, cg_filenames = backmapping.backmap(
        tolerance=tolerance,
        chunking=chunking,
    )

    print("\n--- Backmapping Complete ---")
    print("Backmapped files:", backmapped_filenames)
    print("Backmapped and minimised files:", backmapped_minimised_filenames)
    print("Original atomistic files (if provided):", true_filenames)
    print("Coarse-grained input files:", cg_filenames)
    print("--------------------------\n")

    return backmapped_filenames, backmapped_minimised_filenames, true_filenames, cg_filenames

def main():
    """Parses command-line arguments and launches the backmapping task."""
    parser = argparse.ArgumentParser(
        description="Run inference with a trained Hierarchical Backmapping (HEroBM) model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Argument Groups for Clarity ---
    io_group = parser.add_argument_group("Input and Output Arguments")
    io_group.add_argument("-i", "--input", type=Path, help="Input atomistic PDB file.")
    io_group.add_argument("-it", "--inputtraj", type=Path, help="Input atomistic trajectory file (e.g., .xtc, .trr).")
    io_group.add_argument("-o", "--output", type=Path, default=Path("./output"), help="Output directory for backmapped files.")

    model_group = parser.add_argument_group("Model Configuration Arguments")
    model_group.add_argument("-mo", "--model", type=Path, required=True, help="Path to the trained/deployed model file (.pth).")
    model_group.add_argument("-m", "--mapping", type=str, help="Coarse-grain mapping to use. Overrides model metadata if provided.")
    model_group.add_argument("-b", "--bead-types-filename", type=Path, help="YAML file defining bead types. Overrides model metadata if provided.")
    model_group.add_argument("-a", "--atomistic", action="store_true", default=False, dest="isatomistic", help="Flag if the input is atomistic (not coarse-grained). Use to test model accuracy on MAP->BACKMAP.")

    processing_group = parser.add_argument_group("Processing and Selection Arguments")
    processing_group.add_argument("-s", "--selection", type=str, default="all", help="Atom selection string (MDAnalysis format).")
    processing_group.add_argument("-ts", "--trajslice", type=str, help="Slice of the trajectory (e.g., '900:1000:10').")
    processing_group.add_argument("-d", "--device", type=str, default="cuda:0", help="Torch device to use.")
    processing_group.add_argument("-c", "--chunking", type=int, default=0, help="Enable chunked processing of the input trajectory, setting the max number of atoms per batch.")
    
    minimization_group = parser.add_argument_group("Energy Minimization and Refinement")
    minimization_group.add_argument("-bs", "--bead-stats", type=Path, help="Path to a .csv file with bead-bead distance stats for optional CG minimization. Overrides model metadata if provided.")
    minimization_group.add_argument("-ns", "--num-steps", type=int, default=1000, help="Number of steps for the bead minimization.")
    minimization_group.add_argument("-t", "--tolerance", type=float, default=500.0, help="Energy tolerance for final atomistic minimisation (kJ/(mol nm)).")

    args = parser.parse_args()
    
    # --- Load Model and Metadata ---
    print(f"Loading model from: {args.model}")
    model, model_config, metadata = load_model(str(args.model), device=args.device)
    register_all_fields(model_config)

    # --- Use Metadata as Fallback for Arguments ---
    # Command-line arguments will take precedence over metadata
    if args.mapping is None:
        args.mapping = metadata.pop(MAPPING_KEY)
        if args.mapping:
            print(f"Using mapping from model metadata: {args.mapping}")

    if args.bead_types_filename is None:
        bead_types_path = metadata.pop(BEAD_TYPES_KEY)
        if bead_types_path:
            args.bead_types_filename = Path(bead_types_path)
            print(f"Using bead types file from model metadata: {args.bead_types_filename}")

    # Determine which bead_stats to use (CLI > metadata)
    bead_stats_arg = None
    if args.bead_stats:
        bead_stats_arg = args.bead_stats # Use file path from CLI
    elif BEAD_STATS_KEY in metadata:
        bead_stats_arg = metadata.pop(BEAD_STATS_KEY) # Use CSV content from metadata

    # --- Validate Configuration ---
    if not args.mapping:
         raise ValueError("A mapping must be provided either via the --mapping argument or within the deployed model's metadata.")
    if not args.bead_types_filename:
         logging.warning("bead_types_filename not found in arguments or model metadata. This may be required depending on the mapping.")

    # Create a dictionary from the parsed arguments for HierarchicalBackmapping
    args_dict = vars(args)

    # This seems to be a fixed requirement for your backmapping class
    args_dict["noinvariants"] = True

    # Clean up args that are for this script, not for HierarchicalBackmapping
    num_steps = args_dict.pop('num_steps')
    tolerance = args_dict.pop('tolerance')
    chunking  = args_dict.pop('chunking')
    # We pop bead_stats from the dict as it's handled separately
    args_dict.pop('bead_stats', None) 

    # Ensure output directory exists
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
    
    run_backmapping(
        model,
        model_config,
        args_dict, 
        bead_stats=bead_stats_arg, 
        num_steps=num_steps, 
        tolerance=tolerance,
        chunking=chunking,
    )


if __name__ == "__main__":
    main()
