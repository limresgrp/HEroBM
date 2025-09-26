import argparse
import torch
import glob
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

from herobm.mapper import HierarchicalMapper

torch.set_default_dtype(torch.float32)

# This worker function processes a single file.
# It will be executed in a separate process.
def process_file(worker_args):
    """
    Worker function to process a single input file.
    
    Args:
        worker_args (dict): A dictionary containing all the necessary arguments
                            for processing one file, including 'input', 'inputtraj',
                            and 'output'.
    """
    # Each worker gets its own HierarchicalMapper instance.
    try:
        input_file = worker_args['input']
        output_file = worker_args['output']
        
        print(f"Starting processing for: {Path(input_file).name}")
        
        # We need to construct the full output path for the .npz file
        p = Path(input_file)
        output_dir = Path(output_file)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = output_dir / (p.stem + '.npz')

        # The mapper is initialized with args for a single file.
        mapping = HierarchicalMapper(args_dict=worker_args)
        
        # The map() method processes the single file specified during init.
        m = mapping.map() 
        
        if m is None:
            print(f'Skipping {output_filename} due to an issue during mapping (e.g., >20000 atoms).')
            return None

        m.save_npz(filename=str(output_filename), from_pos_unit='Angstrom', to_pos_unit='Angstrom')
        print(f'✅ Successfully saved: {output_filename}')
        return str(output_filename)
    except Exception as e:
        print(f"❌ Error processing {worker_args.get('input', 'unknown file')}: {e}")
        return None

def build_dataset(args_dict):
    print("Discovering files to process...")

    # --- Step 1: File Discovery (moved from Mapper.__init__) ---
    input_path = Path(args_dict["input"])
    input_traj_path = Path(args_dict["inputtraj"]) if args_dict.get("inputtraj") else None

    # Get a list of basenames to filter by, if a filter file is provided
    input_basenames = None
    if args_dict.get("filter"):
        with open(args_dict["filter"], 'r') as f:
            input_basenames = {line.strip() for line in f.readlines()}

    # Find all input structure/topology files
    if input_path.is_dir():
        input_format = args_dict.get("inputformat", "*")
        input_files = sorted(glob.glob(os.path.join(input_path, f"*.{input_format}")))
    else:
        input_files = [str(input_path)]
    
    # Filter files based on the filter list
    if input_basenames:
        input_files = [f for f in input_files if Path(f).stem in input_basenames]

    # --- Prepare tasks for multiprocessing ---
    tasks = []
    for input_file in input_files:
        # Create a copy of the original args for each task
        task_args = args_dict.copy()
        task_args['input'] = input_file
        task_args['output'] = args_dict['output'] # Pass the output directory

        # Find the corresponding trajectory file, if any
        traj_file = None
        if input_traj_path:
            if input_traj_path.is_dir():
                traj_format = args_dict.get("trajformat", "*")
                # Look for a trajectory file with the same basename
                potential_traj = input_traj_path / (Path(input_file).stem + f".{traj_format}")
                if potential_traj.exists():
                    traj_file = str(potential_traj)
            else:
                # If a single trajectory file is given, assume it applies to the (single) input
                traj_file = str(input_traj_path)
        
        task_args['inputtraj'] = traj_file
        tasks.append(task_args)

    print(f"Found {len(tasks)} files to process.")

    # --- Step 2: Run processing in a multiprocessing pool ---
    # Use slightly fewer than the total number of CPUs to leave resources for the OS.
    num_workers = args_dict.get('workers', None)
    if num_workers is None:
        num_workers = max(1, cpu_count()//2)
    print(f"Starting multiprocessing with {num_workers} workers...")
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_file, tasks)

    successful_files = [r for r in results if r is not None]
    print(f"\n--- All tasks completed. ---")
    print(f"{len(successful_files)} out of {len(tasks)} files were processed successfully.")

    # --- Step 3: Print summary based on the last processed configuration ---
    # To get bead types and sizes, we can run one mapping in the main process
    # or simply instantiate the mapper again.
    if tasks:
        print("\nGenerating configuration snippet...")
        summary_mapper = HierarchicalMapper(args_dict=args_dict)
        config_update_text = f'''Update the training configuration file with the following snippet:
        heads:
            head_output:
                field: node_features
                out_field: node_output
                out_irreps: {summary_mapper.bead_reconstructed_size}x1e
                model: geqtrain.nn.ReadoutModule
        
        type_names:\n'''
        for bt in {x[1]: x[0] for x in sorted(summary_mapper.bead_types_dict.items(), key=lambda x: x[1])}.values():
            config_update_text += f'- {bt}\n'
        print(config_update_text)

    print("Success!")

def parse_command_line(args=None):
    parser = argparse.ArgumentParser(
        description="Build training dataset"
    )

    parser.add_argument(
        "-m",
        "--mapping",
        help="Name of the CG mapping.\n" +
             "It corresponds to the name of the chosen folder relative to the cgmap/data/ folder.\n" +
             "Optionally, the user can specify its own mapping folder by providing an absolute path.",
        type=Path,
        default=None,
        required=True,
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Either Input folder or Input filename of atomistic structure to map.\n" +
             "Supported file formats are those of MDAnalysis (see https://userguide.mdanalysis.org/stable/formats/index.html)" +
             "If a folder is provided, all files in the folder (optionally filtered, see --filter) with specified extension (see --inputformat) will be used as Input file.",
        type=Path,
        default=None,
        required=True,
    )
    parser.add_argument(
        "-if",
        "--inputformat",
        help="Format of input files to consider, e.g., 'pdb'.\n" +
             "By default takes all formats.",
        default='*'
    )
    parser.add_argument(
        "-t",
        "--inputtraj",
        help="Input trjectory file or folder, which contains all input traj files.",
    )
    parser.add_argument(
        "-tf",
        "--trajformat",
        help="Format of input traj files to consider. E.g. 'trr'. By default takes all formats.",
    )
    parser.add_argument(
        "-f",
        "--filter",
        help="Filter file. It is a text file with a list of filenames (without extension) which will be used as input. "+
             "All files inside the input folder which do not appear in the filter file will be skipped.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output folder where to save the npz dataset.\n" +
             "If not provided, it will be the folder of the input.\n" +
             "The filename will be the one of the input with the .npz extension.",
    )
    parser.add_argument(
        "-s",
        "--selection",
        help="Selection of atoms to map. Dafaults to 'all'",
        default="all",
    )
    parser.add_argument(
        "-ts",
        "--trajslice",
        help="Specify a slice of the total number of frames.\n" +
             "Only the sliced frames will be backmapped.",
        type=str,
    )
    parser.add_argument(
        "-b",
        "--bead-types-filename",
        help="YAML file containing the bead type assigned to each bead. Default: 'bead_types.yaml'",
        type=str,
        default="bead_types.yaml",
    )
    parser.add_argument(
        "-c",
        "--cutoff",
        help="Pre-compute graph edges with specified cutoff.\n" +
             "If you set this option, information on preceding and following beads will be included (suggested choice).",
        type=float,
    )
    parser.add_argument('--isatomistic', action='store_true', default=True, help='Specify that the input is atomistic (default)')
    parser.add_argument('-cg', action='store_false', dest='isatomistic', help='Specify that the input is coarse-grained')
    parser.add_argument(
        "-w",
        "--workers",
        help="Number of parallel processes to run for building dataset",
        type=int,
        default=None,
    )

    return parser.parse_args(args=args)

def main(args=None):
    args_dict = parse_command_line(args)
    build_dataset(vars(args_dict))

if __name__ == "__main__":
    main()