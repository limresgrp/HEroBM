import argparse
import re
from pathlib import Path
from collections import defaultdict

# Try to import MDAnalysis, but don't fail if it's not installed.
# The script can still perform the primary merging task without it.
try:
    import MDAnalysis as mda
    MDANALYSIS_AVAILABLE = True
except ImportError:
    MDANALYSIS_AVAILABLE = False


def find_and_map_files(folder_path: Path, pattern: str) -> dict[int, Path]:
    """
    Scans a directory for files matching a pattern and extracts an integer from the filename.

    The pattern should contain '{}' as a placeholder for the integer.
    Example pattern: "min_{}.pdb"

    Args:
        folder_path: The Path object for the directory to scan.
        pattern: The filename pattern to match.

    Returns:
        A dictionary mapping the extracted integer to the full file path.
    """
    if not folder_path.is_dir():
        print(f"Error: Folder not found at '{folder_path}'")
        return {}

    # Convert the user-friendly pattern to a regular expression
    # Replaces {} with a capturing group for one or more digits (\d+)
    regex_pattern = re.compile(pattern.replace("{}", r"(\d+)"))

    file_map = {}
    for file_path in folder_path.iterdir():
        if file_path.is_file():
            match = regex_pattern.match(file_path.name)
            if match:
                # The first captured group is our integer
                file_id = int(match.group(1))
                file_map[file_id] = file_path
    
    print(f"Found {len(file_map)} matching files in '{folder_path}'.")
    return file_map


def merge_pdb_content(path1: Path, path2: Path) -> str:
    """
    Merges the content of two PDB files.

    It takes all lines from the first file (excluding the END tag),
    then appends all ATOM/HETATM records from the second file.
    Finally, it adds an END tag.

    Args:
        path1: Path to the first PDB file.
        path2: Path to the second PDB file.

    Returns:
        A string containing the merged PDB content.
    """
    merged_lines = []

    # Read the first file
    with open(path1, 'r') as f1:
        # Add all lines, but skip any existing END record to avoid premature termination
        for line in f1:
            if not line.strip().startswith('END'):
                merged_lines.append(line)

    # Add a TER record if the first file didn't end with one, for good separation
    if merged_lines and not merged_lines[-1].strip().startswith('TER'):
         merged_lines.append("TER\n")


    # Read the second file and append only ATOM/HETATM records
    with open(path2, 'r') as f2:
        for line in f2:
            if line.strip().startswith(('ATOM', 'HETATM')):
                merged_lines.append(line)

    # Add the final END record
    merged_lines.append("END\n")

    return "".join(merged_lines)


def create_combined_trajectories_by_group(merged_files: list[Path], output_folder: Path):
    """
    Groups PDB files by atom count and creates a separate multi-frame PDB and 
    XTC trajectory for each group.

    Args:
        merged_files: A list of Path objects for the PDB files to combine.
        output_folder: The Path object for the directory to save trajectory files.
    """
    if not MDANALYSIS_AVAILABLE:
        print("\nSkipping combined trajectory creation: MDAnalysis library not found.")
        print("Please install it to enable this feature: pip install MDAnalysis")
        return

    if not merged_files:
        print("\nNo merged files to combine into a trajectory.")
        return

    print("\nCreating combined trajectory files for each unique atom count...")
    
    # 1. Group PDB files by their number of atoms
    atom_count_groups = defaultdict(list)
    print("Scanning files to group by atom count...")
    for pdb_path in merged_files:
        try:
            # Load each PDB to find its atom count
            temp_universe = mda.Universe(str(pdb_path))
            n_atoms = temp_universe.atoms.n_atoms
            atom_count_groups[n_atoms].append(pdb_path)
        except Exception as e:
            print(f"  - WARNING: Could not read '{pdb_path.name}'. Skipping. Error: {e}")
            continue

    if not atom_count_groups:
        print("No valid PDB files could be processed.")
        return

    print("\nFound the following groups:")
    for n_atoms, files in atom_count_groups.items():
        print(f"  - Group with {n_atoms} atoms has {len(files)} frame(s).")

    # 2. For each group, create the trajectory files
    for n_atoms, file_list in atom_count_groups.items():
        if len(file_list) <= 1:
            print(f"\nSkipping trajectory creation for group with {n_atoms} atoms (only {len(file_list)} frame).")
            continue

        print(f"\nBuilding trajectory for group with {n_atoms} atoms...")
        
        # Define unique output names for this group
        output_basename = f"combined_trajectory_{n_atoms}atoms"
        combined_pdb_path = output_folder / f"{output_basename}.pdb"
        combined_xtc_path = output_folder / f"{output_basename}.xtc"

        try:
            # Load all PDBs in the current group as a single trajectory
            universe = mda.Universe(str(file_list[0]), [str(p) for p in file_list])
            all_atoms = universe.select_atoms("all")

            # Write the multi-frame PDB for the group
            all_atoms.write(str(combined_pdb_path), frames='all')
            print(f"  - Created combined PDB: '{combined_pdb_path.name}'")

            # Write the XTC trajectory for the group
            all_atoms.write(str(combined_xtc_path), frames='all')
            print(f"  - Created combined XTC: '{combined_xtc_path.name}'")

        except Exception as e:
            print(f"\nAn error occurred while creating trajectory for group with {n_atoms} atoms: {e}")


def main():
    """Main function to parse arguments and run the merging process."""
    parser = argparse.ArgumentParser(
        description="Merge pairs of PDB files from two folders based on a matching integer in their filenames.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--folder1", 
        type=str, 
        required=True, 
        help="Path to the first folder containing PDB files."
    )
    parser.add_argument(
        "--pattern1", 
        type=str, 
        required=True, 
        help="Filename pattern for the first folder. Use '{}' as a placeholder for the number.\nExample: 'min_{}.pdb'"
    )
    parser.add_argument(
        "--folder2", 
        type=str, 
        required=True, 
        help="Path to the second folder containing PDB files."
    )
    parser.add_argument(
        "--pattern2", 
        type=str, 
        required=True, 
        help="Filename pattern for the second folder. Use '{}' as a placeholder for the number.\nExample: 'relaxed_{}.pdb'"
    )
    parser.add_argument(
        "--output_folder", 
        type=str, 
        required=True, 
        help="Path to the folder where merged PDB files will be saved."
    )
    parser.add_argument(
        "--output_pattern",
        type=str,
        default="merged_{}.pdb",
        help="Filename pattern for the output files. Use '{}' as a placeholder for the number.\nDefault: 'merged_{}.pdb'"
    )

    args = parser.parse_args()

    # Convert string paths to Path objects for easier handling
    folder1_path = Path(args.folder1)
    folder2_path = Path(args.folder2)
    output_folder_path = Path(args.output_folder)

    # Create the output directory if it doesn't exist
    output_folder_path.mkdir(parents=True, exist_ok=True)

    # Find and map files in both directories
    files1_map = find_and_map_files(folder1_path, args.pattern1)
    files2_map = find_and_map_files(folder2_path, args.pattern2)

    # Find the common keys (integers) between the two maps and sort them
    common_ids = sorted(list(files1_map.keys() & files2_map.keys()))

    if not common_ids:
        print("\nNo file pairs with matching numbers were found. Exiting.")
        return

    print(f"\nFound {len(common_ids)} matching file pairs. Starting merge process...")

    merged_file_paths = []
    # Process each matching pair
    for file_id in common_ids:
        file1 = files1_map[file_id]
        file2 = files2_map[file_id]

        # Get merged content
        merged_content = merge_pdb_content(file1, file2)
        
        # Define output path and write the new file
        output_filename = args.output_pattern.format(file_id)
        output_path = output_folder_path / output_filename
        
        with open(output_path, 'w') as out_f:
            out_f.write(merged_content)
        
        merged_file_paths.append(output_path)
        print(f"  - Merged '{file1.name}' and '{file2.name}' -> '{output_path.name}'")

    print("\nMerging complete.")

    # Create combined trajectories if any files were merged
    if merged_file_paths:
        create_combined_trajectories_by_group(merged_file_paths, output_folder_path)


if __name__ == "__main__":
    main()
