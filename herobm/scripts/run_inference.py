import argparse
from typing import Dict
from herobm.backmapping.hierarchical_backmapping import HierarchicalBackmapping

# python run_inference.py -m martini2 -i /capstor/scratch/cscs/dangiole/HEroBM/A2A.A2B/stefano.a2a.gro -s protein --cg -mo /users/dangiole/HEroBM/martini2.protein.deployed_model.pth -d cuda -b bead_types.bbcommon.yaml -bs /users/dangiole/HEroBM/out.csv

def run_backmapping(args_dict: Dict, bead_stats: str = None, num_steps: int = 100, tolerance: float = 500.0):
    if bead_stats:
        from herobm.backmapping.minimize_cg import minimize_bead_distances
        from functools import partial
        func = partial(minimize_bead_distances, csv_filepath=bead_stats, num_steps=num_steps)
        backmapping = HierarchicalBackmapping(args_dict=args_dict, preprocess_npz_func=func)
    else:
        backmapping = HierarchicalBackmapping(args_dict=args_dict)

    backmapped_filenames, backmapped_minimised_filenames, true_filenames, cg_filenames = backmapping.backmap(
        tolerance=tolerance,
    )

    print("Backmapped filenames:", backmapped_filenames)
    print("Backmapped and minimised filenames:", backmapped_minimised_filenames)
    print("True filenames (if applicable):", true_filenames)
    print("Coarse-grained filenames (if applicable):", cg_filenames)

    return backmapped_filenames, backmapped_minimised_filenames, true_filenames, cg_filenames

def main():
    parser = argparse.ArgumentParser(description="Run hierarchical backmapping.")

    # Input and output arguments
    parser.add_argument("-m", "--mapping", type=str, default="martini3", help="Mapping to use (default: martini3)")
    parser.add_argument("-i", "--input", type=str, default=None, help="Input atomistic PDB file")
    parser.add_argument("-it", "--inputtraj", type=str, default=None, help="Input atomistic trajectory file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output directory for backmapped files")

    # Selection and slicing arguments
    parser.add_argument("-s", "--selection", type=str, default="all", help="Atom selection for backmapping (default: all)")
    parser.add_argument("-ts", "--trajslice", type=str, default=None, help="Slice of the trajectory to use (e.g., '900:1000:10')")

    # Coarse-graining and model arguments
    parser.add_argument("--cg", action="store_false", dest="isatomistic", help="Set this flag when backmapping actual CG input")
    parser.add_argument("-mo", "--model", type=str, default=None, help="Path to the trained model file")
    parser.add_argument("-b", "--bead-types-filename", help="YAML file containing the bead type assigned to each bead. Default: 'bead_types.yaml'", type=str,)

    # Device and invariant arguments
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="Torch device to use (default: cuda:0)")

    # Backmapping specific arguments
    parser.add_argument("-bs", "--bead-stats", type=str, default=None, help="csv file with bead2bead distance stats. Pass this to minimize bead pos before backmapping.")
    parser.add_argument("-ns", "--num-steps", type=int, default=1000, help="Number of steps for the bead pos minimization algorithm before backmapping.")
    parser.add_argument("-t", "--tolerance", type=float, default=500.0, help="Energy tolerance for minimisation (kJ/(mol nm)) (default: 500.0)")

    args = parser.parse_args()

    arg_names = ['mapping', 'input', 'inputtraj', 'isatomistic', 'selection', 'trajslice', 'model', 'bead_types_filename', 'output', 'device']
    args_dict = {}
    for arg_name in arg_names:
        arg_value = getattr(args, arg_name)
        if arg_value is not None:
            args_dict[arg_name] = arg_value
    args_dict["noinvariants"] = True  # This is always True

    run_backmapping(args_dict, bead_stats=args.bead_stats, num_steps=args.num_steps, tolerance=args.tolerance)


if __name__ == "__main__":
    main()
