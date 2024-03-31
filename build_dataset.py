import argparse
import json
import os
import glob
import torch
import numpy as np
import yaml

from os.path import dirname, basename
from typing import Dict
from pathlib import Path

from heqbm.mapper import HierarchicalMapper
from heqbm.utils import DataDict
from heqbm.utils.parsing import parse_slice

torch.set_default_dtype(torch.float32)

# EXAMPLE RUNS #
# python build_dataset.py -m martini3_lipids -i /storage_common/angiod/POPC/ -t /storage_common/angiod/POPC/ -if gro -s "resname POPC" -tf trr -l 100 -o /storage_common/angiod/POPC/backmapping/npz/membrane/train/
# python build_dataset.py -m martini3_bbcommon -i /storage_common/angiod/PDB6K/pdb.6k/augment/ -f /storage_common/angiod/PDB6K/set/targets.train.pdb.2.9k -if pdb -s protein -o /storage_common/angiod/PDB6K/backmapping/npz/martini3.bbcommon.2.9k/train
# python build_dataset.py -m ca -i /storage_common/angiod/PDB6K/pdb.6k/augment/ -f /storage_common/angiod/PDB6K/set/targets.train.pdb.2.9k -if pdb -s protein -o /storage_common/angiod/PDB6K/backmapping/npz/ca.2.9k/train

YOUR_PATH_TO_DATA_FOLDER = "/storage_common/angiod/"


def to_npz(dataset):
    return {
        k: v for k, v in dataset.items() if k in [
            DataDict.ATOM_POSITION, DataDict.BEAD_POSITION, DataDict.ATOM_NAMES,
            DataDict.BEAD_NAMES, DataDict.ATOM_TYPES, DataDict.BEAD_TYPES,
            DataDict.BEAD2ATOM_RELATIVE_VECTORS, DataDict.BEAD_RESIDCS,
            DataDict.BB_PHIPSI, DataDict.LEVEL_IDCS_MASK,
            DataDict.LEVEL_IDCS_ANCHOR_MASK, DataDict.BEAD2ATOM_IDCS,
            DataDict.BEAD2ATOM_WEIGHTS, DataDict.BOND_IDCS,
            DataDict.ANGLE_IDCS, DataDict.CELL, DataDict.PBC
        ]
    }

def build_npz_dataset(args_dict, skip_if_existent: bool = False):
    config: Dict[str, str] = dict()

    yaml_config = args_dict.pop("config", None)
    if yaml_config is not None:
        config.update(yaml.safe_load(Path(yaml_config).read_text()))
    
    args_dict = {key: value for key, value in args_dict.items() if value is not None}
    config.update(args_dict)

    if config.get("trajslice", None) is not None:
        config["trajslice"] = parse_slice(config["trajslice"])

    mapping = HierarchicalMapper(config=config)

    input = config.get("input")
    output = config.get("output", input if os.path.isdir(input) else dirname(input))

    if os.path.isdir(input):
        input_folder = input
        input_format = config.get("inputformat", "*")
        filter = config.get("filter", None)
        input_basenames = None
        if filter is not None:
            with open(filter, 'r') as f:
                input_basenames = [line.strip() for line in f.readlines()]
        input_filenames = [
            fname for fname in
            list(glob.glob(os.path.join(input_folder, f"*.{input_format}")))
            if input_basenames is None
            or fname in input_basenames
        ]
    else:
        input_folder = None
        input_filename = input
        input_filenames = [input_filename]
    
    if input_folder is None:
        mapping.map()
        yield to_npz(mapping.dataset), config
    else:
        for input_filename in input_filenames:
            config["input"] = input_filename
            config["inputtraj"] = []
            p = Path(os.path.join(output, basename(input_filename)))
            config["output"] = str(p.with_suffix('.npz'))
            if skip_if_existent and os.path.isfile(config["output"]):
                yield None, None
                continue
            print(f"Mapping structure {input_filename}")
            mapping.map()
            yield to_npz(mapping.dataset), config
    
    config_update_text = f'''Update the training configuration file with the following snippet (excluding quotation marks):
    \n"\neq_out_irreps: {mapping.bead_reconstructed_size}x1o\n\ntype_names:\n'''
    for bt in [x[0] for x in sorted(mapping.bead_types_dict.items(), key=lambda x: x[1])]:
        config_update_text += f'- {bt}\n'
    config_update_text += '"'
    print(config_update_text)

def build_dataset(args_dict):
    for npz_dataset, config in build_npz_dataset(args_dict, skip_if_existent=False):
        if npz_dataset is not None:
            os.makedirs(config["output"], exist_ok=True)
            p = Path(os.path.join(config["output"], basename(config["input"])))
            np.savez(p.rename(p.with_suffix('.npz')), **npz_dataset)

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
    )
    # parser.add_argument(
    #     "-t",
    #     "--trajfolder",
    #     help="Input trjectory folder, which contains all input traj files. Their basename must match that of input file.",
    # )
    # parser.add_argument(
    #     "-tf",
    #     "--trajformat",
    #     help="Format of input traj files to consider. E.g. 'trr'. By default takes all formats.",
    # )
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
        type=int,
    )
    parser.add_argument(
        "-e",
        "--extrakwargs",
        help="Extra arguments to pass to the mapper.",
        type=json.loads,
    )

    return parser.parse_args(args=args)


def main(args=None):
    args_dict = parse_command_line(args)
    build_dataset(args_dict)
    


if __name__ == "__main__":
    main()