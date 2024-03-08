import argparse
import json
import os
from os.path import basename
import glob
import torch
import numpy as np
from typing import Optional
from heqbm.mapper import HierarchicalMapper
from heqbm.utils import DataDict

torch.set_default_dtype(torch.float32)

# EXAMPLE RUNS #
# python build_train_dataset.py -m martini3_lipids -i /storage_common/angiod/POPC/ -t /storage_common/angiod/POPC/ -if gro -s "resname POPC" -tf trr -l 100 -o /storage_common/angiod/POPC/backmapping/npz/membrane/train/
# python build_train_dataset.py -m martini3_bbcommon -i /storage_common/angiod/PDB6K/pdb.6k/augment/ -f /storage_common/angiod/PDB6K/set/targets.train.pdb.2.9k -if pdb -s protein -o /storage_common/angiod/PDB6K/backmapping/npz/martini3.bbcommon.2.9k/train
# python build_train_dataset.py -m ca -i /storage_common/angiod/PDB6K/pdb.6k/augment/ -f /storage_common/angiod/PDB6K/set/targets.train.pdb.2.9k -if pdb -s protein -o /storage_common/angiod/PDB6K/backmapping/npz/ca.2.9k/train

YOUR_PATH_TO_DATA_FOLDER = "/storage_common/angiod/"


def get_ds(
        filename: str,
        mapping_folder: str,
        traj_folder_in: Optional[str] = None,
        traj_format: Optional[str] = None,
        selection: str = 'protein',
        keep_backbone: bool = False,
        frame_limit: int = np.inf,
        extra_kwargs: Optional[dict] = None,
        mapping: Optional[HierarchicalMapper] = None,
        keep_hydrogens: bool = True,
    ):
    conf = {
    "simulation_is_cg": False,
    "keep_hydrogens": keep_hydrogens,
    "structure_filename": filename,
    "mapping_folder": mapping_folder,
    }
    if extra_kwargs is not None:
        conf["extra_kwargs"] = extra_kwargs

    if traj_folder_in is not None:
        if traj_format is None:
            traj_format = "*"
        for traj_filename in glob.glob(os.path.join(traj_folder_in, f"{basename(filename).split('.')[0]}*.{traj_format}")):
            traj_filenames = conf.get("traj_filenames", [])
            traj_filenames.append(traj_filename)
            conf["traj_filenames"] = traj_filenames

    if mapping is None:
        mapping = HierarchicalMapper(config=conf)
    mapping.map(conf, selection=selection, frame_limit=frame_limit)
    dataset = mapping.dataset

    if not keep_backbone:
        dataset[DataDict.BEAD2ATOM_RELATIVE_VECTORS][:, dataset[DataDict.CA_BEAD_IDCS]] = 0.

    npz_ds = {
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

    return mapping, npz_ds

def build_dataset(config):
    os.makedirs(config.output, exist_ok=True)
    if config.filter is not None:
        with open(config.filter, 'r') as f:
            basenames = [line.strip() for line in f.readlines()]
    else:
        basenames = None
    mapping = None
    input_format = config.inputformat or "*"
    for filename in glob.glob(os.path.join(config.inputfolder, f"*.{input_format}")):
        try:
            if basenames is not None:
                file_basename = '.'.join(basename(filename).split('.')[:-1])
                if file_basename not in basenames:
                    continue
            filename_out = os.path.join(config.output, f"{basename(filename).split('.')[0]}.npz")
            if os.path.isfile(filename_out):
                print(f"Dataset {filename_out} is already present. Skipping.")
                continue
            
            mapping, npz_ds = get_ds(
                filename=filename,
                mapping_folder=config.mapping,
                traj_folder_in=config.trajfolder,
                traj_format=config.trajformat,
                selection=config.selection,
                frame_limit=config.limitframes,
                extra_kwargs=config.extrakwargs,
                keep_backbone=True,
                mapping=mapping,
                keep_hydrogens=True,
            )

            print(filename_out, npz_ds[DataDict.ATOM_POSITION].shape)
            if npz_ds is not None:
                np.savez(filename_out, **npz_ds)
                config_update_text = f'''Update the training configuration file with the following snippet (excluding quotation marks):
                \n"\neq_out_irreps: {mapping.bead_reconstructed_size}x1o\n\ntype_names:\n'''
                for bt in [x[0] for x in sorted(mapping.bead_types_dict.items(), key=lambda x: x[1])]:
                    config_update_text += f'- {bt}\n'
                config_update_text += '"'
                print(config_update_text)
        except TypeError as e:
            print(e)
            print(f"Skipping file {filename}. Most probably the resid is messed up")


def parse_command_line(args=None):
    parser = argparse.ArgumentParser(
        description="Build training dataset"
    )

    parser.add_argument(
        "-m", "--mapping", required=True, help="Name of the CG mapping. It corresponds with the name of the chosen folder inside heqbm/data/"
    )
    parser.add_argument(
        "-i", "--inputfolder", required=True, help="Input folder, which contains all input files. Usually they are pdb or gro files.",
    )
    parser.add_argument(
        "-s",
        "--selection",
        help="Selection of atoms. Dafaults to 'all'",
        default="all"
    )
    parser.add_argument(
        "-if",
        "--inputformat",
        help="Format of input files to consider. E.g. 'pdb'. By default takes all formats.",
    )
    parser.add_argument(
        "-t",
        "--trajfolder",
        help="Input trjectory folder, which contains all input traj files. Usually they are trr or xtc files. Their basename must match that of input file.",
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
        required=True,
        help="Path to output file. It must be a .npz file",
    )
    parser.add_argument(
        "-e",
        "--extrakwargs",
        help="Extra arguments to pass to the mapper.",
        type=json.loads,
    )
    parser.add_argument(
        "-l",
        "--limitframes",
        help="Limit the maximum number of frames to extract from each input file.",
        type=int,
    )

    return parser.parse_args(args=args)


def main(args=None):
    config = parse_command_line(args)
    build_dataset(config)
    


if __name__ == "__main__":
    main()