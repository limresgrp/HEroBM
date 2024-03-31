import argparse
from heqbm.utils import DataDict
from heqbm.utils.pdbFixer import joinPDBs
from heqbm.backmapping.hierarchical_backmapping import HierarchicalBackmapping

# python eval.py -c config/backmapping/CG-vince.yaml -d cuda:1 -ts 5000:15000 -o /storage_common/angiod/A2A/CG/Vince/SASA-R12-HUGE/


def eval(args):
    backmapping = HierarchicalBackmapping(args_dict=vars(args))

    for structure_id in range(backmapping.num_structures):
        skip = backmapping.map(
            structure_id,
            skip_if_existent=False,
        )
        if skip:
            continue

        frame_idcs = range(0, len(backmapping.mapping.dataset[DataDict.BEAD_POSITION]))
        n_frames = max(frame_idcs) + 1

        backmapped_u = None
        for frame_index in frame_idcs:
            try:
                backmapping_dataset = backmapping.backmap(
                    frame_index=frame_index,
                    optimise_backbone=True,
                    minimise_dih=False
                )

                backmapped_u = backmapping.to_pdb(
                    backmapping_dataset=backmapping_dataset,
                    n_frames=n_frames,
                    frame_index=frame_index,
                    backmapped_u=backmapped_u,
                )
            except Exception as e:
                print(e)
                print("Skipping.")

        for tag in ['backmapped_min']:
            joinPDBs(backmapping.config.get("output"), tag)


def parse_command_line(args=None):
    parser = argparse.ArgumentParser(
        description="Run backmapping inference"
    )

    parser.add_argument(
        "-c",
        "--config",
        help="Name of the configuration file containing all backmapping inputs.",
    )
    parser.add_argument(
        "-m",
        "--mapping",
        help="Name of the CG mapping. It corresponds to the name of the chosen folder inside heqbm/data/",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Input file or input folder.\n" +
             "If a folder is given, all the files contained in the folder will be backmapped.\n" +
             "The user might provide the --inputformat to filter only files with the specified extension.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output folder.",
    )
    parser.add_argument(
        "-if",
        "--inputformat",
        help="Format of input files to consider.\n" + 
             "E.g. 'pdb' or 'gro'. By default takes all formats.",
    )
    parser.add_argument(
        "-it",
        "--inputtraj",
        nargs='+',
        help="List of trajectory files to load.\n",
    )
    parser.add_argument(
        "-ts",
        "--trajslice",
        help="Specify a slice of the total number of frames.\n" +
             "Only the sliced frames will be backmapped.",
    )
    parser.add_argument(
        "-s",
        "--selection",
        help="Selection of atoms. Dafaults to 'all'.",
    )
    parser.add_argument(
        "-d",
        "--device",
        help="Device to use for running the backmapping inference. Defaults to 'cpu'.",
        default="cpu"
    )
    parser.add_argument(
        "-md",
        "--model",
        help="HEqBM model to use for backmapping.\n" +
             "Could be either a deployed model (.pth) or a training config file (.yaml)",
    )
    parser.add_argument(
        "-mw",
        "--modelweights",
        help="HEqBM model weights to load.\n" +
             "They need to be provided if you load the model from the training config file.\n" +
             "Defaults to 'best_model.pth'.",
    )

    return parser.parse_args(args=args)


def main(args=None):
    eval(parse_command_line(args))
    


if __name__ == "__main__":
    main()