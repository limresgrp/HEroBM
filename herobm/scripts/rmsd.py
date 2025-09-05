import os
import glob
import argparse
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rms

def compute_rmsds(folder):
    bb_rmsds, sc_rmsds, all_rmsds = [], [], []
    for backmapped_filename in glob.glob(os.path.join(folder, "*.backmapped_[0-9]*.pdb")):
        reference_filename = backmapped_filename.replace("backmapped_", "true_")

        backmapped_u = mda.Universe(backmapped_filename)
        ref_u = mda.Universe(reference_filename)

        bb_rmsds.append(rms.rmsd(
            backmapped_u.select_atoms('name N CA C O').positions,
            ref_u.select_atoms('name N CA C O').positions,
            superposition=False
        ))
        sc_rmsds.append(rms.rmsd(
            backmapped_u.select_atoms('protein and not name N CA C O').positions,
            ref_u.select_atoms('protein and not name N CA C O').positions,
            superposition=False
        ))
        all_rmsds.append(rms.rmsd(
            backmapped_u.select_atoms('protein').positions,
            ref_u.select_atoms('protein').positions,
            superposition=False
        ))

    bb_rmsds = np.array(bb_rmsds)
    sc_rmsds = np.array(sc_rmsds)
    all_rmsds = np.array(all_rmsds)

    print(f"Backbone RMSD: mean = {bb_rmsds.mean():.4f}, std = {bb_rmsds.std():.4f}")
    print(f"Side Chain RMSD: mean = {sc_rmsds.mean():.4f}, std = {sc_rmsds.std():.4f}")
    print(f"All Heavy Atoms RMSD: mean = {all_rmsds.mean():.4f}, std = {all_rmsds.std():.4f}")

def main():
    parser = argparse.ArgumentParser(description="Compute RMSDs between backmapped and reference PDB files.")
    parser.add_argument("folder", type=str, help="Folder containing backmapped PDB files")
    args = parser.parse_args()
    compute_rmsds(args.folder)

if __name__ == "__main__":
    main()