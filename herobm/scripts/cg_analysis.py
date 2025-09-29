import os
import glob
from pathlib import Path
import MDAnalysis as mda
import numpy as np
import pandas as pd
from collections import defaultdict
import sys
import argparse
from cgmap.scripts.map import aa2cg
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed


def compute_distances_for_file(filename, mapping, inputtraj=None):
    distances = defaultdict(list)
    try:
        print(f"Analysing CG distances for file {filename}...")
        with tempfile.TemporaryDirectory() as tmpfolder:
            args_dict = {
                "mapping": mapping,
                "input": filename,
                "inputtraj": inputtraj,
                "output": tmpfolder,
                "isatomistic": True,
            }
            cg_filenames, cg_filename_trajs = aa2cg(args_dict)

            for cg_filename, cg_filename_traj in zip(cg_filenames, cg_filename_trajs):
                try:
                    if cg_filename_traj is not None:
                        u = mda.Universe(cg_filename, cg_filename_traj)
                    else:
                        u = mda.Universe(cg_filename)
                    relevant_bead_names = sorted({atom.name for atom in u.atoms})
                    print(f"Universe for {filename} loaded successfully.")
                except Exception as e:
                    print(f"Error loading {cg_filename}: {e}")
                    return distances

                print("Calculating distances for connected beads...")
                for ts in u.trajectory:
                    for segment in u.segments:
                        for i in range(len(segment.residues)):
                            res1 = segment.residues[i]
                            # Intra-residue
                            for j1 in range(len(relevant_bead_names) - 1):
                                for j2 in range(j1 + 1, len(relevant_bead_names)):
                                    bead1_name = relevant_bead_names[j1]
                                    bead2_name = relevant_bead_names[j2]
                                    atom1 = res1.atoms.select_atoms(f"name {bead1_name}")
                                    atom2 = res1.atoms.select_atoms(f"name {bead2_name}")
                                    if atom1 and atom2:
                                        dist = np.linalg.norm(atom2[0].position - atom1[0].position)
                                        key = (res1.resname, bead1_name, res1.resname, bead2_name)
                                        distances[key].append(dist)
                            # Inter-residue
                            if i + 1 < len(segment.residues):
                                res2 = segment.residues[i + 1]
                                if res2.resid == res1.resid + 1:
                                    bb1 = res1.atoms.select_atoms("name BB")
                                    bb2 = res2.atoms.select_atoms("name BB")
                                    if bb1 and bb2:
                                        dist = np.linalg.norm(bb2[0].position - bb1[0].position)
                                        sorted_resnames = sorted([res1.resname, res2.resname])
                                        key = (sorted_resnames[0], "BB", sorted_resnames[1], "BB")
                                        distances[key].append(dist)
        print(f"Completed {filename}")
        return distances
    except Exception as e:
        print(f"Error in {filename}: {e}")
        return distances


def merge_distances(dicts):
    merged = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            merged[k].extend(v)
    return merged


def analyze_distances(input, inputtraj, mapping, output, workers=1):
    if os.path.isdir(input):
        files = glob.glob(os.path.join(input, "*.pdb")) + glob.glob(os.path.join(input, "*.gro"))
    elif input.endswith((".pdb", ".gro")) and os.path.isfile(input):
        files = [input]
    else:
        raise ValueError("Input must be a folder or a .pdb/.gro file")

    if workers == 1:
        results = [compute_distances_for_file(f, mapping, inputtraj) for f in files]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_file = {executor.submit(compute_distances_for_file, f, mapping, inputtraj): f for f in files}
            for future in as_completed(future_to_file):
                results.append(future.result())

    distances = merge_distances(results)

    print("Finished calculating distances. Aggregating results...")
    rows = []
    for key in sorted(distances.keys()):
        resname1, beadname1, resname2, beadname2 = key
        dists = distances[key]
        if dists:
            rows.append({
                "resname1.beadname1": f"{resname1}.{beadname1}",
                "resname2.beadname2": f"{resname2}.{beadname2}",
                "count": len(dists),
                "mean_distance": np.mean(dists),
                "std_distance": np.std(dists),
                "min_distance": np.min(dists),
                "max_distance": np.max(dists),
            })
    df = pd.DataFrame(rows)
    df = df.reindex(columns=["resname1.beadname1", "resname2.beadname2", "count", "mean_distance", "std_distance", "min_distance", "max_distance"])
    df.to_csv(output, index=False)
    print(f"Distance statistics saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="Analyze connected bead distances in a CG structure.")
    parser.add_argument("-i", "--input", required=True, help="Input folder or PDB/GRO file")
    parser.add_argument("-m", "--mapping", required=True, type=Path, help="Path to CG mapping folder")
    parser.add_argument("-t", "--inputtraj", help="Input trajectory file or folder")
    parser.add_argument("-o", "--output", default="out.csv", help="Output CSV file (default: out.csv)")
    parser.add_argument("-w", "--workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    args = parser.parse_args()
    analyze_distances(args.input, args.inputtraj, args.mapping, args.output, workers=args.workers)


if __name__ == "__main__":
    main()
