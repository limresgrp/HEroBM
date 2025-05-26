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


def analyze_distances(input, mapping, output):
    distances = defaultdict(list)

    files = glob.glob(os.path.join(input, "*.pdb")) + glob.glob(os.path.join(input, "*.gro"))
    for filename in files:
        try:
            print(f"Analysing CG distances for file {filename}...")
            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=True) as tmpfile:
                tmp_filename = tmpfile.name
                args_dict = {
                "mapping": mapping,
                "input": filename,
                "output": tmp_filename,
                "isatomistic": True,
                }
                cg_filename = aa2cg(args_dict)
                
                # The rest of the code using cg_filename goes here
                try:
                    u = mda.Universe(cg_filename)
                    relevant_bead_names = sorted({atom.name for atom in u.atoms})
                    print("Universe loaded successfully.")
                except FileNotFoundError:
                    print(f"Error: PDB file not found at {cg_filename}")
                    sys.exit(1)
                except Exception as e:
                    print(f"An error occurred while loading the universe: {e}")
                    sys.exit(1)

                print("Calculating distances for connected beads...")

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
                                    atom1 = atom1[0]
                                    atom2 = atom2[0]
                                    dist = np.linalg.norm(atom2.position - atom1.position)
                                    key = (res1.resname, bead1_name, res1.resname, bead2_name)
                                    distances[key].append(dist)
                        # Inter-residue
                        if i + 1 < len(segment.residues):
                            res2 = segment.residues[i+1]
                            if res2.resid == res1.resid + 1:
                                bb1 = res1.atoms.select_atoms("name BB")
                                bb2 = res2.atoms.select_atoms("name BB")
                                if bb1 and bb2:
                                    bb_atom1 = bb1[0]
                                    bb_atom2 = bb2[0]
                                    dist = np.linalg.norm(bb_atom2.position - bb_atom1.position)
                                    sorted_resnames = sorted([res1.resname, res2.resname])
                                    key = (sorted_resnames[0], 'BB', sorted_resnames[1], 'BB')
                                    distances[key].append(dist)
                    
                print("Completed.")
        except: pass

    print("Finished calculating distances.")
    print("Aggregating results and computing statistics...")
    rows = []
    sorted_keys = sorted(distances.keys())
    for key in sorted_keys:
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
    output_columns = ["resname1.beadname1", "resname2.beadname2", "count", "mean_distance", "std_distance", "min_distance", "max_distance"]
    df = pd.DataFrame(rows)
    df = df.reindex(columns=output_columns)
    df.to_csv(output, index=False)
    print(f"Distance statistics saved to {output}")
    print("Script finished.")

def main():
    parser = argparse.ArgumentParser(description="Analyze connected bead distances in a CG structure.")
    parser.add_argument("-i", "--input", required=True, help="Input folder or PDB/GRO file")
    parser.add_argument("-m", "--mapping", required=True, type=Path, help="Name of the CG mapping.\n"+
                        "It corresponds to the name of the chosen folder relative to the cgmap/data/ folder.\n"+
                        "Optionally, the user can specify its own mapping folder by providing an absolute path.",)
    parser.add_argument("-o", "--output", default="out.csv", help="Output CSV file (default: out.csv)")
    args = parser.parse_args()
    analyze_distances(args.input, args.mapping, args.output)

if __name__ == "__main__":
    main()
