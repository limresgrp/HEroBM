import os
import glob
from typing import List, Tuple
from pdbfixer import PDBFixer
from openmm import Vec3
from openmm.app import Topology

def fixPDB(filename: str, addMissingResidues: bool = False, addHydrogens: bool = True) -> Tuple[Topology, List[Vec3]]:
    fixer = PDBFixer(filename=filename)
    if addMissingResidues:
        fixer.findMissingResidues()
    else:
        fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    if addHydrogens:
        fixer.addMissingHydrogens(7.0)
    # maxSize = max(max((pos[i] for pos in fixer.positions))-min((pos[i] for pos in fixer.positions)) for i in range(3))
    # boxSize = maxSize*Vec3(1, 1, 1)
    # fixer.addSolvent(boxSize)
    return fixer.topology, fixer.positions

def joinPDBs(data_dir, tag):
    i = 1
    files_iterator = sorted(glob.glob(f'{data_dir}/**{tag}_*.pdb', recursive=True))
    if len(files_iterator) == 0:
        return
    out_file = os.path.join(data_dir.replace("**", "multi"), f'multi_{tag}.pdb')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f_out:
        for filename in files_iterator:
            f_out.write(f"MODEL     {i}\n")
            with open(filename, "r") as f_in:
                txt = f_in.read()
                txt = txt.replace("\nEND\n", "\n")
                f_out.write(txt)
            f_out.write(f"ENDMDL\n")
            i += 1
        f_out.write(f"END\n")