"""Keys file to overcome TorchScript constants bug."""

import sys

if sys.version_info[1] >= 8:
    from typing import Final
else:
    from typing_extensions import Final

# Generic keys

STR_SEPARATOR: str = '_'

# DataDict keys

ATOM_POSITION: Final[str] = "atom_pos"
BB_ATOM_POSITION: Final[str] = "bb_atom_position"
CA_ATOM_POSITION: Final[str] = "ca_atom_pos"
CA_ATOM_IDCS: Final[str] = "ca_atom_idcs"
CA_NEXT_DIRECTION: Final[str] = "ca_next_direction"
BEAD_POSITION: Final[str] = "bead_pos"
BEAD_POSITION_ORIGINAL: Final[str] = "bead_pos_orig"
CA_BEAD_POSITION: Final[str] = "ca_bead_pos"
CA_BEAD_IDCS: Final[str] = "ca_bead_idcs"

ATOM_POSITION_PRED: Final[str] = "atom_pos_pred"
BB_ATOM_POSITION_PRED: Final[str] = "bb_atom_position_pred"
ATOM_POSITION_MINIMISATION_TRAJ: Final[str] = "pos_minimisation_traj"
BB_ATOM_POSITION_MINIMISATION_TRAJ: Final[str] = "bb_pos_minimisation_traj"

ATOM_NAMES: Final[str] = "atom_names"
BEAD_NAMES: Final[str] = "bead_names"
ATOM_RESIDCS: Final[str] = "atom_residcs"
BEAD_RESIDCS: Final[str] = "bead_residcs"

ATOM_TYPES: Final[str] = "atom_types"
ATOM_CHAINIDCS: Final[str] = "atom_chain_idcs"
BEAD_CHAINIDCS: Final[str] = "bead_chain_idcs"
MAPPING_IDCS: Final[str] = "mapping_idcs"
BEAD_TYPES: Final[str] = "bead_types"

ATOM_FORCES: Final[str] = "atom_forces"
BEAD_FORCES: Final[str] = "bead_forces"

CELL: Final[str] = "cell"
PBC: Final[str] = "pbc"

BEAD2ATOM_IDCS: Final[str] = "bead2atom_idcs"
BEAD2ATOM_IDCS_MASK: Final[str] = "bead2atom_idcs_mask"

ORIGINAL_FRAMES_IDCS: Final[str] = "orig_frames_idcs"

DIHEDRAL_IDCS: Final[str] = "dihedral_idcs"
PHI_DIH_IDCS: Final[str] = "phi_dih_idcs"
PSI_DIH_IDCS: Final[str] = "psi_dih_idcs"
OMEGA_DIH_IDCS:  Final[str] = "omega_dih_idcs"

BB_PHIPSI:  Final[str] = "bb_phipsi"
BB_PHIPSI_PRED:  Final[str] = "bb_phipsi_pred"

LEVEL_IDCS_MASK:  Final[str] = "lvl_idcs_mask"
LEVEL_IDCS_ANCHOR_MASK:  Final[str] = "lvl_idcs_anchor_mask"
BEAD2ATOM_RELATIVE_VECTORS:  Final[str] = "bead2atom_rel_vectors"
BEAD2ATOM_RELATIVE_VECTORS_PRED:  Final[str] = "bead2atom_rel_vectors_pred"

BOND_IDCS:  Final[str] = "bond_idcs"
ANGLE_IDCS:  Final[str] = "angle_idcs"