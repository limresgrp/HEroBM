"""Keys file to overcome TorchScript constants bug."""

import sys

if sys.version_info[1] >= 8:
    from typing import Final
else:
    from typing_extensions import Final

# DataDict keys

BB_ATOM_POSITION: Final[str] = "bb_atom_position"
CA_ATOM_POSITION: Final[str] = "ca_atom_pos"
CA_ATOM_IDCS: Final[str] = "ca_atom_idcs"

CA_BEAD_POSITION: Final[str] = "ca_bead_pos"
CA_BEAD_IDCS: Final[str] = "ca_bead_idcs"

ATOM_POSITION_PRED: Final[str] = "atom_pos_pred"

BEAD2ATOM_IDCS: Final[str] = "bead2atom_idcs"
BEAD2ATOM_WEIGHTS: Final[str] = "bead2atom_weights"

ORIGINAL_FRAMES_IDCS: Final[str] = "orig_frames_idcs"

PHI_DIH_IDCS: Final[str] = "phi_dih_idcs"
PSI_DIH_IDCS: Final[str] = "psi_dih_idcs"
OMEGA_DIH_IDCS:  Final[str] = "omega_dih_idcs"

BB_PHIPSI:  Final[str] = "bb_phipsi"
BB_PHIPSI_PRED:  Final[str] = "bb_phipsi_pred"

LEVEL_IDCS_MASK:  Final[str] = "lvl_idcs_mask"
LEVEL_IDCS_ANCHOR_MASK:  Final[str] = "lvl_idcs_anchor_mask"
BEAD2ATOM_RELATIVE_VECTORS:  Final[str] = "bead2atom_rel_vectors"
BEAD2ATOM_RELATIVE_VECTORS_PRED:  Final[str] = "bead2atom_rel_vectors_pred"

BEAD2ATOM_RECONSTRUCTED_IDCS:  Final[str] = "bead2atom_reconstructed_idcs"
BEAD2ATOM_RECONSTRUCTED_WEIGHTS:  Final[str] = "bead2atom_reconstructed_weights"

BOND_IDCS:  Final[str] = "bond_idcs"
ANGLE_IDCS:  Final[str] = "angle_idcs"
TORSION_IDCS: Final[str] = "torsion_idcs"