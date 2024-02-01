"""Keys file to overcome TorchScript constants bug."""

import sys

if sys.version_info[1] >= 8:
    from typing import Final
else:
    from typing_extensions import Final

from nequip.data import register_fields

EDGE_ENERGY: Final[str] = "edge_energy"
EDGE_FORCES: Final[str] = "edge_forces"
EDGE_FEATURES: Final[str] = "edge_features"
INVARIANT_EDGE_FEATURES: Final[str] = "inv_edge_features"
EQUIVARIANT_EDGE_FEATURES: Final[str] = "eq_edge_features"
EQUIVARIANT_EDGE_LENGTH_FEATURES: Final[str] = "eq_edge_length_features"
INVARIANT_ATOM_FEATURES: Final[str] = "inv_atom_features"
EQUIVARIANT_ATOM_FEATURES: Final[str] = "eq_atom_features"
EQUIVARIANT_ATOM_LENGTH_FEATURES: Final[str] = "eq_atom_length_features"

EQUIVARIANT_ATOM_INPUT_FEATURES: Final[str] = "eq_atom_in_features"

ATOM_POSITIONS: Final[str] = "atom_pos"

CONTRIBUTIONS_KEY: Final[str] = "contributions"
CURL: Final[str] = "curl"


register_fields(
    node_fields=[INVARIANT_ATOM_FEATURES, EQUIVARIANT_ATOM_FEATURES],
    edge_fields=[EDGE_ENERGY, EDGE_FEATURES, INVARIANT_EDGE_FEATURES, EQUIVARIANT_EDGE_FEATURES, CONTRIBUTIONS_KEY],
)