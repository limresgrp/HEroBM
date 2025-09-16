from cgmap._keys import *  # noqa: F403, F401
from .._keys import *  # noqa: F403, F401

# Also import the module to use in TorchScript, this is a hack to avoid bug:
# https://github.com/pytorch/pytorch/issues/52312
from .. import _keys

PROTEIN_RESNAMES = [
    'ALA', 'GLY', 'ARG', 'ASN', 'ASP', 'GLN', 'GLU', 'HIE',
    'HID', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'SER', 'CYS',
    'TRP', 'TYR', 'GLN', 'PHE', 'PRO', 'THR', 'VAL',
]

MAPPING_KEY: Final[str] = "mapping"
BEAD_TYPES_KEY: Final[str] = "bead_types_filename"
BEAD_STATS_KEY: Final[str] = "bead_stats"