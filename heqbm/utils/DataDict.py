from .._keys import *  # noqa: F403, F401

# Also import the module to use in TorchScript, this is a hack to avoid bug:
# https://github.com/pytorch/pytorch/issues/52312
from .. import _keys

RESNAMES = [
    'ALA', 'GLY', 'ARG', 'ASN', 'ASP', 'GLN', 'GLU', 'HIE',
    'HID', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'SER', 'CYS',
    'TRP', 'TYR', 'GLN', 'PHE', 'PRO', 'THR', 'VAL',
]