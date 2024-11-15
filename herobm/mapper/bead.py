import logging
import re
import itertools
import numpy as np

from typing import Optional, List, Union
from MDAnalysis.core.groups import Atom
from cgmap.mapping.bead import BeadMappingAtomSettings
from cgmap.utils import DataDict
from cgmap.utils.atomType import get_mass_from_name


class HeroBMBeadMappingAtomSettings(BeadMappingAtomSettings):

    def __init__(
        self,
        bead_settings: List[str],
        bead_name: str,
        atom_idnames: List[str],
        num_shared_beads: int
    ) -> None:

        self.bead_name: str = bead_name
        self.atom_idnames: List[str] = atom_idnames
        self.atom_names = [atom_idname.split(DataDict.STR_SEPARATOR)[1] for atom_idname in atom_idnames]
        self.atom_resname = atom_idnames[0].split(DataDict.STR_SEPARATOR)[0]

        self._num_shared_beads: int = num_shared_beads

        self._contributes_to_cm: bool = True
        self._is_cm: bool = False
        self._has_cm: bool = False

        self._has_to_be_reconstructed: bool = False
        self._hierarchy_level: int = -1
        self._hierarchy_name: str = ''
        self._hierarchy_previous_name: str = ''

        # Relative index that keeps track of the order in which atoms are reconstructed inseide the residue.
        # _local_index = 0 is the atom with lowest hierarchy level and name A.
        # _local_index = 1 is the atom with lowest hierarchy level and name B OR the atom with second lowest hierarchy level and name A.
        # and so on...
        self._local_index: int = -1
        # Relative index of the atom to e used as anchor point when reconstructing this atom.
        self._local_index_prev: int = -1

        self._mass: float = get_mass_from_name(self.atom_names[0])
        self._relative_weight: float = 1.
        self._relative_weight_set: bool = False

        for setting in bead_settings:
            try:
                if setting == "!":
                    self.exclude_from_cm()
                    continue
                if setting == "CM":
                    self.set_is_cm()
                    continue

                hierarchy_pattern = 'P(\d+)([A-Z])([A-Z])*'
                result = re.search(hierarchy_pattern, setting)
                if result is not None:
                    groups = result.groups()
                    assert len(groups) == 3
                    self.set_has_to_be_reconstructed()
                    self.set_hierarchy_level(int(groups[0]))
                    if groups[-1] is None:
                        self.set_hierarchy_name(groups[1])
                    else:
                        self.set_hierarchy_previous_name(groups[1])
                        self.set_hierarchy_name(groups[2])
                    continue
                
                weight_pattern = '(\d)/(\d)'
                result = re.search(weight_pattern, setting)
                if result is not None:
                    groups = result.groups()
                    assert len(groups) == 2
                    self.set_relative_weight(float(int(groups[0])/int(groups[1])))
                    continue

                weight_pattern = '(\d*(?:\.\d+)?)'
                result = re.search(weight_pattern, setting)
                if result is not None:
                    groups = result.groups()
                    assert len(groups) == 1
                    self.set_relative_weight(float(groups[0]))
                    continue
            except ValueError as e:
                logging.error(f"Error while parsing mapping for bead {self.bead_name}. Incorrect mapping setting: {setting}")
                raise e
        
        if not self._relative_weight_set:
            self.set_relative_weight(self.relative_weight / self._num_shared_beads)