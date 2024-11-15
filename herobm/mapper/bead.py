import logging
import re
import itertools
import numpy as np

from typing import Optional, List
from MDAnalysis.core.groups import Atom
from cgmap.mapping.bead import BeadMappingAtomSettings, BeadMappingSettings, Bead
from cgmap.utils import DataDict


class HEroBMBeadMappingAtomSettings(BeadMappingAtomSettings):

    def __init__(
        self,
        bead_settings: List[str],
        bead_name: str,
        atom_idnames: List[str],
        num_shared_beads: int
    ) -> None:

        super(HEroBMBeadMappingAtomSettings, self).__init__(bead_settings, bead_name, atom_idnames, num_shared_beads)

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

        self.set_has_to_be_reconstructed(has_to_be_reconstructed=False)
        for setting in bead_settings:
            try:
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
            except ValueError as e:
                logging.error(f"Error while parsing mapping for bead {self.bead_name}. Incorrect mapping setting: {setting}")
                raise e
    
    @property
    def hierarchy_level(self):
        return self._hierarchy_level
    
    @property
    def hierarchy_position(self):
        if self._hierarchy_name == '':
            return 0
        return ord(self._hierarchy_name) - ord('A')

    @property
    def hierarchy_prev_position(self):
        if self._hierarchy_previous_name == '':
            return 0
        return ord(self._hierarchy_previous_name) - ord('A')
    
    def set_local_index(self, local_index: int):
        self._local_index = local_index
    
    def set_local_index_prev(self, local_index_prev: int):
        self._local_index_prev = local_index_prev

    def set_hierarchy_level(self, hierarchy_level: int):
        self._hierarchy_level = hierarchy_level
    
    def set_hierarchy_name(self, hierarchy_name: str):
        self._hierarchy_name = hierarchy_name
    
    def set_hierarchy_previous_name(self, hierarchy_previous_name: str):
        self._hierarchy_previous_name = hierarchy_previous_name


class HEroBMBeadMappingSettings(BeadMappingSettings):

    def __init__(self, bead_idname) -> None:
        super(HEroBMBeadMappingSettings, self).__init__(bead_idname=bead_idname)

        self._atom_settings: List[HEroBMBeadMappingAtomSettings] = []
        self._bead_levels: set[int] = set()       # keep track of all hierarchy levels in the bead
        self._bead_positions: dict[int, int] = {} # for each hierarchy level, keep track of maximum position value

    def update_atom_settings(self, bmas: HEroBMBeadMappingAtomSettings):
        super(HEroBMBeadMappingSettings, self).update_atom_settings(bmas=bmas)
        self._bead_levels.add(bmas.hierarchy_level)
        self._bead_levels = set(sorted(self._bead_levels))
        self._bead_positions[bmas.hierarchy_level] = max(self._bead_positions.get(bmas.hierarchy_level, 0), bmas.hierarchy_position)
        for saved_bmas in self._atom_settings:
            saved_bmas.set_local_index(self.get_bmas_local_index(saved_bmas))
            saved_bmas.set_local_index_prev(self.get_bmas_local_index_prev(saved_bmas))
    
    def get_bmas_local_index(self, bmas: HEroBMBeadMappingAtomSettings):
        return self.get_hierarchy_level_offset(bmas.hierarchy_level) + bmas.hierarchy_position
    
    def get_bmas_local_index_prev(self, bmas: HEroBMBeadMappingAtomSettings):
        return self.get_hierarchy_level_offset(bmas.hierarchy_level - 1) + bmas.hierarchy_prev_position
    
    def get_hierarchy_level_offset(self, hierarchy_level):
        if hierarchy_level <= -1:
            return -1
        offset = 0
        for bead_level in self._bead_levels:
            if bead_level >= hierarchy_level:
                return offset
            offset += self._bead_positions[bead_level] + 1
        raise Exception(f'Hierarchy level {hierarchy_level} is not present in the bead')


class HEroBMBead(Bead):

    def __init__(
            self,
            bms: HEroBMBeadMappingSettings,
            id: int,
            idname: str,
            type: int,
            atoms_offset: int,
            bead2atoms: List[List[str]], # Could have multiple configuration files with different atom namings
            weigth_based_on: str,
            resindex: int = 0,
            resnum: int = 0,
            segid: str = 'A',
    ) -> None:
        
        super(HEroBMBead, self).__init__(bms, id, idname, type, atoms_offset, bead2atoms, weigth_based_on, resindex, resnum, segid)

        self._all_hierarchy_levels = -np.ones((self.n_all_atoms, ), dtype=np.int16)
        self._all_local_index = -np.ones((self.n_all_atoms, ), dtype=np.int16)
        self._all_local_index_anchor = -np.ones((self.n_all_atoms, ), dtype=np.int16)

    def update(
        self,
        atom_idname: str,
        bmas: HEroBMBeadMappingAtomSettings,
        atom: Optional[Atom] = None,
        atom_index: Optional[int] = None,
    ):
        if self._is_newly_created and atom is not None:
            self.resindex = atom.resindex
            self.resnum = atom.resnum
            self.segid = atom.segid
        self._is_newly_created = False
        assert atom_idname in self._eligible_atom_idnames, f"Trying to update bead {self.name} with atom {atom_idname} that does not belong to it."

        conf_ordered_index = None
        updated_config_ordered_atom_idnames = []
        for coaidnames in self._config_ordered_atom_idnames:
            coai_index = np.argwhere(coaidnames == atom_idname)
            if len(coai_index) > 0:
                updated_config_ordered_atom_idnames.append(coaidnames)
                if conf_ordered_index is None:
                    for coaid in coai_index:
                        if self._all_local_index[coaid[0]] == -1:
                            conf_ordered_index = coaid[0]
                            self._alternative_name_index[conf_ordered_index] = coaid[1]
                            break
            else:
                pass
        if conf_ordered_index is None:
            raise Exception(f"Atom with idname {atom_idname} not found in mapping configuration files for bead {self.idname}")
        self._config_ordered_atom_idnames = updated_config_ordered_atom_idnames
        
        self._n_found_atoms += 1

        if atom is not None:
            self._all_atoms.append(atom)
        self._all_atom_idnames.append(atom_idname)
        
        self._all_hierarchy_levels[conf_ordered_index] = bmas.hierarchy_level
        self._all_local_index[conf_ordered_index] = bmas._local_index
        self._all_local_index_anchor[conf_ordered_index] = bmas._local_index_prev

        # All atoms without the '!' flag contribute to the bead position
        # Those with the '!' flag appear with weight 0
        weight = 0.
        if bmas.has_cm:
            weight = 1. * bmas.is_cm
        elif bmas.contributes_to_cm:
            if self.weigth_based_on == "same":
                weight = 1.
            elif self.weigth_based_on == "mass":
                weight = bmas.mass
            else:
                raise Exception(f"{self.weigth_based_on} is not a valid value for 'weigth_based_on'. Use either 'mass' or 'same'")
        weight *= bmas.relative_weight
        self._all_atom_weights[conf_ordered_index] = weight

        idcs_to_update = None
        if atom_index is not None:
            # If atom index is not None, it means that this atom is shared with another bead,
            # and its idname, position and other properties have already been stored.
            # Thus, on this bead we adjust the index to point to the previously saved one (atom_index)
            # and we scale atom indices that are greater to avoid having gaps in the indexing.
            shared_atom_index = self._all_atom_idcs[conf_ordered_index]
            self._all_atom_idcs[conf_ordered_index] = atom_index
            idcs_to_update = np.copy(self._all_atom_idcs[self._all_atom_idcs > shared_atom_index])
            self._all_atom_idcs[self._all_atom_idcs > shared_atom_index] -= 1
        
        if bmas.has_to_be_reconstructed:
            self._reconstructed_conf_ordered_idcs.append(conf_ordered_index)
            self._reconstructed_atom_idnames.append(atom_idname)

        if self._n_found_atoms == self.n_all_atoms:
            self.complete()
        
        return self._all_atom_idcs[conf_ordered_index], idcs_to_update