import re
import copy
import itertools
import numpy as np
from typing import List, Optional
from MDAnalysis.core.groups import Atom
from heqbm.utils.atomType import get_type_from_name

UNSET_V_VALUE = -1


class BeadMappingAtomSettings:

    def __init__(self, bead_settings, bead_name, atom_name) -> None:
        self.bead_name = bead_name
        self.atom_name = atom_name

        self.bead_name: str
        self.atom_name: str

        self._is_cm: bool = False
        self._has_cm: bool = False

        self._hierarchy_level: int = 0
        self._hierarchy_name: str = ''
        self._hierarchy_previous_name: str = ''

        self._local_index: int = -1
        self._local_index_prev: int = -1

        for setting in bead_settings:
            if setting == "CM":
                self.set_is_cm()

            pattern = 'P(\d+)([A-Z])([A-Z])*'
            result = re.search(pattern, setting)
            if result is not None :
                groups = result.groups()
                assert len(groups) == 3
                self.set_hierarchy_level(int(groups[0]))
                if groups[-1] is None:
                    self.set_hierarchy_name(groups[1])
                else:
                    self.set_hierarchy_previous_name(groups[1])
                    self.set_hierarchy_name(groups[2])

    @property
    def is_cm(self):
        return self._is_cm

    @property
    def has_cm(self):
        return self._has_cm
    
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
    
    def set_is_cm(self, is_cm: bool = True):
        self._is_cm = is_cm
        self.set_has_cm(is_cm)
    
    def set_has_cm(self, has_cm: bool = True):
        self._has_cm = has_cm
    
    def set_hierarchy_level(self, hierarchy_level: int):
        self._hierarchy_level = hierarchy_level
    
    def set_hierarchy_name(self, hierarchy_name: str):
        self._hierarchy_name = hierarchy_name
    
    def set_hierarchy_previous_name(self, hierarchy_previous_name: str):
        self._hierarchy_previous_name = hierarchy_previous_name
    
    def set_local_index(self, local_index: int):
        self._local_index = local_index
    
    def set_local_index_prev(self, local_index_prev: int):
        self._local_index_prev = local_index_prev


class BeadMappingSettings:

    def __init__(self, bead_name) -> None:
        self._bead_name = bead_name

        self._bead_name: str
        self._atom_settings: List[BeadMappingAtomSettings] = []
        self._bead_levels: set[int] = set()       # keep track of all hierarchy levels in the bead
        self._bead_positions: dict[int, int] = {} # for each hierarchy level, keep track of maximum position value

    def add_atom_settings(self, bmas: BeadMappingAtomSettings):
        self._atom_settings.append(bmas)
        self.update_atom_settings(bmas)
    
    def update_atom_settings(self, bmas: BeadMappingAtomSettings):
        self._bead_levels.add(bmas.hierarchy_level)
        self._bead_levels = set(sorted(self._bead_levels))
        self._bead_positions[bmas.hierarchy_level] = max(self._bead_positions.get(bmas.hierarchy_level, 0), bmas.hierarchy_position)
        has_cm = bmas.is_cm
        for saved_bmas in self._atom_settings:
            saved_bmas.set_local_index(self.get_bmas_local_index(saved_bmas))
            saved_bmas.set_local_index_prev(self.get_bmas_local_index_prev(saved_bmas))
            if saved_bmas.has_cm:
                has_cm = True
            if has_cm:    
                saved_bmas.set_has_cm()
    
    def get_ordered_bmas(self):
        return sorted(
            self._atom_settings,
            key=lambda x: self.get_bmas_local_index(x),
        )
    
    def get_hierarchy_level_offset(self, hierarchy_level):
        offset = 0
        for bead_level in self._bead_levels:
            if bead_level >= hierarchy_level:
                return offset
            offset += self._bead_positions[bead_level] + 1
        raise Exception(f'Hierarchy level {hierarchy_level} is not present in the bead')
    
    def get_bmas_local_index(self, bmas: BeadMappingAtomSettings):
        return self.get_hierarchy_level_offset(bmas.hierarchy_level) + bmas.hierarchy_position
    
    def get_bmas_local_index_prev(self, bmas: BeadMappingAtomSettings):
        return self.get_hierarchy_level_offset(bmas.hierarchy_level - 1) + bmas.hierarchy_prev_position
    
    def get_bmas_by_atom_name(self, atom_name: str):
        for bmas in self._atom_settings:
            if bmas.atom_name == atom_name:
                return bmas
        return None


class Bead:

    name: str
    type: int

    @property
    def n_atoms(self):
        assert len(self._atoms) == len(self._atom_idcs), \
            f"Bead of type {self.type} has a mismatch between the number of _atoms ({len(self._atoms)})" \
            f"and the number of _atom_idcs ({self._atom_idcs})"
        return len(self._atoms)
    
    @property
    def is_newly_created(self):
        return self._is_newly_created
    
    @property
    def is_complete(self):
        return self._is_complete

    def __init__(
            self,
            name: str,
            type: int,
            atoms: List[str],
            keep_hydrogens: bool,
    ) -> None:
        self.name = name
        self.type = type
        self._is_complete: bool = False
        self._is_newly_created: bool = True
        self.keep_hydrogens = keep_hydrogens
        
        self._missing_atoms_list: List[List[str]] = atoms
        self._config_ordered_atoms: List[List[str]] = copy.deepcopy(atoms)
        self._eligible_atoms: List[str] = set(itertools.chain(*self._missing_atoms_list))
        self._atom_names: List[str] = []
        self._atoms: List[Atom] = []
        self._atom_idcs: List[int] = []
        self._atom_weights: List[float] = []

        self._hierarchy_levels: List[int] = []
        self._local_index: List[int] = []
        self._local_index_prev: List[int] = []
    
    
    def is_missing_atom(self, atom_name: str):
        return atom_name in self._eligible_atoms and atom_name not in self._atom_names
    
    def update(self, atom_name: str, atom: Optional[Atom], _id: int, bmas: BeadMappingAtomSettings):
        self._is_newly_created = False
        assert atom_name in self._eligible_atoms, f"Trying to update bead {self.type} with atom {atom_name} that does not belong to it."
        self._atom_names.append(atom_name)
        self._atoms.append(atom)
        self._atom_idcs.append(_id)
        if bmas.has_cm:
            self._atom_weights.append(1. if bmas.is_cm else 0.)
        else:
            if get_type_from_name(atom_name) != 1:
                self._atom_weights.append(atom.mass if isinstance(atom, Atom) else 0.)
            else:
                self._atom_weights.append(0.)
        
        self._hierarchy_levels.append(bmas.hierarchy_level)
        self._local_index.append(bmas._local_index)
        self._local_index_prev.append(bmas._local_index_prev)

        invalid_confs = []
        atom_mapped = False
        for conf_id, _missing_atoms in enumerate(self._missing_atoms_list):
            if atom_name in _missing_atoms:
                _missing_atoms.remove(atom_name)
                atom_mapped = True
            else:
                invalid_confs.append(conf_id)
        if not atom_mapped:
            raise ValueError(f"Trying to update bead {self.type} with atom {atom_name} that does is already mapped in this bead.")
        if any([not [ma for ma in _missing_atoms if self.keep_hydrogens or get_type_from_name(ma) != 1] for _missing_atoms in self._missing_atoms_list]):
            self.complete()
        return invalid_confs
    
    def complete(self):
        if self.is_complete:
            return
        # We need to reorder the atoms in the bead according to the order of the mapping.
        # This is necessary since when doing backmapping directly from CG we reconstruct
        # atom positions hierarchily following the order of the mapping config, while when
        # reading an atomistic structure the order of the atoms is that of appeareance in the pdb.
        # We need consistency among atomistic and CG, otherwise the NN trained on the order of the
        # atomistic pdbs may swap the prediction order in the CG if atoms appear in different order in the config
        # w.r.t. the order in the pdbs used for training.
        atom_names = np.array(self._atom_names)
        coan_max_len = 0
        for coa in self._config_ordered_atoms:
            coan = np.array([x for x in coa if np.isin(x, atom_names)])
            if len(coan) > coan_max_len:
                coan_max_len = len(coan)
                config_ordered_atom_names = coan
        atom_names_sorted_idcs = np.argsort(atom_names)
        sorting_filter = atom_names_sorted_idcs[np.searchsorted(atom_names[atom_names_sorted_idcs], config_ordered_atom_names)]

        self._atom_names = atom_names[sorting_filter].tolist()
        self._atoms = np.array(self._atoms)[sorting_filter].tolist()
        self._atom_idcs = np.array(self._atom_idcs)[sorting_filter].tolist()
        self._atom_weights = np.array(self._atom_weights)[sorting_filter].tolist()
        self._hierarchy_levels = np.array(self._hierarchy_levels)[sorting_filter].tolist()
        self._local_index = np.array(self._local_index)[sorting_filter].tolist()
        self._local_index_prev = np.array(self._local_index_prev)[sorting_filter].tolist()

        self._is_complete = True


class RBBead(Bead):

    def __init__(self, name: str, type: int, atoms: List[str]) -> None:
        super().__init__(name, type, atoms)
        self._v1_id = UNSET_V_VALUE
        self._v2_id = UNSET_V_VALUE

    def get_v1_id(self):
        return self._v1_id
    
    def get_v2_id(self):
        return self._v2_id

    def set_v1_id(self, _id):
        self._v1_id = _id
    
    def set_v2_id(self, _id):
        self._v2_id = _id


class HierarchicalBead(Bead):

    def __init__(self, name: str, type: int, atoms: List[str], keep_hydrogens: bool) -> None:
        super().__init__(name, type, atoms, keep_hydrogens)