import re
import itertools
import MDAnalysis
import numpy as np
from typing import List, Optional
from MDAnalysis.core.groups import Atom
from heqbm.utils import DataDict
from heqbm.utils.atomType import get_type_from_name


class BeadMappingAtomSettings:

    def __init__(self, bead_settings, bead_name, atom_idname, num_shared_beads: int) -> None:
        self.bead_name: str = bead_name
        self.atom_idname: str = atom_idname

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
                print(f"Error while parsing mapping for bead {self.bead_name}. Incorrect mapping setting: {setting}")
                raise e
        
        if not self._relative_weight_set:
            self.set_relative_weight(self.relative_weight / self._num_shared_beads)

    @property
    def contributes_to_cm(self):
        return self._contributes_to_cm

    @property
    def is_cm(self):
        return self._is_cm

    @property
    def has_cm(self):
        return self._has_cm
    
    @property
    def has_to_be_reconstructed(self):
        return self._has_to_be_reconstructed
    
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
    
    @property
    def relative_weight(self):
        return self._relative_weight
    
    def exclude_from_cm(self):
        self._contributes_to_cm = False

    def set_is_cm(self, is_cm: bool = True):
        self._is_cm = is_cm
        self.set_has_cm(is_cm)
    
    def set_has_cm(self, has_cm: bool = True):
        self._has_cm = has_cm
    
    def set_has_to_be_reconstructed(self, has_to_be_reconstructed: bool = True):
        self._has_to_be_reconstructed = has_to_be_reconstructed
    
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
    
    def set_relative_weight(self, weight: float):
        self._relative_weight_set = True
        self._relative_weight = weight


class BeadMappingSettings:

    def __init__(self, bead_idname) -> None:
        self._bead_idname: str = bead_idname
        self._is_complete = False
        self._bead_reconstructed_size = 0
        self._bead_all_size = 0

        self._atom_settings: List[BeadMappingAtomSettings] = []
        self._bead_levels: set[int] = set()       # keep track of all hierarchy levels in the bead
        self._bead_positions: dict[int, int] = {} # for each hierarchy level, keep track of maximum position value
    
    @property
    def shared_atoms(self):
        return [_as.atom_idname for _as in self._atom_settings if _as._has_to_be_reconstructed and _as._num_shared_beads > 1]

    @property
    def bead_reconstructed_size(self):
        assert self._is_complete
        return self._bead_reconstructed_size
    
    @property
    def bead_all_size(self):
        assert self._is_complete
        return self._bead_all_size

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
    
    def complete(self):
        self.update_relative_weights()
        if not self._is_complete:
            self._bead_reconstructed_size = sum([atom_setting._has_to_be_reconstructed for atom_setting in self._atom_settings])
            self._bead_all_size = len(self._atom_settings)
            self._is_complete = True

    def update_relative_weights(self):
        total_relative_weight = sum([_as.relative_weight for _as in self._atom_settings])
        for bmas in self._atom_settings:
            bmas.set_relative_weight(bmas.relative_weight / total_relative_weight)

    def get_ordered_bmas(self):
        return sorted(
            self._atom_settings,
            key=lambda x: self.get_bmas_local_index(x),
        )
    
    def get_hierarchy_level_offset(self, hierarchy_level):
        if hierarchy_level <= -1:
            return -1
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
    
    def get_bmas_by_atom_idname(self, atom_idname: str):
        for bmas in self._atom_settings:
            if bmas.atom_idname == atom_idname:
                return bmas
        return None


class Bead:

    def __init__(
            self,
            bms: BeadMappingSettings,
            id: int,
            idname: str,
            type: int,
            atoms_offset: int,
            bead2atoms: List[List[str]], # Could have multiple configuration files with different atom namings
            weigth_based_on: str,
            resindex: int = 0,
            resnum: int = 0,
            chainid: str = 'A',
    ) -> None:
        self.bms = bms
        self.id = id
        self.idname = idname
        self.resname, self.name = self.idname.split(DataDict.STR_SEPARATOR)
        self.type = type # Used by the NN
        self.resindex = resindex
        self.resnum = resnum
        self.chainid = chainid
        self.weigth_based_on = weigth_based_on

        self._n_found_atoms = 0
        self._is_complete: bool = False
        self._is_newly_created: bool = True
        
        assert self.n_all_atoms == len(bead2atoms[0])
        self._config_ordered_atom_idnames: List[np.ndarray] = [np.array(b2a) for b2a in bead2atoms]
        self._eligible_atom_idnames: List[str] = set(itertools.chain(*bead2atoms))

        self._all_atoms:        List[Atom] = []
        self._all_atom_idnames: List[str]  = []

        self._all_atom_idcs = np.arange(atoms_offset, atoms_offset + self.n_all_atoms)
        self._all_atom_weights = np.zeros((self.n_all_atoms, ), dtype=np.float32)
        self._all_hierarchy_levels = np.zeros((self.n_all_atoms, ), dtype=np.int16)
        self._all_local_index = np.zeros((self.n_all_atoms, ), dtype=np.int16)
        self._all_local_index_anchor = np.zeros((self.n_all_atoms, ), dtype=np.int16)
        
        self._reconstructed_atom_idnames: List[str]      = []
        self._reconstructed_conf_ordered_idcs: List[int] = []

        self._reconstructed_atom_idcs = np.zeros((self.n_reconstructed_atoms,), dtype=np.int16)
        self._reconstructed_atom_weights = np.zeros((self.n_reconstructed_atoms,), dtype=np.float32)
    
    @property
    def n_all_atoms(self):
        return self.bms.bead_all_size
    
    @property
    def n_reconstructed_atoms(self):
        return self.bms.bead_reconstructed_size
    
    @property
    def is_newly_created(self):
        return self._is_newly_created
    
    @property
    def is_complete(self):
        return self._is_complete
    
    @property
    def all_atom_positions(self):
        if self.is_complete and len(self._all_atoms) > 0:
            return np.stack([atom.position for atom in self._all_atoms], axis=0)
        return None
    
    @property
    def _all_atom_residcs(self):
        return np.array([self.resindex] * self.n_all_atoms)
    
    @property
    def _all_atom_resnums(self):
        return np.array([self.resnum] * self.n_all_atoms)
    
    @property
    def _all_atom_chainidcs(self):
        return np.array([self.chainid] * self.n_all_atoms)
    
    def is_missing_atom(self, atom_idname: str):
        assert not self.is_complete, "Can only call this method before the bead is complete."
        return atom_idname in self._eligible_atom_idnames and ~np.isin(atom_idname, self._all_atom_idnames)
    
    def scale_bead_idcs(self, atom_index_offset: int):
        self._all_atom_idcs -= atom_index_offset

    def update(
        self,
        atom_idname: str,
        bmas: BeadMappingAtomSettings,
        atom: Optional[Atom] = None,
        atom_index: Optional[int] = None,
    ):
        if self._is_newly_created and atom is not None:
            self.resindex = atom.resindex
            self.resnum = atom.resnum
            try:
                self.chainid = atom.chainID
            except MDAnalysis.exceptions.NoDataError:
                self.chainid = 'A'
        self._is_newly_created = False
        assert atom_idname in self._eligible_atom_idnames, f"Trying to update bead {self.type} with atom {atom_idname} that does not belong to it."

        conf_ordered_index = None
        invalid_confs = []
        for conf_id, coaidnames in enumerate(self._config_ordered_atom_idnames):
            coai_index = np.argwhere(coaidnames == atom_idname)
            if len(coai_index) > 0:
                if conf_ordered_index is None:
                    conf_ordered_index = coai_index.item()
            else:
                invalid_confs.append(conf_id)
                self._config_ordered_atom_idnames
        if conf_ordered_index is None:
            raise Exception(f"Atom with idname {atom_idname} not found in mapping configuraiton files for bead {self.idname}")
        
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
        elif isinstance(atom, Atom):
            if not bmas.contributes_to_cm:
                pass
            elif self.weigth_based_on == "mass":
                weight = atom.mass
            elif self.weigth_based_on == "same":
                weight = 1.
            else:
                raise Exception(f"{self.weigth_based_on} is not a valid value for 'weigth_based_on'. Use either 'mass' or 'same'")
        weight *= bmas.relative_weight
        self._all_atom_weights[conf_ordered_index] = weight

        if atom_index is not None:
            # If atom index is not None, it means that this atom is shared with another bead,
            # and its idname, position and other properties have already been stored.
            # Thus, on this bead we adjust the index to point to the previously saved one (atom_index)
            # and we scale atom indices that are greater to avoid having gaps in the indexing.
            shared_atom_index = self._all_atom_idcs[conf_ordered_index]
            self._all_atom_idcs[conf_ordered_index] = atom_index
            self._all_atom_idcs[self._all_atom_idcs > shared_atom_index] -= 1
        
        if bmas.has_to_be_reconstructed:
            self._reconstructed_conf_ordered_idcs.append(conf_ordered_index)
            self._reconstructed_atom_idnames.append(atom_idname)

        if self._n_found_atoms == self.n_all_atoms:
            self.complete()
        
        return invalid_confs, self._all_atom_idcs[conf_ordered_index]
    
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
        self._all_atom_idnames = np.array(self._all_atom_idnames)
        self._reconstructed_atom_idnames = np.array(self._reconstructed_atom_idnames)
        self._reconstructed_conf_ordered_idcs = np.array(self._reconstructed_conf_ordered_idcs)

        # ------------------------------------------------------------------------------------------------------------------ #

        config_ordered_all_atom_idnames, all_atom_idnames_sorted_idcs = self.sort_atom_idnames(self._all_atom_idnames)
        all_sorting_filter = all_atom_idnames_sorted_idcs[np.searchsorted(
            self._all_atom_idnames[all_atom_idnames_sorted_idcs], config_ordered_all_atom_idnames
        )]
        
        if len(self._all_atoms) > 0:
            self._all_atoms = np.array(self._all_atoms)[all_sorting_filter]
        self._all_atom_idnames = np.array(self._all_atom_idnames)[all_sorting_filter]

        config_ordered_reconstructed_atom_idnames, reconstructed_atom_idnames_sorted_idcs = self.sort_atom_idnames(self._reconstructed_atom_idnames)
        reconstructed_sorting_filter = reconstructed_atom_idnames_sorted_idcs[np.searchsorted(
            self._reconstructed_atom_idnames[reconstructed_atom_idnames_sorted_idcs],
            config_ordered_reconstructed_atom_idnames
        )]

        self._reconstructed_atom_idnames = self._reconstructed_atom_idnames[reconstructed_sorting_filter]
        self._reconstructed_conf_ordered_idcs = self._reconstructed_conf_ordered_idcs[reconstructed_sorting_filter]
        self._reconstructed_atom_idcs = self._all_atom_idcs[self._reconstructed_conf_ordered_idcs]
        self._reconstructed_atom_weights = self._all_atom_weights[self._reconstructed_conf_ordered_idcs]

        self._is_complete = True

    def sort_atom_idnames(self, atom_idnames):
        coan_max_len = 0
        for coa in self._config_ordered_atom_idnames:
            coan = np.array([x for x in coa if np.isin(x, atom_idnames)])
            if len(coan) > coan_max_len:
                coan_max_len = len(coan)
                config_ordered_atom_idnames = coan
        atom_idnames_sorted_idcs = np.argsort(atom_idnames)
        return config_ordered_atom_idnames, atom_idnames_sorted_idcs


class HierarchicalBead(Bead):

    def __init__(
            self,
            bms: BeadMappingSettings,
            id: int,
            idname: str,
            type: int,
            atoms_offset: int,
            bead2atoms: List[List[str]],
            weigth_based_on: str,
    ) -> None:
        super().__init__(
            bms=bms,
            id=id,
            idname=idname,
            type=type,
            atoms_offset=atoms_offset,
            bead2atoms=bead2atoms,
            weigth_based_on=weigth_based_on,
        )