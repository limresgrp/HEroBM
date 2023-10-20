import numpy as np
from typing import List, Optional
from heqbm.mapper import Mapper
from heqbm.mapper.bead import Bead
from heqbm.utils import DataDict

class HierarchicalMapper(Mapper):

    bead2atom_relative_vectors_list: List[np.ndarray]

    _level_idcs_mask: np.ndarray = None
    _level_idcs_anchor_mask: np.ndarray = None
    _bead2atom_relative_vectors: np.ndarray = None

    @property
    def level_idcs_mask(self):
        return self._level_idcs_mask
    
    @property
    def level_idcs_anchor_mask(self):
        return self._level_idcs_anchor_mask
    
    @property
    def bead2atom_relative_vectors(self):
        return self._bead2atom_relative_vectors

    @property
    def dataset(self):
        dataset = super().dataset
        dataset.update({k: v for k, v in {
            DataDict.LEVEL_IDCS_MASK: self.level_idcs_mask,
            DataDict.LEVEL_IDCS_ANCHOR_MASK: self.level_idcs_anchor_mask,
            DataDict.BEAD2ATOM_RELATIVE_VECTORS: self.bead2atom_relative_vectors,
        }.items() if v is not None})
        return dataset

    def __init__(self, root: Optional[str] = None) -> None:
        super().__init__(root=root)
    
    def _clear_extra_mappings(self):
        pass
    
    def _initialize_conf_extra_mappings(self):
        pass

    def _store_extra_mappings(self):
        pass
    
    def initialize_extra_map_impl(self):
        pass
    
    def update_extra_map_impl(self, atom_name: str, bead: Bead, _id: int):
        pass
    
    def store_extra_map_impl(self):
        pass

    def initialize_extra_pos_impl(self):
        self.bead2atom_relative_vectors_list = []
    
    def update_extra_pos_impl(self, pos, bead_pos):
        frame_bead2atom_relative_vectors = np.zeros((self.n_beads_instance, self._max_bead_atoms, 3), dtype=float)
        anchor_pos = pos[self._level_idcs_anchor_mask.max(axis=0)]
        anchor_pos[self._level_idcs_anchor_mask[2:].max(axis=0) == -1] = np.repeat(bead_pos, np.sum(self._level_idcs_anchor_mask[2:].max(axis=0) == -1, axis=-1), axis=0)
        frame_bead2atom_relative_vectors[self.bead2atom_idcs_mask_instance] = pos[self.bead2atom_idcs_instance[self.bead2atom_idcs_mask_instance]] - anchor_pos[self.bead2atom_idcs_mask_instance]
        self.bead2atom_relative_vectors_list.append(frame_bead2atom_relative_vectors)

    def store_extra_pos_impl(self):
        self._bead2atom_relative_vectors = np.stack(self.bead2atom_relative_vectors_list, axis=0)
    
    def compute_extra_map_impl(self):
        self._level_idcs_mask = np.zeros((self._max_bead_atoms + 1, self.n_beads_instance, self._max_bead_atoms), dtype=bool)
        self._level_idcs_anchor_mask = -np.ones((self._max_bead_atoms + 1, self.n_beads_instance, self._max_bead_atoms), dtype=int)

        for i, bead in enumerate(self._ordered_beads):
            li = np.array(bead._local_index)
            li_prev = np.array(bead._local_index_prev)
            anchor_idcs = np.array([np.argwhere(li == x)[0].item() if len(np.argwhere(li == x)) >= 1 else 0 for x in li_prev])
            
            for level in range(0, self._max_bead_atoms + 1):
                bead_local_filter = np.argwhere(np.array(bead._hierarchy_levels) == level)
                if len(bead_local_filter) > 0:
                    bead_local_filter = bead_local_filter.flatten()
                    self._level_idcs_mask[level, i, bead_local_filter] = True
                    self._level_idcs_anchor_mask[level, i, bead_local_filter] = self._bead2atom_idcs_instance[i, anchor_idcs[bead_local_filter]]
    
    def initialize_extra_map_impl_cg(self):
        pass

    def update_extra_map_impl_cg(
        self,
        bead_atom_names: np.ndarray,
        bead_name: str,
        mapping_n: int,
        atom_index_offset: int
    ):
        pass

    def store_extra_map_impl_cg(self):
        pass