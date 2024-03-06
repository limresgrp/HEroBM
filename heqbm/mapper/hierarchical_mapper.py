import numpy as np
from typing import Dict, List
from heqbm.mapper import Mapper
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

    def __init__(self, config: Dict) -> None:
        super().__init__(config=config)

    def initialize_extra_pos_impl(self):
        self.bead2atom_relative_vectors_list = []
    
    def update_extra_pos_impl(self, pos):
        frame_bead2atom_relative_vectors = np.zeros((self.num_beads, self.bead_reconstructed_size, 3), dtype=float)
        reconstructed_atom_pos = pos[self.bead2atom_reconstructed_idcs[self._level_idcs_mask.max(axis=0)]]
        anchor_pos = pos[self._level_idcs_anchor_mask.max(axis=0)[self._level_idcs_mask.max(axis=0)]]
        frame_bead2atom_relative_vectors[self.bead2atom_reconstructed_idcs_mask] = reconstructed_atom_pos - anchor_pos
        self.bead2atom_relative_vectors_list.append(frame_bead2atom_relative_vectors)

    def store_extra_pos_impl(self):
        self._bead2atom_relative_vectors = np.stack(self.bead2atom_relative_vectors_list, axis=0)
    
    def compute_extra_map_impl(self):
        self._level_idcs_mask = np.zeros((self.bead_reconstructed_size + 1, self.num_beads, self.bead_reconstructed_size), dtype=bool)
        self._level_idcs_anchor_mask = -np.ones((self.bead_reconstructed_size + 1, self.num_beads, self.bead_reconstructed_size), dtype=int)

        for i, bead in enumerate(self._ordered_beads):
            li = bead._all_local_index
            mask = li >= 0
            li_masked = li[mask]
            li_prev = bead._all_local_index_prev
            li_prev_masked = li_prev[mask]
            anchor_idcs = np.array([np.argwhere(li_masked == x)[0].item() if len(np.argwhere(li_masked == x)) == 1 else -1 for x in li_prev_masked])
            
            for level in range(0, self.bead_reconstructed_size + 1):
                bead_local_filter = np.argwhere(np.array(bead._all_hierarchy_levels[mask]) == level)
                if len(bead_local_filter) > 0:
                    bead_local_filter = bead_local_filter.flatten()
                    self._level_idcs_mask[level, i, bead_local_filter] = True
                    filtered_anchor_idcs = anchor_idcs[bead_local_filter]
                    filtered_anchor_idcs[filtered_anchor_idcs == -1] = bead_local_filter[filtered_anchor_idcs == -1]
                    self._level_idcs_anchor_mask[level, i, bead_local_filter] = bead._reconstructed_atom_idcs[filtered_anchor_idcs]