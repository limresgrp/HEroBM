import numpy as np
from typing import Dict, List
from cgmap.mapping import Mapper
from heqbm.utils import DataDict
import pandas as pd

class HierarchicalMapper(Mapper):

    bead2atom_relative_vectors_list: List[np.ndarray]

    _level_idcs_mask: np.ndarray = None
    _level_idcs_anchor_mask: np.ndarray = None
    _bead2atom_relative_vectors: np.ndarray = None

    _bead2atom_reconstructed_idcs: np.ndarray = None
    _bead2atom_reconstructed_idcs_orig: np.ndarray = None

    _bond_idcs: np.ndarray = None
    _angle_idcs: np.ndarray = None
    _torsion_idcs: np.ndarray = None

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
    def bead2atom_reconstructed_idcs_mask(self):
        if self._bead2atom_reconstructed_idcs is None:
            return None
        return self._bead2atom_reconstructed_idcs == -1

    @property
    def bead2atom_reconstructed_idcs(self):
        if self._bead2atom_reconstructed_idcs is None:
            return None
        return np.ma.masked_array(self._bead2atom_reconstructed_idcs, mask=self.bead2atom_reconstructed_idcs_mask)
    
    @property
    def bead2atom_reconstructed_weights(self):
        if self._bead2atom_reconstructed_weights is None:
            return None
        return np.ma.masked_array(self._bead2atom_reconstructed_weights, mask=self.bead2atom_reconstructed_idcs_mask)

    @property
    def dataset(self):
        dataset = super().dataset
        dataset.update({k: v for k, v in {
            DataDict.LEVEL_IDCS_MASK: self.level_idcs_mask,
            DataDict.LEVEL_IDCS_ANCHOR_MASK: self.level_idcs_anchor_mask,
            DataDict.BEAD2ATOM_RELATIVE_VECTORS: self.bead2atom_relative_vectors,
            DataDict.BOND_IDCS:  self._bond_idcs,
            DataDict.ANGLE_IDCS: self._angle_idcs,
            DataDict.TORSION_IDCS: self._torsion_idcs,
            DataDict.BEAD2ATOM_RECONSTRUCTED_IDCS: self.bead2atom_reconstructed_idcs,
            DataDict.BEAD2ATOM_RECONSTRUCTED_WEIGHTS: self.bead2atom_reconstructed_weights,
        }.items() if v is not None})
        return dataset

    def __init__(self, args_dict) -> None:
        super().__init__(args_dict=args_dict)

    def compute_bead2atom_feats(self):
        self._bead2atom_idcs = -np.ones((self.num_beads, self.bead_all_size), dtype=int)
        self._bead2atom_weights = np.zeros((self.num_beads, self.bead_all_size), dtype=float)

        self._bead2atom_reconstructed_idcs = -np.ones((self.num_beads, self.bead_reconstructed_size), dtype=int)
        self._bead2atom_reconstructed_weights = np.zeros((self.num_beads, self.bead_reconstructed_size), dtype=np.float32)

        self._level_idcs_mask = np.zeros((self.num_beads, self.bead_reconstructed_size + 1, self.bead_reconstructed_size), dtype=bool)
        self._level_idcs_anchor_mask = -np.ones((self.num_beads, self.bead_reconstructed_size + 1, self.bead_reconstructed_size), dtype=int)

        for i, bead in enumerate(self._ordered_beads):
            self._bead2atom_idcs[i, :bead.n_all_atoms] = bead._all_atom_idcs
            self._bead2atom_weights[i, :bead.n_all_atoms] = bead.all_atom_weights / bead.all_atom_weights.sum()

            self._bead2atom_reconstructed_idcs[i, :len(bead._reconstructed_atom_idcs)] = bead._reconstructed_atom_idcs
            self._bead2atom_reconstructed_weights[i, :len(bead._reconstructed_atom_weights)] = bead._reconstructed_atom_weights / bead._reconstructed_atom_weights.sum()

            li = bead._all_local_index
            mask = li >= 0
            li_masked = li[mask]
            li_prev = bead._all_local_index_anchor
            li_prev_masked = li_prev[mask]
            anchor_idcs = np.array([
                np.argwhere(li_masked == x)[0].item()
                if len(np.argwhere(li_masked == x)) == 1
                else -1
                for x in li_prev_masked
            ])
            
            for level in range(0, self.bead_reconstructed_size + 1):
                bead_local_filter = np.argwhere(bead._all_hierarchy_levels[mask] == level)
                if len(bead_local_filter) > 0:
                    bead_local_filter = bead_local_filter.flatten()
                    self._level_idcs_mask[i, level, bead_local_filter] = True
                    filtered_anchor_idcs = anchor_idcs[bead_local_filter]
                    filtered_anchor_idcs[filtered_anchor_idcs == -1] = bead_local_filter[filtered_anchor_idcs == -1]
                    self._level_idcs_anchor_mask[i, level, bead_local_filter] = bead._reconstructed_atom_idcs[filtered_anchor_idcs]

    def initialize_extra_pos_impl(self):
        self.bead2atom_relative_vectors_list = []
    
    def update_extra_pos_impl(self, pos):
        frame_bead2atom_relative_vectors = np.zeros((self.num_beads, self.bead_reconstructed_size, 3), dtype=float)
        reconstructed_atom_pos = pos[self.bead2atom_reconstructed_idcs.data[~self.bead2atom_reconstructed_idcs.mask]]
        anchor_pos = pos[self._level_idcs_anchor_mask.max(axis=1)[self._level_idcs_mask.max(axis=1)]]
        frame_bead2atom_relative_vectors[~self.bead2atom_reconstructed_idcs_mask] = reconstructed_atom_pos - anchor_pos
        self.bead2atom_relative_vectors_list.append(frame_bead2atom_relative_vectors)

    def store_extra_pos_impl(self):
        self._bead2atom_relative_vectors = np.stack(self.bead2atom_relative_vectors_list, axis=0)
    
    def compute_invariants(self):
        bond_min_length, bond_max_length = 1., 2. # Angstrom
        atoms_to_reconstruct_idcs = np.unique(self._bead2atom_reconstructed_idcs)

        # Bonds
        x = self._atom_positions[0]
        y = x - x[:, None]
        y = np.linalg.norm(y, axis=-1)
        z = (y > bond_min_length) * (y < bond_max_length)
        z[np.tril_indices(len(z), k=-1)] = False
        self._bond_idcs = np.stack(np.nonzero(z)).T
        self._bond_idcs = self._bond_idcs[np.all(np.isin(self._bond_idcs, atoms_to_reconstruct_idcs), axis=1)]
        if len(self._bond_idcs) == 0:
            bond_idcs = []
            last_bb_atom_idx = []
            last_bb_atom_name = []
            for idx, an in enumerate(self._atom_names):
                if an in ['N', 'CA', 'C', 'O']:
                    if len(last_bb_atom_idx)>0:
                        if an != 'N':
                            bond_idcs.append([last_bb_atom_idx[-1], idx])
                        elif len(last_bb_atom_idx)>1 and  last_bb_atom_name[-2] == 'C':
                            bond_idcs.append([last_bb_atom_idx[-2], idx])
                    last_bb_atom_idx.append(idx)
                    last_bb_atom_name.append(an)
            if len(bond_idcs) > 0:
                self._bond_idcs = np.stack(bond_idcs)

        
        # Angles
        df1A = pd.DataFrame(self._bond_idcs, columns=['a1', 'a2'])
        df1B = pd.DataFrame(self._bond_idcs, columns=['a2', 'a1'])
        df2  = pd.DataFrame(self._bond_idcs, columns=['a2', 'a3'])
        df3A = df1A.merge(df2, how='outer')
        df3A = df3A.dropna().astype(int)
        df3B = df1B.merge(df2, how='outer')
        df3B = df3B.dropna().astype(int)
        cols = df3B.columns.to_list()
        cols[:2] = cols[1::-1]
        df3B = df3B[cols]
        df3B = df3B[df3B['a1'] != df3B['a3']]
        self._angle_idcs = np.concatenate([df3A.values, df3B.values])
        self._angle_idcs = self._angle_idcs[np.all(np.isin(self._angle_idcs, atoms_to_reconstruct_idcs), axis=1)]

        # Torsions
        df1 = pd.DataFrame(self._angle_idcs, columns=['a1', 'a2', 'a3'])
        df2 = pd.DataFrame(self._angle_idcs, columns=['a2', 'a3', 'a4'])
        df3 = df1.merge(df2, how='outer')
        df3 = df3.dropna().astype(int)
        self._torsion_idcs = df3.values
        self._torsion_idcs = self._torsion_idcs[np.all(np.isin(self._torsion_idcs, atoms_to_reconstruct_idcs), axis=1)]

    def compute_extra_map_impl(self):
        self.compute_invariants()
