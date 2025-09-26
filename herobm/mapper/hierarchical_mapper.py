import numpy as np
from typing import List
from cgmap.mapping import Mapper
from herobm.utils import DataDict
from herobm.mapper.bead import HEroBMBeadMappingAtomSettings, HEroBMBeadMappingSettings, HEroBMBead
import pandas as pd


def list2maskedarray(x, dim=-1):
    # Find the maximum size along the selected dim
    max_dim = max(e.shape[dim] for e in x)
    num_graphs = len(x)
    dims = x[0].shape

    # Prepare output shape
    out_shape = list(dims)
    out_shape[dim] = max_dim
    out_shape = [num_graphs] + out_shape

    # Initialize masked array with -1
    x_padded = np.full(out_shape, -1, dtype=x[0].dtype)
    mask = np.ones(out_shape, dtype=bool)

    # Fill in the values and mask
    for i, e in enumerate(x):
        # Prepare slices for assignment
        slices = [i]
        for d, s in enumerate(e.shape):
            if d == dim or len(dims) + dim == d:
                slices.append(slice(0, s))
            else:
                slices.append(slice(None))
        x_padded[tuple(slices)] = e
        mask[tuple(slices)] = False

    return np.ma.masked_array(x_padded, mask=mask)

class HierarchicalMapper(Mapper):

    bead2atom_relative_vectors_list: List[np.ndarray]

    _level_idcs_mask: np.ndarray = None
    _level_idcs_anchor_mask: np.ndarray = None
    _bead2atom_relative_vectors: np.ndarray = None

    _bead2atom_reconstructed_idcs: np.ndarray = None
    _bead2atom_reconstructed_idcs_orig: np.ndarray = None

    _bead2atom_reconstructed_weights: np.ndarray = None

    _bond_idcs: np.ndarray = None
    _angle_idcs: np.ndarray = None
    _torsion_idcs: np.ndarray = None

    _edge_index: np.ndarray = None
    _edge_cell_shift: np.ndarray = None
    _cutoff: float = None

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
    
    def edge_index(self, cutoff):
        if self._edge_index is not None and self._cutoff == cutoff:
            return self._edge_index
        
        from geqtrain.data.AtomicData import neighbor_list
        from torch import from_numpy
        edge_index = []
        edge_cell_shift = []
        for i, pos in enumerate(self._bead_positions):
            cell = self._cell[i] if self._cell else None
            _edge_index, _edge_cell_shift, _cell = neighbor_list(
            pos=from_numpy(pos),
            r_max=cutoff,
            cell=cell,
            pbc=self._pbc,
            )
            _edge_index = _edge_index.numpy()
            edge_index.append(_edge_index)
            if _edge_cell_shift:
                _edge_cell_shift = _edge_cell_shift.numpy()
                edge_cell_shift.append(_edge_cell_shift)
        
        self._cutoff == cutoff
        self._edge_index = list2maskedarray(edge_index)
        self._edge_cell_shift = list2maskedarray(edge_cell_shift, dim=-2) if edge_cell_shift else None
        return self._edge_index
    
    def edge_cell_shift(self, cutoff):
        if self._edge_cell_shift is not None and self._cutoff == cutoff:
            return self._edge_cell_shift
        self.edge_index(cutoff)
        return self._edge_cell_shift
    
    @property
    def bead_is_same(self):
        assert self._edge_index is not None
        _bead_is_same = []
        bead_resnumbers = self._bead_resnums
        bead_chainids = self._bead_segids
        for _edge_index in self._edge_index:
            resnumbers = bead_resnumbers[_edge_index]
            chainids = bead_chainids[_edge_index]
            is_same = ((resnumbers[1] - resnumbers[0]) == 0) * (chainids[1] == chainids[0]) * (_edge_index[0] > -1)
            _bead_is_same.append(is_same)
        return np.stack(_bead_is_same, axis=0)

    @property
    def bead_is_prev(self):
        assert self._edge_index is not None
        _bead_is_prev = []
        bead_resnumbers = self._bead_resnums
        bead_chainids = self._bead_segids
        for _edge_index in self._edge_index:
            resnumbers = bead_resnumbers[_edge_index]
            chainids = bead_chainids[_edge_index]
            is_prev = ((resnumbers[1] - resnumbers[0]) == -1) * (chainids[1] == chainids[0]) * (_edge_index[0] > -1)
            _bead_is_prev.append(is_prev)
        return np.stack(_bead_is_prev, axis=0)
    
    @property
    def bead_is_next(self):
        assert self._edge_index is not None
        _bead_is_next = []
        bead_resnumbers = self._bead_resnums
        bead_chainids = self._bead_segids
        for _edge_index in self._edge_index:
            resnumbers = bead_resnumbers[_edge_index]
            chainids = bead_chainids[_edge_index]
            is_prev = ((resnumbers[1] - resnumbers[0]) == 1) * (chainids[1] == chainids[0]) * (_edge_index[0] > -1)
            _bead_is_next.append(is_prev)
        return np.stack(_bead_is_next, axis=0)

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
        cutoff = self.config.get("cutoff", None)
        if cutoff is not None:
            dataset.update({k: v for k, v in {
                DataDict.EDGE_INDEX: self.edge_index(cutoff),
                DataDict.EDGE_CELL_SHIFT: self.edge_cell_shift(cutoff),
                DataDict.BEAD_IS_SAME: self.bead_is_same,
                DataDict.BEAD_IS_PREV: self.bead_is_prev,
                DataDict.BEAD_IS_NEXT: self.bead_is_next,
            }.items() if v is not None})
        return dataset

    def __init__(self, args_dict) -> None:
        super().__init__(args_dict=args_dict)
        self.noinvariants = args_dict.get('noinvariants', False)
    
    def _load_mappings(self, bms_class=HEroBMBeadMappingSettings, bmas_class=HEroBMBeadMappingAtomSettings):
        return super(HierarchicalMapper, self)._load_mappings(bms_class=bms_class, bmas_class=bmas_class)
    
    def _create_bead(self, bead_idname: str, bead_class = HEroBMBead):
        return super(HierarchicalMapper, self)._create_bead(bead_idname=bead_idname, bead_class=bead_class)

    def _compute_bead2atom_feats(self):
        self._bead2atom_idcs = -np.ones((self.num_beads, self.bead_all_size), dtype=np.int32)
        self._bead2atom_weights = np.zeros((self.num_beads, self.bead_all_size), dtype=np.float32)

        self._bead2atom_reconstructed_idcs = -np.ones((self.num_beads, self.bead_reconstructed_size), dtype=np.int32)
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

    def _initialize_extra_pos_impl(self):
        self.bead2atom_relative_vectors_list = []
    
    def _update_extra_pos_impl(self, pos):
        frame_bead2atom_relative_vectors = np.zeros((self.num_beads, self.bead_reconstructed_size, 3), dtype=float)
        reconstructed_atom_pos = pos[self.bead2atom_reconstructed_idcs.data[~self.bead2atom_reconstructed_idcs.mask]]
        anchor_pos = pos[self._level_idcs_anchor_mask.max(axis=1)[self._level_idcs_mask.max(axis=1)]]
        frame_bead2atom_relative_vectors[~self.bead2atom_reconstructed_idcs_mask] = reconstructed_atom_pos - anchor_pos
        self.bead2atom_relative_vectors_list.append(frame_bead2atom_relative_vectors)

    def _store_extra_pos_impl(self):
        self._bead2atom_relative_vectors = np.stack(self.bead2atom_relative_vectors_list, axis=0)
    
    def _compute_invariants(self):
        bond_min_length, bond_max_length = 1., 2. # Angstrom
        atoms_to_reconstruct_idcs = np.unique(self._bead2atom_reconstructed_idcs)

        # Bonds
        self._bond_idcs = None
        x = self._atom_positions[0]
        if np.any(x):
            # Compute which atoms are bonded, looking maximum at 20 positions away
            # to avoid quadratic scaling with number of atoms
            bond_idcs = []
            horizon = 20
            for start in range(0, len(x), horizon):
                source_end = min(len(x), start + horizon)
                target_end = min(len(x), start + 2*horizon)
                y = x[start:target_end] - x[start:source_end, None]
                y = np.linalg.norm(y, axis=-1)
                z = (y > bond_min_length) * (y < bond_max_length)
                z[np.tril_indices(len(z), k=-1)] = False
                source_bond_idcs, target_bond_idcs = np.nonzero(z)
                source_bond_idcs += start
                target_bond_idcs += start
                bond_idcs.append(np.stack([source_bond_idcs, target_bond_idcs]))

            self._bond_idcs = np.concatenate(bond_idcs, axis=1).T
            self._bond_idcs = self._bond_idcs[np.all(np.isin(self._bond_idcs, atoms_to_reconstruct_idcs), axis=1)]
        if self._bond_idcs is None or len(self._bond_idcs) == 0:
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
        df3B[['a1', 'a3']] = pd.DataFrame(np.sort(df3B[['a1', 'a3']], axis=1), index=df3B.index)
        df3B = df3B.drop_duplicates()
        self._angle_idcs = np.concatenate([df3A.values, df3B.values])
        self._angle_idcs = self._angle_idcs[np.all(np.isin(self._angle_idcs, atoms_to_reconstruct_idcs), axis=1)]

        # Torsions
        df1 = pd.DataFrame(self._angle_idcs, columns=['a1', 'a2', 'a3'])
        df2 = pd.DataFrame(self._angle_idcs, columns=['a2', 'a3', 'a4'])
        df3 = df1.merge(df2, how='outer')
        df3 = df3.dropna().astype(int)
        self._torsion_idcs = df3.values
        self._torsion_idcs = self._torsion_idcs[np.all(np.isin(self._torsion_idcs, atoms_to_reconstruct_idcs), axis=1)]

    def _compute_extra_map_impl(self):
        if not self.noinvariants:
            self._compute_invariants()
