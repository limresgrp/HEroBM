import torch
from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from heqbm.backmapping.allegro._keys import (
    EQUIVARIANT_ATOM_FEATURES,
    ATOM_POSITIONS,
)
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

@compile_mode("script")
class HierarchicalReconstrucitonModule(GraphModuleMixin, torch.nn.Module):

    def __init__(
        self,
        func: GraphModuleMixin,
        in_field: str = EQUIVARIANT_ATOM_FEATURES,
        out_field: str = ATOM_POSITIONS,
        out_field_irreps: Irreps = Irreps("1x1o"),
    ):
        super().__init__()
        self.func = func
        self.in_field = in_field
        self.out_field = out_field

        irreps_out = func.irreps_out
        irreps_out.update({
            self.out_field: out_field_irreps,
        })

        # check and init irreps
        self._init_irreps(
            irreps_in=func.irreps_in,
            my_irreps_in={in_field: func.irreps_out[in_field]},
            irreps_out=irreps_out,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = self.func(data)

        bead2atom_relative_vectors = data[self.in_field]
        bead_pos = data[AtomicDataDict.POSITIONS_KEY]
        atom_pos_slices = data["atom_pos_slices"]
        idcs_mask = data["bead2atom_idcs"]
        idcs_mask_slices = data["bead2atom_idcs_slices"]
        level_idcs_mask = data["lvl_idcs_mask"]
        level_idcs_mask_slices = data["lvl_idcs_mask_slices"]
        level_idcs_anchor_mask = data["lvl_idcs_anchor_mask"]

        orig_center_atoms = data[AtomicDataDict.ORIG_EDGE_INDEX_KEY][0].unique()
        center_atoms = data[AtomicDataDict.EDGE_INDEX_KEY][0].unique()

        per_bead_reconstructed_atom_pos = []
        for (b2a_idcs_from, b2a_idcs_to), (idcs_mask_from, idcs_mask_to), atom_pos_from in zip(
            zip(idcs_mask_slices[:-1], idcs_mask_slices[1:]),
            zip(level_idcs_mask_slices[:-1], level_idcs_mask_slices[1:]),
            atom_pos_slices[:-1],
        ):
            batch_orig_center_atoms = orig_center_atoms[(orig_center_atoms>=b2a_idcs_from) & (orig_center_atoms<b2a_idcs_to)]
            batch_center_atoms = center_atoms[(orig_center_atoms>=b2a_idcs_from) & (orig_center_atoms<b2a_idcs_to)]
            for h, h_orig, b2a_idcs in zip(batch_center_atoms, batch_orig_center_atoms, idcs_mask[batch_orig_center_atoms]):
                reconstructed_atom_pos = torch.empty((atom_pos_slices[-1], 3), dtype=torch.float32, device=bead2atom_relative_vectors.device)
                reconstructed_atom_pos[:] = torch.nan
                reconstructed_atom_pos[b2a_idcs[b2a_idcs>=0] + atom_pos_from] = bead_pos[h, None, ...]

                for level, (level_idcs_mask_elem, level_anchor_idcs_mask_elem) in enumerate(
                    zip(level_idcs_mask[idcs_mask_from:idcs_mask_to, h_orig-b2a_idcs_from], level_idcs_anchor_mask[idcs_mask_from:idcs_mask_to, h_orig-b2a_idcs_from])
                ):
                    updated_pos = reconstructed_atom_pos[level_anchor_idcs_mask_elem[level_idcs_mask_elem] + atom_pos_from]
                    if level > 0:
                        updated_pos = updated_pos + bead2atom_relative_vectors[h, level_idcs_mask_elem]
                        reconstructed_atom_pos[idcs_mask[h_orig, level_idcs_mask_elem] + atom_pos_from] = updated_pos
                per_bead_reconstructed_atom_pos.append(reconstructed_atom_pos)
        per_bead_reconstructed_atom_pos = torch.stack(per_bead_reconstructed_atom_pos, dim=0)
        
        data[self.out_field] = torch.nanmean(per_bead_reconstructed_atom_pos, dim=0)
        return data