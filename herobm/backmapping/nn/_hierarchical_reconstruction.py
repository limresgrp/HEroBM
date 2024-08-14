import torch
from torch_runstats.scatter import scatter
from geqtrain.nn import GraphModuleMixin
from geqtrain.data import AtomicDataDict
from geqtrain.nn.mace.irreps_tools import reshape_irreps
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from herobm.utils import DataDict

@compile_mode("script")
class HierarchicalReconstructionModule(GraphModuleMixin, torch.nn.Module):

    def __init__(
        self,
        func: GraphModuleMixin,
        num_types: int,
        in_field: str = AtomicDataDict.NODE_OUTPUT_KEY,
        out_field: str = DataDict.ATOM_POSITION,
        out_field_irreps: Irreps = Irreps("1x1o"),
        normalize_b2a_rel_vec: bool = True,
    ):
        super().__init__()
        self.func = func
        self.in_field = in_field
        self.out_field = out_field
        self.normalize_b2a_rel_vec = normalize_b2a_rel_vec

        irreps_out = func.irreps_out
        irreps_out.update({
            self.out_field: out_field_irreps,
        })

        irreps_in = func.irreps_in

        hierarchy_irreps = Irreps(f"{func.irreps_out[in_field].num_irreps}x0e")
        irreps_in.update({
            "atom_pos": Irreps("1x1o"),
            "bead2atom_reconstructed_idcs": hierarchy_irreps,
            "bead2atom_reconstructed_weights": hierarchy_irreps,
            "lvl_idcs_mask": hierarchy_irreps,
            "lvl_idcs_anchor_mask": hierarchy_irreps,
        })

        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={
                in_field: func.irreps_out[in_field],
                "atom_pos": Irreps("1x1o"),
            },
            irreps_out=irreps_out,
        )

        self.reshape = reshape_irreps(func.irreps_out[in_field])

        if self.normalize_b2a_rel_vec:
            self.atom_type2bond_lengths = torch.nn.Parameter(torch.ones((num_types, self.irreps_out[in_field].num_irreps, 1), dtype=torch.float32))

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if AtomicDataDict.NOISE in data:
            data[AtomicDataDict.POSITIONS_KEY] += data[AtomicDataDict.NOISE]
        data = self.func(data)
        if AtomicDataDict.NOISE in data:
            data[AtomicDataDict.POSITIONS_KEY] -= data[AtomicDataDict.NOISE]

        # bead2atom_relative_vectors = data[self.in_field]
        bead2atom_relative_vectors = self.reshape(data[self.in_field])
        if torch.any(torch.isnan(bead2atom_relative_vectors)):
            raise Exception("NaN")
        bead_pos = data[AtomicDataDict.POSITIONS_KEY]
        atom_pos_slices        = data.get("atom_pos_slices", torch.tensor([0, data.get("bead2atom_reconstructed_idcs").max()+1], dtype=int, device=bead_pos.device))
        idcs_mask              = data.get("bead2atom_reconstructed_idcs")
        idcs_mask_slices       = data.get("bead2atom_reconstructed_idcs_slices", torch.tensor([0, len(bead_pos)], dtype=int, device=bead_pos.device))
        weights                = data.get("bead2atom_reconstructed_weights")
        level_idcs_mask        = data.get("lvl_idcs_mask")
        level_idcs_anchor_mask = data.get("lvl_idcs_anchor_mask")
        center_atoms = torch.unique(data[AtomicDataDict.EDGE_INDEX_KEY][0])

        if self.normalize_b2a_rel_vec:
            # with torch.no_grad():
            # norm_factor = bead2atom_relative_vectors.max().item()
            # bead2atom_relative_vectors /= norm_factor
            norm = torch.norm(bead2atom_relative_vectors, dim=-1, keepdim=True)
            bead2atom_relative_vectors = bead2atom_relative_vectors / (norm + 1.e-3)
            bead2atom_relative_vectors = bead2atom_relative_vectors * self.atom_type2bond_lengths[data.get(AtomicDataDict.NODE_TYPE_KEY).squeeze(-1)]

        for (b2a_idcs_from, b2a_idcs_to), atom_pos_from in zip(
            zip(idcs_mask_slices[:-1], idcs_mask_slices[1:]),
            atom_pos_slices[:-1],
        ):
            batch_center_atoms = center_atoms[(center_atoms>=b2a_idcs_from) & (center_atoms<b2a_idcs_to)]

            b2a_idcs = idcs_mask[batch_center_atoms]
            batch_weights = weights[batch_center_atoms]
            reconstructed_atom_pos = torch.empty(len(batch_center_atoms), atom_pos_slices[-1], 3, dtype=torch.float32, device=bead2atom_relative_vectors.device)
            reconstructed_atom_pos[:] = torch.nan

            b2a_idcs_row, b2a_idcs_col = torch.where(b2a_idcs>=0)
            b2a_idcs_col += atom_pos_from
            reconstructed_atom_pos[b2a_idcs_row, b2a_idcs[b2a_idcs_row, b2a_idcs_col]] = bead_pos[center_atoms][b2a_idcs_row]

            batch_level_idcs_mask = level_idcs_mask[batch_center_atoms]
            batch_level_idcs_anchor_mask = level_idcs_anchor_mask[batch_center_atoms]
            for level in range(batch_level_idcs_mask.size(1)):
                if level == 0:
                    continue
                level_idcs_mask_elem = batch_level_idcs_mask[:, level]
                level_anchor_idcs_mask_elem = batch_level_idcs_anchor_mask[:, level]
                mask_row, mask_col = torch.where(level_idcs_mask_elem)
                if len(mask_row) > 0:
                    # mask_col += atom_pos_from ?
                    updated_pos = reconstructed_atom_pos[mask_row, level_anchor_idcs_mask_elem[mask_row, mask_col]]
                    updated_pos = updated_pos + bead2atom_relative_vectors[center_atoms][mask_row, mask_col]
                    reconstructed_atom_pos[mask_row, b2a_idcs[mask_row, mask_col]] = updated_pos
            
            # Re-center predicted atoms' center of mass to the actual bead position
            # !!!!!!!!!!!!!!!!!!!!!!!!!!
            # predicted_atoms_cm = scatter(
            #     reconstructed_atom_pos[b2a_idcs_row, b2a_idcs[b2a_idcs_row, b2a_idcs_col]] * batch_weights[b2a_idcs_row, b2a_idcs_col][:, None],
            #     b2a_idcs_row,
            #     dim=0,
            # )
            # atom_shifts = predicted_atoms_cm - bead_pos[center_atoms]
            # reconstructed_atom_pos[b2a_idcs_row, b2a_idcs[b2a_idcs_row, b2a_idcs_col]] -= atom_shifts[b2a_idcs_row]
        
        data[self.out_field] = torch.nanmean(reconstructed_atom_pos, dim=0)
        return data