import torch
from geqtrain.nn import GraphModuleMixin
from geqtrain.data import AtomicDataDict
from geqtrain.nn.mace.irreps_tools import reshape_irreps
from e3nn.o3 import Irreps
from torch_runstats.scatter import scatter
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
            self.atom_type2bond_lengths = torch.nn.Parameter(torch.ones((num_types + 1, self.irreps_out[in_field].num_irreps, 1), dtype=torch.float32))

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if AtomicDataDict.NOISE_KEY in data:
            data[AtomicDataDict.POSITIONS_KEY] += data[AtomicDataDict.NOISE_KEY]
        data = self.func(data)
        if AtomicDataDict.NOISE_KEY in data:
            data[AtomicDataDict.POSITIONS_KEY] -= data[AtomicDataDict.NOISE_KEY]

        bead2atom_relative_vectors = self.reshape(data[self.in_field])
        if torch.any(torch.isnan(bead2atom_relative_vectors)):
            raise Exception("NaN")
        
        bead_pos = data[AtomicDataDict.POSITIONS_KEY]
        bead_pos_slices        = data.get("pos_slices")
        assert bead_pos_slices is not None, "'pos_slices' must be in data"
        atom_pos_slices        = data.get("atom_pos_slices")
        assert atom_pos_slices is not None, "'atom_pos_slices' must be in data"
        idcs_mask              = data.get("bead2atom_reconstructed_idcs")
        assert idcs_mask is not None, "'bead2atom_reconstructed_idcs' must be in data"
        idcs_mask_slices       = data.get("bead2atom_reconstructed_idcs_slices")
        assert idcs_mask_slices is not None, "'bead2atom_reconstructed_idcs_slices' must be in data"
        weights                = data.get("bead2atom_reconstructed_weights")
        assert weights is not None
        level_idcs_mask        = data.get("lvl_idcs_mask")
        assert level_idcs_mask is not None
        level_idcs_anchor_mask = data.get("lvl_idcs_anchor_mask")
        assert level_idcs_anchor_mask is not None
        bead_types             = data.get(AtomicDataDict.NODE_TYPE_KEY)
        assert bead_types is not None
        bead_types = bead_types.squeeze(-1)
        center_atoms           = torch.unique(data[AtomicDataDict.EDGE_INDEX_KEY][0])
        n_reconstructed_beads  = len(center_atoms)

        # Initialize predicted atom_pos
        reconstructed_atom_pos = torch.empty(n_reconstructed_beads, atom_pos_slices[-1], 3, dtype=torch.float32, device=bead2atom_relative_vectors.device)
        reconstructed_atom_pos[:] = torch.nan

        if self.normalize_b2a_rel_vec:
            # norm_factor = bead2atom_relative_vectors.max().item()
            norm = torch.norm(bead2atom_relative_vectors, dim=-1, keepdim=True)
            bead2atom_relative_vectors = bead2atom_relative_vectors / (norm + 1.e-5)
            bead2atom_relative_vectors = bead2atom_relative_vectors * self.atom_type2bond_lengths[bead_types]

        for b2a_idcs_from, b2a_idcs_to, bead_pos_from, atom_pos_from in zip(
            idcs_mask_slices[:-1], idcs_mask_slices[1:],
            bead_pos_slices[:-1],  atom_pos_slices[:-1],
        ):
            batch_center_atoms = center_atoms[(center_atoms>=b2a_idcs_from) & (center_atoms<b2a_idcs_to)]

            b2a_idcs = idcs_mask[batch_center_atoms]
            batch_weights = weights[batch_center_atoms]

            b2a_idcs_row, b2a_idcs_col = torch.where(b2a_idcs>=0)
            reconstructed_atom_pos[b2a_idcs_row + bead_pos_from, b2a_idcs[b2a_idcs_row, b2a_idcs_col] + atom_pos_from] = bead_pos[center_atoms][b2a_idcs_row + bead_pos_from]

            batch_level_idcs_mask = level_idcs_mask[batch_center_atoms]
            batch_level_idcs_anchor_mask = level_idcs_anchor_mask[batch_center_atoms]
            for level in range(1, batch_level_idcs_mask.size(1)):
                level_idcs_mask_elem = batch_level_idcs_mask[:, level]
                level_anchor_idcs_mask_elem = batch_level_idcs_anchor_mask[:, level]
                mask_row, mask_col = torch.where(level_idcs_mask_elem)
                if len(mask_row) > 0:
                    # Pick position of anchor points
                    updated_pos = reconstructed_atom_pos[mask_row + bead_pos_from, level_anchor_idcs_mask_elem[mask_row, mask_col] + atom_pos_from]
                    # Add predicted vector to positions
                    updated_pos = updated_pos + bead2atom_relative_vectors[center_atoms][mask_row, mask_col]
                    # Store into the correct atom position
                    reconstructed_atom_pos[mask_row + bead_pos_from, b2a_idcs[mask_row, mask_col] + atom_pos_from] = updated_pos
            
            # Re-center predicted atoms' center of mass to the actual bead position
            predicted_atoms_cm = scatter(
                reconstructed_atom_pos[b2a_idcs_row + bead_pos_from, b2a_idcs[b2a_idcs_row, b2a_idcs_col] + atom_pos_from] * batch_weights[b2a_idcs_row, b2a_idcs_col][:, None],
                b2a_idcs_row,
                dim=0,
                dim_size=len(batch_center_atoms),
            )
            atom_shifts = predicted_atoms_cm - bead_pos[batch_center_atoms]
            reconstructed_atom_pos[b2a_idcs_row + bead_pos_from, b2a_idcs[b2a_idcs_row, b2a_idcs_col] + atom_pos_from] -= atom_shifts[b2a_idcs_row]
        
        data[self.out_field] = torch.nanmean(reconstructed_atom_pos, dim=0)
        return data