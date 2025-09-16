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
        num_types: int,
        in_field: str = "node_output",
        out_field_irreps: Irreps = Irreps("1x1e"),
        normalize_b2a_rel_vec: bool = True,
        recenter: bool = True,
        irreps_in = None,
    ):
        super().__init__()
        self.in_field = in_field
        self.out_field = DataDict.ATOM_POSITION
        self.normalize_b2a_rel_vec = normalize_b2a_rel_vec
        self.recenter = recenter


        hierarchy_irreps = Irreps(f"{irreps_in[in_field].num_irreps}x0e")
        irreps_in.update({
            "bead2atom_reconstructed_idcs": hierarchy_irreps,
            "bead2atom_reconstructed_weights": hierarchy_irreps,
            "lvl_idcs_mask": hierarchy_irreps,
            "lvl_idcs_anchor_mask": hierarchy_irreps,
        })

        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={
                "atom_pos": Irreps("1x1o"),
            },
            irreps_out={self.out_field: out_field_irreps},
        )

        self.reshape = reshape_irreps(self.irreps_in[in_field])

        if self.normalize_b2a_rel_vec:
            self.atom_type2bond_lengths = torch.nn.Parameter(torch.ones((num_types + 1, self.irreps_out[in_field].num_irreps, 1), dtype=torch.float32))

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        bead2atom_relative_vectors = self.reshape(data[self.in_field])
        if torch.any(torch.isnan(bead2atom_relative_vectors)):
            raise Exception("NaN")
        
        bead_pos = data[AtomicDataDict.POSITIONS_KEY]
        bead_pos_slices        = data.get("pos_slices")
        assert bead_pos_slices is not None, "'pos_slices' must be in data"
        idcs_mask              = data.get("bead2atom_reconstructed_idcs")
        assert idcs_mask is not None, "'bead2atom_reconstructed_idcs' must be in data"
        
        # JIT-friendly fix for default value creation:
        # Avoid .item() as it breaks the computation graph for TorchScript.
        # Instead, we check for the key and create the default tensor using pure tensor operations.
        atom_pos_slices = data.get("atom_pos_slices")
        if atom_pos_slices is None:
            max_idx = idcs_mask.max() + 1
            zero = torch.tensor(0, device=bead_pos.device, dtype=torch.long)
            atom_pos_slices = torch.stack([zero, max_idx])
        
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
            if self.recenter:
                predicted_atoms_cm = scatter(
                    reconstructed_atom_pos[b2a_idcs_row + bead_pos_from, b2a_idcs[b2a_idcs_row, b2a_idcs_col] + atom_pos_from] * batch_weights[b2a_idcs_row, b2a_idcs_col][:, None],
                    b2a_idcs_row,
                    dim=0,
                    dim_size=len(batch_center_atoms),
                )
                atom_shifts = predicted_atoms_cm - bead_pos[batch_center_atoms]
                reconstructed_atom_pos[b2a_idcs_row + bead_pos_from, b2a_idcs[b2a_idcs_row, b2a_idcs_col] + atom_pos_from] -= atom_shifts[b2a_idcs_row]
        
        atom_pos = torch.nanmean(reconstructed_atom_pos, dim=0)
        if AtomicDataDict.PBC_KEY in data and AtomicDataDict.CELL_KEY in data:
            cell = data[AtomicDataDict.CELL_KEY]
            # Compute cell_shift for each atom position
            # atom_pos is in Cartesian coordinates; convert to fractional, wrap, and convert back
            # cell: (N, 3, 3), atom_pos: (num_atoms, 3)
            # Convert atom_pos to fractional coordinates
            cell_inv = torch.inverse(cell)  # (N, 3, 3)
            atom_pos_frac = torch.einsum("nj,nji->ni", atom_pos, cell_inv)
            # Wrap fractional coordinates into [0, 1)
            atom_pos_frac_wrapped = atom_pos_frac % 1.0
            # Convert back to Cartesian coordinates
            atom_pos = torch.einsum("ni,nij->nj", atom_pos_frac_wrapped, cell)
        data[self.out_field] = atom_pos
        return data
