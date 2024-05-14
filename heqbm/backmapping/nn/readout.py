import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from nequip.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin
from herobm.backmapping.allegro.nn._strided import Linear
from herobm.backmapping.allegro.nn._fc import ScalarMLPFunction
from herobm.backmapping.allegro._keys import (EQUIVARIANT_ATOM_LENGTH_FEATURES)


@compile_mode("script")
class HierarchicalBackmappingReadoutModule(GraphModuleMixin, torch.nn.Module):

    def __init__(
        self,
        num_types: int,
        inv_field: str,
        eq_field: str,
        out_field: str,
        readout_features: bool,
        inv_out_irreps: o3.Irreps = o3.Irreps("2x0e"),
        eq_out_irreps: o3.Irreps = o3.Irreps("4x1o"),
        latent=ScalarMLPFunction,
        normalize_out_features: bool = True,
        irreps_in=None
    ):
        super().__init__()
        self.num_types = num_types
        self.inv_field = inv_field
        self.eq_field = eq_field
        self.out_field = out_field
        self.inv_out_irreps = inv_out_irreps
        self.eq_out_irreps = eq_out_irreps
        self.readout_features = readout_features
        self.normalize_out_features = normalize_out_features

        # check and init irreps
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={
                inv_field: irreps_in[inv_field],
                eq_field: irreps_in[eq_field],
                },
        )

        if self.readout_features:
            self.final_readout = latent(
                mlp_input_dimension=irreps_in[inv_field].dim,
                mlp_latent_dimensions=[512, 512],
                mlp_output_dimension=self.inv_out_irreps.dim,
            )

            self.final_linear = Linear(
                irreps_in[eq_field],
                self.eq_out_irreps,
                shared_weights=False,
                internal_weights=False,
                pad_to_alignment=1,
            )

            self.final_latent = latent(
                mlp_input_dimension=irreps_in[inv_field].dim,
                mlp_latent_dimensions=[512, 512],
                mlp_output_dimension=self.final_linear.weight_numel,
            )

        self.irreps_out.update(
            {
                self.inv_field: self.inv_out_irreps,
                self.out_field: self.eq_out_irreps,
            }
        )

        if self.normalize_out_features:
            self.atom_type2bond_lengths = torch.nn.Parameter(torch.ones((num_types, self.eq_out_irreps.num_irreps, 1), dtype=torch.float32))

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        inv_features = data[self.inv_field]
        eq_features = data[self.eq_field]
        
        if self.readout_features:
            weights = self.final_latent(inv_features)
            data[self.inv_field] = self.final_readout(inv_features)
            
            eq_features = self.final_linear(eq_features, weights).squeeze(dim=-1)
        
        if self.normalize_out_features:
            norm = torch.norm(eq_features, dim=-1, keepdim=True)
            eq_features = eq_features / (norm + 1.e-10)
            eq_features = torch.nan_to_num(eq_features, nan=0.)

            if EQUIVARIANT_ATOM_LENGTH_FEATURES in data:
                eq_features = eq_features * torch.abs(data[EQUIVARIANT_ATOM_LENGTH_FEATURES])[..., None]
            else:
                eq_features = eq_features * self.atom_type2bond_lengths[data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)]

        # if self.normalize_out_features:
        #     to_normalize_filter = ~data[DataDict.LEVEL_IDCS_MASK][1, data[AtomicDataDict.EDGE_INDEX_KEY].unique()] # Exclude hierarchy 1 from normalisation
        #     with torch.no_grad():
        #         norm = torch.norm(eq_features, dim=-1, keepdim=True)
        #         norm_vector = torch.ones_like(eq_features)
        #         norm_vector[to_normalize_filter] /= (norm[to_normalize_filter] + 1e-10)
            
        #     eq_features = eq_features * norm_vector * self.atom_type2bond_lengths[data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)]

        data[self.out_field] = eq_features.squeeze(-2)
        return data


@compile_mode("script")
class ReadoutModule(GraphModuleMixin, torch.nn.Module):

    def __init__(
        self,
        num_types: int,
        inv_field: str,
        inv_out_irreps: o3.Irreps = o3.Irreps("1x0e"),
        latent=ScalarMLPFunction,
        irreps_in=None,
        per_species_bias=None,
    ):
        super().__init__()
        self.inv_field = inv_field
        self.inv_out_irreps = inv_out_irreps

        # check and init irreps
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={
                inv_field: irreps_in[inv_field],
                },
        )

        self.final_readout = latent(
            mlp_input_dimension=irreps_in[inv_field].dim,
            mlp_latent_dimensions=[512, 512],
            mlp_output_dimension=self.inv_out_irreps.dim,
        )

        if per_species_bias is not None:
            assert len(per_species_bias) == num_types
            self.per_species_bias = torch.nn.Parameter(torch.tensor(per_species_bias))
        else:
            self.per_species_bias = torch.nn.Parameter(torch.zeros(num_types))

        self.irreps_out.update(
            {
                self.inv_field: self.inv_out_irreps,
            }
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        out_features = self.final_readout(data[self.inv_field])

        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0].unique()
        species = data[AtomicDataDict.ATOM_TYPE_KEY]
        center_species = species[edge_center]
        out_features[edge_center] += self.per_species_bias[center_species]
        
        data[self.inv_field] = out_features.squeeze(-1)
        return data