from typing import Optional
import math

import torch
from torch_runstats.scatter import scatter

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin

from .. import _keys


class EdgewiseEnergySum(GraphModuleMixin, torch.nn.Module):
    """Sum edgewise energies.

    Includes optional per-species-pair edgewise energy scales.
    """

    out_field: str
    _factor: Optional[float]

    def __init__(
        self,
        # num_types: int,
        field: str = _keys.EDGE_FEATURES,
        out_field: str = AtomicDataDict.PER_ATOM_ENERGY_KEY,
        avg_num_neighbors: Optional[float] = None,
        normalize_edge_energy_sum: bool = True,
        # per_edge_species_scale: Optional[float] = None,
        irreps_in={},
    ):
        """Sum edges into nodes."""
        super().__init__()
        self.field = field
        self.out_field = out_field

        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={field: irreps_in[field]},
            irreps_out={out_field: irreps_in[field]},
        )

        self._factor = None
        if normalize_edge_energy_sum and avg_num_neighbors is not None:
            self._factor = 1.0 / math.sqrt(avg_num_neighbors)

        # self.per_edge_species_scale = per_edge_species_scale
        # if self.per_edge_species_scale is not None:
        #     self.per_edge_scales = torch.nn.Parameter(self.per_edge_species_scale * torch.ones(num_types, num_types))
        # else:
        #     self.register_buffer("per_edge_scales", torch.Tensor())

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        # edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]

        edge_eng = data[self.field]
        species = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        # center_species = species[edge_center]
        # neighbor_species = species[edge_neighbor]

        # if self.per_edge_species_scale:
        #     edge_eng = edge_eng * self.per_edge_scales[
        #         center_species, neighbor_species
        #     ].reshape(-1, *[1 for _ in range(len(edge_eng.shape)-1)])

        atom_eng = scatter(edge_eng, edge_center, dim=0, dim_size=len(species)) # / torch.bincount(edge_center, minlength=len(species)).unsqueeze(1)
        factor: Optional[float] = self._factor  # torchscript hack for typing
        if factor is not None:
            atom_eng = atom_eng * factor

        data[self.out_field] = atom_eng

        return data