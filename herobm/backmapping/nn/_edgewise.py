import torch
import math
from typing import Optional
from einops import rearrange
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin, ScalarMLPFunction
from geqtrain.nn.mace.irreps_tools import reshape_irreps, inverse_reshape_irreps


class EdgewiseReduce(GraphModuleMixin, torch.nn.Module):
    """Sum edgewise features.

    Includes optional per-species-pair edgewise scales.
    """

    out_field: str
    _factor: Optional[float]

    def __init__(
        self,
        field: str,
        out_field: Optional[str] = None,
        readout_latent=ScalarMLPFunction,
        readout_latent_kwargs={},
        head_dim: int = 32,
        use_attention: bool = True,
        irreps_in={},
        avg_num_neighbors: Optional[float] = 5.0,
    ):
        """Sum edges into nodes."""
        super().__init__()
        self.field = field
        self.use_attention = use_attention
        self.out_field = f"weighted_sum_{field}" if out_field is None else out_field
        irreps = irreps_in[field]

        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={field: irreps},
            irreps_out={out_field: irreps},
        )

        if self.use_attention:

            irreps_muls = []
            n_l = {}
            n_dim = 0
            for mul, ir in irreps:
                irreps_muls.append(mul)
                n_l[ir.l] = n_l.get(ir.l, 0) + 1
                n_dim += ir.dim
            assert all([irreps_mul == irreps_muls[0] for irreps_mul in irreps_muls])

            self.irreps_mul = irreps_muls[0]
            self.n_l = n_l
            self.n_dim = n_dim

            if 'mlp_latent_dimensions' not in readout_latent_kwargs:
                readout_latent_kwargs['mlp_latent_dimensions'] = [64, 64]
            if 'zero_init_last_layer_weights' not in readout_latent_kwargs:
                readout_latent_kwargs['zero_init_last_layer_weights'] = True

            self.head_dim = head_dim
            self.isqrtd = math.isqrt(head_dim)

            self.n_scalars = self.irreps_mul * self.n_l[0]

            self.reshape_in = reshape_irreps(irreps)

            self.node_attr_to_query = readout_latent(
                mlp_input_dimension=irreps_in[AtomicDataDict.NODE_ATTRS_KEY].dim,
                mlp_output_dimension=self.n_scalars * self.head_dim,
                **readout_latent_kwargs,
            )

            self.edge_feat_to_key = readout_latent(
                mlp_input_dimension=self.n_scalars,
                mlp_output_dimension= self.n_scalars * self.head_dim,
                **readout_latent_kwargs,
            )

            self.reshape_out = inverse_reshape_irreps(irreps)

            self.irreps_out.update(
                {
                    self.out_field: irreps
                }
            )
        else:
            self.node_attr_to_query = None
            self.edge_feat_to_key = None

        if not self.use_attention:
            self.register_buffer(
                "env_sum_normalization",
                torch.as_tensor([avg_num_neighbors]).rsqrt(),
            )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_feat = data[self.field]

        species = data[AtomicDataDict.NODE_TYPE_KEY].squeeze(-1)
        num_nodes = len(species)

        if self.use_attention and self.node_attr_to_query is not None:
            Q = self.node_attr_to_query(data[AtomicDataDict.NODE_ATTRS_KEY][edge_center])
            Q = rearrange(Q, 'e (c d) -> e c d', c=self.n_scalars, d=self.head_dim)

            K = self.edge_feat_to_key(edge_feat[..., :self.n_scalars])
            K = rearrange(K, 'e (c d) -> e c d', c=self.n_scalars, d=self.head_dim)

            W = torch.einsum('ecd,ecd -> ec', Q, K) * self.isqrtd

            edge_feat = self.reshape_in(edge_feat)
            edge_feat = torch.einsum('emd,em->emd', edge_feat, scatter_softmax(W, edge_center, dim=0))
            edge_feat = self.reshape_out(edge_feat)

        # aggregation step
        data[self.out_field] = scatter(edge_feat, edge_center, dim=0, dim_size=num_nodes)

        if not self.use_attention:
            data[self.out_field] *= self.env_sum_normalization

        return data