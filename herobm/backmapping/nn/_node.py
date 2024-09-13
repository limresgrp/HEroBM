import torch
import torch.nn
import math
from typing import Dict, Optional
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode
from geqtrain.data import AtomicDataDict
from geqtrain.nn._graph_mixin import GraphModuleMixin

@compile_mode("script")
class EmbeddingNodeAttrs(GraphModuleMixin, torch.nn.Module):
    """Select the node embedding based on node type.

    Args:
    """

    def __init__(
        self,
        node_attributes: Dict[str, Dict] = {},
        num_types: Optional[int] = None,
        irreps_in=None,
    ):
        super().__init__()

        attributes_to_embed = {} # k: str field name, v: nn.Embedding layer name
        output_embedding_dim = 0
        for field, values in node_attributes.items():
            if 'embedding_dimensionality' not in values: # this means the attr is not used as embedding
                continue
            emb_layer_name = f"{field}_embedding"
            attributes_to_embed[field] = emb_layer_name
            n_types = values.get('num_types', num_types) + 1
            embedding_dim = values['embedding_dimensionality']
            setattr(self, emb_layer_name, torch.nn.Embedding(n_types, embedding_dim))
            torch.nn.init.normal_(getattr(self, emb_layer_name).weight, mean=0, std=math.isqrt(embedding_dim))
            output_embedding_dim += embedding_dim

        self.attributes_to_embed = attributes_to_embed
        irreps_out = {AtomicDataDict.NODE_ATTRS_KEY: Irreps([(output_embedding_dim, (0, 1))])}
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)


    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        out = []
        for attribute_name, emb_layer_name in self.attributes_to_embed.items():
            x = data[attribute_name].squeeze()
            x = getattr(self, emb_layer_name)(x)
            out.append(x)

        data[AtomicDataDict.NODE_ATTRS_KEY] = torch.cat(out, dim=-1)
        return data