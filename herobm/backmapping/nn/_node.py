import torch
import torch.nn
import math
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode
from geqtrain.data import AtomicDataDict
from geqtrain.nn._graph_mixin import GraphModuleMixin


@compile_mode("script")
class EmbeddingNodeAttrs(GraphModuleMixin, torch.nn.Module):
    """Select the node embedding based on node type.
    Args:
        num_types (int): Total number of different node_types.
        embedding_dim (int): Dimension of the node attribute embedding tensor.
    """

    num_types: int

    def __init__(
        self,
        num_types: int,
        embedding_dim: int = 8,
        irreps_in=None,
    ):
        super().__init__()
        self.embeddings = torch.nn.Embedding(num_types, embedding_dim) # scale_grad_by_freq = False by default
        torch.nn.init.normal_(self.embeddings.weight, mean=0, std=1/math.sqrt(embedding_dim)) # xavier: 1/sqrt(input dim of the layer)

        irreps_out = {AtomicDataDict.NODE_ATTRS_KEY: Irreps([(embedding_dim, (0, 1))])}
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if data.get(AtomicDataDict.NODE_ATTRS_KEY, None) is None:
            type_numbers = data[AtomicDataDict.NODE_TYPE_KEY].squeeze(-1)
            node_attrs = self.embeddings(type_numbers)

            data[AtomicDataDict.NODE_ATTRS_KEY] = node_attrs
        return data