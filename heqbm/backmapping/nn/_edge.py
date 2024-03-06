import torch

from e3nn import o3
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from nequip.nn.radial_basis import BesselBasis
from nequip.nn.cutoffs import PolynomialCutoff



@compile_mode("script")
class RadialBasisClassEdgeEncoding(GraphModuleMixin, torch.nn.Module):
    out_field: str

    def __init__(
        self,
        basis=BesselBasis,
        cutoff=PolynomialCutoff,
        basis_kwargs={},
        cutoff_kwargs={},
        out_field: str = AtomicDataDict.EDGE_EMBEDDING_KEY,
        edge_class_field: str = 'edge_class',
        irreps_in=None,
    ):
        super().__init__()
        self.basis = basis(**basis_kwargs)
        self.cutoff = cutoff(**cutoff_kwargs)
        self.out_field = out_field
        self.edge_class_field = edge_class_field
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: o3.Irreps([(self.basis.num_basis + 1, (0, 1))])},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)
        edge_length = data[AtomicDataDict.EDGE_LENGTH_KEY]
        edge_length_embedded = (
            self.basis(edge_length) * self.cutoff(edge_length)[:, None]
        )
        
        src_index, trg_index = data[AtomicDataDict.EDGE_INDEX_KEY]
        src_classes = data[self.edge_class_field][src_index]
        trg_classes = data[self.edge_class_field][trg_index]
        edge_class_membership =  src_classes == trg_classes

        edge_length_embedded = torch.cat(
            [
                edge_length_embedded,
                edge_class_membership[:, None]
            ],
            dim=1,
        )
        
        data[self.out_field] = edge_length_embedded
        return data