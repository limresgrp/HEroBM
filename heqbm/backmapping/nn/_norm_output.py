import torch
from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from e3nn.util.jit import compile_mode


@compile_mode("script")
class NormOutputLWise(GraphModuleMixin, torch.nn.Module):

    def __init__(
        self,
        func: GraphModuleMixin,
        field: str,
    ):
        super().__init__()
        self.func = func
        self.field = field

        # check and init irreps
        self._init_irreps(
            irreps_in=func.irreps_in,
            my_irreps_in={field: func.irreps_out[field]},
            irreps_out=func.irreps_out,
        )

        self.field_irreps = self.irreps_out[self.field]

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = self.func(data)
        out = data[self.field]

        cum_pos = 0
        with torch.no_grad():
            for irrep in self.field_irreps:
                dim = irrep.ir.dim
                tensor = out[..., cum_pos:cum_pos + dim]
                # tensor[tensor.abs() < 1e-2] *= 0.
                norm = tensor.norm(dim=-1, keepdim=True)
                norm[norm == 0.] += 1.
                out[..., cum_pos:cum_pos + dim] /= norm
                cum_pos += dim


        data[self.field] = out
        return data