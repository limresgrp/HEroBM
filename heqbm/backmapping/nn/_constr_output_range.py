import torch
from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from e3nn.util.jit import compile_mode


@compile_mode("script")
class ConstrOutputRange(GraphModuleMixin, torch.nn.Module):

    def __init__(
        self,
        func: GraphModuleMixin,
        field: str,
        range = [-torch.pi, torch.pi]
    ):
        super().__init__()
        self.func = func
        self.field = field
        self.lower = range[0]
        self.upper = range[1]

        # check and init irreps
        self._init_irreps(
            irreps_in=func.irreps_in,
            my_irreps_in={field: func.irreps_out[field]},
            irreps_out=func.irreps_out,
        )

        self.activation = torch.nn.Sigmoid()
        self.scale = self.upper - self.lower

        self.field_irreps = self.irreps_out[self.field]

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = self.func(data)
        out = data[self.field]
        data[self.field] = self.scale * self.activation(out) + self.lower
        return data