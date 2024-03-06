from typing import List, Union, Optional

import torch

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin

@compile_mode("script")
class RequireGradModule(GraphModuleMixin, torch.nn.Module):

    def __init__(
        self,
        fields: List[str],
        irreps_in=None,
    ):
        super().__init__()
        self.fields = fields

        self._init_irreps(
            irreps_in=irreps_in,
        )
    
    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        for field in self.fields:
            data[field].requires_grad_(True)
        
        return data

@compile_mode("script")
class GradModule(GraphModuleMixin, torch.nn.Module):
    r"""
    """
    sign: float

    def __init__(
        self,
        of: str,
        wrt: str,
        out_field: Optional[str] = None,
        sign: float = 1.0,
        irreps_in=None,
    ):
        super().__init__()
        sign = float(sign)
        self.sign = sign
        self.of = of

        self.wrt = wrt
        if out_field is None:
            self.out_field = f"d({of})/d({self.wrt})"
        else:
            self.out_field = out_field

        # check and init irreps
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={of: irreps_in[of]},
        )

        self.num_irreps = irreps_in[of].num_irreps

        # The gradient of a single scalar w.r.t. something of a given shape and irrep just has that shape and irrep
        # Ex.: gradient of energy (0e) w.r.t. position vector (L=1) is also an L = 1 vector
        self.irreps_out.update(
            {self.out_field: Irreps([(self.num_irreps, ir) for (_, ir) in self.irreps_in[self.wrt]])}
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        # # set req grad
        # wrt_tensors = []
        # old_requires_grad: List[bool] = []
        # for k in self.wrt:
        #     old_requires_grad.append(data[k].requires_grad)
        #     data[k].requires_grad_(True)
        #     wrt_tensors.append(data[k])

        wrt_tensors = [data[self.wrt]]

        # Get grads
        
        nodes, coords = data[self.wrt].shape
        out = torch.zeros((nodes, self.num_irreps, coords), dtype=data[self.of].dtype, device=data[self.of].device)
        for channel in range(self.num_irreps):
            grads = torch.autograd.grad(
                [data[self.of][:, channel].sum()],
                wrt_tensors,
                create_graph=self.training,  # needed to allow gradients of this output during training
                retain_graph=True,
            )
        
            if grads is None:
                # From the docs: "If an output doesn’t require_grad, then the gradient can be None"
                raise RuntimeError("Something is wrong, gradient couldn't be computed")
            out[:, channel] += grads[0]

        data[self.out_field] = out

        # # unset requires_grad_
        # for req_grad, k in zip(old_requires_grad, self.wrt):
        #     data[k].requires_grad_(req_grad)

        return data