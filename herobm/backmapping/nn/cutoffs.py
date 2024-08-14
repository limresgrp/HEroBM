import math
import torch


@torch.jit.script
def _poly_cutoff(x: torch.Tensor, p: float = 6.0) -> torch.Tensor:

    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p))
    out = out + (p * (p + 2.0) * torch.pow(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * torch.pow(x, p + 2.0))

    return out * (x < 1.0)

@torch.jit.script
def polynomial_cutoff(
    x: torch.Tensor, r_max: torch.Tensor, p: float = 6.0
) -> torch.Tensor:
    """Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123


    Parameters
    ----------
    r_max : tensor
        Broadcasts over r_max.

    p : int
        Power used in envelope function
    """
    assert p >= 2.0
    r_max, x = torch.broadcast_tensors(r_max.unsqueeze(-1), x.unsqueeze(0))

    return _poly_cutoff(x / r_max, p=p)

class PolynomialCutoff(torch.nn.Module):
    _factor: float
    p: float

    def __init__(self, r_max: float, p: float = 6):
        r"""Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123


        Parameters
        ----------
        r_max : float
            Cutoff radius

        p : int
            Power used in envelope function
        """
        super().__init__()
        assert p >= 2.0
        self.p = float(p)
        self._factor = 1.0 / float(r_max)

    def forward(self, x):
        """
        Evaluate cutoff function.

        x: torch.Tensor, input distance
        """
        return _poly_cutoff(x * self._factor, p=self.p)


@torch.jit.script
def _tanh_cutoff(x: torch.Tensor, n: float = 6.0) -> torch.Tensor:
    return torch.tanh(1 - torch.pow(x, n)) / math.tanh(1)

@torch.jit.script
def tanh_cutoff(
    x: torch.Tensor, r_max: torch.Tensor, n: float = 6.0
) -> torch.Tensor:
    """Tanh cutoff


    Parameters
    ----------
    r_max : tensor
        Broadcasts over r_max.

    N : int
        Power used in envelope function
    """
    assert n >= 1.0
    r_max, x = torch.broadcast_tensors(r_max.unsqueeze(-1), x.unsqueeze(0))

    return _tanh_cutoff(x / r_max, n=n)

class TanhCutoff(torch.nn.Module):
    _factor: float
    p: float

    def __init__(self, r_max: float, n: float = 6):
        r"""Tanh cutoff


        Parameters
        ----------
        r_max : float
            Cutoff radius

        n : int
            Power used in envelope function
        """
        super().__init__()
        assert n >= 1.0
        self.n = float(n)
        self._factor = 1.0 / float(r_max)

    def forward(self, x):
        """
        Evaluate cutoff function.

        x: torch.Tensor, input distance
        """
        return _tanh_cutoff(x * self._factor, n=self.n)