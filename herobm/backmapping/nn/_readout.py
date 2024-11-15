from typing import Optional, Union
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin, ScalarMLPFunction
from geqtrain.nn.allegro import Linear
from geqtrain.nn.mace.irreps_tools import reshape_irreps, inverse_reshape_irreps


@compile_mode("script")
class ReadoutModule(GraphModuleMixin, torch.nn.Module):

    def __init__(
        self,
        field: str,
        out_field: Optional[str] = None,
        out_irreps: Union[o3.Irreps, str] = None,
        readout_latent=ScalarMLPFunction,
        readout_latent_kwargs={},
        eq_has_internal_weights: bool = False,
        resnet: bool = False,
        irreps_in=None,
    ):
        super().__init__()

        self.field = field
        self.out_field = out_field or field
        self.has_inv_out = False
        self.has_eq_out = False
        self.eq_has_internal_weights = eq_has_internal_weights
        self.resnet = resnet

        in_irreps = irreps_in[field]
        out_irreps = (
            out_irreps if isinstance(out_irreps, o3.Irreps)
            else (
                o3.Irreps(out_irreps) if isinstance(out_irreps, str)
                else irreps_in[self.out_field] if self.out_field in irreps_in
                else in_irreps
            )
        )
        self.out_irreps = out_irreps
        self.out_irreps_muls = [ir.mul for ir in out_irreps]

        # check and init irreps
        my_irreps_in = {field: in_irreps}
        if self.resnet:
            my_irreps_in.update({self.out_field: out_irreps})
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in=my_irreps_in,
            irreps_out={self.out_field: out_irreps},
        )

        self.n_scalars_in = in_irreps.ls.count(0)
        assert self.n_scalars_in > 0

        readout_latent_kwargs['use_layer_norm'] = True
        readout_latent_kwargs.pop('dropout', None)
        self.n_scalars_out = out_irreps.ls.count(0)
        if self.n_scalars_out > 0:
            self.has_inv_out = True
            self.inv_readout = readout_latent(
                mlp_input_dimension=self.n_scalars_in,
                mlp_output_dimension=self.n_scalars_out,
                **readout_latent_kwargs,
            )

        if out_irreps.dim > self.n_scalars_out:
            self.has_eq_out = True
            eq_linear_input_irreps = o3.Irreps([(mul, ir) for mul, ir in in_irreps  if ir.l>0])
            eq_linear_output_irreps = o3.Irreps([(mul, ir) for mul, ir in out_irreps if ir.l>0])
            self.reshape_in = reshape_irreps(eq_linear_input_irreps)
            self.eq_readout = Linear(
                    eq_linear_input_irreps,
                    eq_linear_output_irreps,
                    shared_weights=self.eq_has_internal_weights,
                    internal_weights=self.eq_has_internal_weights,
                    pad_to_alignment=1,
                )

            if not self.eq_has_internal_weights:
                self.weights_emb = readout_latent(
                    mlp_input_dimension=self.n_scalars_in,
                    mlp_output_dimension=self.eq_readout.weight_numel,
                    **readout_latent_kwargs,
                )
            self.reshape_back_features = inverse_reshape_irreps(eq_linear_output_irreps)
        else:
            assert in_irreps.dim == self.n_scalars_in, (
                    f"Module input contains features with irreps which are not scalars ({in_irreps})." +
                    f"However, the irreps of the output is composed of scalars only ({out_irreps})."   +
                    "Please remove non-scalar features from the input, which otherwise would remain unused." +
                    f"If features come from InteractionModule, you can add the parameter 'output_hidden_ls=[0]' in the constructor"
                )
            self.reshape_in = None

        self._resnet_update_coeff: Optional[torch.nn.Parameter] = None # init to None for jit
        if self.resnet:
            assert irreps_in[self.out_field] == out_irreps
            self._resnet_update_coeff = torch.nn.Parameter(torch.tensor([0.0]))
        self.out_irreps_dim = self.out_irreps.dim

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        features = data[self.field]
        out_features = torch.zeros(
            (len(features), self.out_irreps_dim),
            dtype=torch.float32,
            device=features.device
        )

        if self.has_inv_out: # invariant output may be present or not
            out_features[:, :self.n_scalars_out] += self.inv_readout(features[:, :self.n_scalars_in]) # normal mlp on scalar component if present

        # vectorial handling
        if self.has_eq_out and self.reshape_in is not None:
            eq_features = self.reshape_in(features[:, self.n_scalars_in:])
            if self.eq_has_internal_weights: # eq linear layer with its own inner weights
                eq_features = self.eq_readout(eq_features)
            else:
                # else the weights are computed via mlp on scalars
                weights = self.weights_emb(features[:, :self.n_scalars_in])
                eq_features = self.eq_readout(eq_features, weights)
            out_features[:, self.n_scalars_out:] += self.reshape_back_features(eq_features)
        
        if self.resnet:
            assert self._resnet_update_coeff is not None
            old_features = data[self.out_field]
            _coeff = self._resnet_update_coeff.sigmoid()
            coefficient_old = torch.rsqrt(_coeff.square() + 1)
            coefficient_new = _coeff * coefficient_old
            # Residual update
            out_features = coefficient_old * old_features + coefficient_new * out_features

        data[self.out_field] = out_features
        return data