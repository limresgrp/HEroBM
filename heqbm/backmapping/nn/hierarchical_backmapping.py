import math
import functools
import torch

from typing import Callable, Optional, List
from torch_runstats.scatter import scatter

from e3nn import o3
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin
from geqtrain.utils.tp_utils import tp_path_exists

from heqbm.backmapping.allegro.nn._fc import ScalarMLPFunction, ExponentialScalarMLPFunction
from heqbm.backmapping.allegro import _keys
from heqbm.backmapping.allegro.nn._strided import Contracter, MakeWeightedChannels, Linear
from heqbm.backmapping.allegro.nn.cutoffs import cosine_cutoff, polynomial_cutoff

from heqbm.backmapping.mace.blocks import EquivariantProductBasisBlock
from heqbm.backmapping.mace.irreps_tools import reshape_irreps, inverse_reshape_irreps


def pick_mpl_function(func):
    if isinstance(func, Callable):
        return func
    assert isinstance(func, str)
    if func.lower() == "ScalarMLPFunction".lower():
        return ScalarMLPFunction
    if func.lower() == "ExponentialScalarMLPFunction".lower():
        return ExponentialScalarMLPFunction
    raise Exception(f"MLP Funciton {func} not implemented.")


@compile_mode("script")
class HierarchicalBackmappingModule(GraphModuleMixin, torch.nn.Module):
    # saved params
    num_layers: int
    field: str
    inv_out_field: str
    eq_out_field: str
    num_types: int
    env_embed_mul: int
    weight_numel: int
    latent_resnet: bool

    # internal values
    _env_builder_w_index: List[int]
    _env_builder_n_irreps: int
    _input_pad: int

    def __init__(
        self,
        # required params
        num_layers: int,
        num_types: int,
        r_max: float,
        avg_num_neighbors: Optional[float] = None,
        # cutoffs
        r_start_cos_ratio: float = 0.8,
        PolynomialCutoff_p: float = 6,
        per_layer_cutoffs: Optional[List[float]] = None,
        cutoff_type: str = "polynomial",
        # general hyperparameters:
        field: str = AtomicDataDict.EDGE_ATTRS_KEY,
        readout_features: bool = False,
        inv_out_field: Optional[str] = _keys.INVARIANT_EDGE_FEATURES,
        eq_out_field: Optional[str] = _keys.EQUIVARIANT_EDGE_FEATURES,
        inv_out_irreps: o3.Irreps = o3.Irreps("0x0e"),
        eq_out_irreps: o3.Irreps = o3.Irreps("0x1o"),
        edge_invariant_field: str = AtomicDataDict.EDGE_EMBEDDING_KEY,
        node_invariant_field: str = AtomicDataDict.NODE_ATTRS_KEY,
        eq_node_in_feat_field: str = _keys.EQUIVARIANT_ATOM_INPUT_FEATURES,
        eq_node_in_feat_irreps: o3.Irreps = o3.Irreps("0x1o"),
        env_embed_multiplicity: int = 32,
        linear_after_env_embed: bool = True,
        nonscalars_include_parity: bool = True,
        product_correlation: int = 3,
        # MLP parameters:
        env_embed=ScalarMLPFunction,
        env_embed_kwargs={},
        two_body_latent=ScalarMLPFunction,
        two_body_latent_kwargs={},
        latent=ScalarMLPFunction,
        latent_kwargs={},
        latent_resnet: bool = True,
        latent_resnet_update_ratios: Optional[List[float]] = None,
        latent_resnet_update_ratios_learnable: bool = True,
        feature_resnet_update_ratios: Optional[List[float]] = None,
        feature_resnet_update_ratios_learnable: bool = True,
        # Performance parameters:
        pad_to_alignment: int = 1,
        sparse_mode: Optional[str] = None,
        # Other:
        irreps_in=None,
    ):
        super().__init__()
        SCALAR = o3.Irrep("0e")  # define for convinience

        # save parameters
        assert (
            num_layers >= 1
        )  # zero layers is "two body", but we don't need to support that fallback case
        self.num_layers = num_layers
        self.nonscalars_include_parity = nonscalars_include_parity
        self.field = field
        self.inv_out_field = inv_out_field
        self.eq_out_field = eq_out_field
        self.inv_out_irreps = inv_out_irreps
        self.eq_out_irreps = eq_out_irreps
        self.edge_invariant_field = edge_invariant_field
        self.node_invariant_field = node_invariant_field
        self.eq_node_in_feat_field = eq_node_in_feat_field
        self.eq_node_in_feat_irreps = eq_node_in_feat_irreps
        self.latent_resnet = latent_resnet
        self.env_embed_mul = env_embed_multiplicity
        self.r_start_cos_ratio = r_start_cos_ratio
        self.polynomial_cutoff_p = float(PolynomialCutoff_p)
        self.cutoff_type = cutoff_type
        assert cutoff_type in ("cosine", "polynomial")
        self.avg_num_neighbors = avg_num_neighbors
        self.linear_after_env_embed = linear_after_env_embed
        self.num_types = num_types
        self.readout_features = readout_features

        env_embed = pick_mpl_function(env_embed)
        two_body_latent = pick_mpl_function(two_body_latent)
        latent = pick_mpl_function(latent)

        # set up irreps
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                self.field,
                self.edge_invariant_field,
                self.node_invariant_field,
            ],
        )

        # for normalization of features
        # one per layer
        self.register_buffer(
            "env_sum_normalizations",
            # dividing by sqrt(N)
            torch.as_tensor([avg_num_neighbors] * num_layers).reciprocal(),
        )

        latent = functools.partial(latent, **latent_kwargs)
        env_embed = functools.partial(env_embed, **env_embed_kwargs)

        self.latents = torch.nn.ModuleList([])
        self.env_embed_mlps = torch.nn.ModuleList([])
        self.tps = torch.nn.ModuleList([])
        self.products = torch.nn.ModuleList([])
        self.reshape_back_modules = torch.nn.ModuleList([])
        self.linears = torch.nn.ModuleList([])
        self.env_linears = torch.nn.ModuleList([])

        # Embed to the spharm * it as mul
        input_irreps = self.irreps_in[self.field]
        # this is not inherant, but no reason to fix right now:
        assert all(mul == 1 for mul, ir in input_irreps)
        env_embed_irreps = o3.Irreps([(env_embed_multiplicity, ir) for _, ir in input_irreps])
        assert (
            env_embed_irreps[0].ir == SCALAR
        ), "env_embed_irreps must start with scalars"
        self._input_pad = (
            int(math.ceil(env_embed_irreps.dim / pad_to_alignment)) * pad_to_alignment
        ) - env_embed_irreps.dim
        self.register_buffer("_zero", torch.zeros(1, 1))

        # Initially, we have the B(r)Y(\vec{r})-projection of the edges
        # (possibly embedded)
        arg_irreps = env_embed_irreps

        # - begin irreps -
        # start to build up the irreps for the iterated TPs
        tps_irreps = [arg_irreps]

        for layer_idx in range(num_layers):
            ir_out = env_embed_irreps
            # Create higher order terms cause there are more TPs coming
            if layer_idx == self.num_layers - 1:
                # ^ means we're doing the last layer
                # No more TPs follow this, so only need self.eq_out_irreps
                # ir_out = self.eq_out_irreps <- OLD
                if len(self.eq_out_irreps.ls) > 0:
                    ir_out = o3.Irreps([ir for ir in env_embed_irreps if ir.ir.l <= self.eq_out_irreps.lmax])
                else:
                    ir_out = o3.Irreps("0x1o")

            # Prune impossible paths
            ir_out = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in ir_out
                    if tp_path_exists(arg_irreps, env_embed_irreps, ir)
                ]
            )

            # the argument to the next tensor product is the output of this one
            arg_irreps = ir_out
            tps_irreps.append(ir_out)
        # - end build irreps -

        # == Remove unneeded paths ==
        out_irreps = tps_irreps[-1]
        new_tps_irreps = [out_irreps]
        for arg_irreps in reversed(tps_irreps[:-1]):
            new_arg_irreps = []
            for mul, arg_ir in arg_irreps:
                for _, env_ir in env_embed_irreps:
                    if any(i in out_irreps for i in arg_ir * env_ir):
                        # arg_ir is useful: arg_ir * env_ir has a path to something we want
                        new_arg_irreps.append((mul, arg_ir))
                        # once its useful once, we keep it no matter what
                        break
            new_arg_irreps = o3.Irreps(new_arg_irreps)
            new_tps_irreps.append(new_arg_irreps)
            out_irreps = new_arg_irreps

        assert len(new_tps_irreps) == len(tps_irreps)
        tps_irreps = list(reversed(new_tps_irreps))
        del new_tps_irreps

        if len(self.eq_out_irreps.ls) > 0:
            assert tps_irreps[-1].lmax == self.eq_out_irreps.lmax

        tps_irreps_in = tps_irreps[:-1]
        tps_irreps_out = tps_irreps[1:]
        del tps_irreps

        # Environment builder:
        self._env_weighter = MakeWeightedChannels(
            irreps_in=input_irreps,
            multiplicity_out=env_embed_multiplicity,
            pad_to_alignment=pad_to_alignment,
        )

        self._n_scalar_outs: List[int] = []

        # == Build Products and TPs ==
        for layer_idx, (arg_irreps, out_irreps) in enumerate(
            zip(tps_irreps_in, tps_irreps_out)
        ):
            # Make the env embed linear
            if self.linear_after_env_embed:
                self.env_linears.append(
                    Linear(
                        [(env_embed_multiplicity, ir) for _, ir in env_embed_irreps],
                        [(env_embed_multiplicity, ir) for _, ir in env_embed_irreps],
                        shared_weights=True,
                        internal_weights=True,
                    )
                )
            else:
                self.env_linears.append(torch.nn.Identity())
            
            # Make interaction
            interaction_irreps: o3.Irreps = (self.irreps_in[self.field] * env_embed_multiplicity).sort()[0].simplify()

            # Make product
            self.products.append(
                EquivariantProductBasisBlock(
                    node_feats_irreps=interaction_irreps,
                    target_irreps=env_embed_irreps,
                    correlation=product_correlation,
                    num_elements=self.irreps_in[self.node_invariant_field].dim,
                    use_sc=False,
                )
            )

            # Reshape back product so that you can perform tp
            self.reshape_back_modules.append(reshape_irreps(env_embed_irreps))

            # Make TP
            tmp_i_out: int = 0
            instr = []
            n_scalar_outs: int = 0
            full_out_irreps = []
            for i_out, (_, ir_out) in enumerate(out_irreps):
                for i_1, (_, ir_1) in enumerate(arg_irreps):
                    for i_2, (_, ir_2) in enumerate(env_embed_irreps):
                        if ir_out in ir_1 * ir_2:
                            if ir_out == SCALAR:
                                n_scalar_outs += 1
                            instr.append((i_1, i_2, tmp_i_out))
                            full_out_irreps.append((env_embed_multiplicity, ir_out))
                            tmp_i_out += 1
            full_out_irreps = o3.Irreps(full_out_irreps)
            self._n_scalar_outs.append(n_scalar_outs)
            assert all(ir == SCALAR for _, ir in full_out_irreps[:n_scalar_outs])
            tp = Contracter(
                irreps_in1=o3.Irreps(
                    [
                        (
                            (
                                env_embed_multiplicity
                            ),
                            ir,
                        )
                        for _, ir in arg_irreps
                    ]
                ),
                irreps_in2=o3.Irreps(
                    [(env_embed_multiplicity, ir) for _, ir in env_embed_irreps]
                ),
                irreps_out=o3.Irreps(
                    [(env_embed_multiplicity, ir) for _, ir in full_out_irreps]
                ),
                instructions=instr,
                connection_mode=(
                    "uuu"
                ),
                shared_weights=False,
                has_weight=False,
                pad_to_alignment=pad_to_alignment,
                sparse_mode=sparse_mode,
            )
            self.tps.append(tp)

            # Make env embed mlp
            generate_n_weights = (
                self._env_weighter.weight_numel
            )  # the weight for the edge embedding
            if layer_idx == 0:
                # also need weights to embed the edge itself
                # this is because the 2 body latent is mixed in with the first layer
                # in terms of code
                generate_n_weights += self._env_weighter.weight_numel

            # the linear acts after the extractor
            self.linears.append(
                Linear(
                    full_out_irreps,
                    full_out_irreps if layer_idx == self.num_layers - 1 else env_embed_irreps,
                    shared_weights=True,
                    internal_weights=True,
                    pad_to_alignment=pad_to_alignment,
                )
            )

            if layer_idx == 0:
                # at the first layer, we have no invariants from previous TPs
                self.latents.append(
                    two_body_latent(
                        mlp_input_dimension=(
                            (
                                # Node invariants for center and neighbor (chemistry)
                                2 * self.irreps_in[self.node_invariant_field].num_irreps
                                # Plus edge invariants for the edge (radius).
                                + self.irreps_in[self.edge_invariant_field].num_irreps
                            )
                        ),
                        mlp_output_dimension=None,
                        **two_body_latent_kwargs,
                    )
                )
                self._latent_dim = self.latents[-1].out_features
            else:
                self.latents.append(
                    latent(
                        mlp_input_dimension=(
                            (
                                # the embedded latent invariants from the previous layer(s)
                                self.latents[-1].out_features
                                # and the invariants extracted from the last layer's TP:
                                + env_embed_multiplicity * self._n_scalar_outs[layer_idx - 1]
                            )
                        ),
                        mlp_output_dimension=None,
                    )
                )

            # the env embed MLP takes the last latent's output as input
            # and outputs enough weights for the env embedder
            self.env_embed_mlps.append(
                env_embed(
                    mlp_input_dimension=self.latents[-1].out_features,
                    mlp_output_dimension=generate_n_weights,
                )
            )


        self.reshape_back_features = inverse_reshape_irreps(env_embed_irreps)
        
        # ------------------- #

        if self.readout_features and len(self.eq_out_irreps.ls)>0:
            self.final_readout = latent(
                mlp_input_dimension=self.latents[-1].out_features
                + env_embed_multiplicity * self._n_scalar_outs[layer_idx],
                mlp_output_dimension=self.inv_out_irreps.dim,
            )

            self.final_linear = Linear(
                        full_out_irreps,
                        self.eq_out_irreps,
                        shared_weights=False,
                        internal_weights=False,
                        pad_to_alignment=pad_to_alignment,
                    )

            self.final_latent = latent(
                mlp_input_dimension=self.latents[-1].out_features
                + env_embed_multiplicity * self._n_scalar_outs[layer_idx],
                mlp_output_dimension=self.final_linear.weight_numel,
            )

        # ------------------- #

        # - end build modules -

        # - layer resnet update weights -
        if latent_resnet_update_ratios is None:
            # We initialize to zeros, which under the sigmoid() become 0.5
            # so 1/2 * layer_1 + 1/4 * layer_2 + ...
            # note that the sigmoid of these are the factor _between_ layers
            # so the first entry is the ratio for the latent resnet of the first and second layers, etc.
            # e.g. if there are 3 layers, there are 2 ratios: l1:l2, l2:l3
            latent_resnet_update_params = torch.zeros(self.num_layers)
        else:
            latent_resnet_update_ratios = torch.as_tensor(
                latent_resnet_update_ratios, dtype=torch.get_default_dtype()
            )
            assert latent_resnet_update_ratios.min() > 0.0
            assert latent_resnet_update_ratios.min() < 1.0
            latent_resnet_update_params = torch.special.logit(
                latent_resnet_update_ratios
            )
            # The sigmoid is mostly saturated at ±6, keep it in a reasonable range
            latent_resnet_update_params.clamp_(-6.0, 6.0)
        assert latent_resnet_update_params.shape == (
            num_layers,
        ), f"There must be {num_layers} layer resnet update ratios for latents (layer0:layer1, layer1:layer2)"
        if latent_resnet_update_ratios_learnable:
            self._latent_resnet_update_params = torch.nn.Parameter(
                latent_resnet_update_params
            )
        else:
            self.register_buffer(
                "_latent_resnet_update_params", latent_resnet_update_params
            )
        
        if feature_resnet_update_ratios is None:
            # We initialize to zeros, which under the sigmoid() become 0.5
            # so 1/2 * layer_1 + 1/4 * layer_2 + ...
            # note that the sigmoid of these are the factor _between_ layers
            # so the first entry is the ratio for the latent resnet of the first and second layers, etc.
            # e.g. if there are 3 layers, there are 2 ratios: l1:l2, l2:l3
            feature_resnet_update_params = torch.zeros(self.num_layers)
        else:
            feature_resnet_update_ratios = torch.as_tensor(
                feature_resnet_update_ratios, dtype=torch.get_default_dtype()
            )
            assert feature_resnet_update_ratios.min() > 0.0
            assert feature_resnet_update_ratios.min() < 1.0
            feature_resnet_update_params = torch.special.logit(
                feature_resnet_update_ratios
            )
            # The sigmoid is mostly saturated at ±6, keep it in a reasonable range
            feature_resnet_update_params.clamp_(-6.0, 6.0)
        assert feature_resnet_update_params.shape == (
            num_layers,
        ), f"There must be {num_layers} layer resnet update ratios for features (layer0:layer1, layer1:layer2)"
        if feature_resnet_update_ratios_learnable:
            self._feature_resnet_update_params = torch.nn.Parameter(
                feature_resnet_update_params
            )
        else:
            self.register_buffer(
                "_feature_resnet_update_params", feature_resnet_update_params
            )

        # - Per-layer cutoffs -
        if per_layer_cutoffs is None:
            per_layer_cutoffs = torch.full((num_layers + 1,), r_max)
        self.register_buffer("per_layer_cutoffs", torch.as_tensor(per_layer_cutoffs))
        assert torch.all(self.per_layer_cutoffs <= r_max)
        assert self.per_layer_cutoffs.shape == (
            num_layers + 1,
        ), "Must be one per-layer cutoff for layer 0 and every layer for a total of {num_layers} cutoffs (the first applies to the two body latent, which is 'layer 0')"
        assert (
            self.per_layer_cutoffs[1:] <= self.per_layer_cutoffs[:-1]
        ).all(), "Per-layer cutoffs must be equal or decreasing"
        assert (
            self.per_layer_cutoffs.min() > 0
        ), "Per-layer cutoffs must be >0. To remove higher layers entirely, lower `num_layers`."
        self.register_buffer("_zero", torch.as_tensor(0.0))

        self.irreps_out.update(
            {
                self.inv_out_field: o3.Irreps(f"{self.latents[-1].out_features + env_embed_multiplicity * self._n_scalar_outs[layer_idx]}x0e"), # self.inv_out_irreps,
                self.eq_out_field: full_out_irreps, # self.eq_out_irreps,
            }
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]

        edge_attr = data[self.field]
        # pad edge_attr
        if self._input_pad > 0:
            edge_attr = torch.cat(
                (
                    edge_attr,
                    self._zero.expand(len(edge_attr), self._input_pad),
                ),
                dim=-1,
            )

        edge_length = data[AtomicDataDict.EDGE_LENGTH_KEY]
        num_edges: int = len(edge_attr)
        edge_invariants = data[self.edge_invariant_field]
        node_invariants = data[self.node_invariant_field]
        # pre-declare variables as Tensors for TorchScript
        scalars = self._zero
        coefficient_old = scalars
        coefficient_new = scalars
        # Initialize state
        latents = torch.zeros(
            (num_edges, self._latent_dim),
            dtype=edge_attr.dtype,
            device=edge_attr.device,
        )
        active_edges = torch.arange(
            num_edges,
            device=edge_attr.device,
        )

        # For the first layer, we use the input invariants:
        # The center and neighbor invariants and edge invariants
        latent_inputs_to_cat = [
            node_invariants[edge_center],
            node_invariants[edge_neighbor],
            edge_invariants,
        ]
        # The nonscalar features. Initially, the edge data.
        features = edge_attr

        # Node equivariant input features
        # if self.eq_node_in_feat_irreps.dim > 0:
        #     node_features = data[self.eq_node_in_feat_field]

        layer_index: int = 0
        # compute the sigmoids vectorized instead of each loop
        layer_update_coefficients = self._latent_resnet_update_params.sigmoid()

        layer_feature_update_coefficients = self._feature_resnet_update_params.sigmoid()

        # Vectorized precompute per layer cutoffs
        if self.cutoff_type == "cosine":
            cutoff_coeffs_all = cosine_cutoff(
                edge_length,
                self.per_layer_cutoffs,
                r_start_cos_ratio=self.r_start_cos_ratio,
            )
        elif self.cutoff_type == "polynomial":
            cutoff_coeffs_all = polynomial_cutoff(
                edge_length, self.per_layer_cutoffs, p=self.polynomial_cutoff_p
            )
        else:
            # This branch is unreachable (cutoff type is checked in __init__)
            # But TorchScript doesn't know that, so we need to make it explicitly
            # impossible to make it past so it doesn't throw
            # "cutoff_coeffs_all is not defined in the false branch"
            assert False, "Invalid cutoff type"

        # !!!! REMEMBER !!!! update final layer if update the code in main loop!!!
        # This goes through layer0, layer1, ..., layer_max-1
        for latent, env_embed_mlp, env_linear, linear, \
            prod, reshape_back, tp in zip(
            self.latents, self.env_embed_mlps, self.env_linears, self.linears, \
            self.products, self.reshape_back_modules, self.tps
        ):
            # Determine which edges are still in play
            cutoff_coeffs = cutoff_coeffs_all[layer_index]
            prev_mask = cutoff_coeffs[active_edges] > -1e-3
            active_edges = (cutoff_coeffs > -1e-3).nonzero().squeeze(-1)

            # Compute latents
            new_latents = latent(torch.cat(latent_inputs_to_cat, dim=-1)[prev_mask])
            # Apply cutoff, which propagates through to everything else
            new_latents = cutoff_coeffs[active_edges].unsqueeze(-1) * new_latents

            if self.latent_resnet and layer_index > 0:
                this_layer_update_coeff = layer_update_coefficients[layer_index - 1]
                # At init, we assume new and old to be approximately uncorrelated
                # Thus their variances add
                # we always want the latent space to be normalized to variance = 1.0,
                # because it is critical for learnability. Still, we want to preserve
                # the _relative_ magnitudes of the current latent and the residual update
                # to be controled by `this_layer_update_coeff`
                # Solving the simple system for the two coefficients:
                #   a^2 + b^2 = 1  (variances add)   &    a * this_layer_update_coeff = b
                # gives
                #   a = 1 / sqrt(1 + this_layer_update_coeff^2)  &  b = this_layer_update_coeff / sqrt(1 + this_layer_update_coeff^2)
                # rsqrt is reciprocal sqrt
                coefficient_old = torch.rsqrt(this_layer_update_coeff.square() + 1)
                coefficient_new = this_layer_update_coeff * coefficient_old
                # Residual update
                # Note that it only runs when there are latents to resnet with, so not at the first layer
                # index_add adds only to the edges for which we have something to contribute
                latents = torch.index_add(
                    coefficient_old * latents,
                    0,
                    active_edges,
                    coefficient_new * new_latents,
                )
            else:
                # Normal (non-residual) update
                # index_copy replaces, unlike index_add
                latents = torch.index_copy(latents, 0, active_edges, new_latents)

            # From the latents, compute the weights for active edges:
            weights = env_embed_mlp(latents[active_edges])
            w_index: int = 0

            if layer_index == 0:
                # embed initial edge
                env_w = weights.narrow(-1, w_index, self._env_weighter.weight_numel)
                w_index += self._env_weighter.weight_numel
                features_old = self._env_weighter(
                    features[prev_mask], env_w
                )  # features is edge_attr
            else:
                # just take the previous features that we still need
                features_old = features[prev_mask]

            # Extract weights for the environment builder
            env_w = weights.narrow(-1, w_index, self._env_weighter.weight_numel)
            w_index += self._env_weighter.weight_numel

            # Build the local environments
            # This local environment should only be a sum over neighbors
            # who are within the cutoff of the _current_ layer
            # Those are the active edges, which are the only ones we
            # have weights for (env_w) anyway.
            # So we mask out the edges in the sum:
            features_new = self._env_weighter(edge_attr[active_edges], env_w)
            features_old = features_old.reshape(features_new.shape[0], features_new.shape[1], -1)

            this_layer_update_coeff = layer_feature_update_coefficients[layer_index - 1]
            coefficient_old = torch.rsqrt(this_layer_update_coeff.square() + 1)
            coefficient_new = this_layer_update_coeff * coefficient_old

            features = torch.index_add(
                coefficient_old * features_old,
                0,
                active_edges,
                coefficient_new * features_new,
            )

            local_env_per_atom = scatter(
                features,
                edge_center[active_edges],
                dim=0,
            )
            if self.env_sum_normalizations.ndim < 2:
                # it's a scalar per layer
                norm_const = self.env_sum_normalizations[layer_index]
            else:
                # it's per type
                # get shape [N_atom, 1] for broadcasting
                norm_const = self.env_sum_normalizations[
                    layer_index, data[AtomicDataDict.ATOM_TYPE_KEY]
                ].unsqueeze(-1)
            local_env_per_atom = env_linear(local_env_per_atom * norm_const)

            expanded_features_per_atom: torch.Tensor = prod(
                node_feats=local_env_per_atom, sc=None, node_attrs=node_invariants[:edge_center[active_edges].max() + 1]
            )

            expanded_features_per_atom = reshape_back(expanded_features_per_atom)

            # Copy to get per-edge
            # Large allocation, but no better way to do this:
            features = expanded_features_per_atom[edge_center[active_edges]]
            # Now do the TP
            # recursively tp current features with the environment embeddings
            features = tp(features_old, features) / norm_const

            # Get invariants
            # features has shape [z][mul][k]
            # we know scalars are first
            scalars = features[:, :, : self._n_scalar_outs[layer_index]].reshape(
                features.shape[0], -1
            )

            # out_field_list.append(readout(scalars))

            # do the linear
            features = linear(features)

            # For layer2+, use the previous latents and scalars
            # This makes it deep
            latent_inputs_to_cat = [
                latents[active_edges],
                scalars,
            ]

            # increment counter
            layer_index += 1

        # - final layer -
        cutoff_coeffs = cutoff_coeffs_all[layer_index]
        prev_mask = cutoff_coeffs[active_edges] > -1e-3
        final_latent_input = torch.cat(latent_inputs_to_cat, dim=-1)[prev_mask]

        # ------------------------- #
        
        if self.readout_features and len(self.eq_out_irreps.ls) > 0:
            data[self.inv_out_field] = self.final_readout(final_latent_input)

            weights = self.final_latent(final_latent_input)
            data[self.eq_out_field] = self.final_linear(self.reshape_back_features(features), weights).squeeze(dim=-1)
        else:
            data[self.inv_out_field] = final_latent_input
            data[self.eq_out_field] = self.reshape_back_features(features)

        # ------------------------- #

        return data