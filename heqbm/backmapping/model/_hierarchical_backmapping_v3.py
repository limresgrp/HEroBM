import logging
from typing import Optional

from e3nn import o3

from torch.utils.data import ConcatDataset

from nequip.model import builder_utils
from nequip.data import AtomicDataDict
from nequip.nn import SequentialGraphNetwork
from nequip.nn.radial_basis import BesselBasis
from nequip.nn.embedding import (
    OneHotAtomEncoding,
    SphericalHarmonicEdgeAttrs,
)

from heqbm.backmapping.allegro._keys import (
    INVARIANT_EDGE_FEATURES,
    EQUIVARIANT_EDGE_FEATURES,
    INVARIANT_ATOM_FEATURES,
    EQUIVARIANT_ATOM_FEATURES,
    EQUIVARIANT_ATOM_INPUT_FEATURES,
    EQUIVARIANT_EDGE_LENGTH_FEATURES,
    EQUIVARIANT_ATOM_LENGTH_FEATURES,
)
from heqbm.backmapping.allegro.nn import (
    NormalizedBasis,
    EdgewiseEnergySum,
    ExponentialScalarMLPFunction,
)

from heqbm.backmapping.nn import (
    RadialBasisClassEdgeEncoding,
    HierarchicalBackmappingV2Module,
    HierarchicalBackmappingReadoutModule,
)


def HierarchicalBackmappingV3(config, initialize: bool, dataset: Optional[ConcatDataset] = None):
    logging.debug("Building HEqBM model...")

    # Handle avg num neighbors auto
    builder_utils.add_avg_num_neighbors(
        config=config, initialize=initialize, dataset=dataset
    )

    # Handle simple irreps
    if "l_max" in config:
        l_max = int(config["l_max"])
        parity_setting = config["parity"]
        assert parity_setting in ("o3_full", "o3_restricted", "so3")
        irreps_edge_sh = repr(
            o3.Irreps.spherical_harmonics(
                l_max, p=(1 if parity_setting == "so3" else -1)
            )
        )
        nonscalars_include_parity = parity_setting == "o3_full"
        # check consistant
        assert config.get("irreps_edge_sh", irreps_edge_sh) == irreps_edge_sh
        assert (
            config.get("nonscalars_include_parity", nonscalars_include_parity)
            == nonscalars_include_parity
        )
        config["irreps_edge_sh"] = irreps_edge_sh
        config["nonscalars_include_parity"] = nonscalars_include_parity

    layers = {
        # -- Encode --
        # Get various edge invariants
        "one_hot": (
            OneHotAtomEncoding,
            dict(
                node_input_features=config.get("node_input_features", [])
            )
        ),
        "radial_basis": (
            RadialBasisClassEdgeEncoding,
            dict(
                basis=(
                    NormalizedBasis
                    if config.get("normalize_basis", True)
                    else BesselBasis
                ),
                out_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
            ),
        ),
        # Get edge nonscalars
        "spharm": SphericalHarmonicEdgeAttrs,
        # The core model:
        "core": (
            HierarchicalBackmappingV2Module,
            dict(
                field=AtomicDataDict.EDGE_ATTRS_KEY,  # initial input is the edge SH
                edge_invariant_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
                node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
                env_embed=ExponentialScalarMLPFunction,
                inv_out_field=INVARIANT_EDGE_FEATURES,
                eq_out_field=EQUIVARIANT_EDGE_FEATURES,
                readout_features=config.get("readout_pre_pooling", False),
                inv_out_irreps=o3.Irreps(config.get("inv_out_irreps", "2x0e")),
                eq_out_irreps=o3.Irreps(config.get("eq_out_irreps", "4x1o")),
                eq_node_in_feat_field=EQUIVARIANT_ATOM_INPUT_FEATURES,
                eq_node_in_feat_irreps=o3.Irreps(config.get("eq_node_in_irreps", "0x1o")),
            ),
        ),
        # Sum edgewise energies -> per-atom features:

        "per_atom_invariant": (
            EdgewiseEnergySum,
            dict(
                field=INVARIANT_EDGE_FEATURES,
                out_field=INVARIANT_ATOM_FEATURES,
                average_pooling=config.get("inv_average_pooling", False),
            ),
        ),

        "per_atom_equivariant": (
            EdgewiseEnergySum,
            dict(
                field=EQUIVARIANT_EDGE_FEATURES,
                out_field=EQUIVARIANT_ATOM_FEATURES,
                average_pooling=config.get("eq_average_pooling", False),
            ),
        ),

        "per_atom_equivariant_length": (
            EdgewiseEnergySum,
            dict(
                field=EQUIVARIANT_EDGE_LENGTH_FEATURES,
                out_field=EQUIVARIANT_ATOM_LENGTH_FEATURES,
            ),
        ),

        "readout": (
            HierarchicalBackmappingReadoutModule,
            dict(
                inv_field=INVARIANT_ATOM_FEATURES,
                eq_field=EQUIVARIANT_ATOM_FEATURES,
                readout_features=not config.get("readout_pre_pooling", False),
                inv_out_irreps=o3.Irreps(config.get("inv_out_irreps", "2x0e")),
                eq_out_irreps=o3.Irreps(config.get("eq_out_irreps", "4x1o")),
                normalize_out_features=config.get("normalize_out_features", True)
            ),
        )
    }

    model = SequentialGraphNetwork.from_parameters(shared_params=config, layers=layers)

    return model