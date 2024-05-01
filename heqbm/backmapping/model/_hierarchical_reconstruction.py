from geqtrain.nn import GraphModuleMixin
from heqbm.backmapping.nn import HierarchicalReconstructionModule
from heqbm.backmapping.allegro._keys import ATOM_POSITIONS
from geqtrain.data import AtomicDataDict


def HierarchicalReconstruction(
        config,
        model: GraphModuleMixin,
        in_field: str = AtomicDataDict.NODE_OUTPUT_KEY,
        out_field: str = ATOM_POSITIONS,
    ) -> HierarchicalReconstructionModule:
    
    if in_field not in model.irreps_out:
        raise ValueError(f"This model misses outputs for field {in_field}.")

    return HierarchicalReconstructionModule(
        func=model,
        num_types=config['num_types'],
        in_field=in_field,
        out_field=out_field,
    )