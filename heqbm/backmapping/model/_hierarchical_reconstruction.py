from nequip.nn import GraphModuleMixin
from heqbm.backmapping.nn import HierarchicalReconstrucitonModule
from heqbm.backmapping.allegro._keys import (
    EQUIVARIANT_ATOM_FEATURES,
    ATOM_POSITIONS,
)


def HierarchicalReconstruction(
        config,
        model: GraphModuleMixin,
        in_field: str = EQUIVARIANT_ATOM_FEATURES,
        out_field: str = ATOM_POSITIONS,
    ) -> HierarchicalReconstrucitonModule:
    
    if in_field not in model.irreps_out:
        raise ValueError(f"This model misses outputs for field {in_field}.")

    return HierarchicalReconstrucitonModule(
        func=model,
        in_field=in_field,
        out_field=out_field,
    )