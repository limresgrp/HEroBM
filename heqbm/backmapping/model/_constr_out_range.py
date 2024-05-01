from geqtrain.nn import GraphModuleMixin
from nequip.data import AtomicDataDict
from heqbm.backmapping.nn import ConstrOutputRange


def ConstrOutRange(
        config,
        model: GraphModuleMixin,
        field: str = AtomicDataDict.PER_ATOM_ENERGY_KEY,
    ) -> ConstrOutputRange:
    
    if field not in model.irreps_out:
        raise ValueError(f"This model misses outputs for field {field}.")

    return ConstrOutputRange(
        func=model,
        field=field,
    )