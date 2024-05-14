from geqtrain.nn import GraphModuleMixin
from nequip.data import AtomicDataDict
from herobm.backmapping.nn import NormOutputLWise


def NormLWise(
        config,
        model: GraphModuleMixin,
        field: str = AtomicDataDict.PER_ATOM_ENERGY_KEY,
    ) -> NormOutputLWise:
    
    if field not in model.irreps_out:
        raise ValueError(f"This model misses outputs for field {field}.")

    return NormOutputLWise(
        func=model,
        field=field,
    )