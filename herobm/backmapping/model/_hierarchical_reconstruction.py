from geqtrain.nn import GraphModuleMixin
from herobm.backmapping.nn import HierarchicalReconstructionModule
from geqtrain.data import AtomicDataDict

from herobm.utils import DataDict


def HierarchicalReconstruction(
        config,
        model: GraphModuleMixin,
        in_field: str = AtomicDataDict.NODE_OUTPUT_KEY,
        out_field: str = DataDict.ATOM_POSITION,
    ) -> HierarchicalReconstructionModule:
    
    if in_field not in model.irreps_out:
        raise ValueError(f"This model misses outputs for field {in_field}.")

    return HierarchicalReconstructionModule(
        func=model,
        num_types=config['num_types'],
        in_field=in_field,
        out_field=out_field,
    )