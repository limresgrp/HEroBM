from geqtrain.nn import GraphModuleMixin
from herobm.backmapping.nn import HierarchicalReconstructionModule


def HierarchicalReconstruction(config, model: GraphModuleMixin) -> HierarchicalReconstructionModule:

    return HierarchicalReconstructionModule(
        func=model,
        num_types=config['num_types'],
        normalize_b2a_rel_vec=config.get('normalize_b2a_rel_vec', True),
    )