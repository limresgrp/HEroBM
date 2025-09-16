import logging
from geqtrain.nn import SequentialGraphNetwork
from herobm.backmapping.nn import HierarchicalReconstructionModule


def HierarchicalReconstruction(config, model: SequentialGraphNetwork) -> HierarchicalReconstructionModule:

    logging.info("--- Building HierarchicalReconstruction Module ---")
    layers: dict = {
        "wrapped_model": model,
        "hierarchical_reconstruction": HierarchicalReconstructionModule,
    }

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )