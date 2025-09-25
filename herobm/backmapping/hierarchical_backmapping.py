import os
from pathlib import Path
import tempfile
import time
import torch
import shutil
import logging
import yaml

import numpy as np
import MDAnalysis as mda

import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore")

from os.path import join, dirname, basename
from typing import Callable, Dict, List, Optional, Tuple

from herobm.mapper.hierarchical_mapper import HierarchicalMapper
from herobm.utils import DataDict
from herobm.utils.backbone import MinimizeEnergy
from herobm.utils.io import replace_words_in_file


from geqtrain.utils import Config
from geqtrain.train.utils import evaluate_end_chunking_condition
from geqtrain.utils._global_options import register_all_fields
from geqtrain.data import AtomicDataDict
from geqtrain.data._build import dataset_from_config
from geqtrain.data.dataloader import DataLoader
from geqtrain.utils.deploy import load_deployed_model, CONFIG_KEY
from geqtrain.train.components.inference import run_inference


def load_model(model_path_str: str, device: str) -> Tuple[torch.jit.ScriptModule, Config, Dict]:
    """
    Loads a model, intelligently handling both deployed models and training checkpoints.

    Returns:
        A tuple containing:
        - The loaded PyTorch model (torch.jit.ScriptModule).
        - The model's configuration (Config object).
        - A dictionary of metadata (empty if loaded from training).
    """
    metadata = {k: "" for k in DataDict._ALL_METADATA_KEYS}
    try:
        # First, try to load as a deployed model
        model, metadata = load_deployed_model(model_path_str, device=device, freeze=False, extra_metadata=metadata)
        # The config is stored as a YAML string within the metadata
        model_config_str = metadata.get(CONFIG_KEY, "{}")
        model_config = Config(yaml.safe_load(model_config_str))
        logging.info("Successfully loaded deployed model and its metadata.")
    except (ValueError, RuntimeError, FileNotFoundError): 
        # If it fails, fall back to loading from a training session directory
        logging.warning("Could not load as a deployed model. Falling back to loading from a training session.")
        from geqtrain.train.components.checkpointing import CheckpointHandler
        model_path = Path(model_path_str)
        model, model_config = CheckpointHandler.load_model_from_training_session(
            traindir=model_path.parent, 
            model_name=model_path.name, 
            device=device,
        )
        metadata = {} # No metadata available for simple training checkpoints
    return model, model_config, metadata


class HierarchicalBackmapping:

    config: Dict[str, str]
    mapping: HierarchicalMapper
    model: torch.nn.Module

    input_folder: Optional[str]
    model_config: Dict[str, str]
    model_r_max: float

    minimiser: MinimizeEnergy

    def __init__(self, model: torch.nn.Module, model_config: Config, args_dict: Dict, preprocess_npz_func: Callable = None) -> None:
        self.model = model
        
        args_dict.update({
            "noinvariants": True, # Avoid computing angles and dihedrals when Coarse-graining atomistic input, they are used only for training.
            "cutoff": model_config.get("r_max"),
        })

        # Parse Input
        self.mapping = HierarchicalMapper(args_dict=args_dict)
        self.config = self.mapping.config

        self.output_folder = str(self.config.get("output"))
        self.device = self.config.get("device", "cpu")

        self.preprocess_npz_func = preprocess_npz_func
        self.config.update(
            {
                k: v
                for k, v in model_config.items()
                if k not in self.config
            }
        )

        # Initialize energy minimiser for reconstructing backbone
        self.minimiser = MinimizeEnergy()

        ### ------------------------------------------------------- ###
    
    @property
    def num_structures(self):
        return len(self.mapping.input_filenames)
    
    def map(self):
        for mapping in self.mapping():
            yield mapping

    def backmap(self, tolerance: float = 50., frame_idcs: Optional[List[int]] = None, chunking: int = 0):
        
        backmapped_filenames, backmapped_minimised_filenames, true_filenames, cg_filenames = [], [], [], []
        
        for input_filenames_index, (mapping, output_filename) in enumerate(self.map()):
            if self.output_folder is None:
                self.output_folder = dirname(output_filename)
            
            if frame_idcs is None:
                frame_idcs = range(len(mapping))
            n_frames = max(frame_idcs) + 1

            with tempfile.TemporaryDirectory() as tmp:
                npz_filename = join(tmp, 'data.npz')
                mapping.save_npz(filename=npz_filename, from_pos_unit='Angstrom', to_pos_unit='Angstrom')
                
                if self.preprocess_npz_func is not None:
                    ds = mapping.dataset
                    npz_ds = dict(np.load(npz_filename, allow_pickle=True))
                    updated_npz_ds = self.preprocess_npz_func(ds, npz_ds)
                    np.savez(npz_filename, **updated_npz_ds)
                    print(f"npz dataset {npz_filename} correctly preprocessed!")
                
                yaml_filename = join(tmp, 'test.yaml')
                shutil.copyfile(join(dirname(__file__), 'template.test.yaml'), yaml_filename)
                replace_words_in_file(
                    yaml_filename,
                    {
                        "{ROOT}": self.output_folder,
                        "{TEST_DATASET_INPUT}": npz_filename,
                    }
                )
                test_config = Config.from_file(yaml_filename, defaults={})
                self.config.update(test_config)
                self.config['chunking'] = chunking > 0
                self.config['batch_max_atoms'] = chunking
                register_all_fields(self.config)

                dataset = dataset_from_config(self.config, prefix="test")

                dataloader = DataLoader(
                    dataset=dataset,
                    shuffle=False,
                    batch_size=1, # Process one structure/frame at a time
                )

                _backmapped_u = [None] # Use a list to make it mutable

                pbar = tqdm(dataloader, desc=f"Backmapping {basename(output_filename)}")
                for batch_index, data in enumerate(pbar):
                    
                    # Dictionary to collect partial results from each chunk
                    results = {
                        DataDict.BEAD_POSITION: data[AtomicDataDict.POSITIONS_KEY].cpu().numpy(),
                        DataDict.BEAD2ATOM_RELATIVE_VECTORS_PRED: [],
                        DataDict.ATOM_POSITION_PRED: [],
                    }
                    
                    # This loop handles the chunking, just like in the training script
                    already_computed_nodes = None
                    num_batch_center_nodes = len(data[AtomicDataDict.EDGE_INDEX_KEY][0].unique())
                    while True:
                        out, _, center_nodes, _ = run_inference(
                            model=self.model,
                            data=data,
                            device=self.device,
                            config=self.config,
                            already_computed_nodes=already_computed_nodes
                        )
                        if out is not None:
                        
                            # Collect the partial results from this chunk
                            if AtomicDataDict.NODE_FEATURES_KEY in out:
                                results[DataDict.BEAD2ATOM_RELATIVE_VECTORS_PRED].append(out[AtomicDataDict.NODE_FEATURES_KEY].cpu().numpy())
                            if DataDict.ATOM_POSITION in out:
                                results[DataDict.ATOM_POSITION_PRED].append(np.expand_dims(out[DataDict.ATOM_POSITION].cpu().numpy(), axis=0))

                        # Update the state for the next chunk
                        already_computed_nodes = evaluate_end_chunking_condition(
                            already_computed_nodes, center_nodes, num_batch_center_nodes
                        )

                        if already_computed_nodes is None:
                            break # Finished all chunks for this batch

                    # Now, aggregate the results from all chunks for this batch
                    backmapping_dataset = self.mapping.dataset
                    
                    # Concatenate node features from all chunks.
                    aggregated_rvp = np.concatenate(results[DataDict.BEAD2ATOM_RELATIVE_VECTORS_PRED], axis=0)

                    # Stack atom positions (which contain NaNs) and take the mean, ignoring NaNs.
                    # This correctly merges the partial results from all chunks.
                    aggregated_app = np.nanmean(np.concatenate(results[DataDict.ATOM_POSITION_PRED], axis=0), axis=0, keepdims=True)

                    backmapping_dataset.update({
                        DataDict.BEAD_POSITION: results[DataDict.BEAD_POSITION],
                        DataDict.BEAD2ATOM_RELATIVE_VECTORS_PRED: aggregated_rvp,
                        DataDict.ATOM_POSITION_PRED: aggregated_app,
                    })
                    
                    backmapped_u, backmapped_filename, backmapped_minimised_filename, true_filename, cg_filename = self.to_pdb(
                        backmapping_dataset=backmapping_dataset,
                        input_filenames_index=input_filenames_index,
                        n_frames=n_frames,
                        frame_index=batch_index,
                        backmapped_u=_backmapped_u[0],
                        save_CG=True,
                        tolerance=tolerance,
                    )

                    _backmapped_u[0] = backmapped_u
                    backmapped_filenames.append(backmapped_filename)
                    backmapped_minimised_filenames.append(backmapped_minimised_filename)
                    if true_filename is not None:
                        true_filenames.append(true_filename)
                    cg_filenames.append(cg_filename)

        return backmapped_filenames, backmapped_minimised_filenames, true_filenames, cg_filenames

    def to_pdb(
            self,
            backmapping_dataset: Dict,
            input_filenames_index: int,
            n_frames: int,
            frame_index: int,
            backmapped_u: Optional[mda.Universe] = None,
            save_CG: bool = False,
            tolerance: float = 50.,
        ):
        print(f"Saving structures for frame {frame_index}...")
        t = time.time()
        os.makedirs(self.output_folder, exist_ok=True)
        prefix = basename(self.mapping.input_filenames[input_filenames_index])

        # Write pdb file of CG structure
        cg_filename = None
        if save_CG:
            cg_u = build_CG(backmapping_dataset, n_frames, self.mapping.selection.dimensions)
            cg_u.trajectory[frame_index]
            cg_sel = cg_u.select_atoms('all')
            cg_sel.positions = np.nan_to_num(backmapping_dataset[DataDict.BEAD_POSITION])
            cg_filename = join(self.output_folder, f"{prefix}.CG_{frame_index}.pdb")
            with mda.Writer(cg_filename, n_atoms=cg_sel.atoms.n_atoms) as w:
                w.write(cg_sel.atoms)
        
        if backmapped_u is None:
            backmapped_u = build_universe(backmapping_dataset, n_frames, self.mapping.selection.dimensions)
        backmapped_u.trajectory[frame_index]
        positions_pred = backmapping_dataset[DataDict.ATOM_POSITION_PRED][0]
        
        # Write pdb of true atomistic structure (if present)
        true_filename = None
        if DataDict.ATOM_POSITION in backmapping_dataset and not np.all(np.isnan(backmapping_dataset[DataDict.ATOM_POSITION])):
            true_sel = backmapped_u.select_atoms('all')
            true_positions = backmapping_dataset[DataDict.ATOM_POSITION][frame_index]
            true_sel.positions = true_positions[~np.any(np.isnan(positions_pred), axis=-1)]
            true_filename = join(self.output_folder, f"{prefix}.true_{frame_index}.pdb")
            with mda.Writer(true_filename, n_atoms=backmapped_u.atoms.n_atoms) as w:
                w.write(true_sel.atoms)

        # Write pdb of backmapped structure
        backmapped_sel = backmapped_u.select_atoms('all')
        backmapped_sel.positions = positions_pred[~np.any(np.isnan(positions_pred), axis=-1)]
        backmapped_filename = join(self.output_folder, f"{prefix}.backmapped_{frame_index}.pdb")
        with mda.Writer(backmapped_filename, n_atoms=backmapped_u.atoms.n_atoms) as w:
            w.write(backmapped_sel.atoms)
        
        backmapped_minimised_filename = None
        # Write pdb of minimised structure
        if tolerance is not None and tolerance > 0:
            try:
                from herobm.utils.pdbFixer import fixPDB
                from herobm.utils.minimisation import minimise_impl
                topology, positions = fixPDB(backmapped_filename, addHydrogens=True)
                backmapped_minimised_filename = join(self.output_folder, f"{prefix}.backmapped_min_{frame_index}.pdb")

                minimise_impl(
                    topology,
                    positions,
                    backmapped_minimised_filename,
                    restrain_atoms=[],
                    tolerance=tolerance,
                )
            except Exception as e:
                logging.warning(f"Failed to minimise structure {backmapped_filename}: {e}")
                pass

        print(f"Finished saving. Time: {time.time() - t:.2f}s")

        return backmapped_u, backmapped_filename, backmapped_minimised_filename, true_filename, cg_filename

def build_CG(
    backmapping_dataset: dict,
    n_frames: int,
    box_dimensions,
) -> mda.Universe:
    CG_u = mda.Universe.empty(
        n_atoms =       backmapping_dataset[DataDict.NUM_BEADS],
        n_residues =    backmapping_dataset[DataDict.NUM_RESIDUES],
        n_segments =    len(np.unique(backmapping_dataset[DataDict.BEAD_SEGIDS])),
        atom_resindex = close_gaps(backmapping_dataset[DataDict.BEAD_RESIDCS]),
        trajectory =    True, # necessary for adding coordinates
    )
    CG_u.add_TopologyAttr('name',     backmapping_dataset[DataDict.BEAD_NAMES])
    CG_u.add_TopologyAttr('type',     backmapping_dataset[DataDict.BEAD_TYPES])
    CG_u.add_TopologyAttr('resname',  backmapping_dataset[DataDict.RESNAMES])
    CG_u.add_TopologyAttr('resid',    backmapping_dataset[DataDict.RESNUMBERS])
    CG_u.add_TopologyAttr('chainIDs', backmapping_dataset[DataDict.BEAD_SEGIDS])
    
    coordinates = np.empty((
        n_frames,  # number of frames
        backmapping_dataset[DataDict.NUM_BEADS],
        3,
    ))
    CG_u.load_new(coordinates, order='fac')
    add_box(CG_u, box_dimensions)

    return CG_u

def add_box(u: mda.Universe, box_dimensions):
    if box_dimensions is None:
        from MDAnalysis.transformations import set_dimensions
        dim = np.array([100., 100., 100., 90, 90, 90])
        transform = set_dimensions(dim)
        u.trajectory.add_transformations(transform)
    else:
        u.dimensions = box_dimensions

def close_gaps(arr: np.ndarray):
    u = np.unique(arr)
    w = np.arange(0, len(u))
    for u_elem, w_elem in zip(u, w):
        arr[arr == u_elem] = w_elem
    return arr
    
def build_universe(
    backmapping_dataset,
    n_frames,
    box_dimensions,
):
    nan_filter = ~np.any(np.isnan(backmapping_dataset[DataDict.ATOM_POSITION_PRED]), axis=-1)
    nan_filter = np.min(nan_filter, axis=0)
    num_atoms = nan_filter.sum()
    backmapped_u = mda.Universe.empty(
        n_atoms =       num_atoms,
        n_residues =    backmapping_dataset[DataDict.NUM_RESIDUES],
        n_segments =    len(np.unique(backmapping_dataset[DataDict.ATOM_SEGIDS])),
        atom_resindex = close_gaps(backmapping_dataset[DataDict.ATOM_RESIDCS][nan_filter]),
        trajectory    = True # necessary for adding coordinates
    )
    coordinates = np.empty((
        n_frames,  # number of frames
        num_atoms,
        3,
    ))
    backmapped_u.load_new(coordinates, order='fac')
    add_box(backmapped_u, box_dimensions)

    backmapped_u.add_TopologyAttr('name',     backmapping_dataset[DataDict.ATOM_NAMES][nan_filter])
    backmapped_u.add_TopologyAttr('type',     backmapping_dataset[DataDict.ATOM_TYPES][nan_filter])
    backmapped_u.add_TopologyAttr('resname',  backmapping_dataset[DataDict.RESNAMES])
    backmapped_u.add_TopologyAttr('resid',    backmapping_dataset[DataDict.RESNUMBERS])
    backmapped_u.add_TopologyAttr('chainIDs', backmapping_dataset[DataDict.ATOM_SEGIDS][nan_filter])

    return backmapped_u

def get_edge_index(positions: torch.Tensor, r_max: float):
    dist_matrix = torch.norm(positions[:, None, ...] - positions[None, ...], dim=-1).fill_diagonal_(torch.inf)
    return torch.argwhere(dist_matrix <= r_max).T.long()

