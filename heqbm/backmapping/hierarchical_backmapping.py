import glob
import os
import time
import yaml
import torch

import numpy as np
import MDAnalysis as mda
import copy

from os.path import basename
from e3nn import o3
from pathlib import Path
from typing import Dict, List, Optional
from itertools import groupby

from heqbm.mapper.hierarchical_mapper import HierarchicalMapper
from heqbm.backmapping.nn.quaternions import get_quaternions, qv_mult
from heqbm.utils import DataDict
from heqbm.utils.geometry import get_RMSD, get_angles, get_dihedrals
from heqbm.utils.backbone import cat_interleave, MinimizeEnergy
from heqbm.utils.plotting import plot_cg
from heqbm.utils.pdbFixer import fixPDB
from heqbm.utils.minimisation import minimise_impl

from openmm.app import PDBFile

from heqbm.backmapping.allegro._keys import (
    INVARIANT_ATOM_FEATURES,
    EQUIVARIANT_ATOM_FEATURES,
    ATOM_POSITIONS,
)

from nequip.utils import Config
from nequip.utils._global_options import _set_global_options
from nequip.train import Trainer
from nequip.data import AtomicDataDict
from nequip.scripts.deploy import load_deployed_model, R_MAX_KEY


class HierarchicalBackmapping:

    config: Dict[str, str]
    mapping: HierarchicalMapper

    input_folder: Optional[str]
    input_filenames: List[str]
    model_config: Dict[str, str]
    model_r_max: float

    minimizer: MinimizeEnergy

    def __init__(self, args_dict) -> None:
        
        ### --------------------- ###
        self.config: Dict[str, str] = dict()

        config = args_dict.pop("config", None)
        if config is not None:
            self.config.update(yaml.safe_load(Path(config).read_text()))

        args_dict = {key: value for key, value in args_dict.items() if value is not None}
        self.config.update(args_dict)

        def parse_slice(slice_str):
            parts = slice_str.split(':')

            start = None if parts[0] == '' else int(parts[0])
            stop = None if parts[1] == '' else int(parts[1])
            step = None if len(parts) == 2 or parts[2] == '' else int(parts[2])

            return slice(start, stop, step)

        if self.config.get("trajslice", None) is not None:
            self.config["trajslice"] = parse_slice(self.config["trajslice"])


        ### Parse Input ###
        
        self.mapping = HierarchicalMapper(config=self.config)

        self.output_folder = self.config.get("output")
        input = self.config.get("input")
        if os.path.isdir(input):
            self.input_folder = input
            input_format = self.config.get("inputformat", "*")
            self.input_filenames = list(glob.glob(os.path.join(self.input_folder, f"*.{input_format}")))
        else:
            self.input_folder = None
            self.input_filename = input
            self.input_filenames = [self.input_filename]

        ### Load Model ###
            
        print("Loading model...")

        model = self.config.get("model", None)
        if model is None:
            raise Exception("You did not provide the 'model' input parameter.")
        if Path(model).suffix not in [".yaml", ".json"]:
            try:
                deployed_model = os.path.join(os.path.dirname(__file__), '..', '..', deployed_model)
                self.model, metadata = load_deployed_model(
                    deployed_model,
                    device=self.config.get("device"),
                    set_global_options=True,  # don't warn that setting
                )
                # the global settings for a deployed model are set by
                # set_global_options in the call to load_deployed_model above
                self.model_r_max = float(metadata[R_MAX_KEY])
                print("Loaded deployed model")
            except Exception as e:
                raise Exception(
                    f"""Could not load {model}."""
                ) from e
        else:
            model_config_file = os.path.join(os.path.dirname(__file__), '..', '..', model)
            model_config: Optional[Dict[str, str]] = yaml.load(
                Path(model_config_file).read_text(), Loader=yaml.Loader
            ) if model_config_file is not None else None
            self.model, training_model_config = load_model(
                model_dir=None,
                model_config=model_config,
                config=self.config,
            )
            self.model_r_max = float(training_model_config[R_MAX_KEY])
            print("Loaded model from training session.")
        
        self.model.eval()

        ### Initialize energy minimizer for reconstructing backbone ###

        self.minimizer = MinimizeEnergy()

        ### ------------------------------------------------------- ###
    
    @property
    def num_structures(self):
        return len(self.input_filenames)
    
    def plot(self, frame_index=0):
        plot_cg(
            dataset=self.mapping.dataset,
            frame_index=frame_index
        )
    
    def backmap_scl(self, backmapping_dataset: Dict):
        reconstructed_atom_pos = run_b2a_rel_vec_backmapping(dataset=backmapping_dataset, use_predicted_b2a_rel_vec=True)
        backmapping_dataset[DataDict.ATOM_POSITION_PRED] = reconstructed_atom_pos
        return backmapping_dataset
    
    def adjust_beads(self, backmapping_dataset: Dict, device: str = 'cpu', verbose: bool = False):
        backmapping_dataset[DataDict.BEAD_POSITION_ORIGINAL] = np.copy(backmapping_dataset[DataDict.BEAD_POSITION])
        minimizer_data = build_minimizer_data(dataset=backmapping_dataset, device=device)

        # unlock_ca = backmapping_dataset[DataDict.CA_BEAD_POSITION].shape[-2] > 0 and (not self.config.get("lock_ca", True))
        # if unlock_ca:
        #     self.minimizer.minimise(
        #         data=minimizer_data,
        #         dtau=self.config.get("bb_minimisation_dtau", 1e-1),
        #         eps=self.config.get("bb_minimisation_eps_initial", 1e-2),
        #         minimise_dih=False,
        #         unlock_ca=True,
        #         verbose=verbose,
        #     )
        
        #     ca_shifts = minimizer_data["pos"][1::3].detach().cpu().numpy() - backmapping_dataset[DataDict.BEAD_POSITION][0, backmapping_dataset[DataDict.CA_BEAD_IDCS]]
        #     residue_shifts = np.repeat(ca_shifts, minimizer_data["per_residue_beads"].cpu().numpy(), axis=0)
        #     backmapping_dataset[DataDict.BEAD_POSITION][0] += residue_shifts
        return backmapping_dataset, minimizer_data
    
    def optimize_backbone(
            self,
            backmapping_dataset: Dict,
            minimizer_data: Dict,
            lock_ca: bool=False,
            minimise_dih: bool = False,
            device: str = 'cpu',
            verbose: bool = False,
    ):

        # Add predicted dihedrals as dihedral equilibrium values
        minimizer_data = update_minimizer_data(
            minimizer_data=minimizer_data,
            dataset=backmapping_dataset,
            use_only_bb_beads=self.config.get("bb_use_only_bb_beads", False),
            device=device,
        )
        
        # Step 1: initial minimization: adjust bonds, angles and Omega dihedral
        self.minimizer.minimise(
            data=minimizer_data,
            dtau=self.config.get("bb_minimisation_dtau", 1e-1),
            eps=self.config.get("bb_initial_minimisation_eps", 1e-2),
            trace_every=self.config.get("trace_every", 0),
            minimise_dih=False,
            lock_ca=lock_ca,
            verbose=verbose,
        )

        if minimise_dih:
            # Step 2: rotate N and C atoms around the CA-CA axis to find global minimum basin
            minimizer_data = self.rotate_dihedrals_to_minimize_energy(minimizer_data=minimizer_data, dataset=backmapping_dataset)
            
            # Step 3: final minimization: find global minimum
            self.minimizer.minimise(
                data=minimizer_data,
                dtau=self.config.get("bb_minimisation_dtau", 1e-1),
                eps=self.config.get("bb_minimisation_eps", 1e-3),
                trace_every=self.config.get("trace_every", 0),
                minimise_dih=True,
                lock_ca=lock_ca,
                verbose=verbose,
            )

        pos = minimizer_data["pos"].detach().cpu().numpy()
        # real_bb_atom_idcs = minimizer_data["real_bb_atom_idcs"].detach().cpu().numpy()
        backmapping_dataset[DataDict.BB_ATOM_POSITION_PRED] = pos # pos[real_bb_atom_idcs]
        # if DataDict.BB_ATOM_POSITION_MINIMISATION_TRAJ in minimizer_data:
        #     backmapping_dataset[DataDict.BB_ATOM_POSITION_MINIMISATION_TRAJ] = np.stack(
        #         minimizer_data[DataDict.BB_ATOM_POSITION_MINIMISATION_TRAJ],
        #         axis=0
        #     )[:, real_bb_atom_idcs]
        return backmapping_dataset, minimizer_data
        
    def map(
        self,
        index: int,
        skip_if_existent: bool = True,
    ):
        if self.input_folder is None:
            self.mapping.map(self.config)
        else:
            self.input_filename = self.input_filenames[index]
            self.config["input"] = self.input_filename
            self.config["inputtraj"] = []
            self.config["output"] = os.path.join(self.output_folder, '.'.join(basename(self.input_filename).split('.')[:-1]))
            if skip_if_existent and os.path.isdir(self.config["output"]):
                return True
            print(f"Mapping structure {self.input_filename}")
            self.mapping.map(self.config)
        return False
    
    def backmap(
        self,
        frame_index: int,
        verbose: bool = False,
        lock_ca: bool = False,
        minimise_dih: bool = False,
        optimize_backbone: Optional[bool] = None,
    ):
        print(f"Backmapping structure {self.input_filename}")
        backmapping_dataset = self.mapping.dataset
        for k in [
            DataDict.ATOM_POSITION,
            DataDict.ATOM_FORCES,
            DataDict.BEAD_POSITION,
            DataDict.BB_ATOM_POSITION,
            DataDict.BB_PHIPSI,
            DataDict.CA_ATOM_POSITION,
            DataDict.CA_BEAD_POSITION,
            DataDict.BEAD2ATOM_RELATIVE_VECTORS,
        ]:
            if k in backmapping_dataset:
                backmapping_dataset[k] = backmapping_dataset[k][frame_index:frame_index+1]
        
        if optimize_backbone is None:
            optimize_backbone = self.config.get("optimizebackbone", True)
        if DataDict.CA_BEAD_IDCS not in backmapping_dataset or (backmapping_dataset[DataDict.CA_BEAD_IDCS].sum() == 0):
            optimize_backbone = False
        
        device = self.config.get("device", "cpu")
        
        # Adjust CG CA bead positions, if they don't pass structure quality checks
        backmapping_dataset, minimizer_data = self.adjust_beads(
            backmapping_dataset=backmapping_dataset,
            device=device,
            verbose=verbose,
        )

        print("Predicting distance vectors using HEqBM ENN & reconstructing atomistic structure...")
        t = time.time()

        # Predict dihedrals and bead2atom relative vectors
        backmapping_dataset = run_backmapping_inference(
            dataset=backmapping_dataset,
            model=self.model,
            r_max=self.model_r_max,
            use_only_bb_beads=self.config.get("bb_use_only_bb_beads", False),
            device=device,
        )
        
        print(f"Finished. Time: {time.time() - t}")

        if DataDict.BEAD2ATOM_RELATIVE_VECTORS in backmapping_dataset:
            bb_fltr = np.array([bn in ['BB'] for bn in backmapping_dataset[DataDict.BEAD_NAMES]])
            b2a_rev_vec = backmapping_dataset[DataDict.BEAD2ATOM_RELATIVE_VECTORS].copy()

            try:
                print(
                    "RMSD on all b2a relative vectors:", 
                    get_RMSD(backmapping_dataset[DataDict.BEAD2ATOM_RELATIVE_VECTORS_PRED], b2a_rev_vec, ignore_zeroes=True)
                )

                b2a_rev_vec[:, bb_fltr] = 0. # We are not interested in b2a_rev_vec of backbone

                print(
                    "RMSD on side-chain b2a relative vectors:", 
                    get_RMSD(backmapping_dataset[DataDict.BEAD2ATOM_RELATIVE_VECTORS_PRED], b2a_rev_vec, ignore_zeroes=True)
                )
            except:
                pass

        # Backmap backbone
        if optimize_backbone:
            print("Optimizing backbone...")
            t = time.time()

            backmapping_dataset, minimizer_data = self.optimize_backbone(
                backmapping_dataset=backmapping_dataset,
                minimizer_data=minimizer_data,
                lock_ca=lock_ca,
                minimise_dih=minimise_dih,
                device=device,
                verbose=verbose,
            )
        
        # Backmap side chains & ligands
        # if DataDict.ATOM_POSITION_PRED not in backmapping_dataset:
        #     backmapping_dataset = self.backmap_scl(backmapping_dataset=backmapping_dataset)

        if optimize_backbone:
            n_atom_idcs = minimizer_data["n_atom_idcs"].cpu().numpy()
            ca_atom_idcs = backmapping_dataset[DataDict.CA_ATOM_IDCS]
            c_atom_idcs = minimizer_data["c_atom_idcs"].cpu().numpy()
            o_atom_idcs = minimizer_data["o_atom_idcs"].cpu().numpy()
            
            backmapping_dataset[DataDict.ATOM_POSITION_PRED][0, n_atom_idcs + ca_atom_idcs + c_atom_idcs + o_atom_idcs] = backmapping_dataset[DataDict.BB_ATOM_POSITION_PRED]
            # if minimise_dih:
            #     backmapping_dataset = adjust_bb_oxygens(dataset=backmapping_dataset)

                # if DataDict.BB_ATOM_POSITION_MINIMISATION_TRAJ in backmapping_dataset:
                #     backmapping_dataset[DataDict.ATOM_POSITION_MINIMISATION_TRAJ] = np.repeat(
                #         backmapping_dataset[DataDict.ATOM_POSITION_PRED],
                #         len(backmapping_dataset[DataDict.BB_ATOM_POSITION_MINIMISATION_TRAJ]),
                #         axis=0
                #     )
                #     backmapping_dataset[DataDict.ATOM_POSITION_MINIMISATION_TRAJ][:, n_atom_idcs + ca_atom_idcs + c_atom_idcs] = backmapping_dataset[DataDict.BB_ATOM_POSITION_MINIMISATION_TRAJ]
                #     backmapping_dataset = adjust_bb_oxygens(dataset=backmapping_dataset, position_key=DataDict.ATOM_POSITION_MINIMISATION_TRAJ)

        if DataDict.ATOM_POSITION in backmapping_dataset:
            try:
                print(f"RMSD All: {get_RMSD(backmapping_dataset[DataDict.ATOM_POSITION_PRED], backmapping_dataset[DataDict.ATOM_POSITION], ignore_nan=True):4.3f} Angstrom")
                no_bb_fltr = np.array([an not in ["CA", "C", "N", "O"] for an in backmapping_dataset[DataDict.ATOM_NAMES]])
                print(f"RMSD on Side-Chains: {get_RMSD(backmapping_dataset[DataDict.ATOM_POSITION_PRED], backmapping_dataset[DataDict.ATOM_POSITION], fltr=no_bb_fltr, ignore_nan=True):4.3f} Angstrom")
                bb_fltr = np.array([an in ["CA", "C", "N", "O"] for an in backmapping_dataset[DataDict.ATOM_NAMES]])
                print(f"RMSD on Backbone: {get_RMSD(backmapping_dataset[DataDict.ATOM_POSITION_PRED], backmapping_dataset[DataDict.ATOM_POSITION], fltr=bb_fltr, ignore_nan=True):4.3f} Angstrom")
            except:
                pass
        
        print(f"Finished. Time: {time.time() - t}")
        
        return backmapping_dataset
    
    def rotate_dihedrals_to_minimize_energy(self, minimizer_data: Dict, dataset: Dict):
        pred_pos = minimizer_data["pos"].detach().clone()
        ca_pos = torch.from_numpy(dataset[DataDict.CA_BEAD_POSITION][0]).to(pred_pos.device)

        pi_quarters_rotated_pred_pos = rotate_residue_dihedrals(pos=pred_pos, ca_pos=ca_pos, angle=np.pi/4)
        pi_halves_rotated_pred_pos = rotate_residue_dihedrals(pos=pred_pos, ca_pos=ca_pos, angle=np.pi/2)
        pi_three_quarters_rotated_pred_pos = rotate_residue_dihedrals(pos=pred_pos, ca_pos=ca_pos, angle=np.pi*3/4)
        pi_rotated_pred_pos = rotate_residue_dihedrals(pos=pred_pos, ca_pos=ca_pos, angle=np.pi)
        minus_pi_quarters_rotated_pred_pos = rotate_residue_dihedrals(pos=pred_pos, ca_pos=ca_pos, angle=-np.pi/4)
        minus_pi_halves_rotated_pred_pos = rotate_residue_dihedrals(pos=pred_pos, ca_pos=ca_pos, angle=-np.pi/2)
        minus_pi_three_quarters_rotated_pred_pos = rotate_residue_dihedrals(pos=pred_pos, ca_pos=ca_pos, angle=-np.pi*3/4)
        all_rotated_pos = [
            pi_quarters_rotated_pred_pos,
            pi_halves_rotated_pred_pos,
            pi_three_quarters_rotated_pred_pos,
            pi_rotated_pred_pos,
            minus_pi_quarters_rotated_pred_pos,
            minus_pi_halves_rotated_pred_pos,
            minus_pi_three_quarters_rotated_pred_pos,
        ]

        # Rotate and evaluate one residue at a time
        updated_pos = pred_pos.clone()
        baseline_energy = self.minimizer.evaluate_dihedral_energy(minimizer_data, pos=pred_pos)
        for x in range(len(ca_pos)-1):
            temp_updated_pos = updated_pos.clone()
            C_id = x*4+2
            N_id = x*4+4
            energies = []
            for rotated_pos in all_rotated_pos:
                temp_updated_pos[C_id] = rotated_pos[C_id]
                temp_updated_pos[N_id] = rotated_pos[N_id]
                energies.append(self.minimizer.evaluate_dihedral_energy(minimizer_data, pos=temp_updated_pos))
            min_energy_id = torch.stack(energies).argmin()
            best_energy = energies[min_energy_id]
            if best_energy < baseline_energy:
                baseline_energy = best_energy
                updated_pos[C_id] = all_rotated_pos[min_energy_id][C_id]
                updated_pos[N_id] = all_rotated_pos[min_energy_id][N_id]

        minimizer_data["pos"] = updated_pos
        return minimizer_data
    
    def to_pdb(
            self,
            backmapping_dataset: Dict,
            n_frames: int,
            frame_index: int,
            previous_u: Optional[mda.Universe] = None,
            save_CG: bool = False,
        ):
        
        print(f"Saving structures...")
        t = time.time()
        
        u = mda.Universe(self.config.get("input"))

        os.makedirs(self.output_folder, exist_ok=True)

        # Write pdb file of CG structure
        if save_CG:
            CG_u = build_CG(backmapping_dataset, u.dimensions)
            CG_sel = CG_u.select_atoms('all')
            CG_sel.positions = backmapping_dataset.get(DataDict.BEAD_POSITION_ORIGINAL, backmapping_dataset[DataDict.BEAD_POSITION])[0]
            CG_sel.write(os.path.join(self.output_folder, f"CG_{frame_index}.pdb"))
        
        # Write pdb of backmapped structure
        if previous_u is None:
            backmapped_u = build_universe(backmapping_dataset, n_frames, u.dimensions)
        else:
            backmapped_u = previous_u

        backmapped_u.trajectory[frame_index]
        backmapped_sel = backmapped_u.select_atoms('all')
        positions_pred = backmapping_dataset[DataDict.ATOM_POSITION_PRED][0]
        backmapped_sel.positions = positions_pred[~np.any(np.isnan(positions_pred), axis=-1)]
        backmapped_filename = os.path.join(self.output_folder, f"backmapped_{frame_index}.pdb") 
        backmapped_sel.write(backmapped_filename)
        
        # Write pdb of minimised structure
        topology, positions = fixPDB(backmapped_filename, addHydrogens=True)
        backmapped_minimised_filename = os.path.join(self.output_folder, f"backmapped_min_{frame_index}.pdb")
        minimise_impl(
            topology,
            positions,
            backmapped_minimised_filename,
            restrain_atoms=['CA'],
            tolerance=200.,
        )
        
        # Write pdb of true atomistic structure (if present)
        if self.config.get("atomistic", False):
            true_sel = backmapped_u.select_atoms('all')
            true_positions = backmapping_dataset[DataDict.ATOM_POSITION][0]
            true_sel.positions = true_positions[~np.any(np.isnan(positions_pred), axis=-1)]
            true_filename = os.path.join(self.output_folder, f"true_{frame_index}.pdb") 
            true_sel.write(true_filename)

        print(f"Finished. Time: {time.time() - t}")

        return backmapped_u

def build_CG(
    backmapping_dataset: dict,
    box_dimensions,
) -> mda.Universe:
    CG_u = mda.Universe.empty(
        n_atoms =       backmapping_dataset[DataDict.NUM_BEADS],
        n_residues =    backmapping_dataset[DataDict.NUM_RESIDUES],
        atom_resindex = backmapping_dataset[DataDict.BEAD_RESIDCS],
        trajectory =    True, # necessary for adding coordinates
    )
    CG_u.dimensions = box_dimensions
    CG_u.add_TopologyAttr('name',     backmapping_dataset[DataDict.BEAD_NAMES])
    CG_u.add_TopologyAttr('type',     backmapping_dataset[DataDict.BEAD_TYPES])
    CG_u.add_TopologyAttr('resname',  backmapping_dataset[DataDict.RESNAMES])
    CG_u.add_TopologyAttr('resid',    backmapping_dataset[DataDict.RESNUMBERS])
    CG_u.add_TopologyAttr('chainIDs', backmapping_dataset[DataDict.BEAD_CHAINIDCS])

    return CG_u

def build_universe(
    backmapping_dataset,
    n_frames,
    box_dimensions,
):
    nan_filter = ~np.any(np.isnan(backmapping_dataset[DataDict.ATOM_POSITION_PRED][0]), axis=-1)
    num_atoms = nan_filter.sum()
    backmapped_u = mda.Universe.empty(
        n_atoms =       num_atoms,
        n_residues =    backmapping_dataset[DataDict.NUM_RESIDUES],
        atom_resindex = backmapping_dataset[DataDict.ATOM_RESIDCS][nan_filter],
        trajectory    = True # necessary for adding coordinates
    )
    coordinates = np.empty((
        n_frames,  # number of frames
        num_atoms,
        3,
    ))
    backmapped_u.load_new(coordinates, order='fac')

    backmapped_u.dimensions = box_dimensions
    backmapped_u.add_TopologyAttr('name',     backmapping_dataset[DataDict.ATOM_NAMES][nan_filter])
    backmapped_u.add_TopologyAttr('type',     backmapping_dataset[DataDict.ATOM_TYPES][nan_filter])
    backmapped_u.add_TopologyAttr('resname',  backmapping_dataset[DataDict.RESNAMES])
    backmapped_u.add_TopologyAttr('resid',    backmapping_dataset[DataDict.RESNUMBERS])
    backmapped_u.add_TopologyAttr('chainIDs', backmapping_dataset[DataDict.ATOM_CHAINIDCS][nan_filter])

    return backmapped_u

def load_model(config: Dict, model_dir: Optional[Path] = None, model_config: Optional[Dict] = None):
    if model_dir is None:
        assert model_config is not None, "You should provide either 'model_config_file' or 'model_dir' in the configuration file"
        model_dir = os.path.join(model_config.get("root"), model_config.get("fine_tuning_run_name", model_config.get("run_name")))
    model_name = config.get("modelweights", "best_model.pth")
    
    global_config = os.path.join(model_dir, "config.yaml")
    global_config = Config.from_file(str(global_config), defaults={})
    _set_global_options(global_config)
    del global_config

    model, training_model_config = Trainer.load_model_from_training_session(
        traindir=model_dir,
        model_name=model_name,
    )

    model = model.to(config.get("device", "cpu"))
    return model, training_model_config
    
def get_edge_index(positions: torch.Tensor, r_max: float):
    dist_matrix = torch.norm(positions[:, None, ...] - positions[None, ...], dim=-1).fill_diagonal_(torch.inf)
    return torch.argwhere(dist_matrix <= r_max).T.long()

def run_backmapping_inference(dataset: Dict, model: torch.nn.Module, r_max: float, use_only_bb_beads: bool, device: str = 'cpu'):
    
    batch_max_edges = 30000
    batch_max_atoms = 1000
    max_atoms_correction_step = 50

    bead_pos = dataset[DataDict.BEAD_POSITION][0]
    bead_types = dataset[DataDict.BEAD_TYPES]
    bead_residcs = torch.from_numpy(dataset[DataDict.BEAD_RESIDCS]).long()
    if use_only_bb_beads:
        bead_pos = bead_pos[dataset[DataDict.CA_BEAD_IDCS]]
        bead_types = bead_types[dataset[DataDict.CA_BEAD_IDCS]]
    bead_pos = torch.from_numpy(bead_pos).float()
    bead_types = torch.from_numpy(bead_types).long().reshape(-1, 1)
    edge_index = get_edge_index(positions=bead_pos, r_max=r_max)
    batch = torch.zeros(len(bead_pos), device=device, dtype=torch.long)
    bead2atom_idcs = torch.from_numpy(dataset[DataDict.BEAD2ATOM_IDCS]).long()
    bead2atom_weights = torch.from_numpy(dataset[DataDict.BEAD2ATOM_WEIGHTS]).float()
    lvl_idcs_mask = torch.from_numpy(dataset[DataDict.LEVEL_IDCS_MASK]).bool()
    lvl_idcs_anchor_mask = torch.from_numpy(dataset[DataDict.LEVEL_IDCS_ANCHOR_MASK]).long()
    data = {
        AtomicDataDict.POSITIONS_KEY: bead_pos,
        f"{AtomicDataDict.POSITIONS_KEY}_slices": torch.tensor([0, len(bead_pos)]),
        AtomicDataDict.ATOM_TYPE_KEY: bead_types,
        "edge_class": bead_residcs,
        AtomicDataDict.ORIG_EDGE_INDEX_KEY: edge_index,
        AtomicDataDict.EDGE_INDEX_KEY: edge_index,
        AtomicDataDict.BATCH_KEY: batch,
        DataDict.BEAD2ATOM_IDCS: bead2atom_idcs,
        f"{DataDict.BEAD2ATOM_IDCS}_slices": torch.tensor([0, len(bead2atom_idcs)]),
        DataDict.BEAD2ATOM_WEIGHTS: bead2atom_weights,
        f"{DataDict.BEAD2ATOM_WEIGHTS}_slices": torch.tensor([0, len(bead2atom_weights)]),
        DataDict.LEVEL_IDCS_MASK: lvl_idcs_mask,
        f"{DataDict.LEVEL_IDCS_MASK}_slices": torch.tensor([0, len(lvl_idcs_mask)]),
        DataDict.LEVEL_IDCS_ANCHOR_MASK: lvl_idcs_anchor_mask,
        f"{DataDict.ATOM_POSITION}_slices": torch.tensor([0, dataset[DataDict.ATOM_POSITION].shape[1]])
    }

    for v in data.values():
        v.to('cpu')

    already_computed_nodes = None
    chunk = already_computed_nodes is not None

    while True:
        batch_ = copy.deepcopy(data)
        batch_[AtomicDataDict.ORIG_BATCH_KEY] = batch_[AtomicDataDict.BATCH_KEY].clone()

        # Limit maximum batch size to avoid CUDA Out of Memory
        x = batch_[AtomicDataDict.EDGE_INDEX_KEY]
        y = x.clone()
        if already_computed_nodes is not None:
            y = y[:, ~torch.isin(y[0], already_computed_nodes)]
        node_center_idcs = y[0].unique()
        if len(node_center_idcs) == 0:
            return

        while y.shape[1] > batch_max_edges or len(y.unique()) > batch_max_atoms:
            chunk = True
            ### Pick the target edges LESS connected and remove all the node_center_idcs connected to those.
            ### In this way, you prune the "less shared" nodes and keep only nodes that are "clustered",
            ### thus maximizing the number of node_center_idcs while complaining to the self.batch_max_atoms restrain
            
            def get_y_edge_filter(y: torch.Tensor, correction: int):
                target_atom_idcs, count = torch.unique(y[1], return_counts=True)
                less_connected_atom_idcs = torch.topk(-count, max(1, max_atoms_correction_step - correction)).indices
                target_atom_idcs_to_remove = target_atom_idcs[less_connected_atom_idcs]
                node_center_idcs_to_remove = torch.unique(y[0][torch.isin(y[1], target_atom_idcs_to_remove)])
                return ~torch.isin(y[0], node_center_idcs_to_remove), correction
            correction = 0
            y_edge_filter, correction = get_y_edge_filter(y, correction=correction)
            while y_edge_filter.sum() == 0:
                correction += 1
                if correction >= max_atoms_correction_step:
                    print(
                        f"Dataset with index {batch_['dataset_idx'].item()} has at least one center atom with connections"
                            " that exceed 'batch_max_edges' or 'batch_max_atoms'")
                    return
                y_edge_filter, correction = get_y_edge_filter(y, correction=correction)
            y = y[:, y_edge_filter]
            
            #########################################################################################################
        
        x_ulen = len(x[0].unique())

        if chunk:
            batch_[AtomicDataDict.EDGE_INDEX_KEY] = y
            x_edge_filter = torch.isin(x[0], y[0].unique())
            if AtomicDataDict.EDGE_CELL_SHIFT_KEY in batch_:
                batch_[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = batch_[AtomicDataDict.EDGE_CELL_SHIFT_KEY][x_edge_filter]
            batch_[AtomicDataDict.BATCH_KEY] = batch_.get(AtomicDataDict.ORIG_BATCH_KEY, batch_[AtomicDataDict.BATCH_KEY])[y.unique()]
            
        del x
        
        # for slices_key, slices in data.__slices__.items():
        #     batch_[f"{slices_key}_slices"] = torch.tensor(slices, dtype=int)
        batch_["ptr"] = torch.nn.functional.pad(torch.bincount(batch_.get(AtomicDataDict.BATCH_KEY)).flip(dims=[0]), (0, 1), mode='constant').flip(dims=[0])
        
        # Remove all atoms that do not appear in edges and update edge indices
        edge_index = batch_[AtomicDataDict.EDGE_INDEX_KEY]

        edge_index_unique = edge_index.unique()

        ignore_chunk_keys = ["bead2atom_idcs", "lvl_idcs_mask", "lvl_idcs_anchor_mask", "atom_pos"]
        
        for key in batch_.keys():
            if key in [
                AtomicDataDict.BATCH_KEY,
                AtomicDataDict.ORIG_BATCH_KEY,
                AtomicDataDict.EDGE_INDEX_KEY,
                AtomicDataDict.EDGE_CELL_SHIFT_KEY
            ] + ignore_chunk_keys:
                continue
            dim = np.argwhere(np.array(batch_[key].size()) == len(batch_[AtomicDataDict.ORIG_BATCH_KEY])).flatten()
            if len(dim) == 1:
                if dim[0] == 0:
                    batch_[key] = batch_[key][edge_index_unique]
                elif dim[0] == 1:
                    batch_[key] = batch_[key][:, edge_index_unique]
                elif dim[0] == 2:
                    batch_[key] = batch_[key][:, :, edge_index_unique]
                else:
                    raise Exception('Dimension not implemented')

        last_idx = -1
        batch_[AtomicDataDict.ORIG_EDGE_INDEX_KEY] = edge_index
        updated_edge_index = edge_index.clone()
        for idx in edge_index_unique:
            if idx > last_idx + 1:
                updated_edge_index[edge_index >= idx] -= idx - last_idx - 1
            last_idx = idx
        batch_[AtomicDataDict.EDGE_INDEX_KEY] = updated_edge_index

        node_index_unique = edge_index[0].unique()
        del edge_index
        del edge_index_unique

        for k, v in batch_.items():
            batch_[k] = v.to(device)

        with torch.no_grad():
            out = model(batch_)
            predicted_dih = out.get(INVARIANT_ATOM_FEATURES).cpu().numpy()
            predicted_b2a_rel_vec = out.get(EQUIVARIANT_ATOM_FEATURES).cpu().numpy()
            reconstructed_atom_pos = out.get(ATOM_POSITIONS, None)
            if reconstructed_atom_pos is not None:
                reconstructed_atom_pos = reconstructed_atom_pos.cpu().numpy()
        # out = model(batch_)
        # predicted_dih = out.get(INVARIANT_ATOM_FEATURES).detach().cpu().numpy()
        # predicted_b2a_rel_vec = out.get(EQUIVARIANT_ATOM_FEATURES).detach().cpu().numpy()
        # reconstructed_atom_pos = out.get(ATOM_POSITIONS, None)
        # if reconstructed_atom_pos is not None:
        #     reconstructed_atom_pos = reconstructed_atom_pos.detach().cpu().numpy()

        if already_computed_nodes is None:
            dataset[DataDict.BB_PHIPSI_PRED] = np.zeros((1, lvl_idcs_mask.shape[1], predicted_dih.shape[1]), dtype=float)
            dataset[DataDict.BEAD2ATOM_RELATIVE_VECTORS_PRED] = np.zeros((1, lvl_idcs_mask.shape[1], lvl_idcs_mask.shape[2], 3), dtype=float)
            if reconstructed_atom_pos is not None:
                dataset[DataDict.ATOM_POSITION_PRED] = reconstructed_atom_pos[None, ...]
        
        original_nodes = out[AtomicDataDict.ORIG_EDGE_INDEX_KEY][0].unique().cpu().numpy()
        nodes = out[AtomicDataDict.EDGE_INDEX_KEY][0].unique().cpu().numpy()
        
        dataset[DataDict.BB_PHIPSI_PRED][:, original_nodes] = predicted_dih[None, nodes]
        dataset[DataDict.BEAD2ATOM_RELATIVE_VECTORS_PRED][:, original_nodes] = predicted_b2a_rel_vec[None, nodes]
        if reconstructed_atom_pos is not None:
            fltr = np.argwhere(~np.isnan(reconstructed_atom_pos[:, 0])).flatten()
            dataset[DataDict.ATOM_POSITION_PRED][:, fltr] = reconstructed_atom_pos[None, fltr]
        
        del out

        if already_computed_nodes is None:
            if len(node_index_unique) < x_ulen:
                already_computed_nodes = node_index_unique
        elif len(already_computed_nodes) + len(node_index_unique) == x_ulen:
            already_computed_nodes = None
        else:
            assert len(already_computed_nodes) + len(node_index_unique) < x_ulen
            already_computed_nodes = torch.cat([already_computed_nodes, node_index_unique], dim=0)
        
        del batch_
        if already_computed_nodes is None:
            return dataset

def build_minimizer_data(dataset: Dict, device='cpu'):
    no_cappings_filter = [x.split('_')[0] not in ['ACE', 'NME'] for x in dataset[DataDict.ATOM_NAMES]]
    unique_resnums = np.array([x[0] for x in groupby(dataset[DataDict.ATOM_RESNUMBERS][no_cappings_filter])])
    chainidcs = dataset[DataDict.BEAD_CHAINIDCS][dataset[DataDict.CA_BEAD_IDCS]]
    ca_pos = dataset[DataDict.CA_ATOM_POSITION][0]
    n_residues = dataset[DataDict.CA_BEAD_IDCS].sum()

    bond_idcs = []
    bond_eq_val = []
    bond_tolerance = []

    ca_bond_idcs = []
    ca_bond_eq_val = []
    ca_bond_tolerance = []

    angle_idcs = []
    angle_eq_val = []
    angle_tolerance = []

    for id_, resnum, next_resnum in zip(range(0, n_residues-1), unique_resnums[:-1], unique_resnums[1:]):
        atom_id = id_ * 4 # N
        
        bond_idcs.append([atom_id, atom_id+1])
        bond_eq_val.append(1.45) # N - CA bond length
        bond_tolerance.append(0.03) # Permissible values [1.42, 1.48]

        bond_idcs.append([atom_id+1, atom_id+2])
        bond_eq_val.append(1.52) # CA - C bond length
        bond_tolerance.append(0.03) # Permissible values [1.49, 1.55]

        bond_idcs.append([atom_id+2, atom_id+3])
        bond_eq_val.append(1.24) # C - O bond length
        bond_tolerance.append(0.02) # Permissible values [1.22, 1.26]
        
        if resnum == next_resnum - 1 and chainidcs[id_] == chainidcs[id_+1]:
            bond_idcs.append([atom_id+1, atom_id+5])
            bond_eq_val.append(3.81) # CA - CA bond length
            bond_tolerance.append(0.1) # Permissible values [1.71, 1.91]

            bond_idcs.append([atom_id+2, atom_id+4])
            bond_eq_val.append(1.34) # C - N bond length
            bond_tolerance.append(0.03) # Permissible values [1.31, 1.37]

        angle_idcs.append([atom_id, atom_id+1, atom_id+2])
        angle_eq_val.append(1.9216075) # N - CA - C angle value
        angle_tolerance.append(0.035)  # Tolerance 0.035 rad ~ 2 deg

        angle_idcs.append([atom_id+1, atom_id+2, atom_id+3])
        angle_eq_val.append(2.0944) # CA - C - O angle value
        angle_tolerance.append(0.052)  # Tolerance 0.035 rad ~ 3 deg
        
        if resnum == next_resnum - 1 and chainidcs[id_] == chainidcs[id_+1]:
            angle_idcs.append([atom_id+1, atom_id+2, atom_id+4])
            angle_eq_val.append(2.0350539) # CA - C - N angle value
            angle_tolerance.append(0.035)  # Tolerance 0.035 rad ~ 2 deg
        
            angle_idcs.append([atom_id+2, atom_id+4, atom_id+5])
            angle_eq_val.append(2.1275564) # C - N - CA angle value
            angle_tolerance.append(0.035)  # Tolerance 0.035 rad ~ 2 deg
    
    if n_residues > 0:
        atom_id = (id_ + 1) * 4 # N    
        bond_idcs.append([atom_id, atom_id+1])
        bond_eq_val.append(1.45) # N - CA bond length
        bond_tolerance.append(0.03) # Permissible values [1.42, 1.48]

        # bond_idcs.append([atom_id+1, atom_id+2])
        # bond_eq_val.append(1.52) # CA - C bond length
        # bond_tolerance.append(0.03) # Permissible values [1.49, 1.55]

        # bond_idcs.append([atom_id+2, atom_id+3])
        # bond_eq_val.append(1.24) # C - O bond length
        # bond_tolerance.append(0.02) # Permissible values [1.22, 1.26]

        # angle_idcs.append([atom_id, atom_id+1, atom_id+2])
        # angle_eq_val.append(1.9216075) # N - CA - C angle value
        # angle_tolerance.append(0.035)  # Tolerance 0.035 rad ~ 2 deg

        # angle_idcs.append([atom_id+1, atom_id+2, atom_id+3])
        # angle_eq_val.append(2.0944) # CA - C - O angle value
        # angle_tolerance.append(0.052)  # Tolerance 0.035 rad ~ 3 deg
    
    # ----------------------------- #

    bond_idcs = np.array(bond_idcs)
    bond_eq_val = np.array(bond_eq_val)
    bond_tolerance = np.array(bond_tolerance)

    ca_bond_idcs = np.array(ca_bond_idcs)
    ca_bond_eq_val = np.array(ca_bond_eq_val)
    ca_bond_tolerance = np.array(ca_bond_tolerance)

    angle_idcs = np.array(angle_idcs)
    angle_eq_val = np.array(angle_eq_val)
    angle_tolerance = np.array(angle_tolerance)

    movable_pos_idcs = np.ones((len(ca_pos) * 4,), dtype=bool)
    movable_pos_idcs[np.arange(len(ca_pos)) * 4 + 1] = False

    data = {
        "bond_idcs": bond_idcs,
        "bond_eq_val": bond_eq_val,
        "bond_tolerance": bond_tolerance,
        "ca_bond_idcs": ca_bond_idcs,
        "ca_bond_eq_val": ca_bond_eq_val,
        "ca_bond_tolerance": ca_bond_tolerance,
        "angle_idcs": angle_idcs,
        "angle_eq_val": angle_eq_val,
        "angle_tolerance": angle_tolerance,
        "movable_pos_idcs": movable_pos_idcs,
    }

    if DataDict.BEAD_RESIDCS in dataset:
        per_residue_beads = np.bincount(dataset[DataDict.BEAD_RESIDCS] - dataset[DataDict.BEAD_RESIDCS].min())
        data.update({
            "per_residue_beads": per_residue_beads,
        })
    
    for k, v in data.items():
        data[k] = torch.from_numpy(v).to(device)
    
    return data

def update_minimizer_data(minimizer_data: Dict, dataset: Dict, use_only_bb_beads: bool, device: str = 'cpu'):
    no_cappings_filter = [rn not in ['ACE', 'NME'] for rn in dataset[DataDict.ATOM_RESNAMES]]
    unique_residcs = np.array([x[0] for x in groupby(dataset[DataDict.ATOM_RESIDCS][no_cappings_filter])])

    n_atom_idcs =  np.array([an=='N'  and rn not in ['ACE', 'NME'] for an, rn in zip(dataset[DataDict.ATOM_NAMES], dataset[DataDict.ATOM_RESNAMES])])
    ca_atom_idcs = dataset[DataDict.CA_ATOM_IDCS]
    c_atom_idcs =  np.array([an=='C'  and rn not in ['ACE', 'NME'] for an, rn in zip(dataset[DataDict.ATOM_NAMES], dataset[DataDict.ATOM_RESNAMES])])
    o_atom_idcs =  np.array([an=='O'  and rn not in ['ACE', 'NME'] for an, rn in zip(dataset[DataDict.ATOM_NAMES], dataset[DataDict.ATOM_RESNAMES])])

    ca_pos_pred = dataset[DataDict.ATOM_POSITION_PRED][0, ca_atom_idcs]
    n_pos_pred = dataset[DataDict.ATOM_POSITION_PRED][0, n_atom_idcs]
    c_pos_pred = dataset[DataDict.ATOM_POSITION_PRED][0, c_atom_idcs]
    o_pos_pred = dataset[DataDict.ATOM_POSITION_PRED][0, o_atom_idcs]

    ca_names = dataset[DataDict.ATOM_NAMES][ca_atom_idcs]
    n_names =  dataset[DataDict.ATOM_NAMES][n_atom_idcs]
    c_names =  dataset[DataDict.ATOM_NAMES][c_atom_idcs]
    o_names =  dataset[DataDict.ATOM_NAMES][o_atom_idcs]

    atom_pos_pred = cat_interleave(
        [
            n_pos_pred,
            ca_pos_pred,
            c_pos_pred,
            o_pos_pred,
        ],
    ).astype(np.float64)

    atom_names = cat_interleave(
        [
            n_names,
            ca_names,
            c_names,
            o_names,
        ],
    )

    residue_contains_n = []
    residue_contains_c = []
    residue_contains_o = []
    for resid in unique_residcs:
        residue_atom_names = [an.split('_')[-1] for an in dataset[DataDict.ATOM_NAMES][dataset[DataDict.ATOM_RESIDCS] == resid]]
        residue_contains_n.append('N' in residue_atom_names)
        residue_contains_c.append('C' in residue_atom_names)
        residue_contains_o.append('O' in residue_atom_names)
    residue_contains_n = np.array(residue_contains_n)
    residue_contains_c = np.array(residue_contains_c)
    residue_contains_o = np.array(residue_contains_o)
    
    # real_bb_atom_idcs = cat_interleave(
    #     [
    #         residue_contains_n,
    #         np.ones_like(residue_contains_n, dtype=bool),
    #         residue_contains_c,
    #         residue_contains_o,
    #     ],
    # )

    data = {
        "pos": atom_pos_pred,
        "atom_names": atom_names,
        "n_atom_idcs": n_atom_idcs,
        "c_atom_idcs": c_atom_idcs,
        "o_atom_idcs": o_atom_idcs,
        # "real_bb_atom_idcs": real_bb_atom_idcs,
    }

    if DataDict.BB_PHIPSI_PRED in dataset and dataset[DataDict.BB_PHIPSI_PRED].shape[-1] == 2:
        dih = dataset[DataDict.BB_PHIPSI_PRED][0]
        if not use_only_bb_beads:
            dih = dih[dataset[DataDict.CA_BEAD_IDCS]]
        
        dih_idcs = []
        dih_eq_val = []
        chainidcs = dataset[DataDict.BEAD_CHAINIDCS]
        for id_, dih_val in zip(range(0, len(dih)), dih):
            atom_id = id_ * 4 # start from N atom of id_ residue (N1 CA1 C1 O1 N2 CA2 C2 O2...)
            if id_ > 0 and chainidcs[id_-1] == chainidcs[id_]:
                # Phi
                dih_idcs.append([atom_id-1, atom_id, atom_id+1, atom_id+2])
                dih_eq_val.append(dih_val[0])
            if id_ < len(dih)-1 and chainidcs[id_] == chainidcs[id_+1]:
                # Psi
                dih_idcs.append([atom_id, atom_id+1, atom_id+2, atom_id+4])
                dih_eq_val.append(dih_val[1])
                # Peptide bond
                dih_idcs.append([atom_id+1, atom_id+2, atom_id+4, atom_id+5])
                dih_eq_val.append(np.pi)
        dih_idcs = np.array(dih_idcs)
        dih_eq_val = np.array(dih_eq_val)
        data.update({
            "dih_idcs": dih_idcs,
            "dih_eq_val": dih_eq_val,
        })

    for k, v in data.items():
        if v.dtype.type is not np.str_:
            data[k] = torch.from_numpy(v).to(device)
    
    minimizer_data.update(data)

    return minimizer_data

def rotate_residue_dihedrals(pos: torch.TensorType, ca_pos: torch.TensorType, angle: float):
    rot_axes = ca_pos[1:] - ca_pos[:-1]
    rot_axes = rot_axes / torch.norm(rot_axes, dim=-1, keepdim=True)
    rot_axes = rot_axes.repeat_interleave(2 * torch.ones((len(rot_axes),), dtype=int, device=rot_axes.device), dim=0)

    angles_polar = 0.5 * angle * torch.ones((len(rot_axes),), dtype=float, device=rot_axes.device).reshape(-1, 1)

    q_polar = get_quaternions(
        batch=1,
        rot_axes=rot_axes,
        angles=angles_polar
    )

    C_N_O_fltr = torch.zeros((len(pos),), dtype=bool, device=rot_axes.device)
    for x in range(len(ca_pos)-1):
        C_N_O_fltr[x*4+2] = True
        C_N_O_fltr[x*4+4] = True

    v_ = pos[C_N_O_fltr] - ca_pos[:-1].repeat_interleave(2 * torch.ones((len(ca_pos[:-1]),), dtype=int, device=ca_pos.device), dim=0)
    v_rotated = qv_mult(q_polar, v_)

    rotated_pred_pos = pos.clone()
    rotated_pred_pos[C_N_O_fltr] = ca_pos[:-1].repeat_interleave(2 * torch.ones((len(ca_pos[:-1]),), dtype=int, device=ca_pos.device), dim=0) + v_rotated
    return rotated_pred_pos

def run_b2a_rel_vec_backmapping(dataset: Dict, use_predicted_b2a_rel_vec: bool = False):
    b2a_rel_vec_key = DataDict.BEAD2ATOM_RELATIVE_VECTORS_PRED if use_predicted_b2a_rel_vec else DataDict.BEAD2ATOM_RELATIVE_VECTORS
    assert b2a_rel_vec_key in dataset, f"{b2a_rel_vec_key} key is missing from dataset."
    
    bead_pos = torch.from_numpy(dataset[DataDict.BEAD_POSITION])
    idcs_mask = torch.from_numpy(dataset[DataDict.BEAD2ATOM_IDCS])

    level_idcs_mask = torch.from_numpy(dataset[DataDict.LEVEL_IDCS_MASK])
    level_anchor_idcs_mask = torch.from_numpy(dataset[DataDict.LEVEL_IDCS_ANCHOR_MASK])
    bead2atom_relative_vectors = torch.from_numpy(dataset[b2a_rel_vec_key])
    
    per_bead_reconstructed_atom_pos = []
    for h, b2a_idcs in enumerate(idcs_mask):
        reconstructed_atom_pos = torch.empty((len(bead_pos), len(dataset[DataDict.ATOM_NAMES]), 3), dtype=bead_pos.dtype)
        reconstructed_atom_pos[:] = torch.nan
        reconstructed_atom_pos[:, b2a_idcs[b2a_idcs>=0]] = bead_pos[:, h, None, ...]

        for level, (level_idcs_mask_elem, level_anchor_idcs_mask_elem) in enumerate(zip(level_idcs_mask[:, h], level_anchor_idcs_mask[:, h])):
            updated_pos = reconstructed_atom_pos[:, level_anchor_idcs_mask_elem[level_idcs_mask_elem]]
            if level > 0:
                updated_pos = updated_pos + bead2atom_relative_vectors[:, h, level_idcs_mask_elem]
                reconstructed_atom_pos[:, idcs_mask[h, level_idcs_mask_elem]] = updated_pos
        per_bead_reconstructed_atom_pos.append(reconstructed_atom_pos)
    per_bead_reconstructed_atom_pos = torch.stack(per_bead_reconstructed_atom_pos, dim=0)
    
    return per_bead_reconstructed_atom_pos.nanmean(dim=0).cpu().numpy()

def adjust_bb_oxygens(dataset: Dict, position_key: str = DataDict.ATOM_POSITION_PRED):
    atom_CA_idcs = dataset[DataDict.CA_ATOM_IDCS]
    no_cappings_filter = [x.split('_')[0] not in ['ACE', 'NME'] for x in dataset[DataDict.ATOM_NAMES]]
    atom_C_idcs = np.array([ncf and an.split('_')[1] in ["C"] for ncf, an in zip(no_cappings_filter, dataset[DataDict.ATOM_NAMES])])
    atom_O_idcs = np.array([ncf and an.split('_')[1] in ["O"] for ncf, an in zip(no_cappings_filter, dataset[DataDict.ATOM_NAMES])])

    ca_pos = dataset[position_key][:, atom_CA_idcs]

    c_o_vectors = []
    for i in range(ca_pos.shape[1]-2):
        ca_i, ca_ii, ca_iii = ca_pos[:, i], ca_pos[:, i+1], ca_pos[:, i+2]
        ca_i_ca_ii = ca_ii - ca_i
        ca_i_ca_iii = ca_iii - ca_i
        c_o = np.cross(ca_i_ca_ii, ca_i_ca_iii, axis=1)
        c_o = c_o / np.linalg.norm(c_o, axis=-1, keepdims=True) * 1.229 # C-O bond legth
        c_o_vectors.append(c_o)
    # Last missing vectors
    for _ in range(len(c_o_vectors), atom_C_idcs.sum()):
        c_o_vectors.append(c_o)
    c_o_vectors = np.array(c_o_vectors).swapaxes(0,1)

    o_pos = dataset[position_key][:, atom_C_idcs] + c_o_vectors
    dataset[position_key][:, atom_O_idcs] = o_pos

    pos = torch.from_numpy(dataset[position_key]).float()
    omega_dihedral_idcs = torch.from_numpy(dataset[DataDict.OMEGA_DIH_IDCS]).long()
    adjusted_pos = adjust_oxygens(
        pos=pos,
        omega_dihedral_idcs=omega_dihedral_idcs,
    )
    dataset[position_key] = adjusted_pos.cpu().numpy()

    return dataset

def adjust_oxygens(
        pos: torch.Tensor, # (batch, n_atoms, 3)
        omega_dihedral_idcs: torch.Tensor, # (n_dih, 4)
    ):
    batch = pos.size(0)

    # ADJUST DIHEDRAL ANGLE [O, C, N, CA]
    dih_values = get_dihedrals(pos, omega_dihedral_idcs) # (batch, n_dih)

    # omega_dihedral_idcs represent atoms [O, C, N, CA]
    v_ = pos[:, omega_dihedral_idcs[:, 0]] - pos[:, omega_dihedral_idcs[:, 1]]
    v_ = v_.reshape(-1, 3) # (batch * n_peptide_oxygens, xyz)

    rot_axes = pos[:, omega_dihedral_idcs[:, 2]] - pos[:, omega_dihedral_idcs[:, 1]]
    rot_axes = rot_axes / rot_axes.norm(dim=-1, keepdim=True)
    rot_axes = rot_axes.reshape(-1, 3) # (batch * n_peptide_oxygens, xyz)

    # We want that the final dihedral values are 0, so we rotate of dih_values
    angles_polar = dih_values.reshape(-1, 1) # (n_peptide_oxygens, 1)
    q_polar = get_quaternions(               # (batch * n_peptide_oxygens, 4)
        batch=batch,
        rot_axes=rot_axes,
        angles=0.5*angles_polar
    )

    v_rotated = qv_mult(q_polar, v_).reshape(batch, -1, 3) # (batch, n_peptide_oxygens, xyz)
    adjusted_pos = pos.clone()
    adjusted_pos[:, omega_dihedral_idcs[:, 0]] = adjusted_pos[:, omega_dihedral_idcs[:, 1]] + v_rotated

    # ADJUST ANGLE [O, C, N]
    angle_values, b0, b1 = get_angles(adjusted_pos, omega_dihedral_idcs[:, :3], return_vectors=True)
    # angle_values (batch, n_angles)
    # b0 (batch, n_angles, xyz) C->O
    # b1 (batch, n_angles, xyz) C->N

    v_ = adjusted_pos[:, omega_dihedral_idcs[:, 0]] - adjusted_pos[:, omega_dihedral_idcs[:, 1]]
    v_ = v_.reshape(-1, 3) # (batch * n_peptide_oxygens, xyz)

    cross_prod = o3.TensorProduct(
        "1x1o",
        "1x1o",
        "1x1e",
        [(0, 0, 0, "uuu", False)],
        irrep_normalization='none',
    )
    rot_axes = cross_prod(b0, b1) # (batch, n_peptide_oxygens, xyz)
    rot_axes = rot_axes / rot_axes.norm(dim=-1, keepdim=True)
    rot_axes = rot_axes.reshape(-1, 3) # (batch * n_peptide_oxygens, xyz)

    angles_polar = angle_values.reshape(-1, 1) - 2.145 # (n_peptide_oxygens, 1) | O-C-N angle eq value is 122.9 degrees
    angles_polar[angles_polar > np.pi] -= 2*np.pi
    angles_polar[angles_polar < -np.pi] += 2*np.pi
    q_polar = get_quaternions(               # (batch * n_peptide_oxygens, 4)
        batch=batch,
        rot_axes=rot_axes,
        angles=0.5*angles_polar
    )

    v_rotated = qv_mult(q_polar, v_).reshape(batch, -1, 3) # (batch, n_peptide_oxygens, xyz)
    adjusted_pos[:, omega_dihedral_idcs[:, 0]] = adjusted_pos[:, omega_dihedral_idcs[:, 1]] + v_rotated

    return adjusted_pos

# def backmap(config_filename: str, frame_selection: Optional[slice] = None, verbose: bool = False):
#     print(f"Reading input data...")
#     t = time.time()
#     backmapping = HierarchicalBackmapping(config_filename=config_filename)
#     print(f"Finished. Time: {time.time() - t}")

#     frame_idcs = range(0, len(backmapping.mapping.dataset[DataDict.BEAD_POSITION]))
#     if frame_selection is not None:
#         frame_idcs = frame_idcs[frame_selection]
#     n_frames = len(frame_idcs)

#     CG_u, backmapped_u = None, None
#     bb_minimisation_u_list = []
#     for i, frame_index in enumerate(frame_idcs):
#         print(f"Starting backmapping for frame index {frame_index} ({i+1}/{n_frames})")
#         backmapping_dataset = backmapping.backmap(
#             frame_index=frame_index,
#             verbose=verbose,
#         )
#         CG_u, backmapped_u, bb_minimisation_u = backmapping.to_pdb(
#             backmapping_dataset=backmapping_dataset,
#             n_frames=n_frames,
#             frame_index=frame_index,
#             selection=backmapping.config.get("selection", "protein"),
#             folder=backmapping.config.get("output_folder"),
#             previous_u=backmapped_u,
#         )
#         bb_minimisation_u_list.append(bb_minimisation_u)

#     for tag in ['original_CG', 'final_CG', 'backmapped', 'true']:
#         joinPDBs(backmapping.config.get("output_folder"), tag)
    
#     return CG_u, backmapped_u, bb_minimisation_u_list