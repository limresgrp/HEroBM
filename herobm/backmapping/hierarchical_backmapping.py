import os
import tempfile
import time
import torch
import shutil

import numpy as np
import MDAnalysis as mda

import warnings

warnings.filterwarnings("ignore")

from os.path import join, dirname, basename
from typing import Callable, Dict, List, Optional
from MDAnalysis.analysis import align

from herobm.mapper.hierarchical_mapper import HierarchicalMapper
# from herobm.backmapping.nn.quaternions import get_quaternions, qv_mult
from herobm.utils import DataDict
from herobm.utils.geometry import set_phi, set_psi
from herobm.utils.backbone import MinimizeEnergy
from herobm.utils.io import replace_words_in_file

from geqtrain.utils import Config
from geqtrain.data import AtomicDataDict
from geqtrain.data._build import dataset_from_config
from geqtrain.data.dataloader import DataLoader
from geqtrain.scripts.evaluate import load_model, infer


class HierarchicalBackmapping:

    config: Dict[str, str]
    mapping: HierarchicalMapper

    input_folder: Optional[str]
    model_config: Dict[str, str]
    model_r_max: float

    minimiser: MinimizeEnergy

    def __init__(self, args_dict: Dict, preprocess_npz_func: Callable = None) -> None:
        # Load model
        self.model, model_config = load_model(args_dict.get("model"), args_dict.get("device"))
        args_dict.update({
            "noinvariants": True, # Avoid computing angles and dihedrals when Coarse-graining atomistic input, they are used only for training.
            "cutoff": model_config.get("r_max"),
        })

        # Parse Input
        self.mapping = HierarchicalMapper(args_dict=args_dict)
        self.config = self.mapping.config

        self.output_folder = self.config.get("output")
        self.device = self.config.get("device", "cpu")

        self.preprocess_npz_func = preprocess_npz_func
        self.config.update(
            {
                k: v
                for k, v in model_config.items()
                if k not in self.config
                and k not in "skip_chunking"
            }
        )

        # Initialize energy minimiser for reconstructing backbone
        self.minimiser = MinimizeEnergy()

        ### ------------------------------------------------------- ###
    
    @property
    def num_structures(self):
        return len(self.mapping.input_filenames)
    
    def optimise_backbone(
            self,
            backmapping_dataset: Dict,
            optimise_dihedrals: bool = False,
            verbose: bool = False,
    ):
        pred_u = build_universe(backmapping_dataset, 1, self.mapping.selection.dimensions)
        positions_pred = backmapping_dataset[DataDict.ATOM_POSITION_PRED]
        not_nan_filter = ~np.any(np.isnan(positions_pred), axis=-1)
        pred_u.atoms.positions = positions_pred[not_nan_filter]
        
        minimiser_data = build_minimiser_data(dataset=backmapping_dataset)

        # Add predicted dihedrals as dihedral equilibrium values
        minimiser_data = update_minimiser_data(
            minimiser_data=minimiser_data,
            dataset=backmapping_dataset,
        )
        
        # Step 1: initial minimization: adjust bonds, angles and Omega dihedral
        self.minimiser.minimise(
            data=minimiser_data,
            dtau=1e-1, # self.config.get("bb_minimisation_dtau", 1e-1),
            eps=self.config.get("bb_initial_minimisation_eps", 1e-2),
            device=self.device,
            verbose=verbose,
        )

        if optimise_dihedrals:
            self.optimise_dihedrals(minimiser_data=minimiser_data)
        
        minimised_u = build_universe(backmapping_dataset, 1, self.mapping.selection.dimensions)
        minimised_u.atoms.positions = minimiser_data["coords"][not_nan_filter]

        rmsds = align.alignto(
            minimised_u,  # mobile
            pred_u,       # reference
            select='name CA', # selection to operate on
            match_atoms=True
        ) # whether to match atoms

        backmapping_dataset[DataDict.ATOM_POSITION_PRED][not_nan_filter] = np.expand_dims(minimised_u.atoms.positions, axis=0)
        return backmapping_dataset
        
    def map(self):
        for mapping in self.mapping():
            yield mapping

    def backmap(self, tolerance: float = 50., frame_idcs: Optional[List[int]] = None):
        
        backmapped_filenames, backmapped_minimised_filenames, true_filenames, cg_filenames = [], [], [], []
        for input_filenames_index, (mapping, output_filename) in enumerate(self.map()):
            if self.output_folder is None:
                self.output_folder = dirname(output_filename)
            
            if frame_idcs is None:
                frame_idcs = range(0, len(mapping))
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
                test_config.pop('type_names', None)
                test_config.pop('num_types', None)
                self.config.update(test_config)

                dataset = dataset_from_config(self.config, prefix="test_dataset")

                dataloader = DataLoader(
                    dataset=dataset,
                    shuffle=False,
                    batch_size=1,
                )

                results = {
                    DataDict.BEAD_POSITION: [],
                    DataDict.BEAD2ATOM_RELATIVE_VECTORS_PRED: [],
                    DataDict.ATOM_POSITION_PRED: [],
                }
                def collect_chunks(batch_index, chunk_index, out, ref_data, data, pbar, **kwargs):
                    pos_list = results.get(DataDict.BEAD_POSITION)
                    pos_list.append(out[AtomicDataDict.POSITIONS_KEY].cpu().numpy())
                    rvp_list = results.get(DataDict.BEAD2ATOM_RELATIVE_VECTORS_PRED)
                    rvp_list.append(out[AtomicDataDict.NODE_OUTPUT_KEY].cpu().numpy())
                    app_list = results.get(DataDict.ATOM_POSITION_PRED)
                    app_list.append(np.expand_dims(out[DataDict.ATOM_POSITION].cpu().numpy(), axis=0))
                
                _backmapped_u = [None]

                def save_batch(batch_index, **kwargs):
                    backmapping_dataset = self.mapping.dataset
                    backmapping_dataset.update({
                        DataDict.BEAD_POSITION: np.stack(results[DataDict.BEAD_POSITION], axis=0),
                        DataDict.BEAD2ATOM_RELATIVE_VECTORS_PRED: np.concatenate(results[DataDict.BEAD2ATOM_RELATIVE_VECTORS_PRED], axis=0),
                        DataDict.ATOM_POSITION_PRED: np.nanmean(np.stack(results[DataDict.ATOM_POSITION_PRED], axis=0), axis=0),
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

                    results[DataDict.BEAD2ATOM_RELATIVE_VECTORS_PRED].clear()
                    results[DataDict.ATOM_POSITION_PRED].clear()

                infer(
                    dataloader,
                    self.model,
                    self.device,
                    chunk_callbacks=[collect_chunks],
                    batch_callbacks=[save_batch],
                    **{k: v for k, v in self.config.items() if k not in ['model', 'device']},
                )
        
        return backmapped_filenames, backmapped_minimised_filenames, true_filenames, cg_filenames
    
    def optimise_dihedrals(self, minimiser_data: Dict):
        coords = minimiser_data["coords"][0]

        phi_idcs = minimiser_data["phi_idcs"]
        phi_values = minimiser_data["phi_values"]
        psi_idcs = minimiser_data["psi_idcs"]
        psi_values = minimiser_data["psi_values"]
        
        coords = set_phi(coords, phi_idcs, phi_values)
        coords = set_psi(coords, psi_idcs, psi_values)

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
        print(f"Saving structures...")
        t = time.time()
        os.makedirs(self.output_folder, exist_ok=True)
        prefix = basename(self.mapping.input_filenames[input_filenames_index])

        # Write pdb file of CG structure
        cg_filename = None
        if save_CG:
            cg_u = build_CG(backmapping_dataset, n_frames, self.mapping.selection.dimensions)
            cg_u.trajectory[frame_index]
            cg_sel = cg_u.select_atoms('all')
            cg_sel.positions = np.nan_to_num(backmapping_dataset[DataDict.BEAD_POSITION][frame_index])
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
                pass

        print(f"Finished. Time: {time.time() - t}")

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

def build_minimiser_data(dataset: Dict):

    atom_names = dataset[DataDict.ATOM_NAMES]
    dataset_bond_idcs = dataset[DataDict.BOND_IDCS]
    dataset_angle_idcs = dataset[DataDict.ANGLE_IDCS]
    
    bond_idcs = []
    bond_eq_val = []
    bond_tolerance = []

    angle_idcs = []
    angle_eq_val = []
    angle_tolerance = []

    N_CA_filter = np.all(atom_names[dataset_bond_idcs] == ['N', 'CA'], axis=1)
    bond_idcs.append(dataset_bond_idcs[N_CA_filter])
    bond_eq_val.append([1.45] * N_CA_filter.sum())
    bond_tolerance.append([0.02] * N_CA_filter.sum())
    CA_C_filter = np.all(atom_names[dataset_bond_idcs] == ['CA', 'C'], axis=1)
    bond_idcs.append(dataset_bond_idcs[CA_C_filter])
    bond_eq_val.append([1.52] * CA_C_filter.sum())
    bond_tolerance.append([0.02] * CA_C_filter.sum())
    C_O_filter = np.all(atom_names[dataset_bond_idcs] == ['C', 'O'], axis=1)
    bond_idcs.append(dataset_bond_idcs[C_O_filter])
    bond_eq_val.append([1.24] * C_O_filter.sum())
    bond_tolerance.append([0.02] * C_O_filter.sum())
    C_N_filter = np.all(atom_names[dataset_bond_idcs] == ['C', 'N'], axis=1)
    bond_idcs.append(dataset_bond_idcs[C_N_filter])
    bond_eq_val.append([1.32] * C_N_filter.sum())
    bond_tolerance.append([0.02] * C_N_filter.sum())

    N_CA_C_filter = np.all(atom_names[dataset_angle_idcs] == ['N', 'CA', 'C'], axis=1)
    angle_idcs.append(dataset_angle_idcs[N_CA_C_filter])
    angle_eq_val.append([1.9216075] * N_CA_C_filter.sum()) # 110.1 degrees
    angle_tolerance.append([0.035] * N_CA_C_filter.sum())
    CA_C_O_filter = np.all(atom_names[dataset_angle_idcs] == ['CA', 'C', 'O'], axis=1)
    angle_idcs.append(dataset_angle_idcs[CA_C_O_filter])
    angle_eq_val.append([2.1031217] * CA_C_O_filter.sum()) # 120.5 degrees
    angle_tolerance.append([0.035] * CA_C_O_filter.sum())
    CA_C_N_filter = np.all(atom_names[dataset_angle_idcs] == ['CA', 'C', 'N'], axis=1)
    angle_idcs.append(dataset_angle_idcs[CA_C_N_filter])
    angle_eq_val.append([2.0350539] * CA_C_N_filter.sum()) # 116.6 degrees
    angle_tolerance.append([0.035] * CA_C_N_filter.sum())
    O_C_N_filter = np.all(atom_names[dataset_angle_idcs] == ['O', 'C', 'N'], axis=1)
    angle_idcs.append(dataset_angle_idcs[O_C_N_filter])
    angle_eq_val.append([2.14675] * O_C_N_filter.sum()) # 123.0 degrees
    angle_tolerance.append([0.035] * O_C_N_filter.sum())
    C_N_CA_filter = np.all(atom_names[dataset_angle_idcs] == ['C', 'N', 'CA'], axis=1)
    angle_idcs.append(dataset_angle_idcs[C_N_CA_filter])
    angle_eq_val.append([2.1275564] * C_N_CA_filter.sum()) # 121.9 degrees
    angle_tolerance.append([0.035] * C_N_CA_filter.sum())
    
    # ------------------------------------------------------------------------------- #

    bond_idcs = np.concatenate(bond_idcs)
    bond_eq_val = np.concatenate(bond_eq_val)
    bond_tolerance = np.concatenate(bond_tolerance)

    angle_idcs = np.concatenate(angle_idcs)
    angle_eq_val = np.concatenate(angle_eq_val)
    angle_tolerance = np.concatenate(angle_tolerance)

    data = {
        "bond_idcs": bond_idcs,
        "bond_eq_val": bond_eq_val,
        "bond_tolerance": bond_tolerance,
        "angle_idcs": angle_idcs,
        "angle_eq_val": angle_eq_val,
        "angle_tolerance": angle_tolerance,
    }
    
    return data

def update_minimiser_data(minimiser_data: Dict, dataset: Dict):
    atom_names = dataset[DataDict.ATOM_NAMES]
    dataset_torsion_idcs = dataset[DataDict.TORSION_IDCS]

    omega_idcs = dataset_torsion_idcs[np.all(atom_names[dataset_torsion_idcs] == np.array(['CA', 'C', 'N', 'CA']), axis=-1)]
    omega_values = np.array([np.pi] * len(omega_idcs))
    omega_tolerance = np.array([0.436332] * len(omega_idcs)) # 25 deg

    bead_names = dataset[DataDict.BEAD_NAMES]
    bb_bead_idcs = np.isin(bead_names, ['BB'])
    bb_bead_coords = dataset[DataDict.BEAD_POSITION][:, bb_bead_idcs]
    bb_atom_idcs: np.ma.masked_array = dataset[DataDict.BEAD2ATOM_RECONSTRUCTED_IDCS][bb_bead_idcs]
    bb_atom_names = atom_names[bb_atom_idcs]
    bb_atom_names[bb_atom_idcs.mask] = ''
    bb_atom_idcs[~np.isin(bb_atom_names, ['CA', 'C', 'N', 'O'])] = -1
    bb_atom_weights: np.ma.masked_array = dataset[DataDict.BEAD2ATOM_RECONSTRUCTED_WEIGHTS][bb_bead_idcs]

    data = {
        "coords":          dataset[DataDict.ATOM_POSITION_PRED],
        "bb_bead_coords":  bb_bead_coords,
        "bb_atom_idcs":    bb_atom_idcs,
        "bb_atom_weights": bb_atom_weights,
        "atom_names":      atom_names,
        "omega_idcs":      omega_idcs,
        "omega_values":    omega_values,
        "omega_tolerance": omega_tolerance,
    }

    if DataDict.BB_PHIPSI_PRED in dataset and dataset[DataDict.BB_PHIPSI_PRED].shape[-1] == 2:
        pred_torsion_values = dataset[DataDict.BB_PHIPSI_PRED][0, bb_bead_idcs]

        phi_torsion_idcs = dataset_torsion_idcs[np.all(atom_names[dataset_torsion_idcs] == np.array(['C', 'N', 'CA', 'C']), axis=-1)]
        psi_torsion_idcs = dataset_torsion_idcs[np.all(atom_names[dataset_torsion_idcs] == np.array(['N', 'CA', 'C', 'N']), axis=-1)]
        
        phi_idcs = []
        phi_values = []
        psi_idcs = []
        psi_values = []

        for i, (phi_value, psi_value) in enumerate(pred_torsion_values):
            if i > 0:
                phi_idcs.append(phi_torsion_idcs[i - 1])
                phi_values.append(phi_value)
            if i < len(pred_torsion_values) - 1:
                psi_idcs.append(psi_torsion_idcs[i])
                psi_values.append(psi_value)

        phi_idcs = np.stack(phi_idcs, axis=0)
        phi_values = np.stack(phi_values, axis=0)
        psi_idcs = np.stack(psi_idcs, axis=0)
        psi_values = np.stack(psi_values, axis=0)

        data.update({
            "phi_idcs": phi_idcs,
            "phi_values": phi_values,
            "psi_idcs": psi_idcs,
            "psi_values": psi_values,
        })
    
    minimiser_data.update(data)

    return minimiser_data

    # def rotate_dihedrals_to_minimize_energy(self, minimiser_data: Dict, dataset: Dict):
    #     pred_pos = minimiser_data["coords"][0].detach().clone()
    #     ca_pos = pred_pos[np.isin(dataset[DataDict.ATOM_NAMES], ['CA'])]

    #     pi_quarters_rotated_pred_pos = rotate_residue_dihedrals(pos=pred_pos, ca_pos=ca_pos, angle=np.pi/4)
    #     pi_halves_rotated_pred_pos = rotate_residue_dihedrals(pos=pred_pos, ca_pos=ca_pos, angle=np.pi/2)
    #     pi_three_quarters_rotated_pred_pos = rotate_residue_dihedrals(pos=pred_pos, ca_pos=ca_pos, angle=np.pi*3/4)
    #     pi_rotated_pred_pos = rotate_residue_dihedrals(pos=pred_pos, ca_pos=ca_pos, angle=np.pi)
    #     minus_pi_quarters_rotated_pred_pos = rotate_residue_dihedrals(pos=pred_pos, ca_pos=ca_pos, angle=-np.pi/4)
    #     minus_pi_halves_rotated_pred_pos = rotate_residue_dihedrals(pos=pred_pos, ca_pos=ca_pos, angle=-np.pi/2)
    #     minus_pi_three_quarters_rotated_pred_pos = rotate_residue_dihedrals(pos=pred_pos, ca_pos=ca_pos, angle=-np.pi*3/4)
    #     all_rotated_pos = [
    #         pi_quarters_rotated_pred_pos,
    #         pi_halves_rotated_pred_pos,
    #         pi_three_quarters_rotated_pred_pos,
    #         pi_rotated_pred_pos,
    #         minus_pi_quarters_rotated_pred_pos,
    #         minus_pi_halves_rotated_pred_pos,
    #         minus_pi_three_quarters_rotated_pred_pos,
    #     ]

    #     # Rotate and evaluate one residue at a time
    #     updated_pos = pred_pos.clone()
    #     baseline_energy = self.minimiser.evaluate_dihedral_energy(minimiser_data, pos=pred_pos)
    #     for x in range(len(ca_pos)-1):
    #         temp_updated_pos = updated_pos.clone()
    #         C_id = x*4+2
    #         N_id = x*4+4
    #         energies = []
    #         for rotated_pos in all_rotated_pos:
    #             temp_updated_pos[C_id] = rotated_pos[C_id]
    #             temp_updated_pos[N_id] = rotated_pos[N_id]
    #             energies.append(self.minimiser.evaluate_dihedral_energy(minimiser_data, pos=temp_updated_pos))
    #         min_energy_id = torch.stack(energies).argmin()
    #         best_energy = energies[min_energy_id]
    #         if best_energy < baseline_energy:
    #             baseline_energy = best_energy
    #             updated_pos[C_id] = all_rotated_pos[min_energy_id][C_id]
    #             updated_pos[N_id] = all_rotated_pos[min_energy_id][N_id]

    #     minimiser_data["coords"] = updated_pos
    #     return minimiser_data

# def rotate_residue_dihedrals(pos: torch.TensorType, ca_pos: torch.TensorType, angle: float):
#     rot_axes = ca_pos[1:] - ca_pos[:-1]
#     rot_axes = rot_axes / torch.norm(rot_axes, dim=-1, keepdim=True)
#     rot_axes = rot_axes.repeat_interleave(2 * torch.ones((len(rot_axes),), dtype=int, device=rot_axes.device), dim=0)

#     angles_polar = 0.5 * angle * torch.ones((len(rot_axes),), dtype=float, device=rot_axes.device).reshape(-1, 1)

#     q_polar = get_quaternions(
#         batch=1,
#         rot_axes=rot_axes,
#         angles=angles_polar
#     )

#     C_N_O_fltr = torch.zeros((len(pos),), dtype=bool, device=rot_axes.device)
#     for x in range(len(ca_pos)-1):
#         C_N_O_fltr[x*4+2] = True
#         C_N_O_fltr[x*4+4] = True

#     v_ = pos[C_N_O_fltr] - ca_pos[:-1].repeat_interleave(2 * torch.ones((len(ca_pos[:-1]),), dtype=int, device=ca_pos.device), dim=0)
#     v_rotated = qv_mult(q_polar, v_)

#     rotated_pred_pos = pos.clone()
#     rotated_pred_pos[C_N_O_fltr] = ca_pos[:-1].repeat_interleave(2 * torch.ones((len(ca_pos[:-1]),), dtype=int, device=ca_pos.device), dim=0) + v_rotated
#     return rotated_pred_pos

# def adjust_bb_oxygens(dataset: Dict, position_key: str = DataDict.ATOM_POSITION_PRED):
#     atom_CA_idcs = dataset[DataDict.CA_ATOM_IDCS]
#     no_cappings_filter = [x.split('_')[0] not in ['ACE', 'NME'] for x in dataset[DataDict.ATOM_NAMES]]
#     atom_C_idcs = np.array([ncf and an.split('_')[1] in ["C"] for ncf, an in zip(no_cappings_filter, dataset[DataDict.ATOM_NAMES])])
#     atom_O_idcs = np.array([ncf and an.split('_')[1] in ["O"] for ncf, an in zip(no_cappings_filter, dataset[DataDict.ATOM_NAMES])])

#     ca_pos = dataset[position_key][:, atom_CA_idcs]

#     c_o_vectors = []
#     for i in range(ca_pos.shape[1]-2):
#         ca_i, ca_ii, ca_iii = ca_pos[:, i], ca_pos[:, i+1], ca_pos[:, i+2]
#         ca_i_ca_ii = ca_ii - ca_i
#         ca_i_ca_iii = ca_iii - ca_i
#         c_o = np.cross(ca_i_ca_ii, ca_i_ca_iii, axis=1)
#         c_o = c_o / np.linalg.norm(c_o, axis=-1, keepdims=True) * 1.229 # C-O bond legth
#         c_o_vectors.append(c_o)
#     # Last missing vectors
#     for _ in range(len(c_o_vectors), atom_C_idcs.sum()):
#         c_o_vectors.append(c_o)
#     c_o_vectors = np.array(c_o_vectors).swapaxes(0,1)

#     o_pos = dataset[position_key][:, atom_C_idcs] + c_o_vectors
#     dataset[position_key][:, atom_O_idcs] = o_pos

#     pos = torch.from_numpy(dataset[position_key]).float()
#     omega_dihedral_idcs = torch.from_numpy(dataset[DataDict.OMEGA_DIH_IDCS]).long()
#     adjusted_pos = adjust_oxygens(
#         pos=pos,
#         omega_dihedral_idcs=omega_dihedral_idcs,
#     )
#     dataset[position_key] = adjusted_pos.cpu().numpy()

#     return dataset

# def adjust_oxygens(
#         pos: torch.Tensor, # (batch, n_atoms, 3)
#         omega_dihedral_idcs: torch.Tensor, # (n_dih, 4)
#     ):
#     batch = pos.size(0)

#     # ADJUST DIHEDRAL ANGLE [O, C, N, CA]
#     dih_values = get_dihedrals(pos, omega_dihedral_idcs) # (batch, n_dih)

#     # omega_dihedral_idcs represent atoms [O, C, N, CA]
#     v_ = pos[:, omega_dihedral_idcs[:, 0]] - pos[:, omega_dihedral_idcs[:, 1]]
#     v_ = v_.reshape(-1, 3) # (batch * n_peptide_oxygens, xyz)

#     rot_axes = pos[:, omega_dihedral_idcs[:, 2]] - pos[:, omega_dihedral_idcs[:, 1]]
#     rot_axes = rot_axes / rot_axes.norm(dim=-1, keepdim=True)
#     rot_axes = rot_axes.reshape(-1, 3) # (batch * n_peptide_oxygens, xyz)

#     # We want that the final dihedral values are 0, so we rotate of dih_values
#     angles_polar = dih_values.reshape(-1, 1) # (n_peptide_oxygens, 1)
#     q_polar = get_quaternions(               # (batch * n_peptide_oxygens, 4)
#         batch=batch,
#         rot_axes=rot_axes,
#         angles=0.5*angles_polar
#     )

#     v_rotated = qv_mult(q_polar, v_).reshape(batch, -1, 3) # (batch, n_peptide_oxygens, xyz)
#     adjusted_pos = pos.clone()
#     adjusted_pos[:, omega_dihedral_idcs[:, 0]] = adjusted_pos[:, omega_dihedral_idcs[:, 1]] + v_rotated

#     # ADJUST ANGLE [O, C, N]
#     angle_values, b0, b1 = get_angles(adjusted_pos, omega_dihedral_idcs[:, :3], return_vectors=True)
#     # angle_values (batch, n_angles)
#     # b0 (batch, n_angles, xyz) C->O
#     # b1 (batch, n_angles, xyz) C->N

#     v_ = adjusted_pos[:, omega_dihedral_idcs[:, 0]] - adjusted_pos[:, omega_dihedral_idcs[:, 1]]
#     v_ = v_.reshape(-1, 3) # (batch * n_peptide_oxygens, xyz)

#     cross_prod = o3.TensorProduct(
#         "1x1o",
#         "1x1o",
#         "1x1e",
#         [(0, 0, 0, "uuu", False)],
#         irrep_normalization='none',
#     )
#     rot_axes = cross_prod(b0, b1) # (batch, n_peptide_oxygens, xyz)
#     rot_axes = rot_axes / rot_axes.norm(dim=-1, keepdim=True)
#     rot_axes = rot_axes.reshape(-1, 3) # (batch * n_peptide_oxygens, xyz)

#     angles_polar = angle_values.reshape(-1, 1) - 2.145 # (n_peptide_oxygens, 1) | O-C-N angle eq value is 122.9 degrees
#     angles_polar[angles_polar > np.pi] -= 2*np.pi
#     angles_polar[angles_polar < -np.pi] += 2*np.pi
#     q_polar = get_quaternions(               # (batch * n_peptide_oxygens, 4)
#         batch=batch,
#         rot_axes=rot_axes,
#         angles=0.5*angles_polar
#     )

#     v_rotated = qv_mult(q_polar, v_).reshape(batch, -1, 3) # (batch, n_peptide_oxygens, xyz)
#     adjusted_pos[:, omega_dihedral_idcs[:, 0]] = adjusted_pos[:, omega_dihedral_idcs[:, 1]] + v_rotated

#     return adjusted_pos