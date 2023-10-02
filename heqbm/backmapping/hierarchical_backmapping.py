import os
import yaml
import torch

import numpy as np
import MDAnalysis as mda

from e3nn import o3
from pathlib import Path
from typing import Dict, Optional

from heqbm.mapper.hierarchical_mapper import HierarchicalMapper
from heqbm.backmapping.nn.quaternions import get_quaternions, qv_mult
from heqbm.utils import DataDict
from heqbm.utils.atomType import get_type_from_name
from heqbm.utils.geometry import get_RMSD, get_angles, get_dihedrals
from heqbm.utils.backbone import cat_interleave, MinimizeEnergy
from heqbm.utils.plotting import plot_cg
from heqbm.utils.pdbFixer import fixPDB

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

class HierarchicalBackmapping:

    config_filename: str
    config: Dict[str, str]
    mapping: HierarchicalMapper

    model_config: Dict[str, str]
    model_r_max: float

    minimizer: MinimizeEnergy

    def __init__(
        self,
        config_filename: str,
    ) -> None:
        
        ### --------------------- ###

        self.config_filename = config_filename
        self.config: Dict[str, str] = yaml.safe_load(Path(self.config_filename).read_text())

        ### Parse Input ###
        
        self.mapping = HierarchicalMapper(root=self.config.get("mapping_root", None))
        self.mapping.map(self.config, selection=self.config.get("selection", "all"))

        ### Load Model ###

        model_config_file = self.config.get("model_config_file", None)
        model_config: Optional[Dict[str, str]] = yaml.load(
            Path(model_config_file).read_text(), Loader=yaml.Loader
        ) if model_config_file is not None else None
        self.model, training_model_config = load_model(
            model_dir=self.config.get("model_dir", None),
            model_config=model_config,
            config=self.config,
        )
        self.model_r_max = float(training_model_config["r_max"])

        ### Initialize energy minimizer for reconstructing backbone ###

        self.minimizer = MinimizeEnergy()

        ### --------------------- ###
    
    def plot(self, frame_index=0):
        plot_cg(
            dataset=self.mapping.dataset,
            frame_index=frame_index
        )
    
    def backmap_scl(self, backmapping_dataset: Dict):
        reconstructed_atom_pos = run_b2a_rel_vec_backmapping(dataset=backmapping_dataset, use_predicted_b2a_rel_vec=True)
        backmapping_dataset[DataDict.ATOM_POSITION_PRED] = reconstructed_atom_pos
        return backmapping_dataset
    
    def adjust_beads(self, backmapping_dataset: Dict):
        backmapping_dataset[DataDict.BEAD_POSITION_ORIGINAL] = np.copy(backmapping_dataset[DataDict.BEAD_POSITION])
        minimizer_data = build_minimizer_data(dataset=backmapping_dataset)

        unlock_ca = not self.config.get("lock_ca", not self.config.get("simulation_is_cg", False))
        if unlock_ca:
            self.minimizer.minimize(
                data=minimizer_data,
                dtau=self.config.get("bb_minimization_dtau", 1e-1),
                eps=self.config.get("bb_minimization_eps_initial", 1e-2),
                minimize_dih=False,
                unlock_ca=True,
            )
        
            ca_shifts = minimizer_data["pos"][1::3].detach().numpy() - backmapping_dataset[DataDict.BEAD_POSITION][0, backmapping_dataset[DataDict.CA_BEAD_IDCS]]
            residue_shifts = np.repeat(ca_shifts, minimizer_data["per_residue_beads"], axis=0)
            backmapping_dataset[DataDict.BEAD_POSITION][0] += residue_shifts
        return backmapping_dataset, minimizer_data
    
    def optimize_backbone(self, backmapping_dataset: Dict, minimizer_data: Dict):

        # Add predicted dihedrals as dihedral equilibrium values
        minimizer_data = update_minimizer_data(
            minimizer_data=minimizer_data,
            dataset=backmapping_dataset,
            use_only_bb_beads=self.config.get("bb_use_only_bb_beads", False),
        )
        
        # Step 1: initial minimization: Rotate local minimum to find global minimum basin
        self.minimizer.minimize(
            data=minimizer_data,
            dtau=self.config.get("bb_minimization_dtau", 1e-1),
            eps=self.config.get("bb_minimization_eps", 1e-2),
            minimize_dih=False,
        )

        # Step 2: rotate N and C atoms around the CA-CA axis to find global minimum basin
        minimizer_data = self.rotate_dihedrals_to_minimize_energy(minimizer_data=minimizer_data, dataset=backmapping_dataset)
        
        # Step 3: final minimization: find global minimum
        self.minimizer.minimize(
            data=minimizer_data,
            dtau=self.config.get("bb_minimization_dtau", 1e-1),
            eps=self.config.get("bb_minimization_eps", 1e-4),
            minimize_dih=True,
        )

        backmapping_dataset[DataDict.BB_ATOM_POSITION_PRED] = minimizer_data["pos"][minimizer_data["real_bb_atom_idcs"]].detach().numpy()
        return backmapping_dataset, minimizer_data
        
    def backmap(self, frame_index: int, optimize_backbone: bool = True):
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
        
        if DataDict.CA_BEAD_IDCS not in backmapping_dataset or (backmapping_dataset[DataDict.CA_BEAD_IDCS].sum() == 0):
            optimize_backbone = False
        
        # Adjust CG CA bead positions, if they don't pass structure quality checks
        backmapping_dataset, minimizer_data = self.adjust_beads(backmapping_dataset=backmapping_dataset)

        # Predict dihedrals and bead2atom relative vectors
        backmapping_dataset = run_backmapping_inference(
            dataset=backmapping_dataset,
            model=self.model,
            r_max=self.model_r_max,
            use_only_bb_beads=self.config.get("bb_use_only_bb_beads", False),
        )

        if DataDict.BEAD2ATOM_RELATIVE_VECTORS in backmapping_dataset:
            bb_fltr = np.array([x.split('_')[1] in ['BB', 'RE'] for x in backmapping_dataset[DataDict.BEAD_NAMES]])
            b2a_rev_vec = backmapping_dataset[DataDict.BEAD2ATOM_RELATIVE_VECTORS][0].copy()

            try:
                print(
                    "RMSD on all b2a relative vectors:", 
                    get_RMSD(backmapping_dataset[DataDict.BEAD2ATOM_RELATIVE_VECTORS_PRED], b2a_rev_vec, ignore_zeroes=True)
                )

                b2a_rev_vec[bb_fltr] = 0. # We are not interested in b2a_rev_vec of backbone

                print(
                    "RMSD on side-chain b2a relative vectors:", 
                    get_RMSD(backmapping_dataset[DataDict.BEAD2ATOM_RELATIVE_VECTORS_PRED], b2a_rev_vec, ignore_zeroes=True)
                )
            except:
                pass

        # Backmap backbone
        if optimize_backbone:
            backmapping_dataset, minimizer_data = self.optimize_backbone(backmapping_dataset=backmapping_dataset, minimizer_data=minimizer_data)
        
        # Backmap side chains & ligands
        # if DataDict.ATOM_POSITION_PRED not in backmapping_dataset:
        #     backmapping_dataset = self.backmap_scl(backmapping_dataset=backmapping_dataset)
        
        if DataDict.ATOM_POSITION in backmapping_dataset:
            try:
                no_bb_fltr = np.array([an.split('_')[1] not in ["CA", "C", "N", "O"] for an in backmapping_dataset[DataDict.ATOM_NAMES]])
                print(f"RMSD on Side-Chains: {get_RMSD(backmapping_dataset[DataDict.ATOM_POSITION_PRED], backmapping_dataset[DataDict.ATOM_POSITION], fltr=no_bb_fltr):4.3f} Angstrom")
            except:
                pass

        if optimize_backbone:
            n_atom_idcs = minimizer_data["n_atom_idcs"]
            ca_atom_idcs = backmapping_dataset[DataDict.CA_ATOM_IDCS]
            c_atom_idcs = minimizer_data["c_atom_idcs"]
            backmapping_dataset[DataDict.ATOM_POSITION_PRED][0, n_atom_idcs + ca_atom_idcs + c_atom_idcs] = backmapping_dataset[DataDict.BB_ATOM_POSITION_PRED]
            backmapping_dataset = adjust_bb_oxygens(dataset=backmapping_dataset)
        
        return backmapping_dataset
    
    def rotate_dihedrals_to_minimize_energy(self, minimizer_data: Dict, dataset: Dict):
        pred_pos = minimizer_data["pos"].detach().clone()
        ca_pos = torch.from_numpy(dataset[DataDict.CA_BEAD_POSITION][0])

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
            C_id = x*3+2
            N_id = x*3+3
            energies = []
            for rotated_pos in all_rotated_pos:
                temp_updated_pos[C_id] = rotated_pos[C_id]
                temp_updated_pos[N_id] = rotated_pos[N_id]
                energies.append(self.minimizer.evaluate_dihedral_energy(minimizer_data, pos=temp_updated_pos))
            min_energy_id = np.argmin(np.array(energies))
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
            selection: str,
            folder: str,
            atom_filter: Optional[np.ndarray] = None,
            previous_u: Optional[mda.Universe] = None,
        ):
        u = mda.Universe(self.config.get("structure_filename"), *self.config.get("traj_filenames", []))
        u.trajectory[frame_index]

        os.makedirs(folder, exist_ok=True)

        # Write original CG
        bead_resindex = backmapping_dataset[DataDict.BEAD_RESIDCS]
        n_atoms = len(backmapping_dataset[DataDict.BEAD_POSITION][0])
        n_residues = len(np.unique(bead_resindex))
        atom_resindex = np.bincount(bead_resindex)
        atom_resindex = atom_resindex[np.nonzero(atom_resindex)[0]]
        atom_resindex = np.repeat(np.arange(n_residues), atom_resindex)
        CG_u = mda.Universe.empty(n_atoms=n_atoms,
                                n_residues=n_residues,
                                atom_resindex=atom_resindex,
                                trajectory=True) # necessary for adding coordinates
        CG_u.add_TopologyAttr('name', [bn.split('_')[1] for bn in backmapping_dataset[DataDict.BEAD_NAMES]])
        CG_u.add_TopologyAttr('type', ['X' for _ in backmapping_dataset[DataDict.BEAD_CHAINIDCS]])
        CG_u.add_TopologyAttr('resname', [bn.split('_')[0] for bn in backmapping_dataset[DataDict.BEAD_NAMES][np.unique(bead_resindex, return_index=True)[1]]])
        CG_u.add_TopologyAttr('resid', np.unique(bead_resindex))
        CG_u.add_TopologyAttr('chainIDs', backmapping_dataset[DataDict.BEAD_CHAINIDCS])
        CG_sel = CG_u.select_atoms('all')
        CG_sel.positions = backmapping_dataset.get(DataDict.BEAD_POSITION_ORIGINAL, backmapping_dataset[DataDict.BEAD_POSITION])[0]
        CG_sel.write(os.path.join(folder, f"original_CG_{frame_index}.pdb"))

        # Write optimized CG
        if (DataDict.BEAD_POSITION_ORIGINAL in backmapping_dataset):
            optimized_bead_position = backmapping_dataset[DataDict.BEAD_POSITION][0]
            if np.any(~np.isclose(CG_sel.positions, optimized_bead_position)):
                CG_sel.positions = optimized_bead_position
                CG_sel.write(os.path.join(folder, f"final_CG_{frame_index}.pdb"))
        
        # Write backmapped
        atom_resindex = []
        atomnames = []
        resnames = []
        last_res_resname = None
        last_res_atomname = None
        resid = -1
        for an in backmapping_dataset[DataDict.ATOM_NAMES]:
            resname, atomname = an.split(DataDict.STR_SEPARATOR)
            if resname != last_res_resname:
                resid += 1
                last_res_resname = resname
                last_res_atomname = atomname
                resnames.append(resname)
            elif resname == last_res_resname and atomname == last_res_atomname:
                resid += 1
                resnames.append(resname)
            atomnames.append(atomname)
            atom_resindex.append(resid)
        resnames = np.array(resnames)
        atomnames = np.array(atomnames)
        atom_resindex = np.array(atom_resindex)

        if atom_filter is None:
            atom_filter = np.array([True for _ in atomnames])
        if previous_u is None:
            n_atoms = len(backmapping_dataset[DataDict.ATOM_POSITION_PRED][0][atom_filter])
            n_residues = len(backmapping_dataset[DataDict.ATOM_POSITION_PRED][0][atom_filter])
            backmapped_u = mda.Universe.empty(
                n_atoms,
                n_residues=len(np.unique(atom_resindex)),
                atom_resindex=atom_resindex[atom_filter],
                residue_segindex=np.unique(atom_resindex)+1,
                trajectory=True # necessary for adding coordinates
            )
            coordinates = np.empty((
                n_frames,  # number of frames
                n_atoms,
                3,
            ))
            backmapped_u.load_new(coordinates, order='fac')
        else:
            backmapped_u = previous_u
        
        residue_idcs_filter = []
        for resindex, resname in zip(np.unique(atom_resindex), resnames):
            residue_id_filter = np.argwhere(atom_resindex == resindex).flatten()
            residue_atomnames = atomnames[residue_id_filter]
            if 'C' in residue_atomnames \
            and 'O' in residue_atomnames \
            and resname in DataDict.RESNAMES:
                C_id = np.argwhere(residue_atomnames == 'C').item()
                O_id = np.argwhere(residue_atomnames == 'O').item()
                C_index = residue_id_filter[C_id]
                O_index = residue_id_filter[O_id]
                residue_id_filter[C_id:-2] = residue_id_filter[O_id+1:]
                residue_id_filter[-2] = C_index
                residue_id_filter[-1] = O_index
            residue_idcs_filter.append(residue_id_filter)
        residue_idcs_filter = np.concatenate(residue_idcs_filter)

        backmapped_u.add_TopologyAttr('name', atomnames[residue_idcs_filter][atom_filter].tolist())
        backmapped_u.add_TopologyAttr('type', np.array([get_type_from_name(an) for an in backmapping_dataset[DataDict.ATOM_NAMES][atom_filter]])[residue_idcs_filter])
        backmapped_u.add_TopologyAttr('resname', resnames)
        backmapped_u.add_TopologyAttr('resid', np.unique(atom_resindex)+1)
        backmapped_u.add_TopologyAttr('chainIDs', backmapping_dataset[DataDict.ATOM_CHAINIDCS])

        backmapped_u.trajectory[frame_index]
        backmapped_sel = backmapped_u.select_atoms(selection)

        positions_pred = backmapping_dataset[DataDict.ATOM_POSITION_PRED][0][atom_filter]
        positions_pred = np.nan_to_num(positions_pred, nan=0.)
        backmapped_sel.positions = positions_pred[residue_idcs_filter]

        backmapped_filename = os.path.join(folder, f"backmapped_{frame_index}.pdb") 
        backmapped_sel.write(backmapped_filename)
        topology, positions = fixPDB(backmapped_filename)
        PDBFile.writeFile(topology, positions, open(os.path.join(folder, f"backmapped_fixed_{frame_index}.pdb"), 'w'))

        if not self.config.get("simulation_is_cg", False):
            # Write true atomistic
            sel = u.select_atoms(selection)
            true_filename = os.path.join(folder, f"true_{frame_index}.pdb")
            sel.write(true_filename)
            topology, positions = fixPDB(true_filename)
            PDBFile.writeFile(topology, positions, open(os.path.join(folder, f"true_fixed_{frame_index}.pdb"), 'w'))
        return backmapped_u

def load_model(config: Dict, model_dir: Optional[Path] = None, model_config: Optional[Dict] = None):
    if model_dir is None:
        assert model_config is not None, "You should provide either 'model_config_file' or 'model_dir' in the configuration file"
        model_dir = os.path.join(model_config.get("root"), model_config.get("run_name"))
    model_name = config.get("model_relative_weights", "best_model.pth")
    
    global_config = os.path.join(model_dir, "config.yaml")
    global_config = Config.from_file(str(global_config), defaults={})
    _set_global_options(global_config)
    del global_config

    model, training_model_config = Trainer.load_model_from_training_session(
        traindir=model_dir,
        model_name=model_name,
    )

    model = model.to(config.get("model_device", "cuda" if torch.cuda.is_available() else "cpu")).eval()
    return model, training_model_config
    
def get_edge_index(positions: torch.Tensor, r_max: float):
    dist_matrix = torch.norm(positions[:, None, ...] - positions[None, ...], dim=-1).fill_diagonal_(torch.inf)
    return torch.argwhere(dist_matrix <= r_max).T.long()

def run_backmapping_inference(dataset: Dict, model: torch.nn.Module, r_max: float, use_only_bb_beads: bool):
    device = next(model.parameters()).device

    bead_pos = dataset[DataDict.BEAD_POSITION][0]
    bead_types = dataset[DataDict.BEAD_TYPES]
    if use_only_bb_beads:
        bead_pos = bead_pos[dataset[DataDict.CA_BEAD_IDCS]]
        bead_types = bead_types[dataset[DataDict.CA_BEAD_IDCS]]
    bead_pos = torch.from_numpy(bead_pos).float().to(device)
    bead_types = torch.from_numpy(bead_types).long().reshape(-1, 1).to(device)
    edge_index = get_edge_index(positions=bead_pos, r_max=r_max).to(device)
    batch = torch.zeros(len(bead_pos), device=device, dtype=torch.long)
    bead2atom_idcs = torch.from_numpy(dataset[DataDict.BEAD2ATOM_IDCS]).long().to(device)
    lvl_idcs_mask = torch.from_numpy(dataset[DataDict.LEVEL_IDCS_MASK]).bool().to(device)
    lvl_idcs_anchor_mask = torch.from_numpy(dataset[DataDict.LEVEL_IDCS_ANCHOR_MASK]).long().to(device)
    data = {
        AtomicDataDict.POSITIONS_KEY: bead_pos,
        f"{AtomicDataDict.POSITIONS_KEY}_slices": torch.tensor([0, len(bead_pos)]),
        AtomicDataDict.ATOM_TYPE_KEY: bead_types,
        AtomicDataDict.ORIG_EDGE_INDEX_KEY: edge_index,
        AtomicDataDict.EDGE_INDEX_KEY: edge_index,
        AtomicDataDict.BATCH_KEY: batch,
        DataDict.BEAD2ATOM_IDCS: bead2atom_idcs,
        f"{DataDict.BEAD2ATOM_IDCS}_slices": torch.tensor([0, len(bead2atom_idcs)]),
        DataDict.LEVEL_IDCS_MASK: lvl_idcs_mask,
        f"{DataDict.LEVEL_IDCS_MASK}_slices": torch.tensor([0, len(lvl_idcs_mask)]),
        DataDict.LEVEL_IDCS_ANCHOR_MASK: lvl_idcs_anchor_mask,
        f"{DataDict.ATOM_POSITION}_slices": torch.tensor([0, bead2atom_idcs.max().item()+1])
    }

    with torch.no_grad():
        out = model(data)
        predicted_dih = out.get(INVARIANT_ATOM_FEATURES).cpu().numpy()
        predicted_b2a_rel_vec = out.get(EQUIVARIANT_ATOM_FEATURES).cpu().numpy()
        reconstructed_atom_pos = out.get(ATOM_POSITIONS, None)
        if reconstructed_atom_pos is not None:
            reconstructed_atom_pos = reconstructed_atom_pos.cpu().numpy()
    del out

    dataset[DataDict.BB_PHIPSI_PRED] = predicted_dih[None, ...]
    dataset[DataDict.BEAD2ATOM_RELATIVE_VECTORS_PRED] = predicted_b2a_rel_vec[None, ...]
    if reconstructed_atom_pos is not None:
        dataset[DataDict.ATOM_POSITION_PRED] = reconstructed_atom_pos[None, ...]

    return dataset

def build_minimizer_data(dataset: Dict):
    unique_residcs = np.unique(dataset[DataDict.ATOM_RESIDCS])
    ca_pos = dataset[DataDict.CA_ATOM_POSITION][0]
    n_dih = dataset[DataDict.CA_BEAD_IDCS].sum()

    atom_pos_pred = cat_interleave(
        [
            np.zeros_like(ca_pos),
            ca_pos,
            np.zeros_like(ca_pos),
        ],
    ).astype(np.float64)

    bond_idcs = []
    bond_eq_val = []
    bond_tolerance = []

    ca_bond_idcs = []
    ca_bond_eq_val = []
    ca_bond_tolerance = []

    angle_idcs = []
    angle_eq_val = []
    angle_tolerance = []

    for id_, resid, next_resid in zip(range(0, n_dih-1), unique_residcs[:-1], unique_residcs[1:]):
        atom_id = id_ * 3 # N
        
        bond_idcs.append([atom_id, atom_id+1])
        bond_eq_val.append(1.45) # N - CA bond length
        bond_tolerance.append(0.03)

        bond_idcs.append([atom_id+1, atom_id+2])
        bond_eq_val.append(1.52) # CA - C bond length
        bond_tolerance.append(0.03)
        
        if resid == next_resid - 1:
            bond_idcs.append([atom_id+1, atom_id+4])
            bond_eq_val.append(3.81) # CA - CA bond length ranges from 3.77 to 3.85
            bond_tolerance.append(0.05)
            ca_bond_idcs.append([atom_id+1, atom_id+4])
            ca_bond_eq_val.append(3.81) # CA - CA bond length ranges from 3.78 to 3.83
            ca_bond_tolerance.append(0.03)

            bond_idcs.append([atom_id+2, atom_id+3])
            bond_eq_val.append(1.34) # C - N bond length
            bond_tolerance.append(0.03)

        angle_idcs.append([atom_id, atom_id+1, atom_id+2])
        angle_eq_val.append(1.9216075) # N - CA - C angle value
        angle_tolerance.append(0.035)  # Tolerance 0.035 rad ~ 2 deg
        
        if resid == next_resid - 1:
            angle_idcs.append([atom_id+1, atom_id+2, atom_id+3])
            angle_eq_val.append(2.0350539) # CA - C - N angle value
            angle_tolerance.append(0.035)  # Tolerance 0.035 rad ~ 2 deg
        
            angle_idcs.append([atom_id+2, atom_id+3, atom_id+4])
            angle_eq_val.append(2.1275564) # C - N - CA angle value
            angle_tolerance.append(0.035)  # Tolerance 0.035 rad ~ 2 deg
    
    atom_id = (id_ + 1) * 3 # N    
    bond_idcs.append([atom_id, atom_id+1])
    bond_eq_val.append(1.45) # N - CA bond length
    bond_tolerance.append(0.03)

    bond_idcs.append([atom_id+1, atom_id+2])
    bond_eq_val.append(1.52) # CA - C bond length
    bond_tolerance.append(0.03)

    angle_idcs.append([atom_id, atom_id+1, atom_id+2])
    angle_eq_val.append(1.9216075) # N - CA - C angle value
    angle_tolerance.append(0.035)  # Tolerance 0.035 rad ~ 2 deg
    
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

    movable_pos_idcs = np.ones((len(ca_pos) * 3,), dtype=bool)
    movable_pos_idcs[np.arange(n_dih) * 3 + 1] = False

    data = {
        "pos": torch.from_numpy(atom_pos_pred),
        "bond_idcs": torch.from_numpy(bond_idcs),
        "bond_eq_val": torch.from_numpy(bond_eq_val),
        "bond_tolerance": torch.from_numpy(bond_tolerance),
        "ca_bond_idcs": torch.from_numpy(ca_bond_idcs),
        "ca_bond_eq_val": torch.from_numpy(ca_bond_eq_val),
        "ca_bond_tolerance": torch.from_numpy(ca_bond_tolerance),
        "angle_idcs": torch.from_numpy(angle_idcs),
        "angle_eq_val": torch.from_numpy(angle_eq_val),
        "angle_tolerance": torch.from_numpy(angle_tolerance),
        "movable_pos_idcs": torch.from_numpy(movable_pos_idcs),
    }

    if DataDict.BEAD_RESIDCS in dataset:
        per_residue_beads = np.bincount(dataset[DataDict.BEAD_RESIDCS] - dataset[DataDict.BEAD_RESIDCS].min())
        data.update({
            "per_residue_beads": torch.from_numpy(per_residue_beads),
        })
    
    return data

def update_minimizer_data(minimizer_data: Dict, dataset: Dict, use_only_bb_beads: bool):
    unique_residcs = np.unique(dataset[DataDict.ATOM_RESIDCS])
    n_atom_idcs = np.array([an.split('_')[-1]=='N' for an in dataset[DataDict.ATOM_NAMES]])
    ca_atom_idcs = np.array([an.split('_')[-1] in ['CA', 'CH3'] for an in dataset[DataDict.ATOM_NAMES]])
    c_atom_idcs = np.array([an.split('_')[-1]=='C' for an in dataset[DataDict.ATOM_NAMES]])

    residue_contains_n = []
    residue_contains_c = []
    for resid in unique_residcs:
        residue_atom_names = [an.split('_')[-1] for an in dataset[DataDict.ATOM_NAMES][dataset[DataDict.ATOM_RESIDCS] == resid]]
        residue_contains_n.append('N' in residue_atom_names)
        residue_contains_c.append('C' in residue_atom_names)
    residue_contains_n = np.array(residue_contains_n)
    residue_contains_c = np.array(residue_contains_c)

    ca_pos_pred = dataset[DataDict.ATOM_POSITION_PRED][0, ca_atom_idcs]
    n_pos_pred = np.zeros_like(ca_pos_pred)
    n_pos_pred[0] = 2*ca_pos_pred[0] - ca_pos_pred[1]
    n_pos_pred[residue_contains_n] = dataset[DataDict.ATOM_POSITION_PRED][0, n_atom_idcs]
    c_pos_pred = np.zeros_like(ca_pos_pred)
    c_pos_pred[-1] = 2*ca_pos_pred[-1] - ca_pos_pred[-2]
    c_pos_pred[residue_contains_c] = dataset[DataDict.ATOM_POSITION_PRED][0, c_atom_idcs]

    ca_names = dataset[DataDict.ATOM_NAMES][dataset[DataDict.CA_ATOM_IDCS]]
    n_names = np.array(['XXX_N'] * len(ca_names))
    n_names[residue_contains_n] = dataset[DataDict.ATOM_NAMES][n_atom_idcs]
    c_names = np.array(['XXX_C'] * len(ca_names))
    c_names[residue_contains_c] = dataset[DataDict.ATOM_NAMES][c_atom_idcs]

    atom_pos_pred = cat_interleave(
        [
            n_pos_pred,
            ca_pos_pred,
            c_pos_pred,
        ],
    ).astype(np.float64)

    atom_names = cat_interleave(
        [
            n_names,
            ca_names,
            c_names,
        ],
    )

    real_bb_atom_idcs = cat_interleave(
        [
            residue_contains_n,
            np.ones_like(residue_contains_n, dtype=bool),
            residue_contains_c,
        ],
    )

    dih = dataset[DataDict.BB_PHIPSI_PRED][0]
    if not use_only_bb_beads:
        dih = dih[dataset[DataDict.CA_BEAD_IDCS]]

    dih_idcs = []
    dih_eq_val = []
    for id_, dih_val in zip(range(0, len(dih)-2), dih[1:-1]):
        atom_id = id_ * 3 + 2 # start from C of first residue (N1 CA1 C1 N2 CA2 C2 ...) for the phi angle of second residue
        dih_idcs.append([atom_id, atom_id+1, atom_id+2, atom_id+3])
        dih_eq_val.append(dih_val[0])
        atom_id = id_ * 3 + 3 # start from N of second residue (N1 CA1 C1 N2 CA2 C2 ...) for the psi angle of second residue
        dih_idcs.append([atom_id, atom_id+1, atom_id+2, atom_id+3])
        dih_eq_val.append(dih_val[1])
        dih_idcs.append([atom_id+1, atom_id+2, atom_id+3, atom_id+4])
        dih_eq_val.append(np.pi)
    dih_idcs = np.array(dih_idcs)
    dih_eq_val = np.array(dih_eq_val)

    minimizer_data.update({
        "pos": torch.from_numpy(atom_pos_pred),
        "dih_idcs": torch.from_numpy(dih_idcs),
        "dih_eq_val": torch.from_numpy(dih_eq_val),
        "atom_names": atom_names,
        "n_atom_idcs": n_atom_idcs,
        "c_atom_idcs": c_atom_idcs,
        "real_bb_atom_idcs": real_bb_atom_idcs,
    })

    return minimizer_data

def rotate_residue_dihedrals(pos: torch.TensorType, ca_pos: torch.TensorType, angle: float):
    rot_axes = ca_pos[1:] - ca_pos[:-1]
    rot_axes = rot_axes / torch.norm(rot_axes, dim=-1, keepdim=True)
    rot_axes = rot_axes.repeat_interleave(2 * torch.ones((len(rot_axes),), dtype=int), dim=0)

    angles_polar = 0.5 * angle * torch.ones((len(rot_axes),), dtype=float).reshape(-1, 1)

    q_polar = get_quaternions(
        batch=1,
        rot_axes=rot_axes,
        angles=angles_polar
    )

    C_N_fltr = torch.zeros((len(pos),), dtype=bool)
    for x in range(len(ca_pos)-1):
        C_N_fltr[x*3+2] = True
        C_N_fltr[x*3+3] = True

    v_ = pos[C_N_fltr] - ca_pos[:-1].repeat_interleave(2 * torch.ones((len(ca_pos[:-1]),), dtype=int), dim=0)
    v_rotated = qv_mult(q_polar, v_)

    rotated_pred_pos = pos.clone()
    rotated_pred_pos[C_N_fltr] = ca_pos[:-1].repeat_interleave(2 * torch.ones((len(ca_pos[:-1]),), dtype=int), dim=0) + v_rotated
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

def adjust_bb_oxygens(dataset: Dict):
    atom_CA_idcs = dataset[DataDict.CA_ATOM_IDCS]
    atom_C_idcs = np.array([an.split('_')[1] in ["C"] for an in dataset[DataDict.ATOM_NAMES]])
    atom_O_idcs = np.array([an.split('_')[1] in ["O"] for an in dataset[DataDict.ATOM_NAMES]])

    ca_pos = dataset[DataDict.ATOM_POSITION_PRED][0][atom_CA_idcs]

    c_o_vectors = []
    for ca_i, ca_ii, ca_iii in zip(ca_pos[:-2], ca_pos[1:-1], ca_pos[2:]):
        ca_i_ca_ii = ca_ii - ca_i
        ca_i_ca_iii = ca_iii - ca_i
        c_o = np.cross(ca_i_ca_ii, ca_i_ca_iii)
        c_o = c_o / np.linalg.norm(c_o, axis=-1) * 1.229 # C-O bond legth
        c_o_vectors.append(c_o)
    # Last missing vectors
    for _ in range(len(c_o_vectors), atom_C_idcs.sum()):
        c_o_vectors.append(c_o)
    c_o_vectors = np.array(c_o_vectors)

    o_pos = dataset[DataDict.ATOM_POSITION_PRED][0, atom_C_idcs] + c_o_vectors
    dataset[DataDict.ATOM_POSITION_PRED][0, atom_O_idcs] = o_pos

    pos = torch.from_numpy(dataset[DataDict.ATOM_POSITION_PRED]).float()
    omega_dihedral_idcs = torch.from_numpy(dataset[DataDict.OMEGA_DIH_IDCS]).long()
    adjusted_pos = adjust_oxygens(
        pos=pos,
        omega_dihedral_idcs=omega_dihedral_idcs,
    )
    dataset[DataDict.ATOM_POSITION_PRED] = adjusted_pos.cpu().numpy()

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