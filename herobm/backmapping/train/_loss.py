import torch

from geqtrain.data import AtomicDataDict
from geqtrain.train._loss import SimpleLoss
from torch_runstats import Reduction

from herobm.utils.geometry import get_bonds, get_angles, get_dihedrals

class InvariantsLoss(SimpleLoss):

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        **kwargs,
    ):
        pred_key = pred[key]
        ref_key = ref.get(key, torch.zeros_like(pred[key], device=pred[key].device))
        idcs_mask = pred["bead2atom_reconstructed_idcs"]
        idcs_mask_slices = pred["bead2atom_reconstructed_idcs_slices"]
        atom_pos_slices = pred['atom_pos_slices']
        center_atoms = torch.unique(pred[AtomicDataDict.EDGE_INDEX_KEY][0])

        atom_bond_idcs = ref["atom_bond_idx"]
        atom_bond_idcs_slices = ref["atom_bond_idx_slices"]

        bond_pred_list, bond_ref_list = [], []
        for (b2a_idcs_from, b2a_idcs_to), (atom_bond_idx_from, atom_bond_idx_to), atom_pos_from in zip(
            zip(idcs_mask_slices[:-1], idcs_mask_slices[1:]),
            zip(atom_bond_idcs_slices[:-1], atom_bond_idcs_slices[1:]),
            atom_pos_slices[:-1],
        ):
            batch_center_atoms = center_atoms[(center_atoms>=b2a_idcs_from) & (center_atoms<b2a_idcs_to)]
            batch_recon_atom_idcs = idcs_mask[batch_center_atoms].unique()[1:] + atom_pos_from
            batch_atom_bond_idcs = atom_bond_idcs[atom_bond_idx_from:atom_bond_idx_to] + atom_pos_from
            pred_atom_bond_idcs = batch_atom_bond_idcs[torch.all(torch.isin(batch_atom_bond_idcs, batch_recon_atom_idcs), dim=1)]
            bond_pred = get_bonds(pred_key, pred_atom_bond_idcs)
            bond_ref = get_bonds(ref_key, pred_atom_bond_idcs)
            bond_pred_list.append(bond_pred)
            bond_ref_list.append(bond_ref)
        bond_pred = torch.cat(bond_pred_list, axis=0)
        bond_ref = torch.cat(bond_ref_list, axis=0)
        loss_bonds = torch.max(torch.zeros_like(bond_pred), torch.pow(bond_pred - bond_ref, 2) - 0.0009) # accept up to 0.03 Angstrom error
        if torch.any(torch.isnan(loss_bonds)):
            loss_bonds = torch.tensor([torch.inf], device=loss_bonds.device)
        else:
            loss_bonds = loss_bonds.mean()
        
        atom_angle_idcs = ref["atom_angle_idx"]
        atom_angle_idcs_slices = ref["atom_angle_idx_slices"]

        angle_pred_list, angle_ref_list = [], []
        for (b2a_idcs_from, b2a_idcs_to), (atom_angle_idx_from, atom_angle_idx_to), atom_pos_from in zip(
            zip(idcs_mask_slices[:-1], idcs_mask_slices[1:]),
            zip(atom_angle_idcs_slices[:-1], atom_angle_idcs_slices[1:]),
            atom_pos_slices[:-1],
        ):
            batch_center_atoms = center_atoms[(center_atoms>=b2a_idcs_from) & (center_atoms<b2a_idcs_to)]
            batch_recon_atom_idcs = idcs_mask[batch_center_atoms].unique()[1:] + atom_pos_from
            batch_atom_angle_idcs = atom_angle_idcs[atom_angle_idx_from:atom_angle_idx_to] + atom_pos_from
            pred_atom_angle_idcs = batch_atom_angle_idcs[torch.all(torch.isin(batch_atom_angle_idcs, batch_recon_atom_idcs), dim=1)]
            angle_pred = get_angles(pred_key, pred_atom_angle_idcs)
            angle_ref = get_angles(ref_key, pred_atom_angle_idcs)
            angle_pred_list.append(angle_pred)
            angle_ref_list.append(angle_ref)
        angle_pred = torch.cat(angle_pred_list, axis=0)
        angle_ref = torch.cat(angle_ref_list, axis=0)
        
        loss_angles = torch.max(
            torch.zeros_like(angle_pred),
            2 - (0.05) + \
            torch.cos(angle_pred - angle_ref - torch.pi) + \
            torch.sin(angle_pred - angle_ref - torch.pi/2)
        )

        if torch.any(torch.isnan(loss_angles)):
            loss_angles = torch.tensor([torch.inf], device=loss_angles.device)
        else:
            loss_angles = loss_angles.mean()

        if mean:
            return loss_bonds + loss_angles
        else:
            return loss_angles
