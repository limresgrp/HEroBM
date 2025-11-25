import torch
from geqtrain.data import AtomicDataDict
from herobm.utils.geometry import get_bonds, get_angles, get_dihedrals

class InvariantsLoss:

    def __init__(self, **kwargs):
        pass

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        **kwargs,
    ):
        # Get predicted and reference positions
        pred_key = pred[key]
        # Use zeros_like for ref_key if it doesn't exist, assuming missing reference means zero target value or similar handling
        ref_key = ref.get(key, torch.zeros_like(pred[key], device=pred[key].device))

        # Get indices and slices for reconstructing atoms from beads
        idcs_mask = pred["bead2atom_reconstructed_idcs"]
        idcs_mask_slices = pred["bead2atom_reconstructed_idcs_slices"]
        atom_pos_slices = pred['atom_pos_slices'] # Slices for atom positions in the original structure
        # Get unique center atoms involved in edges (used for batching reconstructed atoms)
        center_atoms = torch.unique(pred[AtomicDataDict.EDGE_INDEX_KEY][0])

        # Initialize losses to zero
        loss_bonds = torch.tensor(0.0, device=pred_key.device)
        loss_angles = torch.tensor(0.0, device=pred_key.device)
        loss_dihedrals = torch.tensor(0.0, device=pred_key.device)

        # --- Bond Loss Calculation ---
        # Check if bond indices exist in reference data
        if "atom_bond_idx" in pred and "atom_bond_idx_slices" in pred:
            atom_bond_idcs = pred["atom_bond_idx"]
            atom_bond_idcs_slices = pred["atom_bond_idx_slices"]

            bond_pred_list, bond_ref_list = [], []
            # Iterate through batches defined by bead2atom slices
            for (b2a_idcs_from, b2a_idcs_to), (atom_bond_idx_from, atom_bond_idx_to), atom_pos_from in zip(
                zip(idcs_mask_slices[:-1], idcs_mask_slices[1:]),
                zip(atom_bond_idcs_slices[:-1], atom_bond_idcs_slices[1:]),
                atom_pos_slices[:-1],
            ):
                # Identify center atoms within the current batch's bead2atom range
                batch_center_atoms = center_atoms[(center_atoms>=b2a_idcs_from) & (center_atoms<b2a_idcs_to)]
                # Get unique reconstructed atom indices for this batch (excluding the first element which might be a placeholder)
                # Add atom_pos_from to shift indices to the correct batch offset
                batch_recon_atom_idcs = idcs_mask[batch_center_atoms].unique()[1:] + atom_pos_from
                # Get atom bond indices for the current batch, shifted by atom_pos_from
                batch_atom_bond_idcs = atom_bond_idcs[atom_bond_idx_from:atom_bond_idx_to] + atom_pos_from
                # Filter bond indices to keep only those where both atoms are present in the reconstructed set for this batch
                pred_atom_bond_idcs = batch_atom_bond_idcs[torch.all(torch.isin(batch_atom_bond_idcs, batch_recon_atom_idcs), dim=1)]
                if len(pred_atom_bond_idcs) > 0:
                    # Calculate bond lengths using the filtered indices for predicted and reference positions
                    bond_pred = get_bonds(pred_key, pred_atom_bond_idcs)
                    bond_ref = get_bonds(ref_key, pred_atom_bond_idcs)
                    # Append calculated bonds to lists
                    bond_pred_list.append(bond_pred)
                    bond_ref_list.append(bond_ref)

            # Concatenate bond lists from all batches
            # Check if lists are empty before concatenating
            if bond_pred_list and bond_ref_list:
                bond_pred = torch.cat(bond_pred_list, axis=-1)
                bond_ref = torch.cat(bond_ref_list, axis=-1)
                # Calculate bond loss: squared error, but only for errors exceeding 0.03 Angstrom
                loss_bonds = torch.pow(torch.max(torch.zeros_like(bond_pred), torch.abs(bond_pred - bond_ref) - 0.03), 2)
            # If no bonds were found but keys existed, loss_bonds remains 0.0 as initialized


        # --- Angle Loss Calculation ---
        # Check if angle indices exist in reference data
        if "atom_angle_idx" in pred and "atom_angle_idx_slices" in pred:
            atom_angle_idcs = pred["atom_angle_idx"]
            atom_angle_idcs_slices = pred["atom_angle_idx_slices"]

            angle_pred_list, angle_ref_list = [], []
            # Iterate through batches defined by bead2atom slices
            for (b2a_idcs_from, b2a_idcs_to), (atom_angle_idx_from, atom_angle_idx_to), atom_pos_from in zip(
                zip(idcs_mask_slices[:-1], idcs_mask_slices[1:]),
                zip(atom_angle_idcs_slices[:-1], atom_angle_idcs_slices[1:]),
                atom_pos_slices[:-1],
            ):
                # Identify center atoms within the current batch's bead2atom range
                batch_center_atoms = center_atoms[(center_atoms>=b2a_idcs_from) & (center_atoms<b2a_idcs_to)]
                # Get unique reconstructed atom indices for this batch, shifted by atom_pos_from
                batch_recon_atom_idcs = idcs_mask[batch_center_atoms].unique()[1:] + atom_pos_from
                # Get atom angle indices for the current batch, shifted by atom_pos_from
                batch_atom_angle_idcs = atom_angle_idcs[atom_angle_idx_from:atom_angle_idx_to] + atom_pos_from
                # Filter angle indices to keep only those where all three atoms are present in the reconstructed set for this batch
                pred_atom_angle_idcs = batch_atom_angle_idcs[torch.all(torch.isin(batch_atom_angle_idcs, batch_recon_atom_idcs), dim=1)]
                if len(pred_atom_angle_idcs) > 0:
                    # Calculate angles using the filtered indices for predicted and reference positions
                    angle_pred = get_angles(pred_key, pred_atom_angle_idcs)
                    angle_ref = get_angles(ref_key, pred_atom_angle_idcs)
                    # Append calculated angles to lists
                    angle_pred_list.append(angle_pred)
                    angle_ref_list.append(angle_ref)

            # Concatenate angle lists from all batches
            # Check if lists are empty before concatenating
            if angle_pred_list and angle_ref_list:
                angle_pred = torch.cat(angle_pred_list, axis=-1)
                angle_ref = torch.cat(angle_ref_list, axis=-1)

                # Calculate angle loss: uses a periodic function based on cosine and sine difference
                # This penalizes deviations from the reference angle, considering periodicity
                loss_angles = torch.max(
                    torch.zeros_like(angle_pred),
                    2 - (0.05) + \
                    torch.cos(angle_pred - angle_ref - torch.pi) + \
                    torch.sin(angle_pred - angle_ref - torch.pi/2)
                )
            # If no angles were found but keys existed, loss_angles remains 0.0 as initialized


        # --- Dihedral Loss Calculation ---
        # Check if dihedral indices exist in reference data
        if "atom_dihedral_idx" in pred and "atom_dihedral_idx_slices" in pred:
            atom_dihedral_idcs = pred["atom_dihedral_idx"]
            atom_dihedral_idcs_slices = pred["atom_dihedral_idx_slices"]

            dihedral_pred_list, dihedral_ref_list = [], []

            # Iterate through batches defined by bead2atom slices
            for (b2a_idcs_from, b2a_idcs_to), (atom_dihedral_idx_from, atom_dihedral_idx_to), atom_pos_from in zip(
                zip(idcs_mask_slices[:-1], idcs_mask_slices[1:]),
                zip(atom_dihedral_idcs_slices[:-1], atom_dihedral_idcs_slices[1:]),
                atom_pos_slices[:-1],
            ):
                # Identify center atoms within the current batch's bead2atom range
                batch_center_atoms = center_atoms[(center_atoms>=b2a_idcs_from) & (center_atoms<b2a_idcs_to)]
                # Get unique reconstructed atom indices for this batch, shifted by atom_pos_from
                batch_recon_atom_idcs = idcs_mask[batch_center_atoms].unique()[1:] + atom_pos_from
                # Get atom dihedral indices for the current batch, shifted by atom_pos_from
                batch_atom_dihedral_idcs = atom_dihedral_idcs[atom_dihedral_idx_from:atom_dihedral_idx_to] + atom_pos_from
                # Filter dihedral indices to keep only those where all four atoms are present in the reconstructed set for this batch
                pred_atom_dihedral_idcs = batch_atom_dihedral_idcs[torch.all(torch.isin(batch_atom_dihedral_idcs, batch_recon_atom_idcs), dim=1)]
                if len(pred_atom_dihedral_idcs) > 0:
                    # Calculate dihedrals using the filtered indices for predicted and reference positions
                    dihedral_pred = get_dihedrals(pred_key, pred_atom_dihedral_idcs)
                    dihedral_ref = get_dihedrals(ref_key, pred_atom_dihedral_idcs)
                    # Append calculated dihedrals to lists
                    dihedral_pred_list.append(dihedral_pred)
                    dihedral_ref_list.append(dihedral_ref)

            # Concatenate dihedral lists from all batches
            # Check if lists are empty before concatenating
            if dihedral_pred_list and dihedral_ref_list:
                dihedral_pred = torch.cat(dihedral_pred_list, axis=-1)
                dihedral_ref = torch.cat(dihedral_ref_list, axis=-1)

                # Calculate dihedral loss: using the same periodic function as angle loss
                loss_dihedrals = torch.max(
                    torch.zeros_like(dihedral_pred),
                    2 - (0.05) + \
                    torch.cos(dihedral_pred - dihedral_ref - torch.pi) + \
                    torch.sin(dihedral_pred - dihedral_ref - torch.pi/2)
                )
            # If no dihedrals were found but keys existed, loss_dihedrals remains 0.0 as initialized

        # Combine losses
        if mean:
            # Return sum of mean losses for bonds, angles, and dihedrals
            return loss_bonds.mean() + loss_angles.mean() + loss_dihedrals.mean()
        # Currently the loss is not correctly supported for Metrics purposes
        return loss_bonds
