import torch
import numpy as np
from typing import Optional, Union
from herobm.utils import DataDict

def Rx(theta: float) -> np.ndarray:
    return np.array([[ 1, 0           , 0             ],
                     [ 0, np.cos(theta),-np.sin(theta)],
                     [ 0, np.sin(theta), np.cos(theta)]], dtype=np.float32)
  
def Ry(theta: float) -> np.ndarray:
  return np.array([[ np.cos(theta), 0, np.sin(theta)],
                   [ 0           ,  1, 0            ],
                   [-np.sin(theta), 0, np.cos(theta)]], dtype=np.float32)
  
def Rz(theta: float) -> np.ndarray:
  return np.array([[ np.cos(theta), -np.sin(theta), 0 ],
                   [ np.sin(theta), np.cos(theta) , 0 ],
                   [ 0           , 0            ,   1 ]], dtype=np.float32)

def get_rotation_matrix(phi: float, theta: float, psi: float) -> torch.Tensor:
    """ Get the rotation matrix for a rotation in 3D space by phi, theta and psi

        :param phi: float     | angle to rotate around x axis
        :param psi: float     | angle to rotate around y axis
        :param theta: float   | angle to rotate around z axis
        :return: torch.Tensor | shape (3, 3)
    """
    
    return  torch.from_numpy(Rz(psi) * Ry(theta) * Rx(phi))

def get_bonds(pos: torch.Tensor, bond_idcs: torch.Tensor) -> torch.Tensor:
    """ Compute bond length over specified bond_idcs for every frame in the batch

        :param pos:       torch.Tensor | shape (batch, n_atoms, xyz)
        :param bond_idcs: torch.Tensor | shape (n_bonds, 2)
        :return:          torch.Tensor | shape (batch, n_bonds)
    """

    if len(pos.shape) == 2:
        pos = torch.unsqueeze(pos, dim=0)

    dist_vectors = pos[:, bond_idcs]
    dist_vectors = dist_vectors[:, :, 1] - dist_vectors[:, :, 0]
    return torch.norm(dist_vectors, dim=2)

def get_angles_from_vectors(b0: torch.Tensor, b1: torch.Tensor, return_cos: bool = False) -> torch.Tensor:
    b0n = torch.norm(b0, dim=2, keepdim=False)
    b1n = torch.norm(b1, dim=2, keepdim=False)
    angles = torch.sum(b0 * b1, axis=-1) / b0n / b1n
    clamped_cos = torch.clamp(angles, min=-1., max=1.)
    if return_cos:
        return clamped_cos
    return torch.arccos(clamped_cos)

def get_angles(pos: torch.Tensor, angle_idcs: torch.Tensor, return_vectors = False) -> torch.Tensor:
    """ Compute angle values (in radiants) over specified angle_idcs for every frame in the batch

        :param pos:        torch.Tensor | shape (batch, n_atoms, xyz)
        :param angle_idcs: torch.Tensor | shape (n_angles, 3)
        :return:           torch.Tensor | shape (batch, n_angles)
    """

    if len(pos.shape) == 2:
        pos = torch.unsqueeze(pos, dim=0)

    dist_vectors = pos[:, angle_idcs]
    b0 = -1.0 * (dist_vectors[:, :, 1] - dist_vectors[:, :, 0])
    b1 = (dist_vectors[:, :, 2] - dist_vectors[:, :, 1])
    if return_vectors:
        return get_angles_from_vectors(b0, b1), b0, b1
    return get_angles_from_vectors(b0, b1)

def get_dihedrals(pos, dihedral_idcs):
    if isinstance(pos, torch.Tensor):
        return get_dihedrals_torch(pos, dihedral_idcs)
    return get_dihedrals_numpy(pos, dihedral_idcs)

def get_dihedrals_torch(pos: torch.Tensor, dihedral_idcs: torch.Tensor) -> torch.Tensor:
    """ Compute dihedral values (in radiants) over specified dihedral_idcs for every frame in the batch

        :param pos:        torch.Tensor | shape (batch, n_atoms, xyz)
        :param dihedral_idcs: torch.Tensor | shape (n_dihedrals, 4)
        :return:           torch.Tensor | shape (batch, n_dihedrals)
    """

    if len(pos.shape) == 2:
        pos = torch.unsqueeze(pos, dim=0)
    p = pos[:, dihedral_idcs, :]
    p0 = p[..., 0, :]
    p1 = p[..., 1, :]
    p2 = p[..., 2, :]
    p3 = p[..., 3, :]

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b1 = b1 / torch.linalg.vector_norm(b1, dim=-1, keepdim=True)

    v = b0 - torch.einsum("ijk,ikj->ij", b0, torch.transpose(b1, 1, 2))[..., None] * b1
    w = b2 - torch.einsum("ijk,ikj->ij", b2, torch.transpose(b1, 1, 2))[..., None] * b1

    x = torch.einsum("ijk,ikj->ij", v, torch.transpose(w, 1, 2))
    y = torch.einsum("ijk,ikj->ij", torch.cross(b1, v), torch.transpose(w, 1, 2))

    return torch.atan2(y, x).reshape(-1, dihedral_idcs.shape[0])

def get_dihedrals_numpy(pos: np.ndarray, dihedral_idcs: np.ndarray) -> np.ndarray:
    """ Compute dihedral values (in radiants) over specified dihedral_idcs for every frame in the batch

        :param pos:        torch.Tensor | shape (batch, n_atoms, xyz)
        :param dihedral_idcs: torch.Tensor | shape (n_dihedrals, 4)
        :return:           torch.Tensor | shape (batch, n_dihedrals)
    """

    if len(pos.shape) == 2:
        pos = np.expand_dims(pos, axis=0)
    if len(dihedral_idcs.shape) == 1:
        dihedral_idcs = np.expand_dims(dihedral_idcs, axis=0)
    p = pos[:, dihedral_idcs, :]
    p0 = p[..., 0, :]
    p1 = p[..., 1, :]
    p2 = p[..., 2, :]
    p3 = p[..., 3, :]

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)

    v = b0 - np.einsum("ijk,ikj->ij", b0, np.transpose(b1, (0, 2, 1)))[..., None] * b1
    w = b2 - np.einsum("ijk,ikj->ij", b2, np.transpose(b1, (0, 2, 1)))[..., None] * b1

    x = np.einsum("ijk,ikj->ij", v, np.transpose(w, (0, 2, 1)))
    y = np.einsum("ijk,ikj->ij", np.cross(b1, v), np.transpose(w, (0, 2, 1)))

    return np.arctan2(y, x).reshape(-1, dihedral_idcs.shape[0])

def get_RMSD(
    pred: Union[torch.Tensor, np.ndarray],
    ref:  Union[torch.Tensor, np.ndarray],
    fltr: Optional[np.ndarray] = None,
    ignore_zeroes: bool = False,
    ignore_nan: bool = False,
):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(ref, torch.Tensor):
        ref = ref.cpu().numpy()
    if fltr is not None:
        pred = pred[..., fltr, :]
        ref = ref[..., fltr, :]
    if ignore_nan:
        nan_fltr = ~np.any(np.isnan(pred), axis=-1)
        pred = pred[nan_fltr]
        ref = ref[nan_fltr]
    not_zeroes = np.ones_like(ref).mean(axis=-1).astype(int) if not ignore_zeroes else (~np.all(ref == 0., axis=-1)).astype(int)
    sd =  (np.power(pred - ref, 2).sum(-1)) * not_zeroes
    msd = sd.sum() / not_zeroes.sum()
    return np.sqrt(msd)

def get_dih_loss(
    pred: Union[torch.Tensor, np.ndarray],
    ref:  Union[torch.Tensor, np.ndarray],
    ignore_zeroes: bool = False,
):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(ref, torch.Tensor):
        ref = ref.cpu().numpy()
    
    not_zeroes = np.ones_like(ref).mean(axis=-1).astype(int) if not ignore_zeroes else (~np.all(ref == 0., axis=-1)).astype(int)
    return np.sum(np.mean(2 + np.cos(pred - ref - np.pi) + np.sin(pred - ref - np.pi/2), axis=-1) * not_zeroes) / (not_zeroes.sum() + 1e-6)

def build_peptide_dihedral_idcs(atom_names: np.ndarray, skip_first_res=0):
    CG_orientation_peptide_dihedral_idcs = []
    CG_orientation_peptide_dihedral_id = [0, 0, 0, 0]
    c = 0
    for idx, an in enumerate(atom_names):
        name = an.split(DataDict.STR_SEPARATOR)[-1]
        if name not in ['O', 'C', 'N', 'CA']:
            continue
        if name == 'O':
            CG_orientation_peptide_dihedral_id[0] = idx
            c += 1
        if name == 'C':
            CG_orientation_peptide_dihedral_id[1] = idx
            c += 1
        if name == 'N':
            if skip_first_res > 0:
                skip_first_res -= 1
            else:
                CG_orientation_peptide_dihedral_id[2] = idx
                c += 1
        if name == 'CA':
            if skip_first_res > 0:
                skip_first_res -= 1
            else:
                CG_orientation_peptide_dihedral_id[3] = idx
                c += 1
        if c == 4:
            CG_orientation_peptide_dihedral_idcs.append(CG_orientation_peptide_dihedral_id.copy())
            CG_orientation_peptide_dihedral_id = [0, 0, 0, 0]
            c = 0
    return torch.tensor(CG_orientation_peptide_dihedral_idcs).long()

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def bound_angle(angle):
    result = (angle + np.pi) % (2 * np.pi) - np.pi
    return result

def set_phi(coords: np.ndarray, phi_atom_idcs: np.ndarray, phi_target_values: np.ndarray):
    """
    Rotate residues to match the given phi torsion angles for specified residues
    """

    phi_values = get_dihedrals(coords, phi_atom_idcs)[0]

    # Calculate the rotation matrices for phi and psi rotations
    for phi_value, idcs, phi_target_value in zip(phi_values, phi_atom_idcs, phi_target_values):
        _, N, CA, _ = idcs
        axis = coords[CA] - coords[N]
        R = rotation_matrix(axis, bound_angle(phi_target_value - phi_value))

        # Apply the rotations to all atoms previous to N
        N_coord = np.expand_dims(coords[N], axis=0)
        coords[:N] = np.matmul(coords[:N] - N_coord, R) + N_coord

    return coords

def set_psi(coords: np.ndarray, psi_atom_idcs: np.ndarray, psi_target_values: np.ndarray):
    """
    Rotate residues to match the given phi torsion angles for specified residues
    """

    psi_values = get_dihedrals(coords, psi_atom_idcs)[0]

    # Calculate the rotation matrices for phi and psi rotations
    for psi_value, idcs, psi_target_value in zip(psi_values, psi_atom_idcs, psi_target_values):
        _, CA, C, _ = idcs
        axis = coords[C] - coords[CA]
        R = rotation_matrix(axis, bound_angle(psi_target_value - psi_value))

        # Apply the rotations to all atoms previous to N
        C_coord = np.expand_dims(coords[C], axis=0)
        coords[C:] = np.matmul(coords[C:] - C_coord, R) + C_coord

    return coords