from typing import Optional, Union
import torch
import numpy as np

from heqbm.utils import DataDict

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

def get_dihedrals(pos: torch.Tensor, dihedral_idcs: torch.Tensor) -> torch.Tensor:
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

def get_RMSD(
    pos: Union[torch.Tensor, np.ndarray],
    ref:  Union[torch.Tensor, np.ndarray],
    fltr: Optional[np.ndarray] = None,
    ignore_zeroes: bool = False,
):
    if isinstance(pos, torch.Tensor):
        pos = pos.cpu().numpy()
    if isinstance(ref, torch.Tensor):
        ref = ref.cpu().numpy()
    if fltr is not None:
        pos = pos[..., fltr, :]
        ref = ref[..., fltr, :]
    not_zeroes = np.ones_like(ref).mean(axis=-1).astype(int) if not ignore_zeroes else (~np.all(ref == 0., axis=-1)).astype(int)
    sd =  (np.power(pos - ref, 2).sum(-1)) * not_zeroes
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