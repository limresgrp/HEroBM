import torch
import numpy as np
from typing import List, Optional
from heqbm.utils.geometry import get_bonds, get_angles, get_dihedrals

class Phi:

    def __init__(self, prev_resid: int) -> None:
        self.prev_resid = prev_resid
        self.C_prev = None
        self.N = None
        self.CA = None
        self.C = None
        self.completion = 0
    
    def __call__(self, atom_name, resid, atom_index):
        self.completion += 1
        if atom_name == 'C' and resid == self.prev_resid and self.C_prev is None:
            self.C_prev = atom_index
        elif atom_name == 'N' and resid == self.prev_resid+1 and self.N is None:
            self.N = atom_index
        elif atom_name in ['CA', 'CH3'] and resid == self.prev_resid+1 and self.CA is None:
            self.CA = atom_index
        elif atom_name == 'C' and resid == self.prev_resid+1 and self.C is None:
            self.C = atom_index
        else:
            self.completion -= 1
    
    def is_completed(self):
        return self.completion == 4
    
    def lock(self):
        self.completion = 5
    
    def is_locked(self):
        return self.completion > 4

    def get_idcs(self):
        assert self.is_completed()
        return np.array([self.C_prev, self.N, self.CA, self.C])

class Psi:

    def __init__(self, resid: int) -> None:
        self.resid = resid
        self.N = None
        self.CA = None
        self.C = None
        self.N_next = None
        self.completion = 0
    
    def __call__(self, atom_name, resid, atom_index):
        self.completion += 1
        if atom_name == 'N' and resid == self.resid and self.N is None:
            self.N = atom_index
        elif atom_name in ['CA', 'CH3'] and resid == self.resid and self.CA is None:
            self.CA = atom_index
        elif atom_name == 'C' and resid == self.resid and self.C is None:
            self.C = atom_index
        elif atom_name == 'N' and resid == self.resid+1 and self.N_next is None:
            self.N_next = atom_index
        else:
            self.completion -= 1
    
    def is_completed(self):
        return self.completion == 4
    
    def lock(self):
        self.completion = 5
    
    def is_locked(self):
        return self.completion > 4

    def get_idcs(self):
        assert self.is_completed()
        return np.array([self.N, self.CA, self.C, self.N_next])

class Omega:

    def __init__(self, resid: int) -> None:
        self.resid = resid
        self.O = None
        self.C = None
        self.N_next = None
        self.CA_next = None
        self.completion = 0
    
    def __call__(self, atom_name, resid, atom_index):
        self.completion += 1
        if atom_name == 'O' and resid == self.resid and self.O is None:
            self.O = atom_index
        elif atom_name == 'C' and resid == self.resid and self.C is None:
            self.C = atom_index
        elif atom_name == 'N' and resid == self.resid+1 and self.N_next is None:
            self.N_next = atom_index
        elif atom_name in ['CA', 'CH3'] and resid == self.resid+1 and self.CA_next is None:
            self.CA_next = atom_index
        else:
            self.completion -= 1
    
    def is_completed(self):
        return self.completion == 4
    
    def lock(self):
        self.completion = 5
    
    def is_locked(self):
        return self.completion > 4

    def get_idcs(self):
        assert self.is_completed()
        return np.array([self.O, self.C, self.N_next, self.CA_next])


class MinimizeEnergy(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
    
    def evaluate_bond_energy(self, data, t: int = 0, pos: Optional[np.ndarray] = None, only_ca: bool = False):
        if pos is None:
            pos = data["pos"]
        bond_idcs = data[f"{'ca_' if only_ca else ''}bond_idcs"]
        bond_eq_val = data[f"{'ca_' if only_ca else ''}bond_eq_val"]
        bond_tolerance = data[f"{'ca_' if only_ca else ''}bond_tolerance"]

        bond_val = get_bonds(pos, bond_idcs)[0]
        bond_energy = 1000 * torch.mean(torch.max(torch.pow(bond_val - bond_eq_val, 2) - (bond_tolerance)**2, torch.zeros_like(bond_val)))
        return bond_energy

    def evaluate_angle_energy(self, data, t: int = 0, pos: Optional[np.ndarray] = None):
        if pos is None:
            pos = data["pos"]
        angle_idcs = data["angle_idcs"]
        angle_eq_val = data["angle_eq_val"]
        angle_tolerance = data["angle_tolerance"]

        angle_val = get_angles(pos, angle_idcs)[0]
        angle_energy = 150 * torch.mean(torch.max(torch.pow(angle_val - angle_eq_val, 2) - (angle_tolerance)**2, torch.zeros_like(angle_val)))
        return angle_energy
    
    def evaluate_dihedral_energy(self, data, t: int = 0, pos: Optional[np.ndarray] = None):
        if pos is None:
            pos = data["pos"]
        dih_idcs = data["dih_idcs"]
        dih_eq_val = data["dih_eq_val"]

        dih_val = get_dihedrals(pos, dih_idcs)[0]
        dihedral_energy = torch.mean(2 + torch.cos(dih_val - dih_eq_val - np.pi) + torch.sin(dih_val - dih_eq_val - np.pi/2))
        return dihedral_energy
    
    def evaluate_energy(self, data, t: int, minimize_dih: bool):
        bond_energy = self.evaluate_bond_energy(data)
        angle_energy = self.evaluate_angle_energy(data)

        energy_evaluation = {
            "total_energy": bond_energy + angle_energy,
            "bond_energy": bond_energy,
            "angle_energy": angle_energy,
        }

        if minimize_dih:
            dihedral_energy = self.evaluate_dihedral_energy(data)
            energy_evaluation["dihedral_energy"] = dihedral_energy
            energy_evaluation["total_energy"] = energy_evaluation["total_energy"] + dihedral_energy
        
        return energy_evaluation


    def forward(self, data, t: int, minimize_dih: bool, unlock_ca: bool = False):
        pos = data["pos"]
        pos.requires_grad_(True)
        if unlock_ca:
            ca_bond_energy = self.evaluate_bond_energy(data, only_ca=True)
            energy_evaluation = {
                "total_energy": ca_bond_energy,
                "bond_energy": ca_bond_energy,
            }
        else:
            energy_evaluation = self.evaluate_energy(data, t, minimize_dih)
        
        gradient = -torch.autograd.grad(
            [energy_evaluation.get("total_energy")],
            [pos],
            create_graph=False,
        )[0].detach()

        gradient = torch.nan_to_num(gradient, nan=0.)
        fnorm = gradient.norm(dim=-1, keepdim=True)
        gradient[fnorm.flatten() > .1/data["dtau"]] *= .1/data["dtau"]/fnorm[fnorm.flatten() > .1/data["dtau"]]
        energy_evaluation["gradient"] = gradient

        pos.requires_grad_(False)
        if unlock_ca:
            pos += gradient * data["dtau"]
        else:
            pos[data["movable_pos_idcs"]] += gradient[data["movable_pos_idcs"]] * data["dtau"]
        data["pos"] = pos
        return data, energy_evaluation
    
    def minimize(
        self,
        data,
        dtau=None,
        eps: float = 1e-3,
        minimize_dih: bool = True,
        unlock_ca: bool = False
    ):
        if dtau is not None:
            data["dtau"] = dtau
        last_total_energy = torch.inf
        for t in range(100000):
            data, energy_evaluation = self(data, t, minimize_dih, unlock_ca)
            if not t % 300:
                print(f"Step {t}")
                for k in ["bond_energy", "angle_energy", "dihedral_energy", "total_energy"]:
                    print(f"{k}: {energy_evaluation.get(k, torch.tensor(torch.nan)).item()}")
                current_total_energy = energy_evaluation.get("total_energy")
                if last_total_energy - current_total_energy < eps:
                    break
                last_total_energy = current_total_energy


def cat_interleave(array_list: List[np.ndarray]):
    n_arrays = len(array_list)
    concatenated_array = np.zeros_like(array_list[np.argmax([x.dtype for x in array_list])])
    concatenated_array = np.repeat(concatenated_array, n_arrays, axis=0)
    for i, arr in enumerate(array_list):
        concatenated_array[i::n_arrays] = arr
    return concatenated_array