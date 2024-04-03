import torch
import numpy as np
from typing import List, Optional
from heqbm.utils.geometry import bound_angle, get_bonds, get_angles, get_dihedrals


class MinimizeEnergy(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
    
    def evaluate_bond_energy(self, data, t: int = 0):
        pos = data["coords"]
        bond_idcs = data["bond_idcs"]
        bond_eq_val = data["bond_eq_val"]
        bond_tolerance = data["bond_tolerance"]

        if len(bond_idcs) == 0:
            return torch.tensor(0., device=pos.device, requires_grad=True)
        bond_val = get_bonds(pos, bond_idcs)
        squared_distance = torch.pow(bond_val - bond_eq_val, 2)
        if t == 0:
            threshold = torch.pow(2*bond_eq_val, 2)
            actual_bonds_idcs = torch.argwhere(squared_distance < threshold).flatten()
            data["bond_idcs"] = bond_idcs[actual_bonds_idcs]
            data["bond_eq_val"] = bond_eq_val[actual_bonds_idcs]
            data["bond_tolerance"] = bond_tolerance[actual_bonds_idcs]
        bond_energy = 1000 * torch.mean(torch.max(squared_distance - (bond_tolerance)**2, torch.zeros_like(bond_val)))
        return bond_energy

    def evaluate_angle_energy(self, data, t: int = 0, pos: Optional[np.ndarray] = None):
        if pos is None:
            pos = data["coords"]
        angle_idcs = data["angle_idcs"]
        angle_eq_val = data["angle_eq_val"]
        angle_tolerance = data["angle_tolerance"]

        if len(angle_idcs) == 0:
            return torch.tensor(0., device=pos.device, requires_grad=True)
        angle_val = get_angles(pos, angle_idcs)[0]
        angle_energy = 150 * torch.mean(torch.max(torch.pow(angle_val - angle_eq_val, 2) - (angle_tolerance)**2, torch.zeros_like(angle_val)))
        return angle_energy
    
    def evaluate_torsion_energy(self, data, t: int = 0, pos: Optional[np.ndarray] = None):
        if pos is None:
            pos = data["coords"]
        tor_idcs = data["omega_idcs"]
        tor_eq_val = data["omega_values"]
        tor_tolerance = data["omega_tolerance"]

        tor_val = get_dihedrals(pos, tor_idcs)[0]
        torsion_error = bound_angle(tor_val - tor_eq_val)
        tolerance_bounded_error = torch.sign(torsion_error) * torch.max(torch.abs(torsion_error) - tor_tolerance, torch.zeros_like(tor_val))
        torsion_energy = 100 * torch.sum(
            2 +
            torch.cos(tolerance_bounded_error - np.pi) +
            torch.sin(tolerance_bounded_error - np.pi/2))
        return torsion_energy
    
    def evaluate_energy(self, data, t: int):
        bond_energy = self.evaluate_bond_energy(data, t=t)
        angle_energy = self.evaluate_angle_energy(data, t=t)
        torsion_energy = self.evaluate_torsion_energy(data, t=t)

        energy_evaluation = {
            "bond_energy": bond_energy,
            "angle_energy": angle_energy,
            "torsion_energy": torsion_energy,
            "total_energy": bond_energy + angle_energy + torsion_energy,
        }
        
        return energy_evaluation

    def forward(
        self,
        data, t: int,
    ):
        pos = data["coords"]
        if len(pos) == 0:
            return data, {}
        pos.requires_grad_(True)
        energy_evaluation = self.evaluate_energy(data, t=t)
        
        gradient = torch.autograd.grad(
            [energy_evaluation.get("total_energy")],
            [pos],
            create_graph=False,
            allow_unused=True,
        )

        if gradient is not None:
            gradient = -gradient[0].detach()
            gradient = torch.nan_to_num(gradient, nan=0.)
            fnorm = gradient.norm(dim=-1)
            gradient[fnorm > .1/data["dtau"]] *= .1/data["dtau"]/fnorm[fnorm > .1/data["dtau"]][..., None]
            energy_evaluation["gradient"] = gradient

            pos.requires_grad_(False)
            pos += gradient * data["dtau"]

            weighted_pos = torch.einsum('bijk,ij->bik', torch.nan_to_num(pos[:, data["bb_atom_idcs"]]), data["bb_atom_weights"])
            recentering_vectors = (data["bb_bead_coords"] - weighted_pos)[..., None, :].repeat(1, 1, data["bb_atom_idcs"].shape[-1], 1)
            recentering_vectors[:, data["bb_atom_idcs"] == -1] = 0.
            pos[:, data["bb_atom_idcs"]] += recentering_vectors

            data["coords"] = pos
        return data, energy_evaluation
    
    def minimise(
        self,
        data,
        dtau=None,
        eps: float = 1e-3,
        device: str = 'cpu',
        verbose: bool = False,
    ):
        for k, v in data.items():
            if v.dtype.type is not np.str_:
                data[k] = torch.from_numpy(v).to(device)
        
        if dtau is not None:
            data["dtau"] = dtau
        last_total_energy = torch.inf
        for t in range(1, 100001):
            data, energy_evaluation = self(data, t)
            if energy_evaluation is None:
                return
            if not t % 300:
                if verbose:
                    print(f"Step {t}")
                    for k in ["bond_energy", "angle_energy", "torsion", "total_energy"]:
                        print(f"{k}: {energy_evaluation.get(k, torch.tensor(torch.nan)).item()}")
                current_total_energy = energy_evaluation.get("total_energy")
                if last_total_energy - current_total_energy < eps:
                    break
                last_total_energy = current_total_energy
        
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.detach().cpu().numpy()


def cat_interleave(array_list: List[np.ndarray]):
    n_arrays = len(array_list)
    concatenated_arrays = [np.zeros_like(alist) for alist in array_list]
    concatenated_array = np.concatenate(concatenated_arrays, axis=0)
    for i, arr in enumerate(array_list):
        concatenated_array[i::n_arrays] = arr
    return concatenated_array