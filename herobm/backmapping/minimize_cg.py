import torch
import pandas as pd
import numpy as np
import io
import sys
from typing import Dict, Tuple, List, Optional, Any

def _load_target_distances(csv_filepath: Optional[str] = None, csv_content: Optional[str] = None) -> Tuple[Dict[Tuple, float], Dict[Tuple, float]]:
    """
    Reads target distances from a CSV file path or a raw CSV string.

    Args:
        csv_filepath (str, optional): Path to the CSV file.
        csv_content (str, optional): String containing the CSV data.

    Returns:
        A tuple containing two dictionaries:
        - target_distances: Maps connection type to target distance.
        - target_tolerances: Maps connection type to standard deviation.
    """
    if csv_content:
        print("Reading target distances from provided CSV content...")
        source = io.StringIO(csv_content)
    elif csv_filepath:
        print(f"Reading target distances from {csv_filepath}...")
        source = csv_filepath
    else:
        raise ValueError("Either 'csv_filepath' or 'csv_content' must be provided.")

    try:
        df = pd.read_csv(source)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_filepath}", file=sys.stderr)
        return {}, {}
    except Exception as e:
        print(f"An error occurred while reading the CSV data: {e}", file=sys.stderr)
        return {}, {}

    target_distances = {}
    target_tolerances = {}
    for _, row in df.iterrows():
        resname1, beadname1 = row['resname1.beadname1'].split('.')
        resname2, beadname2 = row['resname2.beadname2'].split('.')

        # Create a consistent key, sorting inter-residue BB-BB connections
        if beadname1 == 'BB' and beadname2 == 'BB' and resname1 != resname2:
            key = tuple(sorted([resname1, resname2])) + ('BB', 'BB')
        else:
            key = (resname1, beadname1, resname2, beadname2)
            
        target_distances[key] = row['mean_distance']
        target_tolerances[key] = row['std_distance']
        
    print(f"Loaded {len(target_distances)} unique target distances.")
    return target_distances, target_tolerances

def _identify_connections(ds: Dict[str, np.ndarray], target_distances: Dict[Tuple, float]) -> List[Tuple[int, int, Tuple]]:
    """
    Identifies bead connections in the dataset based on the provided target distances.
    """
    print("Identifying connections in the input dataset...")
    num_beads = ds['bead_pos'].shape[1]
    connections_to_minimize = []
    
    bead_info = list(zip(
        ds['bead_residcs'], ds['bead_segids'], ds['bead_names'], ds['bead_resnames']
    ))

    for i in range(num_beads):
        res_id1, seg_id1, bead_name1, res_name1 = bead_info[i]
        
        for j in range(i + 1, num_beads):
            res_id2, seg_id2, bead_name2, res_name2 = bead_info[j]

            key = None
            # Intra-residue
            if res_id1 == res_id2 and seg_id1 == seg_id2:
                key = (res_name1, bead_name1, res_name2, bead_name2)
            # Inter-residue (BB-BB)
            elif abs(res_id1 - res_id2) == 1 and seg_id1 == seg_id2 and bead_name1 == 'BB' and bead_name2 == 'BB':
                sorted_resnames = tuple(sorted([res_name1, res_name2]))
                key = sorted_resnames + ('BB', 'BB')
            
            if key in target_distances:
                connections_to_minimize.append((i, j, key))
            # Check for reversed key for intra-residue connections
            elif key is not None and key[:2] != key[2:]:
                reversed_key = (res_name2, bead_name2, res_name1, bead_name1)
                if reversed_key in target_distances:
                     connections_to_minimize.append((i, j, reversed_key))

    print(f"Identified {len(connections_to_minimize)} unique connections to minimize.")
    return connections_to_minimize

def _run_optimization(
    initial_pos: np.ndarray,
    connections: List[Tuple[int, int, Tuple]],
    target_distances: Dict[Tuple, float],
    target_tolerances: Dict[Tuple, float],
    force_constant: float,
    learning_rate: float,
    num_steps: int,
    device: str
) -> np.ndarray:
    """
    Performs the gradient-based optimization of bead positions for all frames.
    """
    num_frames = initial_pos.shape[0]
    minimized_pos_np = np.copy(initial_pos)
    
    pos_tensor = torch.tensor(initial_pos, dtype=torch.float32, device=device)

    idcs_1, idcs_2, targets, tolerances, masses_1, masses_2 = [], [], [], [], [], []
    for idx1, idx2, key in connections:
        idcs_1.append(idx1)
        idcs_2.append(idx2)
        targets.append(target_distances[key])
        tolerances.append(target_tolerances[key])
        masses_1.append(1.0 if key[1] == 'BB' else 0.1)
        masses_2.append(1.0 if key[3] == 'BB' else 0.1)

    t_idcs_1 = torch.tensor(idcs_1, device=device, dtype=torch.long)
    t_idcs_2 = torch.tensor(idcs_2, device=device, dtype=torch.long)
    t_targets = torch.tensor(targets, device=device, dtype=torch.float32)
    t_tolerances = torch.tensor(tolerances, device=device, dtype=torch.float32)
    t_masses_1 = torch.tensor(masses_1, device=device, dtype=torch.float32).unsqueeze(-1)
    t_masses_2 = torch.tensor(masses_2, device=device, dtype=torch.float32).unsqueeze(-1)

    inv_mass_sum = 1.0 / t_masses_1 + 1.0 / t_masses_2
    weight1 = (1.0 / t_masses_1) / inv_mass_sum
    weight2 = (1.0 / t_masses_2) / inv_mass_sum

    print(f"Starting optimization for {num_frames} frames...")
    for frame_idx in range(num_frames):
        current_pos = pos_tensor[frame_idx].clone().requires_grad_(True)
        
        for step in range(num_steps):
            if current_pos.grad is not None:
                current_pos.grad.zero_()

            pos1, pos2 = current_pos[t_idcs_1], current_pos[t_idcs_2]
            disp = pos2 - pos1
            distances = torch.linalg.norm(disp, dim=-1)
            
            deviation = distances - t_targets
            clamped_dev = torch.clamp(torch.abs(deviation) - t_tolerances, min=0.0)
            
            # Harmonic potential: E = k * clamped_dev^2
            # Force magnitude: dE/dr = 2 * k * clamped_dev
            forces_magnitude = (2 * force_constant * clamped_dev * torch.sign(deviation)).unsqueeze(-1)
            force_vec = forces_magnitude * (disp / (distances.unsqueeze(-1) + 1e-8))

            grad_pos1 = -weight1 * force_vec
            grad_pos2 =  weight2 * force_vec
            
            # Manually accumulate gradients
            current_pos.grad = torch.zeros_like(current_pos)
            current_pos.grad.index_add_(0, t_idcs_1, grad_pos1)
            current_pos.grad.index_add_(0, t_idcs_2, grad_pos2)
            
            with torch.no_grad():
                current_pos -= learning_rate * current_pos.grad
        
        minimized_pos_np[frame_idx] = current_pos.detach().cpu().numpy()
        if (frame_idx + 1) % 10 == 0 or frame_idx == num_frames - 1:
            print(f"Finished optimizing frame {frame_idx + 1}/{num_frames}. Final Grad Norm: {current_pos.grad.norm().item():.4f}")
            
    return minimized_pos_np

def minimize_bead_distances(
    ds: Dict[str, np.ndarray], 
    npz_ds: Dict[str, np.ndarray], 
    csv_filepath: Optional[str] = None, 
    csv_content: Optional[str] = None, 
    force_constant: float = 10.0, 
    learning_rate: float = 0.001, 
    num_steps: int = 1000, 
    device: str = 'cpu'
) -> Dict[str, np.ndarray]:
    """
    Minimizes bead positions to match target distances using a harmonic potential.
    This function serves as a wrapper orchestrating the loading, identification,
    and optimization steps.

    Args:
        ds: Dataset with original bead information.
        npz_ds: Dataset loaded from .npz, which will be modified and returned.
        csv_filepath: Path to the CSV file with distance statistics.
        csv_content: String content of the CSV file.
        force_constant: Force constant (k) for the harmonic potential.
        learning_rate: Learning rate for the optimizer.
        num_steps: Number of optimization steps per frame.
        device: PyTorch device ('cpu' or 'cuda:X').

    Returns:
        The updated npz_ds dictionary with minimized 'bead_pos'.
    """
    print("--- Starting Bead Distance Minimization ---")
    
    target_distances, target_tolerances = _load_target_distances(csv_filepath, csv_content)
    if not target_distances:
        print("Could not load target distances. Returning original positions.", file=sys.stderr)
        return npz_ds

    connections = _identify_connections(ds, target_distances)
    if not connections:
        print("No connections to minimize were found. Returning original positions.")
        return npz_ds
        
    initial_pos = ds['bead_pos']
    minimized_pos = _run_optimization(
        initial_pos,
        connections,
        target_distances,
        target_tolerances,
        force_constant,
        learning_rate,
        num_steps,
        device
    )

    npz_ds['bead_pos'] = minimized_pos
    print("--- Bead Distance Minimization Complete ---")
    return npz_ds
