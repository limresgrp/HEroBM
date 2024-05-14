import numpy as np
from typing import Dict, List, Optional, Union
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from herobm.utils import DataDict


def plot_backmapping(reconstructed_atom_pos, dataset, versors_list=[]):
    return plot_backmapping_impl(
        reconstructed_atom_pos=reconstructed_atom_pos[0],
        bead_pos=dataset[DataDict.BEAD_POSITION][0],
        atom_pos=dataset[DataDict.ATOM_POSITION][0],
        versors_list=[x[0] for x in versors_list],
        dataset=dataset,
    )

def plot_backmapping_impl(
    reconstructed_atom_pos: np.ndarray,
    bead_pos: np.ndarray,
    dataset: Dict[str, np.ndarray],
    atom_pos: Optional[np.ndarray] = None,
    versors_list: List[np.ndarray] = [],
):
    subplots = 1
    fig = make_subplots(rows=1, cols=subplots, specs=[[{'type': 'scene'}]*subplots], shared_xaxes=True, horizontal_spacing=0)

    trace_reconstructed_atoms = go.Scatter3d(
        x=reconstructed_atom_pos[:, 0],
        y=reconstructed_atom_pos[:, 1],
        z=reconstructed_atom_pos[:, 2],
        name='predicted atoms',
        text=[f"{resid}:{an}" for an, resid in zip(dataset[DataDict.ATOM_NAMES], dataset[DataDict.ATOM_RESIDCS])],
        mode='markers',
        marker=dict(symbol='circle', color='blue', opacity=0.5, size=5)
    )

    trace_beads = go.Scatter3d(
        x=bead_pos[:, 0],
        y=bead_pos[:, 1],
        z=bead_pos[:, 2],
        name='beads',
        text=dataset[DataDict.BEAD_IDNAMES],
        mode='markers',
        marker=dict(symbol='circle', color='green', opacity=0.5, size=10)
    )

    data = [trace_reconstructed_atoms, trace_beads]

    if atom_pos is not None:
        trace_atoms = go.Scatter3d(
            x=atom_pos[:, 0],
            y=atom_pos[:, 1],
            z=atom_pos[:, 2],
            name='true atoms',
            text=[f"{resid}:{an}" for an, resid in zip(dataset[DataDict.ATOM_NAMES], dataset[DataDict.ATOM_RESIDCS])],
            mode='markers',
            marker=dict(symbol='circle', color='red', opacity=0.5, size=5)
        )
        data.append(trace_atoms)

    colors = ['red', 'blue', 'orange', 'cyan', 'green', 'yellow']

    for idx, versors in enumerate(versors_list):
        beads_v_x = []
        beads_v_y = []
        beads_v_z = []
        for i in range(len(versors)):
            beads_v_x.extend([bead_pos[i, 0], bead_pos[i, 0] + versors[i, 0], None])
            beads_v_y.extend([bead_pos[i, 1], bead_pos[i, 1] + versors[i, 1], None])
            beads_v_z.extend([bead_pos[i, 2], bead_pos[i, 2] + versors[i, 2], None])
        trace_versors = go.Scatter3d(
                x=beads_v_x,
                y=beads_v_y,
                z=beads_v_z,
                name=f'versor_{idx}',
                mode='lines',
                line=dict(color=colors[idx], width=3),
                hoverinfo='none')
        data.append(trace_versors)

    for d in data:
        fig.add_trace(d, row=1, col=1)

    # axes_range = [reconstructed_atom_pos.min() - 0.2, reconstructed_atom_pos.max() + 0.2]
    layout = go.Layout(
        scene=dict(
        xaxis = dict(
            nticks=3,
            #range=axes_range,
            backgroundcolor="rgba(0,0,0,0.2)",
            gridcolor="whitesmoke",
            showbackground=True,
            showgrid=True,
            ),
        yaxis = dict(
            nticks=3,
            #range=axes_range,
            backgroundcolor="rgba(0,0,0,0.1)",
            gridcolor="whitesmoke",
            showbackground=True,
            showgrid=True,
            ),
        zaxis = dict(
            nticks=3,
            #range=axes_range,
            backgroundcolor="rgba(0,0,0,0.4)",
            gridcolor="whitesmoke",
            showbackground=True,
            showgrid=True,
            ),
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        scene_aspectmode='cube',
        autosize=False,
        width=1000,
        height=1000,
    )

    fig.update_layout(layout)
    fig.show()


def plot_cg(dataset: Dict, frame_index: int = 0, residue_filter: str = None):
    return plot_cg_impl(
        bead_pos=dataset[DataDict.BEAD_POSITION][frame_index],
        bead_names=dataset[DataDict.BEAD_IDNAMES],
        bead_types=dataset.get(DataDict.BEAD_TYPES, None),
        bead_residcs=dataset.get(DataDict.BEAD_RESIDCS, None),
        atom_pos=dataset[DataDict.ATOM_POSITION][frame_index] if DataDict.ATOM_POSITION in dataset else None,
        atom_names=dataset.get(DataDict.ATOM_NAMES, None),
        atom_types=dataset.get(DataDict.ATOM_TYPES, None),
        atom_residcs=dataset.get(DataDict.ATOM_RESIDCS, None),
        residue_filter=residue_filter
    )

def plot_cg_impl(
    bead_pos,
    bead_names,
    bead_types=None,
    bead_residcs=None,
    atom_pos=None,
    atom_names=None,
    atom_types=None,
    atom_residcs=None,
    residue_filter: Optional[Union[int, str]] = None,
):
    subplots = 1
    fig = make_subplots(rows=1, cols=subplots, specs=[[{'type': 'scene'}]*subplots], shared_xaxes=True, horizontal_spacing=0)

    if residue_filter is None:
        bead_fltr = np.array([True for _ in bead_names])
    else:
        if isinstance(residue_filter, int):
            bead_fltr = np.array([br == residue_filter for br in bead_residcs])
        else:
            bead_fltr = np.array([bn.split('_')[0] == residue_filter for bn in bead_names])
    if bead_fltr.sum() == 0:
        print(f"No residue has name {residue_filter}")
        return
    bead_pos = bead_pos[bead_fltr]
    bead_names = bead_names[bead_fltr]
    bead_types = bead_types[bead_fltr]
    trace_beads = go.Scatter3d(
        x=bead_pos[:, 0],
        y=bead_pos[:, 1],
        z=bead_pos[:, 2],
        name='beads',
        text=[f"{bn} ({bt})" for bn, bt in zip(bead_names, bead_types)],
        mode='markers',
        marker=dict(symbol='circle', color=bead_types if bead_types is not None else 'green', opacity=0.8, size=10) # colorscale='mygbm'
    )

    data = [trace_beads]
    
    if atom_pos is not None:
        if residue_filter is None:
            atom_fltr = np.array([True for _ in atom_names])
        else:
            if isinstance(residue_filter, int):
                atom_fltr = np.array([ar == residue_filter for ar in atom_residcs])
            else:
                atom_fltr = np.array([an.split('_')[0] == residue_filter for an in atom_names])
        atom_pos = atom_pos[atom_fltr]
        atom_names = atom_names[atom_fltr]
        data.append(
            go.Scatter3d(
                x=atom_pos[:, 0],
                y=atom_pos[:, 1],
                z=atom_pos[:, 2],
                name='atoms',
                text=atom_names,
                mode='markers',
                marker=dict(symbol='circle', color=[get_atom_color_from_atom_type(an) for an in atom_types], opacity=0.5, size=5)
            )
        )

    for d in data:
        fig.add_trace(d, row=1, col=1)

    range = [atom_pos.min(), atom_pos.max()] if atom_pos is not None else None
    layout = go.Layout(
        scene=dict(
        xaxis = dict(
            nticks=3,
            range=range,
            backgroundcolor="rgba(0,0,0,0.2)",
            gridcolor="gray",
            showbackground=True,
            showgrid = True,
            showticklabels = False
            ),
        yaxis = dict(
            nticks=3,
            range=range,
            backgroundcolor="rgba(0,0,0,0.1)",
            gridcolor="gray",
            showbackground=True,
            showgrid = True,
            showticklabels = False
            ),
        zaxis = dict(
            nticks=3,
            range=range,
            backgroundcolor="rgba(0,0,0,0.4)",
            gridcolor="gray",
            showbackground=True,
            showgrid = True,
            showticklabels = False
            ),
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        scene_aspectmode='cube',
        autosize=False,
        width=1000,
        height=1000,
    )

    fig.update_layout(layout)
    fig.show()

def get_atom_color_from_atom_type(atom_type: int):
    colors_dict = {
        1: 'white',
        6: 'gray',
        7: 'blue',
        8: 'red',
        16: 'yellow'
    }

    return colors_dict.get(atom_type, 'black')

def get_atom_color_from_atom_name(atom_name: str):
    colors_dict = {
        'H': 'white',
        'C': 'gray',
        'N': 'blue',
        'O': 'red',
        'S': 'yellow'
    }

    return colors_dict.get(atom_name[0], 'black')