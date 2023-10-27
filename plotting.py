import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_cg_impl(
    dataset,
    frame_index: int,
    bead_filter = None,
    atom_filter = None,
    residue_filter: str = None,
):
    subplots = 1
    fig = make_subplots(rows=1, cols=subplots, specs=[[{'type': 'scene'}]*subplots], shared_xaxes=True, horizontal_spacing=0)

    bead_pos=dataset['bead_pos'][frame_index]
    bead_names=dataset['bead_names']
    bead_types=dataset['bead_types']
    bead2atom_idcs=dataset["bead2atom_idcs"]
    bead2atom_idcs_mask=dataset["bead2atom_idcs_mask"]
    lvl_idcs_mask=dataset['lvl_idcs_mask']
    lvl_idcs_anchor_mask=dataset["lvl_idcs_anchor_mask"]

    atom_pos=dataset.get('atom_pos', None)
    if atom_pos is not None:
        atom_pos=atom_pos[frame_index]
        atom_names=dataset['atom_names']
        rel_vectors=dataset['bead2atom_rel_vectors'][frame_index]

    if bead_filter is not None:
        bead_pos=bead_pos[bead_filter]
        bead_names=bead_names[bead_filter]
        bead_types=bead_types[bead_filter]
        bead2atom_idcs=bead2atom_idcs[bead_filter]
        bead2atom_idcs_mask=bead2atom_idcs_mask[bead_filter]
        lvl_idcs_mask=lvl_idcs_mask[:, bead_filter]
        lvl_idcs_anchor_mask=lvl_idcs_anchor_mask[:, bead_filter]
        if atom_pos is not None:
            rel_vectors=rel_vectors[bead_filter]
    if atom_pos is not None and atom_filter is not None:
        atom_pos=atom_pos[atom_filter]
        atom_names=atom_names[atom_filter]

    if residue_filter is None:
        bead_fltr = np.array([True for _ in bead_names])
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
                marker=dict(symbol='circle', color=[get_atom_color_from_atom_name(an) for an in atom_names], opacity=0.5, size=5)
            )
        )

        for h, (bpos, b2a, b2am, rel_v) in enumerate(zip(
                bead_pos, bead2atom_idcs, bead2atom_idcs_mask, rel_vectors
            )):
            atom_pos_from = np.zeros((bead2atom_idcs.max()+1, 3), dtype=float)
            for level, (am_all, m_all) in enumerate(zip(lvl_idcs_anchor_mask, lvl_idcs_mask)):
                am = am_all[h]
                m = m_all[h]
                if level == 0:
                    atom_pos_from[b2a[b2am]] = np.repeat(bpos[None, :], b2a.count(), axis=0)
                atom_pos_to = np.copy(atom_pos_from)
                atom_pos_from[b2a[m]] = atom_pos_from[am[m]]
                atom_pos_to[b2a[m]] = atom_pos_from[b2a[m]] + rel_v[m]
                apf = atom_pos_from[b2a[m]].reshape(-1, 3)
                apt = atom_pos_to[b2a[m]].reshape(-1, 3)
                pos = np.empty((len(apf) + len(apt), 3), dtype=apf.dtype)
                pos[0::2] = apf
                pos[1::2] = apt
                data.extend(plot_arrow(
                    pos[:, 0].flatten(),
                    pos[:, 1].flatten(),
                    pos[:, 2].flatten(),
                    level
                ))
                atom_pos_from = np.copy(atom_pos_to)

    for d in data:
        fig.add_trace(d, row=1, col=1)

    range = None # [atom_pos.min(), atom_pos.max()] if atom_pos is not None else [bead_pos.min(), bead_pos.max()]
    layout = go.Layout(
        scene=dict(
        xaxis = dict(
            nticks=3,
            range=range,
            backgroundcolor="rgba(0,0,0,0.2)",
            gridcolor="gray",
            showbackground=False,
            showgrid = True,
            showticklabels = False
            ),
        yaxis = dict(
            nticks=3,
            range=range,
            backgroundcolor="rgba(0,0,0,0.1)",
            gridcolor="gray",
            showbackground=False,
            showgrid = True,
            showticklabels = False
            ),
        zaxis = dict(
            nticks=3,
            range=range,
            backgroundcolor="rgba(0,0,0,0.4)",
            gridcolor="gray",
            showbackground=False,
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

def get_atom_color_from_atom_name(atom_name: str):
    colors_dict = {
        'H': 'white',
        'C': 'gray',
        'N': 'blue',
        'O': 'red',
        'S': 'yellow'
    }

    return colors_dict.get(atom_name.split('_')[1][0], 'black')

def get_color_from_hierarchy_level(level: int, n: int):
    colors_dict = {
        0: 'white',
        1: 'yellow',
        2: 'orange',
        3: 'red',
        4: 'purple',
        5: 'blue',
    }

    return [colors_dict.get(level, 'black') for _ in range(n)]

def plot_arrow(x, y, z, level):
    data = []
    norm_list = np.sqrt((x[1::2]-x[:-1:2])**2 + (y[1::2]-y[:-1:2])**2 + (z[1::2]-z[:-1:2])**2)
    for i, norm in enumerate(norm_list):
        xxl = [x[2*i], x[2*i+1], None]
        yyl = [y[2*i], y[2*i+1], None]
        zzl = [z[2*i], z[2*i+1], None]
        data.append(
            go.Scatter3d(
            x=xxl,
            y=yyl,
            z=zzl,
            mode="lines",
            hoverinfo="none",
            line={
                "color": get_color_from_hierarchy_level(level, len(x)),
                "width": 3,
            }
            )
        )

        u = [(x[2*i+1]-x[2*i])/norm, None]
        v = [(y[2*i+1]-y[2*i])/norm, None]
        w = [(z[2*i+1]-z[2*i])/norm, None]
        xxc = [x[2*i+1], None]
        yyc = [y[2*i+1], None]
        zzc = [z[2*i+1], None]
        data.append(
            go.Cone(x=xxc, y=yyc, z=zzc, u=u, v=v, w=w,
                anchor="tip",
                hoverinfo="none",
                sizemode='absolute',
                sizeref=0.3,
                colorscale=[
                    [0, get_color_from_hierarchy_level(level, 1)[0]],
                    [1, get_color_from_hierarchy_level(level, 1)[0]]
                ],
                showscale=False,
            )
        )

    return data