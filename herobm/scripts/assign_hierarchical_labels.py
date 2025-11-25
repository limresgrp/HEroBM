from __future__ import annotations

import argparse
import logging
import re
import string
from collections import Counter, defaultdict, deque
from itertools import product
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import MDAnalysis as mda
import numpy as np
import yaml
from MDAnalysis.topology import guessers
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tempfile
from cgmap.mapping.mapper import Mapper


LABEL_RE = re.compile(r"^P(\d+)([A-Z]+)$")


def _normalize_values(values: List[str], target_len: int) -> List[str]:
    values = [v.strip() for v in values]
    if len(values) < target_len:
        values = values + [""] * (target_len - len(values))
    elif len(values) > target_len:
        values = values[:target_len]
    return values


@dataclass
class LabelInfo:
    level: int
    anchor_letter: Optional[str]
    label_letter: str

    @property
    def as_string(self) -> str:
        if self.level == 1:
            return f"P{self.level}{self.label_letter}"
        return f"P{self.level}{self.anchor_letter}{self.label_letter}"


def parse_label(label: Optional[str]) -> Optional[LabelInfo]:
    if not label:
        return None
    match = LABEL_RE.match(label.strip())
    if not match:
        return None
    level = int(match.group(1))
    suffix = match.group(2)
    if level == 1:
        return LabelInfo(level=level, anchor_letter=None, label_letter=suffix)
    if len(suffix) < 2:
        return None
    return LabelInfo(level=level, anchor_letter=suffix[0], label_letter=suffix[1:])


def is_label_column(values: List[str]) -> bool:
    return any(LABEL_RE.match(v.strip()) for v in values)


def next_letter(used: Iterable[str]) -> str:
    used_set = set(used)
    alphabet = string.ascii_uppercase
    # Allow multi-letter labels if needed
    for length in range(1, 4):
        for combo in product(alphabet, repeat=length):
            candidate = "".join(combo)
            if candidate not in used_set:
                return candidate
    raise RuntimeError("Ran out of label letters.")


class MappingEntry:
    def __init__(self, key: str, raw_value: str):
        self.key = key.strip()
        self.atom_names = [name.strip() for name in self.key.split(",")]

        tokens = str(raw_value).split()
        if not tokens:
            raise ValueError(f"Mapping for {self.key} is empty")
        self.bead_names = [b.strip() for b in tokens[0].split(",")]
        self.num_beads = len(self.bead_names)

        self.columns: List[Dict] = []
        for token in tokens[1:]:
            values = _normalize_values(token.split(","), self.num_beads)
            self.columns.append({"values": values, "is_label": False})

        self.label_index: Optional[int] = None
        self.existing_labels = [""] * self.num_beads
        for idx, col in enumerate(self.columns):
            if is_label_column(col["values"]):
                col["is_label"] = True
                self.label_index = idx
                self.existing_labels = [
                    val if LABEL_RE.match(val) else "" for val in col["values"]
                ]
                break

    def set_labels(self, labels: List[str]):
        self.existing_labels = _normalize_values(labels, self.num_beads)

    def render(self, new_labels: List[str]) -> str:
        labels = _normalize_values(
            [lbl if lbl is not None else "" for lbl in new_labels], self.num_beads
        )
        tokens: List[str] = []
        label_written = False
        for col in self.columns:
            if col["is_label"]:
                tokens.append(",".join(labels))
                label_written = True
            else:
                tokens.append(",".join(_normalize_values(col["values"], self.num_beads)))
        if not label_written:
            tokens.insert(0, ",".join(labels))
        # Drop empty trailing tokens for cleaner output
        tokens = [tok for tok in tokens if tok.strip() != ""]
        return "  ".join([",".join(self.bead_names)] + tokens)


@dataclass
class BeadMember:
    atom: mda.core.groups.Atom
    entry: MappingEntry
    bead_index: int
    existing_label: str


def load_mapping_entries(mapping_file: Path) -> Tuple[Dict, List[MappingEntry]]:
    conf = yaml.safe_load(mapping_file.read_text())
    atoms_conf = conf.get("atoms", {})
    entries = [MappingEntry(key, value) for key, value in atoms_conf.items()]
    return conf, entries


def build_residue_pairs(
    aa_universe: mda.Universe, cg_universe: mda.Universe, resname: str
) -> List[Tuple[mda.core.groups.Residue, mda.core.groups.Residue]]:
    aa_residues = sorted(
        [res for res in aa_universe.residues if res.resname == resname],
        key=lambda r: (r.segid, r.resid),
    )
    cg_residues = sorted(
        [res for res in cg_universe.residues if res.resname == resname],
        key=lambda r: (r.segid, r.resid),
    )
    if len(aa_residues) != len(cg_residues):
        logging.warning(
            "Found %d %s residues in atomistic vs %d in CG. Pairing by order.",
            len(aa_residues),
            resname,
            len(cg_residues),
        )
    return list(zip(aa_residues, cg_residues))


def build_adjacency(residue: mda.core.groups.Residue) -> Dict[int, List[int]]:
    adjacency: Dict[int, List[int]] = defaultdict(list)
    atoms = residue.atoms
    universe_atoms = atoms.universe.atoms
    bonds = guessers.guess_bonds(atoms, atoms.positions, box=atoms.universe.dimensions)
    for i, j in bonds:
        ai = universe_atoms[int(i)]
        aj = universe_atoms[int(j)]
        if ai.resindex != residue.resindex or aj.resindex != residue.resindex:
            continue
        adjacency[ai.index].append(aj.index)
        adjacency[aj.index].append(ai.index)
    return adjacency


def select_initial_anchors(
    dist_to_bead: Dict[int, float],
    atoms_by_index: Dict[int, mda.core.groups.Atom],
    threshold: float,
) -> List[int]:
    if not dist_to_bead:
        return []
    min_dist = min(dist_to_bead.values())
    anchors = [idx for idx, dist in dist_to_bead.items() if dist - min_dist <= threshold]
    if not anchors:
        anchors = [min(dist_to_bead, key=dist_to_bead.get)]
    return sorted(anchors, key=lambda idx: (dist_to_bead[idx], atoms_by_index[idx].name))


def assign_labels_for_bead(
    bead_position: np.ndarray,
    members: List[BeadMember],
    adjacency: Dict[int, List[int]],
    distance_threshold: float,
    overwrite_existing: bool,
) -> Dict[int, LabelInfo]:
    if not members:
        return {}

    atoms_by_index = {m.atom.index: m.atom for m in members}
    bead_indices = set(atoms_by_index.keys())
    filtered_adjacency = {
        idx: [n for n in adjacency.get(idx, []) if n in bead_indices] for idx in bead_indices
    }
    dist_to_bead = {
        idx: float(np.linalg.norm(atom.position - bead_position))
        for idx, atom in atoms_by_index.items()
    }

    assigned: Dict[int, LabelInfo] = {}
    used_letters: Dict[int, set[str]] = defaultdict(set)

    if not overwrite_existing:
        for member in members:
            info = parse_label(member.existing_label)
            if info:
                assigned[member.atom.index] = info
                used_letters[info.level].add(info.label_letter)

    if not any(info.level == 1 for info in assigned.values()):
        for anchor_idx in select_initial_anchors(dist_to_bead, atoms_by_index, distance_threshold):
            if anchor_idx not in assigned:
                letter = next_letter(used_letters[1])
                info = LabelInfo(level=1, anchor_letter=None, label_letter=letter)
                assigned[anchor_idx] = info
                used_letters[1].add(letter)

    queue = deque(sorted(assigned.keys(), key=lambda idx: (assigned[idx].level, dist_to_bead[idx])))

    while queue:
        current = queue.popleft()
        current_info = assigned[current]
        child_level = current_info.level + 1
        anchor_letter = current_info.label_letter
        neighbors = sorted(
            [n for n in filtered_adjacency.get(current, []) if n not in assigned],
            key=lambda n: np.linalg.norm(atoms_by_index[n].position - atoms_by_index[current].position),
        )
        for neighbor in neighbors:
            letter = next_letter(used_letters[child_level])
            used_letters[child_level].add(letter)
            assigned[neighbor] = LabelInfo(level=child_level, anchor_letter=anchor_letter, label_letter=letter)
            queue.append(neighbor)

    while len(assigned) < len(bead_indices):
        remaining = [idx for idx in bead_indices if idx not in assigned]
        start_idx = min(remaining, key=lambda idx: dist_to_bead[idx])
        letter = next_letter(used_letters[1])
        used_letters[1].add(letter)
        assigned[start_idx] = LabelInfo(level=1, anchor_letter=None, label_letter=letter)
        queue.append(start_idx)
        while queue:
            current = queue.popleft()
            current_info = assigned[current]
            child_level = current_info.level + 1
            anchor_letter = current_info.label_letter
            neighbors = sorted(
                [n for n in filtered_adjacency.get(current, []) if n not in assigned],
                key=lambda n: np.linalg.norm(atoms_by_index[n].position - atoms_by_index[current].position),
            )
            for neighbor in neighbors:
                letter = next_letter(used_letters[child_level])
                used_letters[child_level].add(letter)
                assigned[neighbor] = LabelInfo(level=child_level, anchor_letter=anchor_letter, label_letter=letter)
                queue.append(neighbor)

    return assigned


def assign_residue_labels(
    entries: List[MappingEntry],
    aa_residue: mda.core.groups.Residue,
    cg_residue: mda.core.groups.Residue,
    distance_threshold: float,
    overwrite_existing: bool,
) -> Dict[str, Dict[int, str]]:
    atom_lookup = {atom.name: atom for atom in aa_residue.atoms}
    used_names = set()
    members_by_bead: Dict[str, List[BeadMember]] = defaultdict(list)

    for entry in entries:
        selected_atom: Optional[mda.core.groups.Atom] = None
        for candidate in entry.atom_names:
            if candidate in atom_lookup and candidate not in used_names:
                selected_atom = atom_lookup[candidate]
                used_names.add(candidate)
                break
        if selected_atom is None:
            continue
        for bead_idx, bead_name in enumerate(entry.bead_names):
            members_by_bead[bead_name].append(
                BeadMember(
                    atom=selected_atom,
                    entry=entry,
                    bead_index=bead_idx,
                    existing_label=entry.existing_labels[bead_idx],
                )
            )

    cg_atoms = {atom.name: atom for atom in cg_residue.atoms}
    adjacency = build_adjacency(aa_residue)
    labels: Dict[str, Dict[int, str]] = defaultdict(dict)

    for bead_name, members in members_by_bead.items():
        bead_atom = cg_atoms.get(bead_name)
        if bead_atom is None:
            logging.warning("Bead %s not found in CG residue %s%d", bead_name, cg_residue.resname, cg_residue.resid)
            continue
        assigned = assign_labels_for_bead(
            bead_atom.position, members, adjacency, distance_threshold, overwrite_existing
        )
        for member in members:
            label_info = assigned.get(member.atom.index)
            if label_info:
                labels[member.entry.key][member.bead_index] = label_info.as_string
    return labels


def compute_residue_graph(
    entries: List[MappingEntry],
    aa_residue: mda.core.groups.Residue,
    cg_residue: mda.core.groups.Residue,
    distance_threshold: float,
    overwrite_existing: bool,
) -> Tuple[Dict[Union[int, str], np.ndarray], List[Tuple[Union[int, str], Union[int, str], int]], Dict[int, LabelInfo]]:
    atom_lookup = {atom.name: atom for atom in aa_residue.atoms}
    used_names = set()
    members_by_bead: Dict[str, List[BeadMember]] = defaultdict(list)

    for entry in entries:
        selected_atom: Optional[mda.core.groups.Atom] = None
        for candidate in entry.atom_names:
            if candidate in atom_lookup and candidate not in used_names:
                selected_atom = atom_lookup[candidate]
                used_names.add(candidate)
                break
        if selected_atom is None:
            continue
        for bead_idx, bead_name in enumerate(entry.bead_names):
            members_by_bead[bead_name].append(
                BeadMember(
                    atom=selected_atom,
                    entry=entry,
                    bead_index=bead_idx,
                    existing_label=entry.existing_labels[bead_idx],
                )
            )

    cg_atoms = {atom.name: atom for atom in cg_residue.atoms}
    adjacency = build_adjacency(aa_residue)

    positions: Dict[Union[int, str], np.ndarray] = {}
    edges: List[Tuple[Union[int, str], Union[int, str], int]] = []
    assigned_info: Dict[int, LabelInfo] = {}

    for bead_name, members in members_by_bead.items():
        bead_atom = cg_atoms.get(bead_name)
        if bead_atom is None:
            continue
        positions[f"bead:{bead_name}"] = bead_atom.position
        assigned = assign_labels_for_bead(
            bead_atom.position, members, adjacency, distance_threshold, overwrite_existing
        )
        if not assigned:
            continue
        level_letter_to_idx = {(info.level, info.label_letter): idx for idx, info in assigned.items()}
        for idx, info in assigned.items():
            assigned_info[idx] = info
            positions[idx] = aa_residue.atoms.universe.atoms[idx].position
        for idx, info in assigned.items():
            if info.level == 1:
                edges.append((f"bead:{bead_name}", idx, info.level))
                continue
            parent_idx = level_letter_to_idx.get((info.level - 1, info.anchor_letter))
            if parent_idx is not None:
                edges.append((parent_idx, idx, info.level))
            else:
                edges.append((f"bead:{bead_name}", idx, info.level))

    return positions, edges, assigned_info


def _project_positions(positions: Dict[Union[int, str], np.ndarray]) -> Dict[Union[int, str], np.ndarray]:
    keys = list(positions.keys())
    coords = np.stack([positions[k] for k in keys])
    coords_centered = coords - coords.mean(axis=0)
    if coords_centered.shape[0] > 1:
        _, _, vh = np.linalg.svd(coords_centered, full_matrices=False)
        basis = vh[:2].T
        coords2d = coords_centered @ basis
    else:
        coords2d = np.hstack([coords_centered[:, :1], np.zeros_like(coords_centered[:, :1])])
    return {k: coords2d[i] for i, k in enumerate(keys)}


def plot_hierarchy(
    positions: Dict[Union[int, str], np.ndarray],
    edges: List[Tuple[Union[int, str], Union[int, str], int]],
    assigned_info: Dict[int, LabelInfo],
    aa_residue: mda.core.groups.Residue,
    output_path: Path,
):
    positions2d = _project_positions(positions)

    bead_keys = [k for k in positions2d if isinstance(k, str)]
    atom_keys = [k for k in positions2d if not isinstance(k, str)]

    fig, ax = plt.subplots(figsize=(6, 6))

    if bead_keys:
        bead_coords = np.array([positions2d[k] for k in bead_keys])
        ax.scatter(bead_coords[:, 0], bead_coords[:, 1], marker="s", s=120, c="tab:blue", label="beads")
        for k, (x, y) in zip(bead_keys, bead_coords):
            ax.text(x, y, k.replace("bead:", ""), ha="center", va="center", fontsize=4, color="white", weight="bold")

    levels = {assigned_info[idx].level for idx in atom_keys if idx in assigned_info}
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:orange", "tab:green", "tab:red"])
    level_colors = {level: color_cycle[(level - 1) % len(color_cycle)] for level in sorted(levels)}

    if atom_keys:
        atom_coords = np.array([positions2d[k] for k in atom_keys])
        colors = [level_colors.get(assigned_info.get(k, LabelInfo(1, None, "")).level, "gray") for k in atom_keys]
        ax.scatter(atom_coords[:, 0], atom_coords[:, 1], marker="o", s=30, c=colors, edgecolors="k", label="atoms")
        for k, (x, y) in zip(atom_keys, atom_coords):
            atom = aa_residue.universe.atoms[k]
            label = assigned_info.get(k, None)
            txt = f"{atom.name}"
            if label:
                txt += f" ({label.as_string})"
            ax.text(x, y, txt, fontsize=4, ha="center", va="bottom")

    for parent, child, level in edges:
        p = positions2d.get(parent)
        c = positions2d.get(child)
        if p is None or c is None:
            continue
        color = level_colors.get(level, "gray")
        ax.annotate(
            "",
            xy=(c[0], c[1]),
            xytext=(p[0], p[1]),
            arrowprops=dict(arrowstyle="->", lw=1.2, color=color),
        )

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(f"Hierarchy for {aa_residue.resname} resid {aa_residue.resid}")
    ax.axis("equal")
    ax.legend(loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_hierarchy_plotly(
    positions: Dict[Union[int, str], np.ndarray],
    edges: List[Tuple[Union[int, str], Union[int, str], int]],
    assigned_info: Dict[int, LabelInfo],
    aa_residue: mda.core.groups.Residue,
    output_path: Path,
):
    bead_keys = [k for k in positions if isinstance(k, str)]
    atom_keys = [k for k in positions if not isinstance(k, str)]

    levels = {assigned_info[idx].level for idx in atom_keys if idx in assigned_info}
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    level_colors = {level: color_cycle[(level - 1) % len(color_cycle)] for level in sorted(levels)}

    traces = []
    if bead_keys:
        bead_coords = np.array([positions[k] for k in bead_keys])
        traces.append(
            go.Scatter3d(
                x=bead_coords[:, 0],
                y=bead_coords[:, 1],
                z=bead_coords[:, 2],
                mode="markers+text",
                marker=dict(symbol="square", size=6, color="rgba(31,119,180,0.8)"),
                text=[k.replace("bead:", "") for k in bead_keys],
                textposition="top center",
                name="beads",
            )
        )

    if atom_keys:
        atom_coords = np.array([positions[k] for k in atom_keys])
        colors = [level_colors.get(assigned_info.get(k, LabelInfo(1, None, "")).level, "gray") for k in atom_keys]
        texts = []
        for k in atom_keys:
            atom = aa_residue.universe.atoms[k]
            label = assigned_info.get(k, None)
            txt = atom.name
            if label:
                txt += f" ({label.as_string})"
            texts.append(txt)
        traces.append(
            go.Scatter3d(
                x=atom_coords[:, 0],
                y=atom_coords[:, 1],
                z=atom_coords[:, 2],
                mode="markers+text",
                marker=dict(size=4, color=colors, line=dict(width=1, color="black")),
                text=texts,
                textposition="top center",
                name="atoms",
            )
        )

    for parent, child, level in edges:
        p = positions.get(parent)
        c = positions.get(child)
        if p is None or c is None:
            continue
        color = level_colors.get(level, "gray")
        traces.append(
            go.Scatter3d(
                x=[p[0], c[0]],
                y=[p[1], c[1]],
                z=[p[2], c[2]],
                mode="lines",
                line=dict(color=color, width=3),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Hierarchy for {aa_residue.resname} resid {aa_residue.resid}",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        legend=dict(orientation="h"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def merge_labels(
    entries: List[MappingEntry],
    collected: Dict[str, Dict[int, List[str]]],
    overwrite_existing: bool,
) -> Dict[str, List[str]]:
    final_labels: Dict[str, List[str]] = {}
    for entry in entries:
        labels = []
        entry_labels = collected.get(entry.key, {})
        for bead_idx in range(entry.num_beads):
            if entry.existing_labels[bead_idx] and not overwrite_existing:
                labels.append(entry.existing_labels[bead_idx])
                continue
            candidates = entry_labels.get(bead_idx, [])
            if candidates:
                most_common = Counter(candidates).most_common(1)[0][0]
                labels.append(most_common)
            else:
                labels.append(entry.existing_labels[bead_idx])
        final_labels[entry.key] = labels
    return final_labels


def generate_temp_cg(atomistic: Path, mapping_yaml: Path, mapping_folder: Optional[Path]) -> Tuple[tempfile.TemporaryDirectory, Path]:
    if mapping_folder is None:
        mapping_folder = mapping_yaml.parent
    mapping_folder = mapping_folder.resolve()
    if not mapping_folder.exists():
        raise FileNotFoundError(f"Mapping folder '{mapping_folder}' not found.")
    tmpdir = tempfile.TemporaryDirectory()
    cg_path = Path(tmpdir.name) / "temp_cg.pdb"
    mapper = Mapper(
        {
            "mapping": str(mapping_folder),
            "input": str(atomistic),
            "output": str(cg_path),
            "isatomistic": True,
            "align": False,
        }
    )
    mapper.map()
    mapper.save(filename=str(cg_path))
    return tmpdir, cg_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute hierarchical backmapping labels for a mapping yaml using atomistic and CG structures."
        )
    )
    parser.add_argument("-m", "--mapping", required=True, type=Path, help="Mapping yaml file to update.")
    parser.add_argument("-a", "--atomistic", required=True, type=Path, help="Atomistic PDB file.")
    parser.add_argument("-c", "--cg", required=False, type=Path, help="Coarse-grained PDB file aligned to the atomistic one. If omitted, a temporary CG will be generated via Mapper.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output yaml (defaults to overwrite input).")
    parser.add_argument(
        "--mapping-folder",
        type=Path,
        default=None,
        help="Optional mapping folder to pass to Mapper when generating CG (defaults to parent of the mapping yaml).",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.25,
        help="Atoms within this distance of the closest atom to a bead are grouped at hierarchy level 1.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Recompute labels even when they are already present in the mapping.",
    )
    parser.add_argument(
        "--plot-hierarchy",
        type=Path,
        help="Optional path to save a hierarchy plot (2D projection with arrows showing dependencies).",
    )
    parser.add_argument(
        "--plot-resid",
        type=int,
        default=None,
        help="Resid of the residue to plot (defaults to the first matching residue).",
    )
    parser.add_argument(
        "--plotly-hierarchy",
        type=Path,
        help="Optional path to save an interactive 3D hierarchy plot (html via plotly).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    conf, entries = load_mapping_entries(args.mapping)
    resname = conf.get("molecule")
    if resname is None:
        raise ValueError("Mapping file must contain a 'molecule' field.")

    temp_dirs: List[tempfile.TemporaryDirectory] = []
    try:
        cg_path = args.cg
        if cg_path is None:
            tmpdir, cg_path = generate_temp_cg(args.atomistic, args.mapping, args.mapping_folder)
            temp_dirs.append(tmpdir)
            logging.info("Temporary CG generated at %s", cg_path)

        aa_u = mda.Universe(str(args.atomistic))
        cg_u = mda.Universe(str(cg_path))

        residue_pairs = build_residue_pairs(aa_u, cg_u, resname)
        if not residue_pairs:
            raise RuntimeError(f"No residues named {resname} found in provided structures.")

        collected: Dict[str, Dict[int, List[str]]] = defaultdict(lambda: defaultdict(list))
        for aa_res, cg_res in residue_pairs:
            res_labels = assign_residue_labels(entries, aa_res, cg_res, args.distance_threshold, args.overwrite_existing)
            for entry_key, bead_dict in res_labels.items():
                for bead_idx, label in bead_dict.items():
                    collected[entry_key][bead_idx].append(label)

        final_labels = merge_labels(entries, collected, args.overwrite_existing)

        for entry in entries:
            entry.set_labels(final_labels[entry.key])
            conf["atoms"][entry.key] = entry.render(final_labels[entry.key])

        output_path = args.output if args.output is not None else args.mapping
        output_path.write_text(yaml.safe_dump(conf, sort_keys=False))
        logging.info("Updated mapping saved to %s", output_path)

        if args.plot_hierarchy or args.plotly_hierarchy:
            target_pair = None
            if args.plot_resid is not None:
                for aa_res, cg_res in residue_pairs:
                    if aa_res.resid == args.plot_resid:
                        target_pair = (aa_res, cg_res)
                        break
                if target_pair is None:
                    logging.warning("Resid %s not found; defaulting to first %s residue.", args.plot_resid, resname)
            if target_pair is None and residue_pairs:
                target_pair = residue_pairs[0]
            if target_pair is None:
                logging.error("No residue available to plot.")
            else:
                positions, edges, assigned_info = compute_residue_graph(
                    entries,
                    target_pair[0],
                    target_pair[1],
                    args.distance_threshold,
                    args.overwrite_existing,
                )
                if not positions:
                    logging.warning("No positions found for plotting; skipping figure.")
                else:
                    if args.plot_hierarchy:
                        plot_hierarchy(positions, edges, assigned_info, target_pair[0], args.plot_hierarchy)
                        logging.info("Hierarchy plot saved to %s", args.plot_hierarchy)
                    if args.plotly_hierarchy:
                        plot_hierarchy_plotly(positions, edges, assigned_info, target_pair[0], args.plotly_hierarchy)
                        logging.info("Interactive hierarchy saved to %s", args.plotly_hierarchy)
    finally:
        for td in temp_dirs:
            try:
                td.cleanup()
            except Exception as e:
                logging.warning("Failed to clean temporary directory: %s", e)


if __name__ == "__main__":
    main()
