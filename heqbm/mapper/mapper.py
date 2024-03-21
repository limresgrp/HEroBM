import copy
import re
import os
import sys
import torch
import yaml
import traceback
import numpy as np
import MDAnalysis as mda
import pandas as pd

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from MDAnalysis.core.groups import Atom
from ase.cell import Cell

from .bead import BeadMappingSettings, BeadMappingAtomSettings, Bead, HierarchicalBead
from heqbm.utils import DataDict
from heqbm.utils.backbone import Phi, Psi, Omega
from heqbm.utils.atomType import get_type_from_name
from heqbm.utils.geometry import get_dihedrals


class Mapper():

    u: mda.Universe = None
    bead_types_filename: str = "bead_types.yaml"

    _keep_hydrogens: bool = False
    _weigth_based_on: str = 'mass'

    _atom2bead: dict[str, str] = {}
    _bead2atom: dict[str, List[str]] = {}
    _bead_types: dict[str, int] = {}
    _bead_mapping_settings: dict[str, BeadMappingSettings] = {}
    _bead_cm: dict[str, str] = {}

    _incomplete_beads: List[HierarchicalBead] = []
    _complete_beads: List[HierarchicalBead] = []
    _ordered_beads: List[HierarchicalBead] = []
    _max_bead_reconstructed_atoms: int = 0
    _max_bead_all_atoms: int = 0
    _n_beads: int = 0
    _n_atoms: int = 0

    _bead2atom_reconstructed_idcs: np.ndarray = None
    _bead2atom_reconstructed_idcs_orig: np.ndarray = None
    _bead2atom_pos_idcs: np.ndarray = None
    _bead2atom_pos_weights: np.ndarray = None

    _atom_positions: np.ndarray = None
    _bb_atom_positions: np.ndarray = None
    _bb_atom_idcs: np.ndarray = None
    _ca_atom_positions: np.ndarray = None
    _ca_atom_idcs: np.ndarray = None
    _bead_positions: np.ndarray = None
    _ca_bead_positions: np.ndarray = None
    _ca_bead_idcs: np.ndarray = None
    _ca_next_directions: np.ndarray = None

    _atom_idnames: np.ndarray = None
    _atom_resnames: np.ndarray = None
    _atom_names: np.ndarray = None
    _atom2idcs_dict: Dict[str, int] = {}

    _bead_idnames: np.ndarray = None
    _bead_resnames: np.ndarray = None
    _bead_names: np.ndarray = None

    _atom_residcs: np.ndarray = None
    _atom_resnums: np.ndarray = None
    
    _bead_residcs: np.ndarray = None
    _bead_resnums: np.ndarray = None
    
    _atom_forces: np.ndarray = None
    _atom_chainidcs: np.ndarray = None
    _bead_chainidcs: np.ndarray = None
    _cell: np.ndarray = None
    _pbc: np.ndarray = np.array([True, True, True])

    _phi_dihedral_idcs: np.ndarray = None
    _psi_dihedral_idcs: np.ndarray = None
    _omega_dihedral_idcs: np.ndarray = None
    _phi_dict: dict[int, Phi] = {}
    _psi_dict: dict[int, Psi] = {}
    _omega_dict: dict[int, Omega] = {}
    _bb_phi_psi_values: np.ndarray = None

    _bead2atom_idcs_mask: np.ndarray = None

    _bond_idcs: np.ndarray = None
    _angle_idcs: np.ndarray = None

    # Properties of mapping
    
    @property
    def bead_reconstructed_size(self):
        return self._max_bead_reconstructed_atoms
    
    @property
    def bead_all_size(self):
        return self._max_bead_all_atoms
    
    @property
    def bead_types_dict(self):
        return self._bead_types
    
    @property
    def n_bead_types(self):
        return len(self._bead_types)
    
    @property
    def bead2atom_idcs_mask(self):
        return self._bead2atom_idcs_mask

    # Properties of mapped instance
    
    @property
    def bead2atom_reconstructed_idcs(self):
        if self._bead2atom_reconstructed_idcs is None:
            return None
        return np.ma.masked_array(self._bead2atom_reconstructed_idcs, mask=~self.bead2atom_reconstructed_idcs_mask)
    
    @property
    def bead2atom_reconstructed_weights(self):
        if self._bead2atom_reconstructed_weights is None:
            return None
        return np.ma.masked_array(self._bead2atom_reconstructed_weights, mask=~self.bead2atom_reconstructed_idcs_mask)
    
    @property
    def bead2atom_reconstructed_idcs_mask(self):
        if self._bead2atom_reconstructed_idcs is None:
            return None
        return self._bead2atom_reconstructed_idcs != -1
    
    @property
    def mapping_idcs_instance(self):
        if self._bead2atom_reconstructed_idcs is None:
            return None
        return [list(row[row != -1]) for row in self._bead2atom_reconstructed_idcs]
    
    @property
    def dihedral_idcs(self):
        try:
            return np.concatenate(
                [
                    self._phi_dihedral_idcs,
                    self._psi_dihedral_idcs,
                    self._omega_dihedral_idcs,
                ], axis=0
            )
        except:
            return None
    
    @property
    def resnames(self):
        if self._atom_resnames is None or self._atom_residcs is None:
            return None
        return self._atom_resnames[
            np.concatenate(
                ([0], np.where(self._atom_residcs[:-1] != self._atom_residcs[1:])[0] + 1)
                )
            ]
    
    @property
    def resnumbers(self):
        if self._atom_resnums is None:
            return None
        return self._atom_resnums[
            np.concatenate(
                ([0], np.where(self._atom_residcs[:-1] != self._atom_residcs[1:])[0] + 1)
                )
            ]
    
    @property
    def num_residues(self):
        if self._atom_chainidcs is None or self._atom_residcs is None:
            return None
        chain_residue_names = []
        for chain, resindex in zip(self._atom_chainidcs, self._atom_residcs):
            chain_residue_names.append(f'{str(chain)}_{str(resindex)}')
        return len(np.unique(chain_residue_names))
    
    @property
    def num_atoms(self):
        if self._atom_names is None:
            return None
        return len(self._atom_names)

    @property
    def atom_types(self):
        if self._atom_names is None:
            return None
        return np.array([get_type_from_name(name) for name in self._atom_names])

    @property
    def num_beads(self):
        if self._bead_idnames is None:
            return None
        return len(self._bead_idnames)
    
    @property
    def bead_types(self):
        if self._bead_idnames is None:
            return None
        return np.array([self._bead_types[idname] for idname in self._bead_idnames])
    
    @property
    def dataset(self):
        return {k: v for k, v in {
            DataDict.NUM_RESIDUES: self.num_residues,
            DataDict.RESNAMES:     self.resnames,
            DataDict.RESNUMBERS:   self.resnumbers,

            DataDict.NUM_ATOMS:       self.num_atoms,
            DataDict.ATOM_POSITION:   self._atom_positions,
            DataDict.ATOM_NAMES:      self._atom_names,
            DataDict.ATOM_TYPES:      self.atom_types,
            DataDict.ATOM_RESNAMES:   self._atom_resnames,
            DataDict.ATOM_RESIDCS:    self._atom_residcs,
            DataDict.ATOM_RESNUMBERS: self._atom_resnums,
            DataDict.ATOM_CHAINIDCS:  self._atom_chainidcs,
            DataDict.ATOM_FORCES:     self._atom_forces,

            DataDict.NUM_BEADS:       self.num_beads,
            DataDict.BEAD_POSITION:   self._bead_positions,
            DataDict.BEAD_IDNAMES:    self._bead_idnames,
            DataDict.BEAD_RESNAMES:   self._bead_resnames,
            DataDict.BEAD_NAMES:      self._bead_names,
            DataDict.BEAD_TYPES:      self.bead_types,
            DataDict.BEAD_RESIDCS:    self._bead_residcs,
            DataDict.BEAD_RESNUMBERS: self._bead_resnums,
            DataDict.BEAD_CHAINIDCS:  self._bead_chainidcs,

            DataDict.BOND_IDCS:  self._bond_idcs,
            DataDict.ANGLE_IDCS: self._angle_idcs,
            DataDict.CELL:       self._cell,
            DataDict.PBC:        self._pbc,

            # DataDict.MAPPING_IDCS:        self.mapping_idcs_instance,
            DataDict.BEAD2ATOM_IDCS:      self.bead2atom_reconstructed_idcs,
            DataDict.BEAD2ATOM_WEIGHTS:   self.bead2atom_reconstructed_weights,
            # DataDict.BEAD2ATOM_IDCS_MASK: self.bead2atom_reconstructed_idcs_mask,

            DataDict.BB_ATOM_POSITION: self._bb_atom_positions,
            DataDict.BB_PHIPSI: self._bb_phi_psi_values,
            DataDict.CA_ATOM_POSITION: self._ca_atom_positions,
            DataDict.CA_ATOM_IDCS: self._ca_atom_idcs,
            DataDict.CA_BEAD_POSITION: self._ca_bead_positions,
            DataDict.CA_BEAD_IDCS: self._ca_bead_idcs,
            
            DataDict.PHI_DIH_IDCS: self._phi_dihedral_idcs,
            DataDict.PSI_DIH_IDCS: self._psi_dihedral_idcs,
            DataDict.OMEGA_DIH_IDCS: self._omega_dihedral_idcs,
        }.items() if v is not None}

    def __init__(self, config: Dict) -> None:
        mapping_folder = config.get("mapping", None)
        if mapping_folder is None:
            raise Exception(
                """
                You must provide the mapping folder.
                Add 'mapping_folder: name-of-mapping-folder' in the config file.
                mappings are specified in a mapping folder inside 'heqbm/data/'
                """
            )
        self._root = os.path.join(os.path.dirname(__file__), '..', 'data', mapping_folder)
        self._keep_hydrogens = config.get("keep_hydrogens", True)
        self._weigth_based_on = config.get("weigth_based_on", "mass")

        # Iterate configuration files and load all mappings
        self._clear_mappings()
        self._load_mappings()

    def _clear_mappings(self):
        self._atom2bead.clear()
        self._bead2atom.clear()
        self._bead_types.clear()

        self._incomplete_beads.clear()
        self._complete_beads.clear()
        self._ordered_beads.clear()

        self._bead2atom_reconstructed_idcs: np.ndarray = None
        self._bead2atom_pos_idcs: np.ndarray = None
        self._bead2atom_pos_weights: np.ndarray = None
        self._bead2atom_reconstructed_weights: np.ndarray = None

        self._atom_positions: np.ndarray = None
        self._bb_atom_positions: np.ndarray = None
        self._bb_atom_idcs: np.ndarray = None
        self._ca_atom_positions: np.ndarray = None
        self._bead_positions: np.ndarray = None
        self._ca_bead_positions: np.ndarray = None
        self._ca_bead_idcs: np.ndarray = None

        self._atom_idnames: np.ndarray = None
        self._atom_resnames: np.ndarray = None
        self._atom_names: np.ndarray = None
        self._atom2idcs_dict: Dict[str, int] = {}
        
        self._bead_idnames: np.ndarray = None
        self._bead_resnames: np.ndarray = None
        self._bead_names: np.ndarray = None
        
        self._atom_forces: np.ndarray = None
        self._atom_chainidcs: np.ndarray = None
        self._bead_chainidcs: np.ndarray = None
        self._cell: np.ndarray = None
        self._pbc: np.ndarray = np.array([True, True, True])

        self._bond_idcs: np.ndarray = None
        self._angle_idcs: np.ndarray = None

    def _load_mappings(self):
        
        # Load bead types file, if existent.
        # It contains the bead type to identify each bead inside the NN
        # Different beads could have the same bead type (for example, all backbone beads could use the same bead type)
        bead_types_filename = os.path.join(self._root, self.bead_types_filename)
        if os.path.isfile(bead_types_filename):
            bead_types_conf: dict = yaml.safe_load(Path(bead_types_filename).read_text())
        else:
            bead_types_conf: dict = dict()
        
        # Iterate mapping files -> 1 mapping file = 1 residue mapping
        for filename in os.listdir(self._root):
            if filename == self.bead_types_filename:
                continue
            
            conf: dict = OrderedDict(yaml.safe_load(Path(os.path.join(self._root, filename)).read_text()))
            
            mol = conf.get("molecule")
            
            _conf_bead2atom = OrderedDict({})

            for atom, bead_settings_str in conf.get("atoms").items():
                if not self._keep_hydrogens and atom.startswith('H'):
                    continue
                
                all_bead_settings = bead_settings_str.split()
                bead_names = all_bead_settings[0].split(',')

                for i, bn in enumerate(bead_names):
                    bead_idname = DataDict.STR_SEPARATOR.join([mol, bn])
                    atom_idname = DataDict.STR_SEPARATOR.join([mol, atom])
                    bead_settings = [x.split(',')[i] for x in all_bead_settings[1:]]
                    
                    atom2bead_list = self._atom2bead.get(atom_idname, [])
                    atom2bead_list.append(bead_idname)
                    self._atom2bead[atom_idname] = atom2bead_list
                    bms = self._bead_mapping_settings.get(bead_idname, BeadMappingSettings(bead_idname))
                    bmas = bms.get_bmas_by_atom_idname(atom_idname)
                    if bmas is None:
                        bmas = BeadMappingAtomSettings(bead_settings, bead_idname, atom_idname, num_shared_beads=len(bead_names))
                        bms.add_atom_settings(bmas)
                        self._bead_mapping_settings[bead_idname] = bms
                    
                    bead2atom: List[str] = _conf_bead2atom.get(bead_idname, [])
                    if len(bead2atom) == 0 and bead_idname not in self._bead_types:
                        bead_type = bead_types_conf.get(bead_idname, max(bead_types_conf.values(), default=-1)+1)
                        bead_types_conf[bead_idname] = bead_type
                        self._bead_types[bead_idname] = bead_type
                    assert atom_idname not in bead2atom, f"{atom_idname} is already present in bead {bead_idname}. Duplicate mapping"
                    bead2atom.append(atom_idname)
                    _conf_bead2atom[bead_idname] = bead2atom
            
            for bms in self._bead_mapping_settings.values():
                bms.complete()
                self._max_bead_reconstructed_atoms = max(self._max_bead_reconstructed_atoms, bms.bead_reconstructed_size)
                self._max_bead_all_atoms = max(self._max_bead_all_atoms, bms.bead_all_size)
                
            for bead_idname, bead2atom in _conf_bead2atom.items():
                _bead2atom = self._bead2atom.get(bead_idname, [])
                _bead2atom.append(bead2atom)
                self._bead2atom[bead_idname] = _bead2atom
        
        with open(bead_types_filename, 'w') as outfile:
            yaml.dump(bead_types_conf, outfile, default_flow_style=False)
        
        for k, b2a in self._bead2atom.items():
            len_base = self.get_bead2atom_len(b2a[0])
            assert all([self.get_bead2atom_len(v) == len_base for v in b2a]), f"Configurations for bead type {k} have different number of atoms"

        ### Compute global mask and maximum bead size ###
        self._bead2atom_idcs_mask = np.zeros((self.n_bead_types, self.bead_reconstructed_size), dtype=bool)
                
        for bead_idname, b2a in self._bead2atom.items():
            self._bead2atom_idcs_mask[self._bead_types[bead_idname], :self.get_bead2atom_len(b2a[0])] = True
    
    def get_bead2atom_len(self, b2a: List[str]):
        return len([atom_name for atom_name in b2a if self._keep_hydrogens or get_type_from_name(atom_name) != 1])

    def _load_extra_mappings(self, bead_splits, atom_name, bead_idname):
        return
    
    def map(self, conf):
        # New mapping, clear last records
        self._incomplete_beads.clear()
        self._complete_beads.clear()
        self._ordered_beads.clear()
        self._n_beads = 0
        self._n_atoms = 0
        self.trajslice = conf.get("trajslice", None)

        self.u = mda.Universe(conf.get("input"), *conf.get("inputtraj", []), **conf.get("extrakwargs", {}))

        selection = conf.get("selection")
        if not self._keep_hydrogens:
            selection = selection + ' and not (element H)'
        try:
            self.sel = self.u.select_atoms(selection)
        except AttributeError:
            selection = selection.replace('element', 'type')
            self.sel = self.u.select_atoms(selection)
        
        if conf.get("atomistic", False):
            return self.map_impl()
        return self.map_impl_cg()
    
    def map_impl_cg(
        self,
        conf_id: int = 0 # Which configuration to select for naming atoms
    ):
        
        try:
            self._bead_chainidcs = self.sel.chainIDs
        except mda.exceptions.NoDataError:
            self._bead_chainidcs = np.array(['A'] * self.sel.n_atoms)
        
        self._bead_idnames = np.array([
            f"{resname}{DataDict.STR_SEPARATOR}{bead}"
            for resname, bead in zip(self.sel.resnames, self.sel.names)
        ])

        self._ca_bead_idcs = np.array([bn.split('_')[-1] in ['BB'] for bn in self._bead_idnames])

        bead_positions = []
        ca_atom_positions = []
        ca_bead_positions = []
        cell_sizes = []
        
        try:
            traj = self.u.trajectory if self.trajslice is None else self.u.trajectory[self.trajslice]
            for ts in traj:
                bead_pos = self.sel.positions
                if ts.dimensions is not None:
                    cell_sizes.append(Cell.fromcellpar(ts.dimensions)[:])
                bead_positions.append(bead_pos)
                ca_atom_positions.append(bead_pos[self._ca_bead_idcs])
                ca_bead_positions.append(bead_pos[self._ca_bead_idcs])
                
        except ValueError as e:
            print(f"Error rading trajectory: {e}. Skipping missing trajectory frames.")
        self._bead_positions =  np.stack(bead_positions, axis=0)
        self._ca_atom_positions = np.stack(ca_atom_positions, axis=0)
        self._ca_bead_positions = np.stack(ca_bead_positions, axis=0)
        self._cell = np.stack(cell_sizes, axis=0) if len(cell_sizes) > 0 else np.zeros((3, 3), dtype=np.float32)

        atom_idnames =   []
        atom_resnames =  []
        atom_names =     []
        atom_residcs =   []
        atom_resnums =   []
        
        bead_resnames =  []
        bead_names =     []
        bead_residcs =   []
        bead_resnums =   []

        atom_chainidcs = []
        bead2atom_reconstructed_idcs = -np.ones((len(self._bead_idnames), self.bead_reconstructed_size), dtype=int)

        atom_idnames_missing_multiplicity = {}
        atom_idnames2index = {}
        for h, (bead_idname, sel_bead) in enumerate(zip(self._bead_idnames, self.sel.atoms)):
            bead = self._create_bead(bead_idname)
            bead_resnames.append(bead.resname)
            bead_names.append(bead.name)

            for atom_idname in bead._config_ordered_atom_idnames[conf_id]:
                if atom_idnames_missing_multiplicity.get(atom_idname, 0) == 0:
                    atom_idnames.append(atom_idname)
                    atom_resname, atom_name = atom_idname.split(DataDict.STR_SEPARATOR)
                    atom_resnames.append(atom_resname)
                    atom_names.append(atom_name)
                    atom_residcs.append(sel_bead.resindex)
                    atom_resnums.append(sel_bead.resnum)
                    atom_chainidcs.append(self._bead_chainidcs[h])

                    atom_idnames_missing_multiplicity[atom_idname] = len(np.unique(self._atom2bead[atom_idname])) - 1
                    atom_index = None
                else:
                    atom_idnames_missing_multiplicity[atom_idname] -= 1
                    atom_index = atom_idnames2index.get(atom_idname)

                atom_index, _ = self._update_bead(bead, atom_idname, atom_index=atom_index)
                atom_idnames2index[atom_idname] = atom_index
            
            self._check_bead_completeness(bead)
            bead2atom_reconstructed_idcs[h, :bead.n_reconstructed_atoms] = bead._reconstructed_atom_idcs
            bead_residcs.append(sel_bead.resindex)
            bead_resnums.append(sel_bead.resnum)

        self._atom_idnames = np.array(atom_idnames)
        self._atom_resnames = np.array(atom_resnames)
        self._atom_names = np.array(atom_names)
        self._atom_residcs = np.array(atom_residcs)
        self._atom_residcs -= self._atom_residcs.min()
        self._atom_resnums = np.array(atom_resnums)
        self._bead_resnames = np.array(bead_resnames)
        self._bead_names = np.array(bead_names)
        self._bead_residcs = np.array(bead_residcs)
        self._bead_resnums = np.array(bead_resnums)
        self._bead_residcs -= self._bead_residcs.min()
        self._atom_chainidcs = np.array(atom_chainidcs)
        self._bead2atom_reconstructed_idcs = np.array(bead2atom_reconstructed_idcs)
        self._bead2atom_reconstructed_idcs_orig = np.array(bead2atom_reconstructed_idcs)
        self._bb_atom_idcs = np.array([an in ['CA', 'N', 'C'] for an in self._atom_names])
        self._ca_atom_idcs = np.array([an in ['CA'] for an in self._atom_names])
        self._ca_bead_idcs = np.array([bn in ['BB'] for bn in self._bead_names])

        batch, _, xyz = self._bead_positions.shape
        self._atom_positions =  np.zeros((batch, len(self._atom_names), xyz), dtype=self._bead_positions.dtype)

        self.compute_bead2atom_idcs_and_weights()
        self.compute_extra_map_impl()
        self.compute_dihedral_idcs()
            
    def map_impl(self):

        atom_resnames =  []
        atom_names =     []
        
        bead_idnames =   []
        bead_resnames =  []
        bead_names =     []
        bead_residcs =   []
        bead_resnums =   []
        bead_chainidcs = []

        excluded_atoms = []

        last_resindex = -1
        for atom in self.sel.atoms:
            try:
                atom_idname = DataDict.STR_SEPARATOR.join([atom.resname, re.sub(r'^(\d+)\s*(.+)', r'\2\1', atom.name)])

                # This check is necessary to complete beads on residue change.
                # This allows having beads with incomplete atoms that do not remain pending
                if atom.resindex > last_resindex:
                    for bead in self._incomplete_beads:
                        self._complete_bead(bead)
                        self._check_bead_completeness(bead)
                    last_resindex = atom.resindex
                
                # Get existing beads which contain the current atom.
                # If no beads exist, create a new one.
                # Multiple beads could be retrieved, in case an atom is shared among multiple beads
                beads = self._get_incomplete_bead_from_atom_idname(atom_idname)

                # Iterate the retrieved beads.
                atom_index = None
                for bead in beads:
                    # If the bead is newly created, record its info.
                    if bead.is_newly_created:
                        bead_idnames.append(bead.idname)
                        bead_resnames.append(bead.resname)
                        bead_names.append(bead.name)
                        bead_residcs.append(atom.resindex)
                        bead_resnums.append(atom.resnum)
                        try:
                            bead_chainidcs.append(atom.chainID)
                        except mda.exceptions.NoDataError:
                            bead_chainidcs.append('A')
                    
                    # Update the bead object with the current atom properties
                    atom_index, idcs_to_update = self._update_bead(bead, atom_idname, atom=atom, atom_index=atom_index)

                    if idcs_to_update is not None:
                        for bead2update in self._incomplete_beads:
                            if bead2update is bead:
                                continue
                            atomid2update_mask = np.isin(bead2update._all_atom_idcs, idcs_to_update)
                            if sum(atomid2update_mask) > 0:
                                for atomid2update in bead2update._all_atom_idcs[atomid2update_mask]:
                                    bead2update._all_atom_idcs[bead2update._all_atom_idcs >= atomid2update] -= 1
                    
                    self._check_bead_completeness(bead)
                
            except AssertionError:
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb) # Fixed format
                tb_info = traceback.extract_tb(tb)
                filename, line, func, text = tb_info[-1]

                print('An error occurred on line {} in statement {}'.format(line, text))
                raise
            except Exception as e:
                print(f"Missing {atom_idname} in mapping file")
                atom_to_exclude = '_'.join([atom.resname, atom.name])
                if atom_to_exclude not in excluded_atoms:
                    selection = selection + f' and not (resname {atom.resname} and name {atom.name})'
                    excluded_atoms.append(atom_to_exclude)
        # Complete all beads. Missing atoms will be ignored.
        for bead in self._incomplete_beads:
            self._complete_bead(bead)
            self._check_bead_completeness(bead)

        self._bead_idnames = np.array(bead_idnames)
        self._bead_resnames = np.array(bead_resnames)
        self._bead_names = np.array(bead_names)
        self._bead_residcs = np.array(bead_residcs)
        self._bead_residcs -= self._bead_residcs.min()
        self._bead_resnums = np.array(bead_resnums)
        self._bead_chainidcs = np.array(bead_chainidcs)

        self._atom_idnames = np.empty(self._n_atoms, dtype="<U32")
        self._atom_residcs = np.empty(self._n_atoms, dtype=int)
        self._atom_resnums = np.empty(self._n_atoms, dtype=int)
        self._atom_chainidcs = np.empty(self._n_atoms, dtype="<U32")

        for bead in self._ordered_beads:
            self._atom_idnames[bead._all_atom_idcs] = bead._all_atom_idnames
            self._atom_residcs[bead._all_atom_idcs] = bead._all_atom_residcs
            self._atom_resnums[bead._all_atom_idcs] = bead._all_atom_resnums
            self._atom_chainidcs[bead._all_atom_idcs] = bead._all_atom_chainidcs
        
        for atom_idname in self._atom_idnames:
            atom_resname, atom_name = atom_idname.split(DataDict.STR_SEPARATOR)
            atom_resnames.append(atom_resname)
            atom_names.append(atom_name)
        
        self._atom_resnames = np.array(atom_resnames)
        self._atom_names = np.array(atom_names)
        
        self._bb_atom_idcs = np.array([an in ['CA', 'N', 'C', 'O'] for an in self._atom_names])
        self._ca_atom_idcs = np.array([an in ['CA'] for an in self._atom_names])
        self._ca_bead_idcs = np.array([bn in ['BB'] for bn in self._bead_names])
        
        self.compute_bead2atom_idcs_and_weights()
        self.compute_extra_map_impl()
        
        # Read trajectory and map atom coords to bead coords
        atom_positions = []
        bb_atom_positions = []
        ca_atom_positions = []
        bead_positions = []
        ca_bead_positions = []
        cell_sizes = []

        all_atom_idcs = []
        build_all_atom_idcs = True

        self.initialize_extra_pos_impl()

        try:
            traj = self.u.trajectory if self.trajslice is None else self.u.trajectory[self.trajslice]
            for ts in traj:

                pos = np.empty((self._n_atoms, 3), dtype=float)
                pos[...] = np.nan
                for bead in self._ordered_beads:
                    if build_all_atom_idcs:
                        all_atom_idcs.append(bead._all_atom_idcs)
                    pos[bead._all_atom_idcs] = bead.all_atom_positions
                atom_positions.append(pos)

                if ts.dimensions is not None:
                    cell_sizes.append(Cell.fromcellpar(ts.dimensions)[:])
                
                if len(self._bb_atom_idcs) > 0:
                    bb_atom_positions.append(pos[self._bb_atom_idcs])
                if len(self._ca_atom_idcs) > 0:
                    ca_atom_positions.append(pos[self._ca_atom_idcs])
                
                not_nan_pos = np.nan_to_num(pos)
                bead_pos = np.sum(not_nan_pos[self._bead2atom_pos_idcs] * self._bead2atom_pos_weights[..., None], axis=1)
                bead_positions.append(bead_pos)
                if len(self._ca_bead_idcs) > 0:
                    ca_bead_positions.append(bead_pos[self._ca_bead_idcs])
                self.update_extra_pos_impl(pos)

                build_all_atom_idcs = False
        
        except Exception as e:
            print(f"Error rading trajectory: {e}. Skipping missing trajectory frames.")
        self._atom_positions =  np.stack(atom_positions, axis=0)
        try:
            self._bb_atom_positions =  np.stack(bb_atom_positions, axis=0)
            self._ca_atom_positions = np.stack(ca_atom_positions, axis=0)
            self._ca_bead_positions = np.stack(ca_bead_positions, axis=0)
        except:
            pass
        self._bead_positions =  np.stack(bead_positions, axis=0)
        self._cell = np.stack(cell_sizes, axis=0) if len(cell_sizes) > 0 else np.zeros((3, 3), dtype=np.float32)
        self.all_atom_idcs = np.concatenate(all_atom_idcs)

        self.store_extra_pos_impl()
        self.compute_invariants()
        self.compute_dihedral_idcs()

    def compute_extra_map_impl(self):
        pass

    def initialize_extra_pos_impl(self):
        pass

    def update_extra_pos_impl(self, pos):
        pass

    def store_extra_pos_impl(self):
        pass
    
    def _get_incomplete_bead_from_atom_idname(self, atom_idname: str) -> List[Bead]:
        bead_idnames = np.unique(self._atom2bead[atom_idname])
        beads = []
        for bead_idname in bead_idnames:
            found = False
            for bead in self._incomplete_beads:
                if bead.idname.__eq__(bead_idname) and bead.is_missing_atom(atom_idname):
                    beads.append(bead)
                    found = True
                    break
            if not found:
                beads.append(self._create_bead(bead_idname))
        return beads

    def _update_bead(
        self,
        bead: Bead,
        atom_idname: str,
        atom: Optional[Atom]=None,
        atom_index: Optional[int]=None,
    ):
        bmas = self._bead_mapping_settings.get(bead.idname).get_bmas_by_atom_idname(atom_idname)
        if atom_index is not None:
            self._n_atoms -= 1
        return bead.update(atom_idname, bmas, atom=atom, atom_index=atom_index)
    
    def _create_bead(self, bead_idname: str):
        bms = self._bead_mapping_settings.get(bead_idname)
        bead = HierarchicalBead(
            bms=bms,
            id=self._n_beads,
            idname=bead_idname,
            type=self._bead_types[bead_idname],
            atoms_offset=self._n_atoms,
            bead2atoms=copy.deepcopy(self._bead2atom[bead_idname]),
            weigth_based_on=self._weigth_based_on,
        )
        self._incomplete_beads.append(bead)
        self._ordered_beads.append(bead)
        self._n_beads += 1
        self._n_atoms += bead.n_all_atoms
        return bead
    
    def _complete_bead(self, bead: Bead):
        bead.complete()
        self._check_bead_completeness(bead)
    
    def _check_bead_completeness(self, bead: Bead):
        ''' Keep track of complete and incomplete beads, retaining the correct ordering.
        If a bead is complete, it is removed from the incomplete list and added to the complete list.
        '''
        if bead.is_complete and not (bead in self._complete_beads):
            self._complete_beads.append(bead)
            # Comment for martini3_like ###########
            # if self._keep_hydrogens:
            #     if bead in self._incomplete_beads:
            #         self._incomplete_beads.remove(bead)
            #######################################
            if bead in self._incomplete_beads:
                self._incomplete_beads.remove(bead)
            return True
        return False
    
    def compute_bead2atom_idcs_and_weights(self):
        ### Initialize instance mapping ###
        self._bead2atom_reconstructed_idcs = -np.ones((self.num_beads, self.bead_reconstructed_size), dtype=int)
        self._bead2atom_pos_idcs = -np.ones((self.num_beads, self.bead_all_size), dtype=int)
        self._bead2atom_pos_weights = np.zeros((self.num_beads, self.bead_all_size), dtype=float)
        self._bead2atom_reconstructed_weights = np.zeros((self.num_beads, self.bead_reconstructed_size), dtype=float)

        for i, bead in enumerate(self._ordered_beads):
            ### Build instance bead2atom_idcs and weights ###
            self._bead2atom_pos_idcs[i, :bead.n_all_atoms] = bead._all_atom_idcs
            self._bead2atom_pos_weights[i, :bead.n_all_atoms] = bead.all_atom_weights / bead.all_atom_weights.sum()
            try:
                self._bead2atom_reconstructed_idcs[i, :bead.n_reconstructed_atoms] = bead._reconstructed_atom_idcs
                self._bead2atom_reconstructed_weights[i, :bead.n_reconstructed_atoms] = bead._reconstructed_atom_weights / bead._reconstructed_atom_weights.sum()
            except:
                self._bead2atom_reconstructed_idcs[i, :len(bead._reconstructed_atom_idcs)] = bead._reconstructed_atom_idcs
                self._bead2atom_reconstructed_weights[i, :len(bead._reconstructed_atom_weights)] = bead._reconstructed_atom_weights / bead._reconstructed_atom_weights.sum()
    
    def compute_dihedral_idcs(self):
        phi_dihedral_idcs = []
        psi_dihedral_idcs = []
        omega_dihedral_idcs = []

        phi_dict = {}
        psi_dict = {}
        omega_dict = {}
        no_cappings_filter = [x not in ['ACE', 'NME'] for x in self._atom_resnames]
        unique_residcs = np.unique(self._atom_residcs[no_cappings_filter])
        for prev_resid, curr_resid in zip(unique_residcs[:-1], unique_residcs[1:]):
            if curr_resid != prev_resid + 1:
                continue
            phi = Phi(prev_resid)
            psi = Psi(curr_resid)
            omega = Omega(curr_resid)
            phi_dict[prev_resid] = phi
            psi_dict[curr_resid] = psi
            omega_dict[curr_resid] = omega

        for atom_index, (atom_name, resindex) in enumerate(zip(self._atom_names, self._atom_residcs)):
            if atom_name not in ['C', 'N', 'CA', 'O']:
                continue
            for dih_dict, idcs_list in zip([phi_dict, psi_dict, omega_dict], [phi_dihedral_idcs, psi_dihedral_idcs, omega_dihedral_idcs]):
                if resindex in dih_dict:
                    for offset in [0, 1]:
                        dih = dih_dict.get(resindex - offset)
                        if dih is None:
                            continue
                        dih(atom_name, resindex, atom_index)
                        if dih.is_completed():
                            idcs = dih.get_idcs()
                            if idcs is not None:
                                idcs_list.append(idcs)
                            dih.lock()

        self._phi_dihedral_idcs = np.array(phi_dihedral_idcs, dtype=int)
        self._psi_dihedral_idcs = np.array(psi_dihedral_idcs, dtype=int)
        self._omega_dihedral_idcs = np.array(omega_dihedral_idcs, dtype=int)
        self._phi_dict = phi_dict
        self._psi_dict = psi_dict
        self._omega_dict = omega_dict

        # Compute phi & psi dihedral values

        if self._atom_positions is not None and len(self._phi_dihedral_idcs) > 0:
            pos = torch.from_numpy(self._atom_positions)
            phi_val = get_dihedrals(pos=pos, dihedral_idcs=self._phi_dihedral_idcs).numpy()
            psi_val = get_dihedrals(pos=pos, dihedral_idcs=self._psi_dihedral_idcs).numpy()

            # if you want to keep only the ca beads
            # -> phipsi_val = np.zeros((len(phi_val), self._ca_bead_idcs.sum(), 2), dtype=float)
            phipsi_val = np.zeros((len(phi_val), len(self._ca_bead_idcs), 2), dtype=float)
            fltr_residcs = []
            dih_dict_iter = iter(zip(self._phi_dict.items(), self._psi_dict.items()))
            (phi_prev_resid, phi), (psi_curr_resid, psi) = next(dih_dict_iter)
            last_curr_resid = phi_prev_resid
            for global_resid, local_resid in enumerate(np.unique(self._atom_residcs)):
                if local_resid == last_curr_resid + 1:
                    assert psi_curr_resid == phi_prev_resid + 1
                    assert phi.is_locked() == psi.is_locked()
                    if phi.is_locked():
                        fltr_residcs.append(global_resid)
                    try:
                        (phi_prev_resid, phi), (psi_curr_resid, psi) = next(dih_dict_iter)
                    except StopIteration:
                        break
                    last_curr_resid = phi_prev_resid
            fltr_residcs = np.array(fltr_residcs)

            # if you want to keep only the ca beads
            # -> updated_ca_bead_fltr = self._ca_bead_fltr
            updated_ca_bead_fltr = np.nonzero(self._ca_bead_idcs)[0][fltr_residcs]
            phipsi_val[:, updated_ca_bead_fltr, 0] = phi_val
            phipsi_val[:, updated_ca_bead_fltr, 1] = psi_val

            self._bb_phi_psi_values = phipsi_val
    
    def compute_invariants(self):
        bond_min_length, bond_max_length = 1., 2. # Angstrom
        atoms_to_reconstruct_idcs = np.unique(self._bead2atom_reconstructed_idcs)

        # Bonds
        x = self._atom_positions[0]
        y = x - x[:, None]
        y = np.linalg.norm(y, axis=-1)
        z = (y > bond_min_length) * (y < bond_max_length)
        z[np.tril_indices(len(z), k=-1)] = False
        self._bond_idcs = np.stack(np.nonzero(z)).T
        self._bond_idcs = self._bond_idcs[np.all(np.isin(self._bond_idcs, atoms_to_reconstruct_idcs), axis=1)]
        
        # Angles
        df1 = pd.DataFrame(self._bond_idcs, columns=['a1', 'a2'])
        df2 = pd.DataFrame(self._bond_idcs, columns=['a2', 'a3'])
        df3 = df1.merge(df2, how='outer')
        df3 = df3.dropna().astype(int)
        self._angle_idcs = df3.values
        self._angle_idcs = self._angle_idcs[np.all(np.isin(self._angle_idcs, atoms_to_reconstruct_idcs), axis=1)]