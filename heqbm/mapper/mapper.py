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

from pathlib import Path
from typing import Dict, List, Optional
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
    _valid_configurations_list: dict[str, list] = {}
    _valid_configurations: dict[str, np.ndarray] = {}

    _atom2bead: dict[str, str] = {}
    _bead2atom: dict[str, List[str]] = {}
    _bead_types: dict[str, int] = {}
    _bead_mapping_settings: dict[str, BeadMappingSettings] = {}
    _bead_cm: dict[str, str] = {}

    _incomplete_beads: List[HierarchicalBead] = []
    _complete_beads: List[HierarchicalBead] = []
    _ordered_beads: List[HierarchicalBead] = []
    _max_bead_atoms: int = 0
    _max_bead_cm_atoms: int = 0

    _bead2atom_idcs_instance: np.ndarray = None
    _bead2atom_cm_idcs_instance: np.ndarray = None
    _weights_instance: np.ndarray = None

    _atom_positions: np.ndarray = None
    _bb_atom_positions: np.ndarray = None
    _bb_atom_idcs: np.ndarray = None
    _ca_atom_positions: np.ndarray = None
    _ca_atom_idcs: np.ndarray = None
    _bead_positions: np.ndarray = None
    _ca_bead_positions: np.ndarray = None
    _ca_bead_idcs: np.ndarray = None
    _ca_next_directions: np.ndarray = None

    _atom_resnames: np.ndarray = None
    _atom_names: np.ndarray = None
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
    def bead_size(self):
        return self._max_bead_atoms
    
    @property
    def bead_cm_size(self):
        return self._max_bead_cm_atoms
    
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
    def bead2atom_idcs_instance(self):
        if self._bead2atom_idcs_instance is None:
            return None
        return np.ma.masked_array(self._bead2atom_idcs_instance, mask=~self.bead2atom_idcs_mask_instance)
    
    @property
    def bead2atom_idcs_mask_instance(self):
        if self._bead2atom_idcs_instance is None:
            return None
        return self._bead2atom_idcs_instance != -1
    
    @property
    def mapping_idcs_instance(self):
        if self._bead2atom_idcs_instance is None:
            return None
        return [list(row[row != -1]) for row in self._bead2atom_idcs_instance]
    
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
        if self._bead_names is None:
            return None
        return len(self._bead_names)
    
    @property
    def bead_types(self):
        if self._bead_names is None:
            return None
        return np.array([self._bead_types[name] for name in self._bead_names])
    
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
            DataDict.BEAD_NAMES:      self._bead_names,
            DataDict.BEAD_TYPES:      self.bead_types,
            DataDict.BEAD_RESIDCS:    self._bead_residcs,
            DataDict.BEAD_RESNUMBERS: self._bead_resnums,
            DataDict.BEAD_CHAINIDCS:  self._bead_chainidcs,

            DataDict.BOND_IDCS:  self._bond_idcs,
            DataDict.ANGLE_IDCS: self._angle_idcs,
            DataDict.CELL:       self._cell,
            DataDict.PBC:        self._pbc,

            DataDict.MAPPING_IDCS:        self.mapping_idcs_instance,
            DataDict.BEAD2ATOM_IDCS:      self.bead2atom_idcs_instance,
            DataDict.BEAD2ATOM_IDCS_MASK: self.bead2atom_idcs_mask_instance,

            DataDict.BB_ATOM_POSITION: self._bb_atom_positions,
            DataDict.BB_PHIPSI: self._bb_phi_psi_values,
            DataDict.CA_ATOM_POSITION: self._ca_atom_positions,
            DataDict.CA_ATOM_IDCS: self._ca_atom_idcs,
            DataDict.CA_NEXT_DIRECTION: self._ca_next_directions,
            DataDict.CA_BEAD_POSITION: self._ca_bead_positions,
            DataDict.CA_BEAD_IDCS: self._ca_bead_idcs,
            
            DataDict.PHI_DIH_IDCS: self._phi_dihedral_idcs,
            DataDict.PSI_DIH_IDCS: self._psi_dihedral_idcs,
            DataDict.OMEGA_DIH_IDCS: self._omega_dihedral_idcs,
        }.items() if v is not None}

    def __init__(self, config: Dict) -> None:
        mapping_folder = config.get("mapping_folder", None)
        if mapping_folder is None:
            raise Exception(
                """
                You must provide the mapping folder.
                Add 'mapping_folder: name-of-mapping-folder' in the config file.
                mappings are specified in a mapping folder inside 'heqbm/data/'
                """
            )
        self._root = os.path.join(os.path.dirname(__file__), '..', 'data', mapping_folder)
        self._keep_hydrogens = config.get("keep_hydrogens", False)
        self._weigth_based_on = config.get("weigth_based_on", "mass")
        self._max_bead_atoms = config.get("max_bead_atoms", self._max_bead_atoms)

        # Iterate configuration files and load all mappings
        self._clear_mappings()
        self._load_mappings()

    def _clear_mappings(self):
        self._valid_configurations_list: dict[str, list] = {}
        self._valid_configurations: dict[str, np.ndarray] = {}

        self._atom2bead.clear()
        self._bead2atom.clear()
        self._bead_types.clear()

        self._incomplete_beads.clear()
        self._complete_beads.clear()
        self._ordered_beads.clear()

        self._bead2atom_idcs_instance: np.ndarray = None
        self._bead2atom_cm_idcs_instance: np.ndarray = None
        self._weights_instance: np.ndarray = None

        self._atom_positions: np.ndarray = None
        self._bb_atom_positions: np.ndarray = None
        self._bb_atom_idcs: np.ndarray = None
        self._ca_atom_positions: np.ndarray = None
        self._bead_positions: np.ndarray = None
        self._ca_bead_positions: np.ndarray = None
        self._ca_bead_idcs: np.ndarray = None
        self._ca_next_directions: np.ndarray = None

        self._atom_resnames: np.ndarray = None
        self._atom_names: np.ndarray = None
        self._bead_names: np.ndarray = None
        self._atom_forces: np.ndarray = None
        self._atom_chainidcs: np.ndarray = None
        self._bead_chainidcs: np.ndarray = None
        self._cell: np.ndarray = None
        self._pbc: np.ndarray = np.array([True, True, True])

        self._bond_idcs: np.ndarray = None
        self._angle_idcs: np.ndarray = None

        self._clear_extra_mappings()
    
    def _clear_extra_mappings(self):
        pass

    def _load_mappings(self):
        bead_types_filename = os.path.join(self._root, self.bead_types_filename)
        if os.path.isfile(bead_types_filename):
            bead_types_conf: dict = yaml.safe_load(Path(bead_types_filename).read_text())
        else:
            bead_types_conf: dict = {}
        for filename in os.listdir(self._root):
            if filename == self.bead_types_filename:
                continue
            conf: dict = yaml.safe_load(Path(os.path.join(self._root, filename)).read_text())
            mol = conf.get("molecule")
            valid_confs_list = self._valid_configurations_list.get(mol, [])
            valid_confs_list.append(0 if len(valid_confs_list) == 0 else valid_confs_list[-1] + 1)
            self._valid_configurations_list[mol] = valid_confs_list
            _conf_bead2atom = {}
            self._initialize_conf_extra_mappings()

            for atom, bead_settings_str in conf.get("atoms").items():
                if not self._keep_hydrogens and atom.startswith('H'):
                    continue
                all_bead_settings = bead_settings_str.split()
                bead_names = all_bead_settings[0].split(',')

                for i, bn in enumerate(bead_names):
                    bead_name = "_".join([mol, bn])
                    atom_idname = "_".join([mol, atom])
                    bead_settings = [x.split(',')[i] for x in all_bead_settings[1:]]
                    
                    atom2bead_list = self._atom2bead.get(atom_idname, [])
                    atom2bead_list.append(bead_name)
                    self._atom2bead[atom_idname] = atom2bead_list
                    bms = self._bead_mapping_settings.get(bead_name, BeadMappingSettings(bead_name))
                    bmas = bms.get_bmas_by_atom_idname(atom_idname)
                    if bmas is None:
                        bmas = BeadMappingAtomSettings(bead_settings, bead_name, atom_idname, num_shared_beads=len(bead_names))
                        bms.add_atom_settings(bmas)
                        self._bead_mapping_settings[bead_name] = bms
                    
                    bead2atom: List[str] = _conf_bead2atom.get(bead_name, [])
                    if len(bead2atom) == 0 and bead_name not in self._bead_types:
                        bead_type = bead_types_conf.get(bead_name, max(bead_types_conf.values(), default=-1)+1)
                        bead_types_conf[bead_name] = bead_type
                        self._bead_types[bead_name] = bead_type
                    assert atom_idname not in bead2atom, f"{atom_idname} is already present in bead {bead_name}. Duplicate mapping"
                    bead2atom.append(atom_idname)
                    _conf_bead2atom[bead_name] = bead2atom
            
            for bead_name, bms in self._bead_mapping_settings.items():
                bms.update_relative_weights()
                self._max_bead_atoms = max(self._max_bead_atoms, bms.bead_atoms)
                self._max_bead_cm_atoms = max(self._max_bead_cm_atoms, bms.bead_cm_atoms)
                
            for bead_name, bead2atom in _conf_bead2atom.items():
                _bead2atom = self._bead2atom.get(bead_name, [])
                _bead2atom.append(bead2atom)
                self._bead2atom[bead_name] = _bead2atom
            self._store_extra_mappings()
        
        with open(bead_types_filename, 'w') as outfile:
            yaml.dump(bead_types_conf, outfile, default_flow_style=False)
        
        for mol, valid_confs_list in self._valid_configurations_list.items():
            self._valid_configurations[mol] = np.array(valid_confs_list)
        
        for k, b2a in self._bead2atom.items():
            len_base = self.get_bead2atom_len(b2a[0])
            assert all([self.get_bead2atom_len(v) == len_base for v in b2a]), f"Configurations for bead type {k} have different number of atoms"

        ### Compute global mask and maximum bead size ###
        self._bead2atom_idcs_mask = np.zeros((self.n_bead_types, self._max_bead_atoms), dtype=bool)
                
        for bead_name, b2a in self._bead2atom.items():
            self._bead2atom_idcs_mask[self._bead_types[bead_name], :self.get_bead2atom_len(b2a[0])] = True
    
    def get_bead2atom_len(self, b2a: List[str]):
        return len([atom_name for atom_name in b2a if self._keep_hydrogens or get_type_from_name(atom_name) != 1])
    
    def _initialize_conf_extra_mappings(self):
        pass

    def _load_extra_mappings(self, bead_splits, atom_name, bead_name):
        return

    def _store_extra_mappings(self, bead_names):
        pass
    
    def map(self, conf, selection = 'protein', frame_limit: Optional[int] = None):
        # New mapping, clear last records
        self._incomplete_beads.clear()
        self._complete_beads.clear()
        self._ordered_beads.clear()

        self.u = mda.Universe(conf.get("structure_filename"), *conf.get("traj_filenames", []), **conf.get("extra_kwargs", {}))
        
        if conf.get("simulation_is_cg", True):
            return self.map_impl_cg(selection=selection, frame_limit=frame_limit)
        return self.map_impl(selection=selection, frame_limit=frame_limit)
    
    def map_impl_cg(self, selection, frame_limit=None):
        sel = self.u.select_atoms(selection)

        try:
            self._bead_chainidcs = sel.chainIDs
        except mda.exceptions.NoDataError:
            self._bead_chainidcs = np.array(['A'] * sel.n_atoms)
        
        self._bead_names = np.array([f"{resname}_{bead}" for resname, bead in zip(sel.resnames, sel.names)])
        self._ca_bead_idcs = np.array([bn.split('_')[-1] in ['BB'] for bn in self._bead_names])

        bead_positions = []
        ca_atom_positions = []
        ca_bead_positions = []
        ca_next_directions = []
        cell_sizes = []
        
        try:
            traj = self.u.trajectory if frame_limit is None else self.u.trajectory[:frame_limit]
            for ts in traj:
                bead_pos = sel.positions
                if ts.dimensions is not None:
                    cell_sizes.append(Cell.fromcellpar(ts.dimensions)[:])
                bead_positions.append(bead_pos)
                ca_atom_positions.append(bead_pos[self._ca_bead_idcs])
                ca_bead_positions.append(bead_pos[self._ca_bead_idcs])

                ca_next_direction = np.zeros_like(bead_pos[self._ca_bead_idcs])
                ca_next_direction[:-1] = bead_pos[self._ca_bead_idcs][1:] - bead_pos[self._ca_bead_idcs][:-1]
                ca_next_directions.append(ca_next_direction)
                
        except ValueError as e:
            print(f"Error rading trajectory: {e}. Skipping missing trajectory frames.")
        self._bead_positions =  np.stack(bead_positions, axis=0)
        self._ca_atom_positions = np.stack(ca_atom_positions, axis=0)
        self._ca_bead_positions = np.stack(ca_bead_positions, axis=0)
        self._ca_next_directions = np.stack(ca_next_directions, axis=0)
        self._cell = np.stack(cell_sizes, axis=0) if len(cell_sizes) > 0 else np.zeros((3, 3), dtype=np.float32)

        mapping_n = 0

        atom_resnames = []
        atom_names = []
        atom_residcs = []
        atom_resnums = []
        bead_residcs = []
        bead_resnums = []
        atom_chainidcs = []
        bead2atom_idcs_instance = -np.ones((len(self._bead_names), self._max_bead_atoms), dtype=int)

        self.initialize_extra_map_impl_cg()

        # chainid2residoffset is used to adjust resindex values  in pdbs with multiple chains, because the resindex are repeated across chains
        # unique_chain_idcs = np.insert(np.unique(self._bead_chainidcs), 0, '')

        # chainid2residoffset = {}
        # for prev_chainid, chainid in zip(unique_chain_idcs[:-1], unique_chain_idcs[1:]):
        #         chainid2residoffset[chainid] = chainid2residoffset.get(prev_chainid, 0) + \
        #                                        resnums[self._bead_chainidcs == prev_chainid].max(initial=0) + \
        #                                        -resnums[self._bead_chainidcs == chainid].min() + \
        #                                        1

        atom_index_offset = 0
        _id = 0
        atom_idnames_missing_multiplicity = {}
        atom_idnames2index = {}
        for h, (bead_name, sel_bead) in enumerate(zip(self._bead_names, sel.atoms)):
            bead_atom_idnames = np.array([
                atom_idname for atom_idname in self._bead2atom[bead_name][mapping_n]
                if (self._bead_mapping_settings.get(bead_name).get_bmas_by_atom_idname(atom_idname)._has_to_be_reconstructed)
            ])
            new_bead_atom_idnames = np.array(
                [atom_idname for atom_idname in bead_atom_idnames if atom_idnames_missing_multiplicity.get(atom_idname, 0) == 0]
            )

            for atom_idname in new_bead_atom_idnames:
                atom_resname, atom_name = atom_idname.split(DataDict.STR_SEPARATOR)
                atom_resnames.append(atom_resname)
                atom_names.append(atom_name)

            bead = self._create_bead(bead_name)
            _idcs = []
            for atom_idname in bead_atom_idnames:
                increase_id = False
                if atom_idname in new_bead_atom_idnames:
                    _idcs.append(_id + atom_index_offset)
                    increase_id = True
                else:
                    _idcs.append(atom_idnames2index[atom_idname])
                _ = self._update_bead(bead, atom_idname, None, _idcs[-1])
                self._check_bead_completeness(bead)
                
                if atom_idnames_missing_multiplicity.get(atom_idname, 0) > 0:
                    atom_idnames_missing_multiplicity[atom_idname] -= 1
                else:
                    atom_idnames_missing_multiplicity[atom_idname] = len(np.unique(self._atom2bead[atom_idname])) - 1
                    if atom_idnames_missing_multiplicity[atom_idname] > 0:
                        atom_idnames2index[atom_idname] = _id
                
                self.update_extra_map_impl_cg(bead_atom_idnames, bead_name, mapping_n, atom_index_offset)
                if increase_id:
                    _id += 1

            bead2atom_idcs_instance[h, :len(bead_atom_idnames)] = np.array(_idcs)
            # atom_residcs.extend([sel_bead.resindex + chainid2residoffset.get(self._bead_chainidcs[h], 0)] * len(new_bead_atom_idnames))
            # bead_residcs.append(sel_bead.resindex + chainid2residoffset.get(self._bead_chainidcs[h], 0))
            atom_residcs.extend([sel_bead.resindex] * len(new_bead_atom_idnames))
            atom_resnums.extend([sel_bead.resnum] * len(new_bead_atom_idnames))
            bead_residcs.append(sel_bead.resindex)
            bead_resnums.append(sel_bead.resnum)
            atom_chainidcs.extend([self._bead_chainidcs[h]] * len(new_bead_atom_idnames))

        self._atom_resnames = np.array(atom_resnames)
        self._atom_names = np.array(atom_names)
        self._atom_residcs = np.array(atom_residcs)
        self._atom_residcs -= self._atom_residcs.min()
        self._atom_resnums = np.array(atom_resnums)
        self._bead_residcs = np.array(bead_residcs)
        self._bead_resnums = np.array(bead_resnums)
        self._bead_residcs -= self._bead_residcs.min()
        self._atom_chainidcs = np.array(atom_chainidcs)
        self._bead2atom_idcs_instance = np.array(bead2atom_idcs_instance)
        self._bb_atom_idcs = np.array([an in ['CA', 'N', 'C'] for an in self._atom_names])
        self._ca_atom_idcs = np.array([an in ['CA'] for an in self._atom_names])
        self._ca_bead_idcs = np.array([bn.split('_')[-1] in ['BB'] for bn in self._bead_names])

        batch, _, xyz = self._bead_positions.shape
        self._atom_positions =  np.zeros((batch, self._bead2atom_idcs_instance.max() + 1, xyz), dtype=self._bead_positions.dtype)

        self.store_extra_map_impl_cg()
        
        self.compute_extra_map_impl()

        # self.compute_dihedral_idcs()
            
    def map_impl(self, selection, frame_limit=None):
        if not self._keep_hydrogens:
            selection = selection + ' and not (element H)'
        try:
            sel = self.u.select_atoms(selection)
        except AttributeError:
            selection = selection.replace('element', 'type')
            sel = self.u.select_atoms(selection)
        try:
            temp_atom_chainidcs = sel.chainIDs
        except mda.exceptions.NoDataError:
            temp_atom_chainidcs = np.array(['A'] * sel.n_atoms)

        # Iterate elements in the system
        atom_resnames = []
        atom_names = []
        bead_names = []
        excluded_atoms = []
        atom_residcs = []
        atom_resnums = []
        bead_residcs = []
        bead_resnums = []
        bead_chainidcs = []

        self.initialize_extra_map_impl()

        _id = 0
        # chainid2residoffset = {}
        # try:
            # unique_chain_idcs = np.insert(np.unique(temp_atom_chainidcs), 0, '')
            # # chainid2residoffset is used to adjust resindex values in pdbs with multiple chains, because the resindex are repeated across chains
            # for prev_chainid, chainid in zip(unique_chain_idcs[:-1], unique_chain_idcs[1:]):
            #     chainid2residoffset[chainid] = chainid2residoffset.get(prev_chainid, 0) + \
            #                                    resnums[temp_atom_chainidcs == prev_chainid].max(initial=0) + \
            #                                    -resnums[temp_atom_chainidcs == chainid].min() + \
            #                                    1
        # except mda.exceptions.NoDataError:
        #     pass

        for h, atom in enumerate(sel.atoms):
            try:
                # Read atom properties
                atom_idname = DataDict.STR_SEPARATOR.join([atom.resname, re.sub(r'^(\d+)\s*(.+)', r'\2\1', atom.name)])
                try:
                    beads = self._get_incomplete_bead_from_atom_idname(atom_idname)
                except:
                    atom_idname = atom_idname.replace("T1", "")
                    beads = self._get_incomplete_bead_from_atom_idname(atom_idname)

                for bead in beads:
                    if bead.is_newly_created:
                        bead_names.append(bead.name)
                        # bead_residcs.append(atom.resindex + chainid2residoffset.get(temp_atom_chainidcs[h], 0))
                        bead_residcs.append(atom.resindex)
                        bead_resnums.append(atom.resnum)
                        bead_chainidcs.append(temp_atom_chainidcs[h])
                    invalid_confs = self._update_bead(bead, atom_idname, atom, _id)
                        
                    self._check_bead_completeness(bead)
                    for ic in invalid_confs:
                        valid_confs = self._valid_configurations[atom.resname]
                        if ic in valid_confs:
                            valid_confs = np.delete(valid_confs, np.nonzero(valid_confs == ic)[0])
                        self._valid_configurations[atom.resname] = valid_confs
                    self.update_extra_map_impl(atom_idname, bead, _id)
                _id += 1

                atom_resnames.append(atom.resname)
                atom_names.append(atom.name)
                # atom_residcs.append(atom.resindex + chainid2residoffset.get(temp_atom_chainidcs[h], 0))
                atom_residcs.append(atom.resindex)
                atom_resnums.append(atom.resnum)
                
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
        # Complete all beads. Missing atoms will be ignored
        for bead in self._incomplete_beads:
            self._complete_bead(bead)
            self._check_bead_completeness(bead)
        # Uncomment for martini3_like ###########
        if not self._keep_hydrogens:
            for bead in self._complete_beads:
                self._incomplete_beads.remove(bead)
        #########################################
        self._atom_resnames = np.array(atom_resnames)
        self._atom_names = np.array(atom_names)
        self._bead_names = np.array(bead_names)
        self._atom_residcs = np.array(atom_residcs)
        self._atom_residcs -= self._atom_residcs.min()
        self._atom_resnums = np.array(atom_resnums)
        self._bead_residcs = np.array(bead_residcs)
        self._bead_residcs -= self._bead_residcs.min()
        self._bead_resnums = np.array(bead_resnums)
        self._bead_chainidcs = np.array(bead_chainidcs)
        self._bb_atom_idcs = np.array([an.split('_')[-1] in ['CA', 'N', 'C'] for an in self._atom_names])
        self._ca_atom_idcs = np.array([an.split('_')[-1] in ['CA'] for an in self._atom_names])
        self._ca_bead_idcs = np.array([bn.split('_')[-1] in ['BB'] for bn in self._bead_names])

        self.store_extra_map_impl()

        sel = self.u.select_atoms(selection)
        try:
            self._atom_chainidcs = sel.chainIDs
        except mda.exceptions.NoDataError:
            self._atom_chainidcs = np.array(['A'] * sel.n_atoms)
        
        # Extract the indices of the atoms in the trajectory file for each bead
        self.compute_bead2atom_idcs_and_weights()

        self.compute_extra_map_impl()

        self.compute_invariants(selection=sel)
        
        # Read trajectory and map atom coords to bead coords
        atom_positions = []
        bb_atom_positions = []
        ca_atom_positions = []
        bead_positions = []
        ca_next_directions = []
        ca_bead_positions = []
        atom_forces = []
        cell_sizes = []

        self.initialize_extra_pos_impl()

        read_forces = True
        try:
            traj = self.u.trajectory if frame_limit is None else self.u.trajectory[:frame_limit]
            for ts in traj:
                pos = sel.positions
                if ts.dimensions is not None:
                    cell_sizes.append(Cell.fromcellpar(ts.dimensions)[:])
                
                atom_positions.append(pos)
                if len(self._bb_atom_idcs) > 0:
                    bb_atom_positions.append(pos[self._bb_atom_idcs])
                if len(self._ca_atom_idcs) > 0:
                    ca_atom_positions.append(pos[self._ca_atom_idcs])
                weighted_pos = pos[self._bead2atom_cm_idcs_instance] * self._weights_instance[..., None]
                bead_pos = np.sum(weighted_pos, axis=1)
                bead_positions.append(bead_pos)
                ca_next_direction = np.zeros_like(bead_pos)
                if len(self._ca_bead_idcs) > 0:
                    ca_bead_positions.append(bead_pos[self._ca_bead_idcs])
                    ca_next_direction[np.nonzero(self._ca_bead_idcs)[0][:-1]] = bead_pos[self._ca_bead_idcs][1:] - bead_pos[self._ca_bead_idcs][:-1]
                    
                ca_next_directions.append(ca_next_direction)
                self.update_extra_pos_impl(pos, bead_pos)
                if read_forces:
                    try:
                        forces = sel.forces
                        atom_forces.append(forces)
                    except:
                        print("Missing information on Forces")
                        read_forces = False
        except Exception as e:
            print(f"Error rading trajectory: {e}. Skipping missing trajectory frames.")
        self._atom_positions =  np.stack(atom_positions, axis=0)
        try:
            self._bb_atom_positions =  np.stack(bb_atom_positions, axis=0)
            self._ca_atom_positions = np.stack(ca_atom_positions, axis=0)
            self._ca_bead_positions = np.stack(ca_bead_positions, axis=0)
        except:
            pass
        self._ca_next_directions = np.stack(ca_next_directions, axis=0)
        self._bead_positions =  np.stack(bead_positions, axis=0)
        self._atom_forces = np.stack(atom_forces, axis=0) if len(atom_forces) > 0 else None
        self._cell = np.stack(cell_sizes, axis=0) if len(cell_sizes) > 0 else np.zeros((3, 3), dtype=np.float32)

        self.store_extra_pos_impl()

        # Extract dihedrals from topology (if present) or compute those of backbone
        # self.compute_dihedral_idcs(selection=sel)
    
    def initialize_extra_map_impl(self):
        pass

    def update_extra_map_impl(self, atom_idname: str, bead: Bead, _id: int):
        pass

    def store_extra_map_impl(self):
        pass

    def compute_extra_map_impl(self):
        pass

    def initialize_extra_pos_impl(self):
        pass

    def update_extra_pos_impl(self, pos, bead_pos):
        pass

    def store_extra_pos_impl(self):
        pass

    def initialize_extra_map_impl_cg(self):
        pass

    def update_extra_map_impl_cg(self, bead_atom_names: np.ndarray, bead_name: str, mapping_n: int, atom_index_offset: int):
        pass

    def store_extra_map_impl_cg(self):
        pass
    
    def _get_incomplete_bead_from_atom_idname(self, atom_idname: str):
        bead_names = np.unique(np.array(self._atom2bead[atom_idname]))
        beads = []
        for bead_name in bead_names:
            found = False
            for bead in self._incomplete_beads:
                if bead.name.__eq__(bead_name) and bead.is_missing_atom(atom_idname):
                    beads.append(bead)
                    found = True
                    break
            if not found:
                beads.append(self._create_bead(bead_name))
        return beads
    
    def _update_bead(self, bead: Bead, atom_idname: str, atom: Atom, _id: int):
        bmas = self._bead_mapping_settings.get(bead.name).get_bmas_by_atom_idname(atom_idname)
        return bead.update(atom_idname, atom, _id, bmas)
    
    def _create_bead(self, bead_name: str):
        bead = HierarchicalBead(
            name=bead_name,
            type=self._bead_types[bead_name],
            atoms=copy.deepcopy(self._bead2atom[bead_name]),
            weigth_based_on=self._weigth_based_on,
            keep_hydrogens=self._keep_hydrogens,
        )
        self._incomplete_beads.append(bead)
        self._ordered_beads.append(bead)
        return bead
    
    def _complete_bead(self, bead: Bead):
        bead.complete()
        self._check_bead_completeness(bead)
    
    def _check_bead_completeness(self, bead: Bead):
        if bead.is_complete and not (bead in self._complete_beads):
            self._complete_beads.append(bead)
            # Comment for martini3_like ###########
            if self._keep_hydrogens:
                if bead in self._incomplete_beads:
                    self._incomplete_beads.remove(bead)
            #######################################
            return True
        return False
    
    def compute_bead2atom_idcs_and_weights(self):
        ### Initialize instance mapping ###
        self._bead2atom_idcs_instance = -np.ones((self.num_beads, self.bead_size), dtype=int)
        self._bead2atom_cm_idcs_instance = -np.ones((self.num_beads, self.bead_cm_size), dtype=int)
        self._weights_instance = np.zeros((self.num_beads, self.bead_cm_size), dtype=float)

        for i, bead in enumerate(self._ordered_beads):
            ### Build instance bead2atom_idcs and weights ###
            self._bead2atom_idcs_instance[i, :bead.n_atoms] = np.array(bead._atom_idcs)
            self._bead2atom_cm_idcs_instance[i, :bead.n_cm_atoms] = np.array(bead._atom_cm_idcs)
            weights = np.array(bead._atom_weights)
            self._weights_instance[i, :bead.n_cm_atoms] = weights / weights.sum()
    
    def compute_dihedral_idcs(self, selection = None):
        # if selection is not None:
        #     try:
        #         self._dihedral_idcs = selection.intra_dihedrals.indices.astype(int)
        #         return
        #     except:
        #         pass
        phi_dihedral_idcs = []
        psi_dihedral_idcs = []
        omega_dihedral_idcs = []

        phi_dict = {}
        psi_dict = {}
        omega_dict = {}
        no_cappings_filter = [x.split('_')[0] not in ['ACE', 'NME'] for x in self._atom_names]
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

        for atom_index, (name, resindex) in enumerate(zip(self._atom_names, self._atom_residcs)):
            resname, atom_name = name.split('_')
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
    
    def compute_invariants(self, selection):
        bond_idcs_from_top, angle_idcs_from_top = True, True
        atoms_to_reconstruct_idcs = np.unique(self._bead2atom_idcs_instance)[1:]
        try:
            self._bond_idcs = selection.intra_bonds.indices
        except:
            try:
                selection.guess_bonds()
                self._bond_idcs = selection.intra_bonds.indices
            except:
                x = selection.positions
                y = x - x[:, None]
                y = np.linalg.norm(y, axis=-1)
                z = (y > 1.) * (y < 2.1)
                z[np.tril_indices(len(z), k=-1)] = False
                self._bond_idcs = np.stack(np.nonzero(z)).T
                bond_idcs_from_top = False
        try:
            self._angle_idcs = selection.intra_angles.indices
        except:
            df1 = pd.DataFrame(self._bond_idcs, columns=['a1', 'a2'])
            df2 = pd.DataFrame(self._bond_idcs, columns=['a2', 'a3'])
            df3 = df1.merge(df2, how='outer')
            df3 = df3.dropna().astype(int)
            self._angle_idcs = df3.values
            angle_idcs_from_top = False
        if not self._keep_hydrogens:
            bond_atoms_are_valid = np.zeros_like(self._bond_idcs, dtype=bool)
            angle_atoms_are_valid = np.zeros_like(self._angle_idcs, dtype=bool)
            for _id, atom in enumerate(selection.atoms):
                # Adjust atom indices to account for the absence of hydrogens
                if bond_idcs_from_top:
                    bond_atoms_are_valid[self._bond_idcs == atom.index] = True
                    self._bond_idcs[self._bond_idcs == atom.index] = _id
                if angle_idcs_from_top:
                    angle_atoms_are_valid[self._angle_idcs == atom.index] = True
                    self._angle_idcs[self._angle_idcs == atom.index] = _id
            self._bond_idcs = self._bond_idcs[np.all(bond_atoms_are_valid, axis=1)]
            self._angle_idcs = self._angle_idcs[np.all(angle_atoms_are_valid, axis=1)]
        self._bond_idcs = self._bond_idcs[np.all(np.isin(self._bond_idcs, atoms_to_reconstruct_idcs), axis=1)]
        self._angle_idcs = self._angle_idcs[np.all(np.isin(self._angle_idcs, atoms_to_reconstruct_idcs), axis=1)]