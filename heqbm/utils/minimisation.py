import time
from typing import List
import numpy as np
from os.path import dirname
from openmm.app import *
from openmm import *
from openmm.unit import kelvin, nanometer, picoseconds


def minimise(
    pdb_in_filename: str,
    pdb_out_filename: str,
    tolerance: float = 100, # Value in Kj/(mol nm)
    restrain_atoms: List[str] = [],
):
        print('Reading pdb file...')
        pdb = PDBFile(pdb_in_filename)
        minimise_impl(
                pdb.topology,
                pdb.positions,
                pdb_out_filename,
                tolerance,
                restrain_atoms,
        )

def minimise_impl(
    topology,
    positions,
    pdb_out_filename: str,
    tolerance: float = 50, # Value in Kj/(mol nm)
    restrain_atoms: List[str] = [],
):
        modeller = Modeller(topology, positions)

        forcefield = ForceField('amber14-all.xml')
        system = forcefield.createSystem(
                modeller.topology,
                nonbondedMethod=NoCutoff,
                nonbondedCutoff=1*nanometer,
                constraints=HBonds
        )
        system.setDefaultPeriodicBoxVectors(Vec3(100., 0., 0.), Vec3(0., 100., 0.), Vec3(0., 0., 100.))

        # --- RESTRAINTS --- #

        restraint = CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2')
        system.addForce(restraint)
        restraint.addGlobalParameter('k', 1e6*unit.kilojoules_per_mole/unit.nanometers**2)
        restraint.addPerParticleParameter('x0')
        restraint.addPerParticleParameter('y0')
        restraint.addPerParticleParameter('z0')

        for atom in modeller.topology.atoms():
                if atom.name in restrain_atoms:
                        restraint.addParticle(atom.index, modeller.positions[atom.index])

        ######################

        integrator = LangevinMiddleIntegrator(300*kelvin, 1/picoseconds, 0.001*picoseconds)
        platform = Platform.getPlatformByName('CUDA')
        prop = dict(CudaPrecision='single')
        simulation = Simulation(modeller.topology, system, integrator, platform, prop)

        print('Running minimisation...')
        t = time.time()

        simulation.context.setPositions(modeller.positions)
        simulation.minimizeEnergy(tolerance=tolerance*unit.kilojoules_per_mole/unit.nanometer, maxIterations=10000)
        state = simulation.context.getState(getPositions=True, getEnergy=True, getForces=True)
        f = np.array([[a.x,a.y,a.z] for a in state.getForces()])
        if np.linalg.norm(f, axis=1).mean() > 10 * tolerance:
                print(f"Failed minimisation. Force: {np.linalg.norm(f, axis=1).mean()}.")
                print("Check box dimension, as it could be too small and cause energy minimisation to fail.")
                return
        
        print(f"Finished. Time: {time.time() - t}")
        print('Saving...')
        os.makedirs(dirname(pdb_out_filename), exist_ok=True)
        positions = simulation.context.getState(getPositions=True).getPositions()[:modeller.topology._numAtoms]
        PDBFile.writeFile(simulation.topology, positions, open(pdb_out_filename, 'w'), keepIds=True)
        print('Done')