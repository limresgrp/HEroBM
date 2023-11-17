import time
import numpy as np
from os.path import dirname
from openmm.app import *
from openmm import *
from openmm.unit import kelvin, nanometer, picoseconds


def minimise(
    pdb_in_filename: str,
    pdb_out_filename: str,
    tolerance: float, # Value in Kj/(mol nm)
):
        print('Reading pdb file...')
        pdb = PDBFile(pdb_in_filename)
        forcefield = ForceField('amber14-all.xml')
        modeller = Modeller(pdb.topology, pdb.positions)

        system = forcefield.createSystem(
                modeller.topology,
                nonbondedMethod=PME,
                nonbondedCutoff=1*nanometer,
                constraints=HBonds
        )
        # system.setDefaultPeriodicBoxVectors(Vec3(100., 0., 0.), Vec3(0., 100., 0.), Vec3(0., 0., 100.))

        # --- RESTRAINTS --- #

        # restraint = CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2')
        # restraint.addGlobalParameter('k', 1000.*unit.kilojoules_per_mole/unit.nanometers**2)
        # restraint.addPerParticleParameter('x0')
        # restraint.addPerParticleParameter('y0')
        # restraint.addPerParticleParameter('z0')

        # for atom in modeller.topology.atoms():
        #         if atom.name in RESTRAIN_ATOMS:
        #                 restraint.addParticle(atom.index, modeller.positions[atom.index])

        # system.addForce(restraint)

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
                print(f"Failed minimisation for {pdb_in_filename}. Force: {np.linalg.norm(f, axis=1).mean()}.")
                print("Check box dimension, as it could be too small and cause energy minimisation to fail.")
                return
        
        print(f"Finished. Time: {time.time() - t}")
        print('Saving...')
        os.makedirs(dirname(pdb_out_filename), exist_ok=True)
        positions = simulation.context.getState(getPositions=True).getPositions()
        PDBFile.writeFile(simulation.topology, positions, open(pdb_out_filename, 'w'))
        print('Done')