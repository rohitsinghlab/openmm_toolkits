"""
TODO:
1. while creating a system:
    - Ensure that there is a good sampling of alternate conformations
    - Ensure that the solvent ionization is neutralized---set `neutralize=True`
    - Set the `hydrogenMass=1.5amu` and `constraints=HBonds` to be able to set timesteps to 4 femtoseconds
    - compute the missing loops?
2. while minimizing energy and equilibration:
    - restrain the protein backbone & ligand, only let the water adjust first.
    - then restrain the protein backbone, allow the ligand to equilibrate
3. while doing simulation:
    - ensure that free energy stabilizes with time.
"""
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from openmmforcefields.generators import SystemGenerator
from openff.toolkit import Molecule
from openmmtools import alchemy
from openmmtools.states import ThermodynamicState, CompoundThermodynamicState
from pymbar import MBAR
from openmmtools import alchemy
from pdbfixer import PDBFixer
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm 
from openff.toolkit.utils.toolkits import ToolkitRegistry, AmberToolsToolkitWrapper

# Create a registry with only AmberTools
toolkit_registry = ToolkitRegistry([AmberToolsToolkitWrapper()])


import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def fix_pdb(protein_url, fixed_url):
    fixer = PDBFixer(protein_url)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    fixer.removeHeterogens(keepWater=False)
    with open(fixed_url, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    return

from openmmforcefields.generators import GAFFTemplateGenerator


def boundingbox(positions):
    xmin = positions[0][0]
    xmax = positions[0][0]
    ymin = positions[0][1]
    ymax = positions[0][1]
    zmin = positions[0][2]
    zmax = positions[0][2]
    for i in range(len(positions)):
        x = positions[i][0]
        y = positions[i][1]
        z = positions[i][2]
        # print('type of x ', type(x))
        # print('Site ', i, ' Coord ', positions[i])
        if(x > xmax):
            xmax = x
        if(x < xmin):
            xmin = x
        if(y > ymax):
            ymax = y
        if(y < ymin):
            ymin = y
        if(z > zmax):
            zmax = z
        if(z < zmin):
            zmin = z
    return [ (xmin*angstrom,xmax*angstrom), 
             (ymin*angstrom,ymax*angstrom), 
             (zmin*angstrom,zmax*angstrom)] 

def prepare_system(protein_pdb, ligand_sdfs, platform, save_loc = None):
    protein_pdb = PDBFile(protein_pdb)
    forcefield  = ForceField("amber14-all.xml", "amber14/tip3pfb.xml") 

    ligand_mols    = []
    n_ligand_atoms = 0
    lig_positions  = []
    modeller          = Modeller(protein_pdb.topology, protein_pdb.positions)
    ligand_pos_start  = len(protein_pdb.positions)
    for lsdf in ligand_sdfs:
        ligand_mol = Molecule.from_file(lsdf)
        n_ligand_atoms += len(ligand_mol.conformers[0])
        lig_position    = ligand_mol.conformers[0].to('angstrom').magnitude
        lig_positions  += [Vec3(lig_position[i][0], lig_position[i][1], lig_position[i][2])
                           for i in range(lig_position.shape[0])] # angstrom
        ligand_mols.append(ligand_mol)
        modeller.add(ligand_mol.to_topology().to_openmm(),
                     Quantity([Vec3(0.1 * x[0].magnitude, 
                                    0.1 * x[1].magnitude,
                                    0.1 * x[2].magnitude) for x in ligand_mol.conformers[0]], unit = nanometer))


    ligand_locs    = list(range(ligand_pos_start, ligand_pos_start + n_ligand_atoms))
    template_gen = GAFFTemplateGenerator(molecules = ligand_mols)
    forcefield.registerTemplateGenerator(template_gen.generator)
    
    
    bbox              = boundingbox(lig_positions + list(protein_pdb.positions.value_in_unit(angstrom)))

    bbox    = [ bbox[i][1]-bbox[i][0] for i in range(3) ]
    padding = 2.*nanometer
    xBoxvec = Vec3((bbox[0]+padding)/nanometer, 0., 0.)*nanometer
    yBoxvec = Vec3(0.0, (bbox[1]+padding)/nanometer, 0.)*nanometer
    zBoxvec = Vec3(0.0, 0.0, (bbox[2]+padding)/nanometer)*nanometer
    logger.info(f"\t===Bounding box: x-axis {xBoxvec}")
    logger.info(f"\t                 y-axis {yBoxvec}")
    logger.info(f"\t                 z-axis {zBoxvec}")

    ## Additional parameters 
    # padding= 10 * angstroms,
    # positiveIon="Na+", negativeIon="Cl-",
    # ionicStrength= 0.1 * molar, neutralize=False
    modeller.addSolvent(forcefield,
                        boxVectors = (xBoxvec, yBoxvec, zBoxvec))
    ## conformers are usually expressed as nanometer
    system            = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, 
                                               constraints=HBonds, rigidWater = True, 
                                               removeCMMotion=False, hydrogenMass = 1*amu)
    system.addForce(MonteCarloBarostat(1 * atmospheres, 300 * kelvin, 25))
    
    ## since we added the ligands the last, we can identify the ligand locations the following way
    n_all_atoms    = system.getNumParticles()
    logger.debug(f"All atom count: {n_all_atoms}")

    if save_loc is not None:
        logging.info("Saving the solvated complex...")
        with open(save_loc, "w") as f:
            PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)

    return system, modeller, ligand_locs


def compute_alchemical_system_with_region(system, ligand_locs, n_lambda_states=11, 
                                          temperature=300.0*kelvin, 
                                          pressure=1.0*bar):
    factory = alchemy.AbsoluteAlchemicalFactory(alchemical_pme_treatment='exact',
                                                alchemical_rf_treatment='exact',
                                                disable_alchemical_dispersion_correction=True)    
    alchemical_region = alchemy.AlchemicalRegion(alchemical_atoms=ligand_locs,
                                                 annihilate_electrostatics=True,
                                                 annihilate_sterics=True)
    alchemical_system = factory.create_alchemical_system(system, alchemical_region)
    ## Identify alchemical regions: i.e. regions that are successively made to decouple from the original system
    # Create compound states at different lambda values
    lambda_values = np.linspace(0.0, 1.0, n_lambda_states)
    alchemical_states = []

    for i, lambda_value in enumerate(lambda_values):
        name  = f'lambda{i}'
        state = alchemy.AlchemicalState.from_system(alchemical_system)

        # Set lambda values - adjust these as needed for your specific case
        if lambda_value < 0.5:
            # First half: turn off charges
            state.lambda_electrostatics = 1.0 - 2.0 * lambda_value
            state.lambda_sterics = 1.0
        else:
            # Second half: turn off vdW
            state.lambda_electrostatics = 0.0
            state.lambda_sterics = 1.0 - 2.0 * (lambda_value - 0.5)

        tstate = ThermodynamicState(system = alchemical_system,
                                   temperature = temperature,
                                   pressure = pressure)
        alchemical_states.append(CompoundThermodynamicState(
            thermodynamic_state=tstate, 
            composable_states=[state]))
    return alchemical_system, alchemical_states


def minimize_and_equilibrate(alchemical_system, alchemical_states, modeller, platform, 
                             n_lambda_states=11,
                             temperature=300.0*kelvin, friction_coeff=1.0/picosecond, 
                             int_timestep=2.0*femtoseconds, minimize_energy_iter=1000):
    # Create simulation context for each state
    contexts    = []
    simulations = []

    for state in tqdm(alchemical_states, "Equilibrating different lambda parameters"):
        integrator = openmm.LangevinIntegrator(temperature, friction_coeff, int_timestep)
        simulation = app.Simulation(
            modeller.topology,
            alchemical_system,
            integrator,
            platform,
            {'Precision': 'mixed'}
        )
        simulation.context.setPositions(modeller.positions)

        # Minimize energy
        simulation.minimizeEnergy(maxIterations=minimize_energy_iter)

        # Equilibrate
        simulation.context.setVelocitiesToTemperature(temperature)
        simulation.step(2.5*minimize_energy_iter)  # Short equilibration

        simulations.append(simulation)
        contexts.append(simulation.context)
    return simulations, contexts


def do_fep_simulation(simulations, contexts, alchemical_states, 
                      outputfld, n_lambda_states = 11, n_iterations = 1000, 
                      nsteps_per_iteration = 1000, output_frequency = 10):
    u_kn = np.zeros([n_lambda_states, n_lambda_states, n_iterations])
    for iteration in tqdm(range(n_iterations), desc = "Generating simulation"):
        # Run simulations at each state
        for k, simulation in enumerate(simulations):
            # Run a simulation step at the current lambda value
            simulation.step(nsteps_per_iteration)

            # Get positions and box vectors
            positions = simulation.context.getState(getPositions=True).getPositions()
            box_vectors = simulation.context.getState().getPeriodicBoxVectors()

            # Compute energies at all lambda values for this state
            for l, state in enumerate(alchemical_states):
                # Set the context parameters for this alchemical state
                pvmap = {"MonteCarloPressure" : state.pressure, 
                        "MonteCarloTemperature" : state.temperature,
                        "lambda_sterics" : state.lambda_sterics,
                        "lambda_electrostatics": state.lambda_electrostatics}
                for parameter, value in pvmap.items():
                    if parameter in contexts[k].getParameters():
                        contexts[k].setParameter(parameter, value)

                # Compute potential energy
                energy = contexts[k].getState(getEnergy=True).getPotentialEnergy()
                u_kn[k, l, iteration] = energy.value_in_unit(kilojoules_per_mole)

                # Reset context parameters to original state
                pvmap  = {"MonteCarloPressure" : alchemical_states[k].pressure, 
                        "MonteCarloTemperature" : alchemical_states[k].temperature,
                        "lambda_sterics" : alchemical_states[k].lambda_sterics,
                        "lambda_electrostatics": alchemical_states[k].lambda_electrostatics}
                for parameter, value in pvmap.items():
                    if parameter in contexts[k].getParameters():
                        contexts[k].setParameter(parameter, value)

        # Save checkpoint periodically
        if (iteration + 1) % output_frequency == 0:
            np.save(f'{outputfld}/fep_energies_iter{iteration+1}.npy', u_kn[:,:,:iteration+1])
    # Run production simulations and collect data
    return u_kn

def final_FEP(u_kn, temperature = 300 * kelvin):
    # Reshape for MBAR
    n_states = u_kn.shape[0]
    n_samples = u_kn.shape[2]

    # Reshape for MBAR input
    u_kln = np.zeros([n_states, n_states, n_samples])
    N_k = np.zeros([n_states], dtype=np.int32)

    for k in range(n_states):
        u_kln[k] = u_kn[k]
        N_k[k] = n_samples

    # Initialize MBAR
    print("Analyzing with MBAR...")
    beta = 1.0 / (temperature.value_in_unit(kelvin) * 8.3144621E-3)  # in kJ/mol
    mbar = MBAR(u_kln, N_k, verbose=True)

    # Calculate free energy differences
    results = mbar.compute_free_energy_differences(compute_uncertainty=True)

    # Extract results
    deltaF = results["Delta_f"]
    dDeltaF = results["dDelta_f"]

    # Print free energy differences
    print("\n\tFree Energy Differences (in kJ/mol):")
    print("\tState\tΔF\tδΔF")
    for i in range(n_states-1):
        print(f"{i} -> {i+1}\t{deltaF[i, i+1]:.2f}\t±{dDeltaF[i, i+1]:.2f}")

    # Total binding free energy
    total_deltaF = deltaF[0, -1]
    total_error = dDeltaF[0, -1]
    print(f"\nTotal Binding Free Energy: {total_deltaF:.2f} ± {total_error:.2f} kJ/mol")
    return total_deltaF, total_error

@hydra.main(version_base = None, config_name = "calmodulin-1", config_path="configs")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve = True)
    # number of iterations
    n_iterations           = cfg["n_iterations"]  # Total iterations
    nsteps_per_iteration   = cfg["n_steps_per_iterations"]
    min_energy_iterations  = cfg["min_energy_iterations"]
    out_frequency          = cfg["output_save_freq"]
    
    # states
    n_lambda_states        = cfg["n_lambda_states"]
    temperature            = cfg["temperature_kelvin"] * kelvin
    pressure               = cfg["pressure_bar"] * bar
    timestep               = cfg["timestep_femtoseconds"] * femtoseconds
    friction_coeff         = cfg["friction_coeff"]
    protein_file           = cfg["protein_pdb"]
    ligand_sdfs            = cfg["ligand_sdfs"]
    outputfld              = cfg["outputfld"]
    tsv_out_file           = cfg["tsv_out"]
    
    os.makedirs(outputfld, exist_ok = True)
    
    if cfg["fix_protein_pdb"]:
        assert "fixed_protein_pdb" in cfg, "Fixed PDB location is not available."
        logger.info("Fixing the protein PDB")
        fix_protein_file   = cfg["fixed_protein_pdb"]
        fix_pdb(protein_file, fix_protein_file)
        protein_file       = fix_protein_file
        logger.info(f"Fixed protein PDB saved to {protein_file}")
        
    # load device
    device    = cfg["device"]
    platform  = Platform.getPlatform(device)
    logger.info("Preparing the Protein + Ligand system")
    system, modeller, ligand_locs = prepare_system(protein_file, 
                                                   ligand_sdfs, 
                                                   platform)
    logger.info("Setting up alchemical system and alchemical regions")
    alchemical_system, alchemical_states = compute_alchemical_system_with_region(system, ligand_locs,
                                                                                 n_lambda_states=n_lambda_states,
                                                                                 temperature=temperature, 
                                                                                 pressure=pressure)
    
    logger.info("Equilibrating the system at different lambda parameters")
    simulations, contexts = minimize_and_equilibrate(alchemical_system, alchemical_states, 
                                                     modeller, platform, n_lambda_states=n_lambda_states,
                                                     temperature=temperature, friction_coeff=friction_coeff/picosecond, 
                                                     int_timestep=timestep, minimize_energy_iter=min_energy_iterations)
    
    logger.info("Running the simulations")
    u_kn = do_fep_simulation(simulations, contexts, alchemical_states, 
                             outputfld, n_lambda_states = n_lambda_states, n_iterations = n_iterations, 
                             nsteps_per_iteration = nsteps_per_iteration, output_frequency = out_frequency)
    
    total_deltaF, total_error = final_FEP(u_kn, 
                                          temperature = temperature)
    
    results = {"protein_pdb" : [Path(protein_file).resolve()], 
               "ligand_pdb"  : [",".join([str(Path(ligand_sdf).resolve()) for ligand_sdf in ligand_sdfs])],
               "total_deltaF": [total_deltaF],
               "total_error" : [total_error], 
               "n_iterations": [n_iterations], 
               "n_steps_per_iterations": [nsteps_per_iteration],
               "min_energy_iterations" : [min_energy_iterations],
               "n_lambda_states" : [n_lambda_states], 
               "temperature (K)" : [temperature.value_in_unit(kelvin)],
               "pressure (bar)"  : [pressure.value_in_unit(bar)],
               "timestep (femtosecond)": [timestep.value_in_unit(femtoseconds)],
               "friction coeff"  : [friction_coeff]
              }
    pd.DataFrame(results).to_csv(tsv_out_file, sep = "\t", index = None, mode = "a", 
                                 header = not os.path.exists(tsv_out_file))
    
    return

if __name__ == "__main__":
    main()