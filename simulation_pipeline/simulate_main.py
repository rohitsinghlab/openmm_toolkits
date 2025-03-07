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
import time

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
        # if ligand_mol.partial_charges is None:
        #     ligand_mol.assign_partial_charges(partial_charge_method='am1bcc',
        #                                       toolkit_registry=toolkit_registry)
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

def simulate(system, modeller, platform, temperature = 300*kelvin, 
            friction_coeff = 1.0/picosecond, 
            timestep       = 2.0 * femtosecond, 
            n_timesteps    = 100000, 
            output_pdb     = "output.pdb", 
            save_freq      = 1000):
    integrator = LangevinIntegrator(temperature, 
                                    friction_coeff,
                                    timestep)
    simulation = Simulation(modeller.topology, system, 
                            integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()
    simulation.reporters.append(PDBReporter(output_pdb, save_freq))
    simulation.reporters.append(StateDataReporter(stdout, save_freq, step=True, 
                                potentialEnergy=True, temperature=True))
    simulation.step(n_timesteps)

def setup_logging(logfileloc, logger):
    # Create file handler - this will be our only output destination
    file_handler = logging.FileHandler(logfileloc)
    file_handler.setLevel(logging.DEBUG)
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)
    return 


@hydra.main(version_base = None, config_name = "3htb", config_path="configs")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve = True)
    # number of iterations

    setup_logging(cfg["_outputlog"], logger)

    n_steps                = cfg["n_steps"]  # Total iterations
    out_frequency          = cfg["output_save_freq"]
    
    # states
    temperature            = cfg["temperature_kelvin"] * kelvin
    timestep               = cfg["timestep_femtoseconds"] * femtoseconds
    friction_coeff         = cfg["friction_coeff"]
    protein_file           = cfg["protein_pdb"]
    ligand_sdfs            = cfg["ligand_sdfs"]
    output_url             = cfg["_outputurl"]
    
    if cfg["fix_protein_pdb"]:
        assert "fixed_protein_pdb" in cfg, "Fixed PDB location is not available."
        logger.info("Fixing the protein PDB")
        fix_protein_file   = cfg["fixed_protein_pdb"]
        fix_pdb(protein_file, fix_protein_file)
        protein_file       = fix_protein_file
        logger.info(f"Fixed protein PDB saved to {protein_file}")
        
    # load device
    start     = time.time()
    device    = cfg["device"]
    platform  = Platform.getPlatform(device)
    logger.info("Preparing the Protein + Ligand system")
    system, modeller, ligand_locs = prepare_system(protein_file, 
                                                   ligand_sdfs, 
                                                   platform)
    logger.info("Running the simulations")
    u_kn = simulate(system, modeller, platform, temperature = temperature, 
                    friction_coeff = friction_coeff/picosecond, 
                    timestep = timestep, 
                    n_timesteps = n_steps, 
                    output_pdb = output_url, 
                    save_freq = out_frequency)
    elapsed_time = time.time() - start
    logger.info(f"Total time elapsed: {elapsed_time:2f} seconds")
    return

if __name__ == "__main__":
    main()