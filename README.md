Required packages (and versions):
1. Install openmm using the following command:
    
    ```python
    micromamba install -c conda-forge openmm cudatoolkit=11.8
    ```
    
2. Install openmmtools with
    
    ```python
    micromamba -c conda-forge openmmtools=1.11
    ```
    
3. Install openff-toolkit with 
    
    ```python
    micromamba -c conda-forge openff-toolkit
    ```
    
4. Import `openmmforcefields`
    
    ```python
    micromamba install -c conda-forge openmmforcefields
    ```
    
5. Import `acpype` and `parmed`
6. Install `openbabel`
    
    [openbabel commands](https://www.notion.so/openbabel-commands-1a8f8e94bfb980b0a4f1e8972f6c1c21?pvs=21)
    
7. Additional headache that needed to be resolved:
    
    **antechamber**
    
    ```python
    # created a softlink to antechamber at the ~/.local/bin folder
    pushd ~/.local/bin/; ln -s /hpc/group/singhlab/user/kd312/minimamba/envs/yank/bin/antechamber .; popd
    
    # additionally, change the antechamber sh file by replacing 
    this_script_dir="/hpc/group/singhlab/user/kd312/minimamba/envs/yank/bin"
    
    ```
    
    **parmchk2**
    
    ```python
    # do the same as above: created a softlink to parmchk2 at the ~/.local/bin folder
    pushd ~/.local/bin/; ln -s /hpc/group/singhlab/user/kd312/minimamba/envs/yank/bin/parmchk2 .; popd
    
    # additionally, change the parmchk2 sh file by replacing 
    this_script_dir="/hpc/group/singhlab/user/kd312/minimamba/envs/yank/bin"
    ```
    
8. Additional changes at `openmmforcefields/generators/template_generators.py`:
    
    ```python
    ## update line 848 in the openmmforcefields/generators/template_generators.py to
    
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
    ```
    
9. Same changes in `openff/toolkit/utils/ambertools_wrapper.py` line 202


## Some important conversion commands

- CIF to PDB conversion
    
    ```jsx
    obabel input.cif -O output.pdb
    ```
    
- PDB to SDF for ligands
    
    ```jsx
    obabel input.pdb -O output.sdf
    ```
    
- To preserve partial charge, use the following command
    
    ```python
    obabel -ipdb complex.pdb -omol2 -O ligand.mol2 -l LIG --partialcharge gasteiger
    ```
