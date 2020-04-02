Transient Time Correlation Function (TTCF)
==========================================
    
This code is intended to runs LAMMPS to get a mother trajectory and then take a range of mirrored child trajectories as required for the TTCF method. 
The batch running is done using SimWrapPy, which should be automatically downloaded (from https://github.com/edwardsmith999/SimWrapPy)

The first aim is to match the work of Bernardi et al (2012) (https://doi.org/10.1063/1.4746121), before moving to more complex systems.
The case is Couette flow, which is set up by creating a set of imaginary atoms which act as tethering sites.
These do not interact with the system except to apply a force to the wall atom they are tethered to. 
These sites then slide and pull the walls as required for Couette flow. 
The liquid domain is set to have a size y, a wall region "wallwidth" is then added to top and bottom and a buffer which is empty to prevent tethered molecule from leaving the domain.
A Langevin thermostat is applied to the outer part of the walls to remove the heat generated by shearing.

The lattice region includes both wall and liquid molecules, created with,

    create_atoms 1 region latticeregion

We then define upper/lower to be the region which contains the wall atoms and create a duplicate set of molecules with type 4 and 5 to be bottom and top sites respectively:

    create_atoms    4 region lower
    create_atoms    5 region upper
    group            lowersites type 4
    group            uppersites type 5

These type 4 and type 5 atoms are then set to not interact with anything else (they are sites or ghost atoms)

    fix                4 uppersites setforce 0.0 0.0 0.0
    fix                5 lowersites setforce 0.0 0.0 0.0

The real wall particles are then tethered harmonically to these sites

    bond_style       harmonic
    bond_coeff       1 150.0 0.0
    create_bonds     many lowersites lower 1 0.0 0.0001
    create_bonds     many uppersites upper 1 0.0 0.0001

The sites themselves are given a motion and so drag the particles. 

    velocity	    uppersites set ${srate} 0.0 0.0 units box
    velocity	    lowersites set -${srate} 0.0 0.0 units box

Note that the wall particle themselves are not given any velocity.

In order to obtain the required quantities from the MD, the x component of force between the wall tethering bonds and the tethering site, the LAMMPS `compute_bond_local` is used. However, this only returns the magnitude of force and not the components.
As a result, we need to patch LAMMPS to allow a statement of the form,

    group wallandsites union sites wall
    compute fx wallandsites bond/local fx
    compute sumfx all reduce sum c_fx

which gets the sum of the bonds between walls and sites in the x direction. 
The patch for lammps is provided in the repository, which can be applied by copying to the lammps/src directory and applied with,

    git apply lammps_bond_local.patch

This has been submitted as a pull request to the main LAMMPS branch (https://github.com/lammps/lammps/pull/1667) and will be included in the 2019 Autumn/Winter release.


Quickstart
===========

Mother
------

We discuss the key parts of the mother.in files which create the branches to be used in the TTCF method. The rest of the file sets up a simple Couette flow example (wall described above, nose hoover thermostat). This can then be changed as needed to any other system setup.

First of all, we need to equilibrate a molecular channel to be used as the initial file for the mother trajectory which generates a bunch of children. This assumes we have a copy of the LAMMPS executable in the current directory (or added to path), called lmp. The equilibration is run using,

    mpiexec -n 1 ./lmp -in mother_equil.in

This will run equilibration for 10000 steps, where the equilibration is run using mother_equil.in which creates the system and runs a number of timesteps, saving to final_mother.dat file.
The actual running of this script can also be handled by the run.py file in the run_mother function.
where Nequil is an input specifying how many timesteps to run for.

Next, we read in the restart file from mother_equil.in called final_mother.dat, and then loops in blocks of Ninloop=300 creating the starting point for each child trajectories to branch off. The final step saves a new final_mother.dat based on the last step we got to. 

![alt text](https://github.com/edwardsmith999/TTCF/blob/master/mother.png)

For this run which creates a range of restart (branch) files for each of the child trajectories, we use
        
    mpiexec -n 1 ./lmp -in mother_restart.in

The actual code for the blocks of runs creating branch files can be seen in bottom of the mother_restart.in as follows,

    variable Ninloop equal   300

    #Loop over lots of input cases
    label loop
    variable index loop 1000

        run	           ${Ninloop}
        write_data    branch*.dat
        
    next index
    jump SELF loop

Again, the restart case can be managed using the run_mother function of run.py and the Ninloop and number of loops to run specified. 

Children
--------

We then run a batch run starting from each of the branch.dat trajectories (and its mirror) to create child runs using python code with simwraplib. You will need numpy installed then run.

    python run_children.py

Tested with python 2.7. The number of jobs running in parallel can be set by changing 

    #Number of processors to use
    ncpus = 6

in run.py. This creates a folder called study which has each of the child runs with a file called 
output.txt. The content of this is set in child.in

github.com/edwardsmith999/TTCF

In its current form, run.py saves the run data to a pickle file and also calculates dissipation from the initial condition file. Together these are the key quantities needed for the TTCF.


Post Process
------------
The output files from run.py create a pickle file of TTCF_run.p which has the dissipation function and a record of all time history of all outputs saved to output.txt in the child folder. From this, we can calculate the TTCF, with the key utilities for this saved in 

![alt text](https://github.com/edwardsmith999/TTCF/blob/master/children.png)


Each of the child trajectories creates data in the study folder, the result can be read into Python by scripts in,

    analyse_data.py

which loop over the ttcf folders in the study directory and averages all of them.
The entire trajectory histories are saved by run.py to a Numpy binary file, which can be processed as follows

    plot_summary.py

which gives the following output,

![alt text](https://github.com/edwardsmith999/TTCF/blob/master/TTCF_MOP_VIrial.png)

The MOP and Virial stresses obtained directly from the average over all trajectories (here 40,000 in total using 20,000 mirror. 
Note there is still a difference in magnitude between the TTCF and direct so they are plotted on different figures. 

![alt text](https://github.com/edwardsmith999/TTCF/blob/master/TTCF_fijwall_dispfn.png)

The sum of the force between liquid and wall molecules (fij) and the dissipation function (forces in the x direction between wall tethering sites and molecules), both directly averaged and obtained using the TTCF

![alt text](https://github.com/edwardsmith999/TTCF/blob/master/Delhomm_DAV_vs_TTCF.png)

Delta F, the difference between top and bottom forcing between wall and fluid. 

Note there is an optional output can be switched on in mother,

    fix		        6 all ave/chunk 5 40 1000 layers density/mass vx temp  c_stress[1] c_stress[2]  c_stress[3]  c_stress[4] file profile.wall.2d

which can be plotted using

    python plot_chunks.py 

to check the system is evolving as expected.

To Do
=====

 - The calculation of Pxy and dissipation function are not correct
 - The mother trajectory when we take a "brach" must be matched to the child trajectory
 - Match results of Bernardi et al (2012) (https://doi.org/10.1063/1.4746121)
