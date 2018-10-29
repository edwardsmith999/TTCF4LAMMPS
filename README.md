Transient Time Correlation Function (TTCF)
==========================================
    
This code is intended to runs LAMMPS to get a mother trajectory and then take a range of mirrored child trajectories as required for the TTCF method. The first aim is to match the work of Bernardi et al (2012) (https://doi.org/10.1063/1.4746121), before moving to more complex systems.

The batch running is done using SimWrapPy, which should be automatically downloaded (from https://github.com/edwardsmith999/SimWrapPy)

Quickstart
===========


Mother
------

First of all, we need to create the mother trajectory. We need to copy the LAMMPS executable to the current directory (or add to path). We will assume this is called lmp,


    mpiexec -n 1 ./lmp -in mother.in

This will run equilibration for 1000,000 steps and then loops in blocks of 300 creating the starting point for the child trajectories to branch off. This can be seen in bottom of the mother.in as follows,


    #Equilibration
    run	           100000
    write_data output_equil.dat

    #Loop over lots of input cases
    label loop
    variable index loop 1000

        run	           300
        write_data output*.dat
        
    next index
    jump SELF loop

The rest of the file sets up a Couette flow example, although this can be changed as needed.

Children
--------

We then run a batch run starting from each of these trajectories (and its mirror) to create child runs using python code with simwraplib. You will need numpy installed.

    python run_children.py

The number if sequential running jobs can be set by changing 

    #Number of processors to use
    ncpus = 6

in run_children.py. This creates a folder called study which has each of the child runs.


Post Process
------------
Each of the child trajectories creates data in the study folder, the final result can be obtained by running

    analyse_data.py

which loops over the folders in the study directory and averages all of them.

To Do
=====

 - The calculation of Pxy and dissipation function are not correct
 - The mother trajectory when we take a "brach" must be matched to the child trajectory
 - Match results of Bernardi et al (2012) (https://doi.org/10.1063/1.4746121)
