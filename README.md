Transient Time Correlation Function (TTCF)
==========================================

The code aims at creating a user friendly interface to implement the TTCF method in LAMMPS molecular dynamics simulation. The benchmark example shown here is particularly simple, and can be implemented on local machines, but the script is designed to be employed on HPC clusters for multi-core runs. 
The goal it to match the work of Borszak et al (2002) (https://doi.org/10.1080/00268970210137275), where the computed the shear viscosity of a homogeneous atomic system using TTCF. 
The dynamics is described by SLLOD equations. The computed the shear pressure over direct nonequilibrium trajectories, and compared the direct average (DAV) with TTCF.


The TTCF algorithm requires to integrate the phase space averace of the correlation between the quantity of interest, measured along a nonequilibrium trajectory, with the dissipation function at t=0 (the initial time of the nonequilibrium trajectory) 
```math
\langle B(t) \rangle = \int_0^t \langle \Omega(0)B(s)\rangle ds 
```

The average is perfmormed over nonequilibrium trajectories initial conditions sampled from the equilibrium ensemble associated to the system. The easiest way to achieve this is to follow the system over an equilibrium \textit{mother} trajectory. After a thermalization to ensure the system is in thermodynamic equilibrium, the state of the system (set of all positions and momenta) is periodically sampled. The procedure is shown is the figure below

![alt text](https://github.com/edwardsmith999/TTCF/blob/master/mother.png)

After this process, a series of nonequilibrium \textit{daughter} runs are perfomed, where their initial conditions are the states sampled from the equilibrium trajectories. 

![alt text](https://github.com/edwardsmith999/TTCF/blob/master/children.png)

From each initial state, three further mirrored (two in the figure) states are generated. These futher initial states guarantee that the phase average of the dissipation function is identically null and hence the convergence of the integral is ensured. The following mappings are used in this script

```math
\bigl(x_i\;,\;y_i\;,\;z_i\;,\;p_{xi}\;,\;p_{yi}\;,\;p_{zi}\bigr)\longrightarrow\bigl(x_i\;,\;y_i\;,\;z_i\;,\;p_{xi}\;,\;p_{yi}\;,\;p_{zi}\bigr)\\
```
```math
\bigl(x_i\;,\;y_i\;,\;z_i\;,\;p_{xi}\;,\;p_{yi}\;,\;p_{zi}\bigr)\longrightarrow\bigl(x_i\;,\;y_i\;,\;z_i\;,\;-p_{xi}\;,\;-p_{yi}\;,\;-p_{zi}\bigr)\\
```
```math
\bigl(x_i\;,\;y_i\;,\;z_i\;,\;p_{xi}\;,\;p_{yi}\;,\;p_{zi}\bigr)\longrightarrow\bigl(-x_i\;,\;y_i\;,\;z_i\;,\;-p_{xi}\;,\;p_{yi}\;,\;p_{zi}\bigr)\\
```
```math
\bigl(x_i\;,\;y_i\;,\;z_i\;,\;p_{xi}\;,\;p_{yi}\;,\;p_{zi}\bigr)\longrightarrow\bigl(-x_i\;,\;y_i\;,\;z_i\;,\;p_{xi}\;,\;-p_{yi}\;,\;-p_{zi}\bigr)
```

Hence, for each sampled state, four nonequilibrium runs are generated. 
A compact TTCF implementation can be written within a single LAMMPS input file using the following structure

	System setup

	Run equilibrium thermalization

	Save state

	Loop over Daughters

		Load state

		Run equilibrium decorrelation

		Save state

		Loop over Mappings
 
			Load state

   			Apply mapping

			Run nonequilibrium daughter

		end Loop
 
	end Loop	


Each single block is translated into LAMMPS commands as follows:


	########### System setup ###########

                #Declaration of all variables and simulation parameters (Type 1)
		
		variable rho equal 0.8442                               #Density
		variable Npart equal 256                                #Number of particles
		variable T equal 0.722                                  #Temperature 
		variable L equal (${Npart}/${rho})^(1.0/3)              #system size 
		variable rc equal 2^(1/6)                               #Interaction radius for Lennard-Jones (effectively WCA potential)
		variable k_B equal 1                                    #Boltzmann Constant
  
		variable srate equal 1                                  #Shear rate applied   

		########################################################################################################
                #Declaration of all variables and simulation parameters (Type 2). 
		#These variables will be implemented in Python, and hence they are not declared in the LAMMPS script. 
  		#They are shown here for clarity

		Ndaughters=1000                                         #Total number of initial states generated

		Maps=[0,7,36,35]					#Selected mapping
		Nmappings=4						#Total number of mappings

		Nsteps_Thermalization = 10000                          	#Lenght of thermalization run
		Nsteps_Decorrelation  = 10000				#Lenght of decorrelation runs
		Nsteps_Daughter       = 1000                            #Lenght of nonequilibrium runs

		Delay=10    						#Frequency (in timesteps) for output generation along the nonequilibrium runs
		Nsteps_eff=int(Nsteps_Daughter/Delay)+1			#Effective number of timesteps of the output
  
		Nbins=100 						#Number of bins for profile output
		Bin_Width=1.0/float(Nbins)				#Bin width for profile output 

		dt = 0.0025                                             #Bin width for profile output

  		rand_seed = 12345    					#Seed for random initial velocity generation

		########################################################################################################
                #End of parameter declaration
                


 		units		    lj
		dimension	    3
		atom_style      full 
		neigh_modify	delay 0 every 1
		boundary		p p p
	
		lattice         fcc ${rho}
		region          simbox prism 0 ${L} 0 ${L} 0 ${L} 0 0 0 units box
		create_box      1 simbox 
		create_atoms    1 region simbox

		group           fluid region simbox

		mass            * 1.0
		pair_style      lj/cut ${rc}

		pair_coeff       1 1 1.0 1.0

		velocity        fluid create $T ${rand_seed}
  
                timestep ${dt}
		variable Thermo_damp equal 10*${dt}
		




The declared variables are either used by LAMMPS only (type 1) and directly declared within the input file, or managed by the python interface (type 2) and hence not declared in the input file, and shown here just for clarity purpose. The remaining set of commands are standard creation of simulation box, atom positions and velociites, and interatomic potential.

	########### Run equilibrium thermalization ###########

		fix NVT_thermalization all nvt temp ${T} ${T} ${Thermo_damp} tchain 1
		run ${Nsteps_Thermalization}
		unfix NVT_thermalization
  
	########### Save state ###########

 		fix snapshot all store/state 0 x y z vx vy vz

The command fix store/state allow to save the state of the system without handling any file. The state can be restored by the set of commands
	########### Load State ###########
 
		change_box all  xy final 0

		variable px atom f_snapshot[1]
		variable py atom f_snapshot[2]
		variable pz atom f_snapshot[3]
		variable vx atom f_snapshot[4]
		variable vy atom f_snapshot[5]
		variable vz atom f_snapshot[6]

		set             atom * x v_px 
		set             atom * y v_py 
		set             atom * z v_pz 
		set             atom * vx v_vx 
		set             atom * vy v_vy 
		set             atom * vz v_vz
  
Where the command change_box is needed only of the SLLOD dynamics is employed. The variable commands regain the output of the fix store/state, and later assign it to the position of velocity of each atom.
Note that no info about the thermostat is saved by the store/state command. Should the thermostat be relevant for the simulation, the save and load state must be replaced by 

	########### Save state ###########

 		write_restart snapshot.rst

	########### Load State ###########
 
		read_restart snapshot.rst

And each fix nvt command should have the same ID throughout the whole run.
The mapping procedure work is the following way: in order to keep a general algorithm, each single dimension can be independenlty mirrored. There are 6 total dimensions, namely x,y,z,vx,v,z. Thus, a mapping can be identified by a set of six digits, each of which can be either 0 (no reflection) or 1 (reflection). For instance, the sequence 101100 identifies the following mapping

	(101100) = (-x , y , -z , -vx , vy , vz )
 
There are a total of 2^6 independent mappings, hence the string corresponding to the selected mapping can be translated into a number from 0 to 63 by simply converting the string from a binary to a decimal number. The mappings selected here are 

	( x , y , z ,  vx ,  vy ,  vz ) = 000000 = 0  (original state)
 	( x , y , z , -vx , -vy , -vz ) = 000111 = 7  (time reversal)
  	(-x , y , z , -vx ,  vy ,  vz ) = 100100 = 36 (x-reflection)
  	(-x , y , z ,  vx , -vy , -vz ) = 100011 = 35 (time reversal + x-reflection)

and the mapping is applied by the following commands

	########### Mapping application ###########
 
		#variable map equal to a number from 0 to 63
  
		variable mpx equal     floor((${map})/(2^5))
		variable mpy equal     floor((${map}-(${mpx}*2^5))/(2^4))
		variable mpz equal     floor((${map}-(${mpx}*2^5)-(${mpy}*2^4))/(2^3))
		variable mvx equal     floor((${map}-(${mpx}*2^5)-(${mpy}*2^4)-(${mpz}*2^3))/(2^2))
		variable mvy equal     floor((${map}-(${mpx}*2^5)-(${mpy}*2^4)-(${mpz}*2^3)-(${mvx}*2^2))/(2^1))
		variable mvz equal     floor((${map}-(${mpx}*2^5)-(${mpy}*2^4)-(${mpz}*2^3)-(${mvx}*2^2)-(${mvy}*2^1))/(2^0))

		variable        px atom x+((xhi-2*x)*${mpx})
		variable        py atom y+((yhi-2*y)*${mpy})
		variable        pz atom z+((zhi-2*z)*${mpz})
		variable        vx atom vx-(2*vx*${mvx})
		variable        vy atom vy-(2*vy*${mvy})
		variable        vz atom vz-(2*vz*${mvz})

		set             atom * x  v_px 
		set             atom * y  v_py 
		set             atom * z  v_pz 
		set             atom * vx v_vx 
		set             atom * vy v_vy 
		set             atom * vz v_vz

where the first block of commands traslates back a decimal number into its binary representation and selects each single digit, the second block calculates the corresponding reflected dimension (1 inverts sign, 0 leaves unchanged), and the third block updates the positions and momenta.
The last two blocks are the equilibrium decorrelation process and the daughter setup. 

	########### Run equilibrium decorrelation ###########

 		include ./load_state.lmp
    		fix NVT_sampling all nvt temp ${T} ${T} ${Thermo_damp} tchain 1
    		run ${Nsteps_Decorrelation}
		unfix NVT_sampling
    		fix snapshot all store/state 0 x y z vx vy vz


	########### Run nonequilibrium daughter trajectory ###########
 
 		# Apply the external field
		variable        vx_shear atom vx+${srate}*y     
		set             atom * vx v_vx_shear            
	

		####SET DYNAMICS OF THE DAUGHTER TRAJECTORY####	
	
		fix     box_deform all deform 1 xy erate ${srate} remap v units box
		fix     NVT_SLLOD all nvt/sllod temp ${T} ${T} ${Thermo_damp}












   
This code is inended to runs LAMMPS to get a mother trajectory and then take a range of mirrored child trajectories as required for the TTCF method. 
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

    create_atoms    2 region latticeregion
    ...
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
In this way, the TTCF force adds in energy of the form 

```math
\dot{H}=\sum_ik(\textbf{r}^w_i-\textbf{r}^l_i)v
```
where the position of the wall site is $\textbf{r}^w_i$ and the position of the corresponding real molecule tethered to it is $\textbf{r}^l_i$. 
The sliding speed of the wall $v$ is the value give by `${srate}` in the velocity statement above.

In order to obtain the required quantities from the MD, the x component of force between the wall tethering bonds and the tethering site, the LAMMPS `compute_bond_local` is used. 


** Historical note for pre-2019 LAMMPS**

Onlythe magnitude of force and not the components was returned in LAMMPS prior to 2019.
As a result, we needed to patch LAMMPS to allow a statement of the form,

    group wallandsites union sites wall
    compute fx wallandsites bond/local fx
    compute sumfx all reduce sum c_fx

which gets the sum of the bonds between walls and sites in the x direction. 
The patch for lammps is provided in the repository, which can be applied by copying to the lammps/src directory and applied with,

    git apply lammps_bond_local.patch

This has been accepyed from the pull request to the main LAMMPS branch (https://github.com/lammps/lammps/pull/1667) as of the 2019 Autumn/Winter release.


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
    variable index loop   ${Nchild}

	    run   ${Ninloop}
	    write_data    branch${index}.dat pair ij
        
    next index
    jump SELF loop

Again, the restart case can be managed using the run_mother function of run.py and the Ninloop and number of loops to run specified. 

Children
--------

The children are then restarted from the various restart files created by the mother trajectory. We use phase space mirroring to generate complementary trajectories,

```math
\bigl(x_i\;,\;y_i\;,\;z_i\;,\;p_{xi}\;,\;p_{yi}\;,\;p_{zi}\bigr)\longrightarrow\bigl(x_i\;,\;y_i\;,\;z_i\;,\;p_{xi}\;,\;p_{yi}\;,\;p_{zi}\bigr)\\
```
```math
\bigl(x_i\;,\;y_i\;,\;z_i\;,\;p_{xi}\;,\;p_{yi}\;,\;p_{zi}\bigr)\longrightarrow\bigl(x_i\;,\;y_i\;,\;z_i\;,\;-p_{xi}\;,\;-p_{yi}\;,\;-p_{zi}\bigr)\\
```
```math
\bigl(x_i\;,\;y_i\;,\;z_i\;,\;p_{xi}\;,\;p_{yi}\;,\;p_{zi}\bigr)\longrightarrow\bigl(-x_i\;,\;y_i\;,\;z_i\;,\;-p_{xi}\;,\;p_{yi}\;,\;p_{zi}\bigr)\\
```
```math
\bigl(x_i\;,\;y_i\;,\;z_i\;,\;p_{xi}\;,\;p_{yi}\;,\;p_{zi}\bigr)\longrightarrow\bigl(-x_i\;,\;y_i\;,\;z_i\;,\;p_{xi}\;,\;-p_{yi}\;,\;-p_{zi}\bigr)
```

which we can imagine this as follows,

![alt text](https://github.com/edwardsmith999/TTCF/blob/master/children.png)

Only two mirrors per output are shown for clarity.

To implement this, we want a batch run starting from each of the branch.dat trajectories to create child runs. 

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
