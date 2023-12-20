Transient Time Correlation Function (TTCF) [![CI](https://github.com/edwardsmith999/TTCF/actions/workflows/main.yml/badge.svg)](https://github.com/edwardsmith999/TTCF/actions/workflows/main.yml) 
==========================================

The code aims at creating a user-friendly interface to implement the Transient Time Correlation Function (TTCF) method in the [LAMMPS](https://www.lammps.org/) Molecular Dynamics Simulator. This provides a way to get better statistics from molecular dynamics simulations for cases where the forcing is very weak. For example, the typical shear rates in wall-driven flow typical in Tribology experiments or the pressure gradient driving flow in fluid dynamics experiments, which at the molecular scale would be too small to measure. 

Quickstart
----------

The TTCF code is designed to run in LAMMPS. First, the Python interface for LAMMPS (https://docs.lammps.org/Python_module.html) must be installed. Installation instructions are provided on the LAMMPS page for Windows, Linux and Mac here: https://docs.lammps.org/Python_install.html.

After installation, to check the Python LAMMPS interfaces works as expected, open Python and try to import the lammps module.

    python

    >>> import lammps
    >>> lmp = lammps.lammps()

To use the TTCF, start by running the example case, first clone the repository,

    git clone https://github.com/edwardsmith999/TTCF.git

next install the prerequisite,

    mpi4py
    numpy
    matplotlib
    
then navigate to the TTCF folder and run,

    cd TTCF
    python run_TTCF.py

For mpi4py, a version of MPI is required, either [mpich](https://www.mpich.org/) or [openMPI](https://www.open-mpi.org/) should work. This allows the code to run in parallel, which should speed up the example by as many cores as you run it on, for example if you have a 4 core CPU,

    mpiexec -n 4 python run_TTCF.py

will divide the work over 4 processes. The example should run fairly quickly and gives the velocity profile for the case of SLLOD shearing flow, comparing direct averaging (DAV) to the transient time correlation function (TTCF). For the case of shear rate of 1, set in `system_setup.in`,

    variable srate equal 1

the results should look like,

![alt text](https://github.com/edwardsmith999/TTCF/blob/master/figures/TTCF_vs_DAV_SLLOD.png)

The TTCF provided better statistics than the direct averaging at low shear rates, to see this, we can change the example code under `system_setup.in` we can change,

    variable srate equal 0.0001 

which will appear as follows,

![alt text](https://github.com/edwardsmith999/TTCF/blob/master/figures/TTCF_vs_DAV_SLLOD_lowstrain.png)

We can already see the TTCF is giving less fluctuation about the expected linear profile.
If we go to even lower shear rates,

    variable srate equal 0.000001

this becomes even more apparent,

![alt text](https://github.com/edwardsmith999/TTCF/blob/master/figures/TTCF_vs_DAV_SLLOD_lowerstrain.png)

You can adapt the script to your own example by changing the LAMMPS code as needed. The theory behind the TTCF and software is discussed below. 

Theory
------
The benchmark example shown here is particularly simple, and can be implemented on local machines, but the script is designed to be employed on HPC clusters for multi-core runs. 
The goal it to match the work of Borszak et al (2002) (https://doi.org/10.1080/00268970210137275), which computed shear viscosity of a homogeneous atomic system using the TTCF. 
The dynamics is described by the SLLOD equations, shear pressure is computed over direct nonequilibrium trajectories with the TTCF and compared to the direct average (DAV).


The TTCF algorithm requires integration of the phase space average of the correlation between the quantity of interest, measured along a nonequilibrium trajectory, with the dissipation function, $\Omega$, at t=0 (the initial time of the nonequilibrium trajectory) 
```math
\langle B(t) \rangle =\langle B(0) \rangle+ \int_0^t \langle \Omega(0)B(s)\rangle ds 
```

The average is performed over nonequilibrium trajectories initial conditions sampled from the equilibrium ensemble associated with the system. The easiest way to achieve this is to follow the system over an equilibrium \textit{mother} trajectory. After a thermalization to ensure the system is in thermodynamic equilibrium, the state of the system (set of all positions and momenta) is periodically sampled. The procedure is shown is the figure below

![alt text](https://github.com/edwardsmith999/TTCF/blob/master/figures/mother.png)

After this process, a series of nonequilibrium \textit{daughter} runs are perfomed, where their initial conditions are the states sampled from the equilibrium trajectories. 

![alt text](https://github.com/edwardsmith999/TTCF/blob/master/figures/children.png)

From each initial state, three further mirrored states are generated (two in the figure). These further initial states guarantee that the phase average of the dissipation function is identically null and hence the convergence of the integral is ensured. The following mappings are used in this script

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


A compact TTCF implementation can be written within a single LAMMPS input file using the following psudocode structure

	System setup

	Run equilibrium thermalization

	Save state

	Loop over Daughters

		Load state

		Run equilibrium sampling

		Save state

		Loop over Mappings
 
			Load state

   			Apply mapping

			Run nonequilibrium daughter

		end Loop
 
	end Loop	


The implementation can be found in the file LAMMPS_script.in. The file is equivalent to the python implemenetation which will be described later and can be directly used for MD simulations. However, it generates two output files for each daughter trajectory, making it impractical for realistic calculations. It is here used just as a reference to better understand the rationale behing the python interface. The entire TTCF calculation is here performed via a single run. The systems repeatedly switches between equilibrium (mother) and nonequilibrium (daughter) trajectory as shown in the abode pseudocode. Each block of commands is quite straightforward. However, the generation and loading the generation of the sample from the mother trajectory, and loading it need to be clarified.
In order ot avoid writing/reading from files, the instantaneous state of the system is stored via the following command, which temporarily save the variables specified (positions and momenta) in the structure called snapshot

 		fix snapshot all store/state 0 x y z vx vy vz

The coordinates can then be loaded via the following sets of commands
 
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
  
Where the command change_box is needed only if the SLLOD dynamics is employed. The otuput of the fix store/state (f_snapshot) is assigned to the declared variables, which then overwrite the existing positions and velocities.
Note that no info about the thermostat can be saved by the store/state command. Should the thermostat be relevant for the simulation, the operations must be replaced by 

	########### Save state ###########

 		write_restart snapshot.rst

	########### Load State ###########
 
		read_restart snapshot.rst

And each fix nvt command should have the same ID throughout the whole run, bot for the mother and the daughter trajectory.
The second point which needs clarification is the mapping procedure. It  works is the following way: in order to keep a general algorithm, each single dimension can be independenlty mirrored. There are 6 total dimensions, namely x,vx,y,vy,z,vz. Thus, a mapping can be identified by a set of six digits, each of which can be either 0 (no reflection) or 1 (reflection). For instance, the sequence 101100 identifies the following mapping

	(101100) = (-x , vx , -y , -vy , z , vz )
 
There are a total of 2^6 independent mappings and the string corresponding to the selected mapping can be translated into a number from 0 to 63 by simply converting the string from a binary to a decimal number. The mappings selected here are 

	(  x ,  vx , y ,  vy , z ,  vz ) = 000000 = 0  (original state)
 	(  x , -vx , y , -vy , z , -vz ) = 010101 = 21  (time reversal)
  	( -x , -vx , y ,  vy , z ,  vz ) = 110000 = 48 (x-reflection)
  	( -x ,  vx , y , -vy , z , -vz ) = 100101 = 37 (time reversal + x-reflection)

and the mapping is applied by the following commands

	########### Mapping application ###########
 
		#variable map equal to a number from 0 to 63
  
		variable mpx equal     floor((${map})/(2^5))
		variable mvy equal     floor((${map}-(${mpx}*2^5))/(2^4))
		variable mpy equal     floor((${map}-(${mpx}*2^5)-(${mpy}*2^4))/(2^3))
		variable mvy equal     floor((${map}-(${mpx}*2^5)-(${mpy}*2^4)-(${mpz}*2^3))/(2^2))
		variable mpz equal     floor((${map}-(${mpx}*2^5)-(${mpy}*2^4)-(${mpz}*2^3)-(${mvx}*2^2))/(2^1))
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

where the first block of commands traslates back a decimal number into its binary representation and selects each separate digit, the second block calculates the corresponding reflected dcoordinate (1 inverts sign, 0 leaves unchanged), and the third block updates the positions and momenta.

The proposed examples produces both a profile quantity (associated to a specific point of the system) and global quantity (associated to the enire system), in order to provide a general scheme for TTCF caluclation. The profile variable is the velocity, whereas the global variables are the shear pressure and the dissipation function, respectively. In this script, the dissipation function must always be the last variable listed on the fin ave/time command. For the SLLOD equation, we have
```math
\Omega(t)=\dfrac{Vp_{xy}(t)}{k_B T} 
```
Note that the dissipation function at t=0 can be computed either from the mother or from the daughter trajectory. However, the LAMMPS implementation of the SLLOD algorithm contains various errors which result in a mismatch between the two calculations. However, the errors have minor effects of the final outcome. For simplicity, in the dissipation function is here monitored over the entire daughter trajectory. 
The same setup can be used with different systems, provided the various parameters, the dynamics, and the dissipation function are properly modified.
The Python interface described here aims at managing the entire LAMMPS simulation without the need to any output file for each nonequilibrium run. Since TTCF calculation requires thousands, or up to million nonequilibrium runs, the file management can become cumbersome and substantially decrease the performaces in HPC clusters.
The script uses the python LAMMPS interface (https://docs.lammps.org/Python_head.html), which allows to manage the LAMMPS run from python directly. As such, the otuput produced by the fix ave/time and fix ave/chuck commands are not written on file, but taken as input by Python.
The Python script run_TTCF.py is strictured as follows:

SPLIT THE SIMULATION INTO N SINGLE-CORE RUNS (N # OF CORES SELECTED) 
------

	comm = MPI.COMM_WORLD
	irank = comm.Get_rank()
	nprocs = comm.Get_size()
	t1 = MPI.Wtime()
	root = 0

DECLARATION OF VARIABLES AND STRUCTURES
------

Here all the parameters needed by the Python scripts are declared. 
These are the parameters requied by Python. The rest of the info about the simulation (temperature, density, etc) are stored in the LAMMPS input file which will be uploaded into a python object (see below).
If needed, these parameters will be passed to LAMMPS via proper functions.

   

	Tot_Daughters         = 100
	Maps                  = [0,21,48,37]
	Nsteps_Thermalization = 10000
	Nsteps_Decorrelation  = 10000
	Nsteps_Daughter       = 1000
	Delay                 = 10
	Nbins                 = 100
	dt                    = 0.0025


	Nmappings=len(Maps)
	Ndaughters=int(np.ceil(Tot_Daughters/nprocs))
	Nsteps=int(Nsteps_Daughter/Delay)+1
	Bin_Width=1.0/float(Nbins)
	Thermo_damp = 10*dt


The dynamics of the nonequilibrium daughter trajectory is then declared. The user should translate each LAMMPS command into a string which is then appendend to the block. The order identical to that of a LAMMPS script. The blocks are respectively: definition of the dynamics (apply ext. field, set dynamics), definition and caluclation of profile variables, definition and caluclation of global variables (including the dissipation function, which must be the last output quantity listed in the related command)

	setlist = []

	setlist.append("variable vx_shear atom vx+${srate}*y")
	setlist.append("set atom * vx v_vx_shear")
	setlist.append("fix box_deform all deform 1 xy erate ${srate} remap v units box")
	setlist.append("fix NVT_SLLOD all nvt/sllod temp ${T} ${T} " + str(Thermo_damp))
 
	profile_variables = ['vx']
	setlist.append("compute profile_layers all chunk/atom bin/1d y lower "+str(Bin_Width)+" units reduced")
	setlist.append("fix Profile_variables all ave/chunk 1 1 {} profile_layers {} ave one".format(Delay, ' '.join(profile_variables)))
	
	setlist.append("compute        shear_T all temp/deform")
	setlist.append("compute        shear_P all pressure shear_T ")
	setlist.append("variable       Omega equal -c_shear_P[4]*(xhi-xlo)*(yhi-ylo)*(zhi-zlo)*${srate}/(${k_B}*${T})")
	global_variables = ['c_shear_P[4]', 'v_Omega']
	setlist.append("fix Global_variables all ave/time 1 1 {} {} ave one".format(Delay, ' '.join(global_variables)))


	
CREATION OF LAMMPS OBJECT
------


This operation associates a LAMMPS input file to a LAMMPS object.
The file "System setup.in" contains only the declaration of the remaining parameters and the initialization of the system. The command line arguments are '-sc', 'none' (no video output),
'-log', 'none' (no log file), '-var', 'rand_seed' , seed_v (the random seed to initialize the velocities, different for each processor). The last command sets the timestep for the integration of the equations of motion. The parameter is declared in the script, and appended to the LAMMPS object via lmp.command()
      
	args = ['-sc', 'none','-log', 'none','-var', 'rand_seed' , seed_v]
	lmp = lammps(comm=MPI.COMM_SELF, cmdargs=args)
	L = PyLammps(ptr=lmp)
	nlmp = lmp.numpy
	lmp.file("system_setup.in")
	lmp.command("timestep " + str(dt))
 	lmp.command("timestep " + str(dt))


RUN THERMALIZATION
------

This block appends to the existing LAMMPS object (loaded from system_setup.in) the set of commands listed in the function, which represent the equilibrium dynamics (mother trajectory) of the system. Since multiple runs are perfomred, ant the end of each run all the fixes and computes must be discarded.


	run_mother_trajectory(lmp,Nsteps_Thermalization,Thermo_damp)

	def  run_mother_trajectory(lmp,Nsteps_Decorrelation,Thermo_damp):

		lmp.command("fix NVT_equilibrium all nvt temp ${T} ${T} " +  str(Thermo_damp) + " tchain 1")
    		lmp.command("run " + str(Nsteps_Decorrelation))
   		lmp.command("unfix NVT_equilibrium")

   		return None

LOOP OVER THE DAUGHTER TRAJECTORIES
------

This block appends to the existing LAMMPS input file the set of commands listed in the "Run equilibrium thermalization" block.
Before the decorrelation, the inital state is loaded, and after the decorrelation, a new initial state is produced. As the load state is called several times, the set of operations are stored in the separate file "load_state.lmp", and included in the script with the command include ./load_state.lmp.
In the last block, the structures where the ouptut will be saved are initialized to 0

	for Nd in range(Ndaughters):

		lmp.command("include ./load_state.lmp")
		lmp.command("fix NVT_sampling all nvt temp ${T} ${T} ${Thermo_damp} tchain 1")
		lmp.command("run " + str(Nsteps_Decorrelation))
		lmp.command("unfix NVT_sampling")
		lmp.command("fix snapshot all store/state 0 x y z vx vy vz")


   		DAV_profile_partial[:,:,:] = 0
    		DAV_global_partial[:,:]    = 0
        
    		integrand_profile_partial[:,:,:] = 0
    		integrand_global_partial[:,:]    = 0

LOOP OVER THE MAPPINGS
------

The first block of command set the proper mapping (selected by PYthon form the Maps list provided), load the initial state generated, apply the mapping (stored in the separate file "mapping.lmp"), and set the daughter dynamics (file "set_daughter.lmp"), that is, applying the external field and define the SLLOD dynamics

      for Nm in range(Nmappings):
              
        	lmp.command("variable map equal " + str(Maps[Nm]))
        	lmp.command("include ./load_state.lmp")
        	lmp.command("include ./mappings.lmp")
        	lmp.command("include ./set_daughter.lmp")
	 
The secon block defines the various computes and outputs. REMINDER: THE COMMANDS MUST MATCH THE ONES DECLARED AT THE BEGINNING OF THE PYTHON SCRIPT.

       	 	lmp.command(computestr)
        	lmp.command(profilestr)
        	lmp.command("compute	shear_T all temp/deform")     
        	lmp.command("compute    shear_P all pressure shear_T")
       	 	lmp.command("variable   Omega equal -c_shear_P[4]*(xhi-xlo)*(yhi-ylo)*(zhi-zlo)*${srate}/(${k_B}*${T})")
       	 	lmp.command(globalstr)
	  
The command "run 0 pre yes post yes" is necessary only for the SLLOD dynamics. It is normally redundant.

        	lmp.command("run 0 pre yes post yes")
	 
The next two commands take the output generated by the fix ave/time and fix ave/chunk and store it in the proper structure. The commands are nested in speficic function for clarity.
The Python routine to perfom this operation is "nlmp.extract_fix". After the first output, the dissipation function at t=0 is saved. REMINDER: THE DISSIPATION FUNCTION MUST ALWAYS BE THE LAST VARIABLE LISTED IN THE FIX AVE/TIME   

          	data_profile[0, :, :]= get_profiledata(profile_variables, Nbins)
        	data_global[0, :] = get_globaldata(global_variables)

  		omega = data_global[0, -1] 

The script now loops over the nonequilibrium trajectory timesteps. The ouptu is generated every N (Delay varaible) timesteps. Hence, LAMMPS runs for N steps, the output ic caluclated and stored, then the run is resumed for futher N steps. The run option "pre yes post no" can subtaintially speed up the simulation. They shoud however be used with caution, as the output might be corrupted if both options are set to "no"

    		for t in range(1 , Nsteps_eff , 1):
      
        		lmp.command("run " + str(Delay) + " pre yes post no")
			data_profile[t, :, :]= get_profiledata(profile_variables, Nbins)
        		data_global[t, :] = get_globaldata(global_variables)


The computes and fixes related to the nonequilibrium trajectory are deleted

        	lmp.command("unfix Profile_variables")
        	lmp.command("unfix Global_variables")
        	lmp.command("uncompute profile_layers")
       	 	lmp.command("uncompute shear_T")
        	lmp.command("uncompute shear_P")
       
        	lmp.command("include ./unset_daughter.lmp")

The output is summed to the previous mappings, and the TTCF integrand function updated.

        	DAV_profile_partial  += data_profile[:,:,:]
        	DAV_global_partial   += data_global[:,:]
        
        	integrand_profile_partial += data_profile[:,:,:]*omega
        	integrand_global_partial  += data_global[:,:]*omega

The integration of the correlation is performed. Since the numerical integration is a linear operation, the average can either be perfomred before or after the integration. For simplicity, here each single independent daughter (average over the 4 mappings) is integrated

     	TTCF_profile_partial = TTCF_integration_profile(integrand_profile_partial, dt*Delay, Nsteps_eff , Nbins, avechunk_ncol )
    	TTCF_global_partial  = TTCF_integration_global(integrand_global_partial , dt*Delay, Nsteps_eff , avetime_ncol )  
    
    
 The initial value is added
 
   	 TTCF_profile_partial += DAV_profile_partial[0,:,:]
   	 TTCF_global_partial  += DAV_global_partial[0,:]

And averages over the 4 mappings 

   	 DAV_profile_partial  /= Nmappings   
   	 DAV_global_partial   /= Nmappings 
   	 TTCF_profile_partial /= Nmappings   
   	 TTCF_global_partial  /= Nmappings 

The result is then used to update mean and variance. The algorithm to compute mean and avariance is the Welford algorithm, and it is defined as follows
```math
s^2_n= \dfrac{n-2}{n-1}s^2_{n-1}+\dfrac{(x_n-\bar{x}_{n-1})^2}{n}
```
```math
\bar{x}_n= \dfrac{n-1}{n}\bar{x}_{n-1}+\dfrac{x_n}{n}
```
	Count += 1

	if Count >1
		TTCF_profile_var= update_var(TTCF_profile_partial, TTCF_profile_mean, TTCF_profile_var, Count)      
		DAV_profile_var= update_var(DAV_profile_partial, DAV_profile_mean, DAV_profile_var, Count)
		TTCF_global_var= update_var(TTCF_global_partial, TTCF_global_mean, TTCF_global_var, Count)   
		DAV_global_var= update_var(DAV_global_partial, DAV_global_mean, DAV_global_var, Count)
      
    	TTCF_profile_mean= update_mean(TTCF_profile_partial, TTCF_profile_mean, Count)     
   	 DAV_profile_mean= update_mean(DAV_profile_partial, DAV_profile_mean, Count)
    	TTCF_global_mean= update_mean(TTCF_global_partial, TTCF_global_mean, Count)
    	DAV_global_mean= update_mean(DAV_global_partial, DAV_global_mean, Count)





FINALIZE THE SIMULATION
----
Here the releveant quantity is selected from the profile quantities (by default the fix ave/chuck command adds two futher info). If more than one variable is computed, e.g three, then the last three variables must be selected

	TTCF_profile_mean = TTCF_profile_mean[:,:,-1]
	DAV_profile_mean  = DAV_profile_mean[:,:,-1]

	TTCF_profile_var = TTCF_profile_var[:,:,-1]
	DAV_profile_var  = DAV_profile_var[:,:,-1]

The variance of the mean is computed (computed variance of the sample over the number of samples)

	TTCF_global_var/= float(Count)
	DAV_global_var /= float(Count)
	TTCF_profile_var /= float(Count)
	DAV_profile_var  /= float(Count)

Summed across the independent parallel runs

	TTCF_profile_mean_total = sum_over_MPI(TTCF_profile_mean, irank)
	DAV_profile_mean_total = sum_over_MPI(DAV_profile_mean, irank)
	TTCF_profile_var_total = sum_over_MPI(TTCF_profile_var, irank)
	DAV_profile_var_total = sum_over_MPI(DAV_profile_var, irank)

	TTCF_global_mean_total = sum_over_MPI(TTCF_global_mean, irank)
	DAV_global_mean_total = sum_over_MPI(DAV_global_mean, irank)
	TTCF_global_var_total = sum_over_MPI(TTCF_global_var, irank)
	DAV_global_var_total = sum_over_MPI(DAV_global_var, irank)

And normalized again over the number of runs. Finally, the standard error is computed as the square root of the variance of the mean.
	
	if irank == root:
    		TTCF_profile_mean_total = TTCF_profile_mean_total/float(nprocs)
    		DAV_profile_mean_total  = DAV_profile_mean_total/float(nprocs)
    		TTCF_profile_var_total  = TTCF_profile_var_total/float(nprocs)
    		DAV_profile_var_total   = DAV_profile_var_total/float(nprocs)
    
    		TTCF_global_mean_total = TTCF_global_mean_total/float(nprocs)
    		DAV_global_mean_total  = DAV_global_mean_total/float(nprocs)
    		TTCF_global_var_total  = TTCF_global_var_total/float(nprocs)
    		DAV_global_var_total   = DAV_global_var_total/float(nprocs)
    

    		TTCF_profile_SE_total  = np.sqrt(TTCF_profile_var_total)
    		DAV_profile_SE_total   = np.sqrt(DAV_profile_var_total)
    		TTCF_global_SE_total   = np.sqrt(TTCF_global_var_total)
    		DAV_global_SE_total    = np.sqrt(DAV_global_var_total)

The script plots the output using matplotlib, which should look as follows (note that due to random seed, the exact peaks might be different but trends should be the same),

![alt text](https://github.com/edwardsmith999/TTCF/blob/master/TTCF_vs_DAV_SLLOD.png)

The variables are then saved on file

    np.savetxt('profile_DAV.txt', DAV_profile_mean_total)
    np.savetxt('profile_TTCF.txt', TTCF_profile_mean_total)
    
    np.savetxt('profile_DAV_SE.txt', DAV_profile_SE_total)
    np.savetxt('profile_TTCF_SE.txt', TTCF_profile_SE_total)
    
    np.savetxt('global_DAV.txt', DAV_global_mean_total)
    np.savetxt('global_TTCF.txt', TTCF_global_mean_total)
    
    np.savetxt('global_DAV_SE.txt', DAV_global_SE_total)
    np.savetxt('global_TTCF_SE.txt', TTCF_global_SE_total)




