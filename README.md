Transient Time Correlation Function (TTCF) [![CI](https://github.com/edwardsmith999/TTCF/actions/workflows/main.yml/badge.svg)](https://github.com/edwardsmith999/TTCF/actions/workflows/main.yml) 
==========================================

The code provides a user-friendly interface to implement the Transient Time Correlation Function (TTCF) method in the [LAMMPS](https://www.lammps.org/). This provides better signal-to-noise ratios from non-equilibrium molecular dynamics (NEMD) simulations where the forcing is very weak. For example, the low shear rates used in tribology or rheology experiments or the small pressure gradients driving flow in fluid dynamics experiments, which are innaccesible to standard NEMD simulations. TTCF is a generalisation of the popular Green-Kubo method for systems driven out of equilbrium.

Citiations
----------

Please cite the following if you use this code for your own research: 
1. L. Maffioli, J. P. Ewen, E. R. Smith, S. Varghese, P. J. Daivis, D. Dini, and B. D. Todd. TTCF4LAMMPS: A toolkit for simulation of the non-equilibrium behaviour of molecular fluids at experimentally accessible shear rates. Computer Physics Communications 109205 (2024). https://doi.org/10.1016/j.cpc.2024.109205
2. L. Maffioli, E. R. Smith, J. P. Ewen, P. J. Daivis, D. Dini and B. D. Todd. Slip and Stress From Low Strain-Rate Nonequilibrium Molecular Dynamics: The Transient-Time Correlation Function Technique. The Journal of Chemical Physics 156, 184111 (2022). https://doi.org/10.1063/5.0088127 - if you use/modify the confined system example, or use TTCF to measure slip lengths or for barostatted systems at constant pressure
3. I. Borzsak, P. T. Cummings, D. J. Evans. Shear viscosity of a simple fluid over a wide range of strain rates. Molecular Physics, 100, 2735-2738 (2002). https://doi.org/10.1080/00268970210137275 - if you use/modify the bulk system example.

Quickstart
----------

The quickest way to try out TTCF4LAMMPS is to download using the Python package manager,

    pip install TTCF4LAMMPS

This is built on the unofficial Python packaged version of LAMMPS (https://pypi.org/project/lammps/), which uses mpich to provide parallisation and requires Python 3.7 or later.
This will probably be fine for most users as it will be fairly up to date and should run with your own input files. However, if you are working on a supercomputer, have a version of LAMMPS already installed with Python bindings, or you have developed custom modules, then the Python interface for LAMMPS (https://docs.lammps.org/Python_module.html) can be installed manually. Installation instructions are provided on the LAMMPS page for Windows, Linux and Mac here: https://docs.lammps.org/Python_install.html.
After installation by `pip` or manually, to check the Python LAMMPS interfaces works as expected, open Python and try to import the lammps module.

    python

    >>> import lammps
    >>> lmp = lammps.lammps()
    
Then, TTCF4LAMMPS is simply a package, so you can clone the repository,

    git clone https://github.com/edwardsmith999/TTCF.git

install any missing prerequisite from,

    mpi4py
    numpy
    matplotlib

and ensure you can import TTCF, either `sys.path.append` to include the path to the TTCF folder or running

    python setup.py install
    
and check TTCF works with

    from TTCF import TTCF
    test = TTCF.TTCF(global_variables=[], profile_variables=['vx'], Nsteps=51, Nbins=100, Nmappings=4)

To use the TTCF, start by running the example case, navigate to the TTCF folder (use `git clone https://github.com/edwardsmith999/TTCF.git TTCF` if you haven't already) and run,

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

The average is performed over nonequilibrium trajectories initial conditions sampled from the equilibrium ensemble associated with the system. The easiest way to achieve this is to evolve the system over an equilibrium *mother* trajectory. After a thermalization to ensure the system is in thermodynamic equilibrium, the state of the system (set of all positions and momenta) is periodically sampled. The procedure is shown is the figure below

![alt text](https://github.com/edwardsmith999/TTCF/blob/master/figures/mother.png)

At each sample point on the mother, a series of nonequilibrium *daughter* runs are perfomed, where their initial conditions are the states sampled from the mother trajectories. 

![alt text](https://github.com/edwardsmith999/TTCF/blob/master/figures/children.png)

In the figure, each initial state and a mirror is generated and run with an applied thermodynamic force (e.g. shared walls, SLLOD forcing, temperature gradient).
In the example above, the original trajectory and three further mirrored states are generated. These 4 initial states guarantee that the phase average of the dissipation function is identically null and hence the convergence of the integral is ensured. The following mappings are used in this script

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


The implementation can be found in the file LAMMPS_script.in. The file is equivalent to the python implemenetation which will be described later and can be directly used for MD simulations. However, it generates two output files for each daughter trajectory, making it impractical for realistic calculations. It is here used just as a reference to better understand the rationale behing the python interface. The entire TTCF calculation is here performed via a single run. The systems repeatedly switches between equilibrium (mother) and nonequilibrium (daughter) trajectory as shown in the above pseudocode. Each block of commands is quite straightforward. However, the generation and loading of the sample from the mother trajectory needs to be clarified.
In order ot avoid writing/reading from files, the instantaneous state of the system is stored via the following command, which temporarily save the variables specified (positions and momenta) in the structure called a snapshot in LAMMPS. This is not written to disk but stored in an array in memory (RAM),

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

And each fix nvt command should have the same ID throughout the whole run, both for the mother and the daughter trajectory.
The second point which needs clarification is the mapping procedure. It  works is the following way: in order to keep a general algorithm, each single dimension can be independenlty mirrored. There are 6 total dimensions, namely x,vx,y,vy,z,vz. Thus, a mapping can be identified by a set of six digits, each of which can be either 0 (no reflection) or 1 (reflection). For instance, the sequence 101100 identifies the following mapping

	(101100) = (-x , vx , -y , -vy , z , vz )
 
There are a total of 2^6 independent mappings and the string corresponding to the selected mapping can be translated into a number from 0 to 63 by simply converting the string from a binary to a decimal number. The mappings selected here are 

	(  x ,  vx , y ,  vy , z ,  vz ) = 000000 = 0  (original state)
 	(  x , -vx , y , -vy , z , -vz ) = 010101 = 21  (time reversal)
  	( -x , -vx , y ,  vy , z ,  vz ) = 110000 = 48 (x-reflection)
  	( -x ,  vx , y , -vy , z , -vz ) = 100101 = 37 (time reversal + x-reflection)

and the mapping is applied in LAMMPS by the following commands

	########### Mapping application ###########
		set             atom * x  v_px 
		set             atom * y  v_py 
		set             atom * z  v_pz 
		set             atom * vx v_vx 
		set             atom * vy v_vy 
		set             atom * vz v_vz

which updates the positions and momenta.

The proposed examples produces both a profile quantity (associated to a specific point of the system) and global quantity (associated to the enire system), in order to provide a general scheme for TTCF caluclation. The profile variable is the velocity, whereas the global variables are the shear pressure and the dissipation function, respectively. In this script, the dissipation function must always be the last variable listed on the fin ave/time command. For the SLLOD equation, we have
```math
\Omega(t)=\dfrac{Vp_{xy}(t)}{k_B T} 
```
Note that the dissipation function at t=0 can be computed either from the mother or from the daughter trajectory. However, the LAMMPS implementation of the SLLOD algorithm contains various errors which result in a mismatch between the two calculations. However, the errors have minor effects of the final outcome. For simplicity, in the dissipation function is here monitored over the entire daughter trajectory. 
The same setup can be used with different systems, provided the various parameters, the dynamics, and the dissipation function are properly modified.


The Python interface described here aims at managing the entire LAMMPS simulation without the need to any output file for each nonequilibrium run. Since TTCF calculation requires thousands, or up to million nonequilibrium runs, the file management can become cumbersome and substantially decrease the performaces in HPC clusters.
The script uses the python LAMMPS interface (https://docs.lammps.org/Python_head.html), which allows to manage the LAMMPS run from python directly. As such, the otuput produced by the fix ave/time and fix ave/chuck commands are not written on file, but taken as input by Python.
The Python script run_TTCF.py is structured as follows:

HOW THE PARALLELISM WORKS - SPLIT THE SIMULATION INTO N SINGLE-CORE RUNS (N # OF CORES SELECTED) 
------
Each core is assigned to a LAMMPS run. of N cores, N independent mother trajectories are generated, as well as the releted daughter trajectories.

	comm = MPI.COMM_WORLD
	irank = comm.Get_rank()
	nprocs = comm.Get_size()
	t1 = MPI.Wtime()
	root = 0

 Each run is characterized by a different randon seed, based on the MPI processor rank (called `irank`) which is used to randomize the inital momenta (see below).
 
	np.random.seed(irank)
	seed_v = str(int(np.random.randint(1, 1e5 + 1)))

	#Define LAMMPS object and initialise
	args = ['-sc', 'none','-log', 'none','-var', 'rand_seed' , seed_v]
	lmp = lammps(comm=MPI.COMM_SELF, cmdargs=args)

This is shown graphically for two processors here,

![alt text](https://github.com/edwardsmith999/TTCF/blob/master/figures/children_MPI.png)

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


The dynamics of the nonequilibrium daughter trajectory is then declared. The user should translate each LAMMPS command into a string which is then appendend to the block. The order appended to the Python list must be identical to that of a LAMMPS script. The blocks are respectively: definition of the dynamics (apply ext. field, set dynamics), definition and caluclation of profile variables, definition and caluclation of global variables (including the dissipation function, which must be the last output quantity listed in the related command)

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
The file "System setup.in" contains only the declaration of the remaining parameters and the initialization of the system. The command line arguments are '-sc', 'none' (no standard output to terminal), '-log', 'none' (no log file), '-var', 'rand_seed' , seed_v (the random seed to initialize the velocities, different for each processor). The last command sets the timestep for the integration of the equations of motion. The parameter is declared in the script, and appended to the LAMMPS object via lmp.command(). IMPORTANT: if multiple, independent, single-core instances of this Python script are launched, then the random seed should be carefully chosen. Best method would be to read it from a file, where all the seeds required are stored.
      
	args = ['-sc', 'none','-log', 'none','-var', 'rand_seed' , seed_v]
	lmp = lammps(comm=MPI.COMM_SELF, cmdargs=args)
	L = PyLammps(ptr=lmp)
	nlmp = lmp.numpy
	lmp.file("system_setup.in")
	lmp.command("timestep " + str(dt))
 	
RUN THERMALIZATION
------

This block appends to the existing LAMMPS object (loaded from system_setup.in) the set of commands listed in the function, which represent the equilibrium dynamics (mother trajectory) of the system. Since multiple runs are perfrmed, ant the end of each run all the fixes and computes must be discarded. At the end of the equilibrium run, the state of the system is saved 


	run_mother_trajectory(lmp,Nsteps_Thermalization,Thermo_damp)

	def  run_mother_trajectory(lmp,Nsteps_Decorrelation,Thermo_damp):

		lmp.command("fix NVT_equilibrium all nvt temp ${T} ${T} " +  str(Thermo_damp) + " tchain 1")
    		lmp.command("run " + str(Nsteps_Decorrelation))
   		lmp.command("unfix NVT_equilibrium")

   		return None
     
At the end of the equilibrium run, the state of the system is saved via the following

	state = save_state(lmp, "snapshot")

	def save_state(lmp, statename, save_variables=["x", "y", "z", "vx", "vy", "vz"]):

    		state = {}
    		state['name'] = statename
    		state['save_variables'] = save_variables
    		cmdstr = "fix " + statename + " all store/state 0 {}".format(' '.join(save_variables))
    		lmp.command(cmdstr)

    		return state

LOOP OVER THE DAUGHTER TRAJECTORIES
------

The script loop over the number of daughter trajectories (excluding the mappings). At each step, the last saved state of the system is loaded, 


	load_state(lmp, state)
	def load_state(lmp, state):
    
    		cmdstr = "change_box all  xy final 0\n"
    		for i, s in enumerate(state['save_variables']):
       		varname = "p"+s 
       	 	cmdstr += "variable " + varname + " atom f_"+state['name']+"["+str(i+1)+"]\n"
        	cmdstr += "set             atom * " + s + " v_"+varname+"\n"

    		for line in cmdstr.split("\n"):
        		lmp.command(line)
    		return None

and the equilibrium run is carried on, until the system is fully decorrelated from the last saved state, after which a new state owerwrite the exisiting saved one.

	run_mother_trajectory(lmp,Nsteps_Decorrelation,Thermo_damp)
   	state = save_state(lmp, "snapshot")

LOOP OVER THE MAPPINGS
------

Each step repesent a single mapped daughter trajectory. The last generated sample is first loaded, and then modified accordingly to the current mapping. The conversion from deciaml number to six-digits binary string is performed direcly by Python, unlike in the LAMMPS script example.

	load_state(lmp, state)
 	apply_mapping(lmp, Maps[Nm])

  	def apply_mapping(lmp, map_index):

    		map_list=["x","y","z"]
    		N=len(map_list)
    
    		ind=map_index
     
    		cmdstr=""
    		for i in range(N):

       		mp = ind % 2
       		ind = np.floor(ind/2)
    
        	cmdstr += "variable map atom  v"+map_list[N-1-i]+"-(2*v"+map_list[N-1-i]+"*"+str(mp)+")\n"
        	cmdstr += "set atom * v"+map_list[N-1-i]+" v_map\n"


       	 	mp=ind % 2
        	ind = np.floor(ind/2)

        	cmdstr += "variable map atom  "+map_list[N-1-i]+"+(("+map_list[N-1-i]+"hi-2*"+map_list[N-1-i]+")*"+str(mp)+")\n"
        	cmdstr += "set atom * "+map_list[N-1-i]+" v_map\n"

    		for line in cmdstr.split("\n"):
        		lmp.command(line)

    		return None
	 


The set of commands related to the daughter trajectory and declared in the first section of the script are then loaded and discarded at the end of the daughter run

	set_list(lmp, setlist)
 	unset_list(lmp, setlist)

During the daughter run, the output quantities are repeatedly accessed and stored via the PyLAMMPS built-in function "nlmp.extract_fix" (called here within the Python function "get_fix_data"). The function must be called the precise timestep the output is produced. A loop cycles over the lenght of the simulation ot extract the output with the specified frequency. The total run is hence fragmented in a series of short run between outputs. The option " pre yes post no" can significanlty improve the performaces, but should be carefully tested, as it can impact the produced output. The first command "run 0" is here required to trigger the box deformation induced by SLLOD dynamics. It is likely an unintended effect, and can be removed for different systems, or different LAMMPS versions.

	lmp.command("run 0 pre yes post yes")
	data_profile[0, :, :]= get_fix_data(lmp, "Profile_variables", profile_variables, Nbins)
        data_global[0, :] = get_fix_data(lmp, "Global_variables", global_variables)
        omega = data_global[0, -1] 

    
        for t in range(1, Nsteps):
            lmp.command("run " + str(Delay) + " pre yes post no")
            data_profile[t, :, :]= get_fix_data(lmp, "Profile_variables", profile_variables, Nbins)
            data_global[t, :] = get_fix_data(lmp, "Global_variables", global_variables)
	    
The last commands adds the produced output with the ones generated within the same initial state, 

	ttcf.add_mappings(data_profile, data_global, omega)
And the average over the four mappings is then integrated once the loop over the mappings has been performed. The phase average and the integration can be swapped since both are linear operators. The integration uses a second order Simpson method. 

	ttcf.integrate(dt*Delay)


 Which is defined as follows,

 	def integrate(self, step):

        	#Perform the integration
        	self.TTCF_profile_partial = TTCF_integration(self.integrand_profile_partial, step)
        	self.TTCF_global_partial  = TTCF_integration(self.integrand_global_partial, step)

        	#Add the initial value (t=0) 
        	self.TTCF_profile_partial += self.DAV_profile_partial[0,:,:]
        	self.TTCF_global_partial  += self.DAV_global_partial[0,:]

        	#Average over the mappings and update the Count (# of children trajectories generated excluding the mappings)
        	self.DAV_profile_partial  /= self.Nmappings   
        	self.DAV_global_partial   /= self.Nmappings 
        	self.TTCF_profile_partial /= self.Nmappings   
        	self.TTCF_global_partial  /= self.Nmappings 

        	self.Count += 1

        	if self.Count >1:
        
            		self.TTCF_profile_var= update_var(self.TTCF_profile_partial, self.TTCF_profile_mean, self.TTCF_profile_var, self.Count)      
            		self.DAV_profile_var= update_var(self.DAV_profile_partial, self.DAV_profile_mean, self.DAV_profile_var, self.Count)
            		self.TTCF_global_var= update_var(self.TTCF_global_partial, self.TTCF_global_mean, self.TTCF_global_var, self.Count)   
            		self.DAV_global_var= update_var(self.DAV_global_partial, self.DAV_global_mean, self.DAV_global_var, self.Count)
          
        	self.TTCF_profile_mean= update_mean(self.TTCF_profile_partial, self.TTCF_profile_mean, self.Count)     
        	self.DAV_profile_mean= update_mean(self.DAV_profile_partial, self.DAV_profile_mean, self.Count)
        	self.TTCF_global_mean= update_mean(self.TTCF_global_partial, self.TTCF_global_mean, self.Count)
        	self.DAV_global_mean= update_mean(self.DAV_global_partial, self.DAV_global_mean, self.Count)

        	self.DAV_profile_partial[:,:,:] = 0
        	self.DAV_global_partial[:,:]    = 0
            
        	self.integrand_profile_partial[:,:,:] = 0
        	self.integrand_global_partial[:,:]    = 0
	 
After the integration has been performed, the results is used to update the total mean and variance. The two quantities can be update using the one-passage Welford algorithm.
```math
s^2_n= \dfrac{(n-2)s^2_{n-1}+(x_n-\bar{x}_{n-1})(x_n-\bar{x}_{n})}{n-1}
```
```math
\bar{x}_n= \dfrac{(n-1)\bar{x}_{n-1}+x_n}{n}
```
Based on the above formula, the variance must always be computed starting from the second element of the sequence.

After the final process, the cycles starts over again from the last generated sample.

FINALIZE THE SIMULATION
----

Once all the trajectories have been generated. The script loops over the processors and averages the results. 

	ttcf.finalise_output(irank, comm)
 The final output is the mean of the desired quantities and their standard error (SE). IMPORTANT: if each process has a single daughter, then the variance of the process is zero, and the total variance is 0 likewise. The fianl standard error is correctly computed ONLY IF there are 2 or more daughters in each process.
 
