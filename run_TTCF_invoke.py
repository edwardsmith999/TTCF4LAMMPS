#!/usr/bin/python3
import numpy as np
from lammps import lammps
from mpi4py import MPI

from TTCF import utils
from TTCF import TTCF

#This code is run using MPI - each processes will
#run this same bit code with its own memory
comm = MPI.COMM_WORLD
irank = comm.Get_rank()
nprocs = comm.Get_size()
t1 = MPI.Wtime()
root = 0
print("Proc {:d} out of {:d} procs".format(irank+1,nprocs), flush=True)

#Define lengths for all runs, number of Daughters, etc
Tot_Daughters         = 1000
Maps                  = [0,21,48,37]
Nsteps_Thermalization = 10000
Nsteps_Decorrelation  = 10000
Nsteps_Daughter       = 500
Delay                 = 10
Nbins                 = 100
dt                    = 0.0025
showplots             = True

#Set the the GPU forcde calculation (1 GPU calculation, 0 no use of GPU).
#IMPORTANT: THE GPU PACKAGE REQUIRES ONE TO INSERT THE SPECIFIC GPU MODEL. PLEASE EDIT THE COMMAND ACCORDINGLY.
Use_GPU = 0

#Derived quantities
Nmappings=len(Maps)
Ndaughters=int(np.ceil(Tot_Daughters/nprocs))
Nsteps=int(Nsteps_Daughter/Delay)+1
Bin_Width=1.0/float(Nbins)
Thermo_damp = 10*dt

if Ndaughters<3:
    print("Number of daughter trajectories must be larger than 2")
    comm.Abort()

# Here we can define all variables, computes and fixs in lammps 
#format that we want to set for each daughter
# and turn off at the end
setlist = []
# =============== Forces =========================
#For the case of SLLOD these apply the forces
setlist.append("variable vx_shear atom vx+${srate}*y")
setlist.append("set atom * vx v_vx_shear")
setlist.append("fix box_deform all deform 1 xy erate ${srate} remap v units box")
setlist.append("fix NVT_SLLOD all nvt/sllod temp ${T} ${T} " + str(Thermo_damp))
# ============= Outputs =========================
#Here we define all the computes to get outputs from the simulation
#Profile 1D chunks here, can add anything from https://docs.lammps.org/fix_ave_chunk.html
profile_variables = ['vx']
#Define bin discretization
setlist.append("compute profile_layers all chunk/atom bin/1d y lower "+str(Bin_Width)+" units reduced")
#Profile (ave/chunk fix)
setlist.append("fix Profile_variables all ave/chunk 1 1 {} profile_layers {} ave one".format(Delay, ' '.join(profile_variables)))
#Define computes to get Omega
setlist.append("compute        shear_T all temp/deform")
setlist.append("compute        shear_P all pressure shear_T ")
setlist.append("variable       Omega equal -c_shear_P[4]*(xhi-xlo)*(yhi-ylo)*(zhi-zlo)*${srate}/(${k_B}*${T})")
#And global (ave/time fix) variables, often custom computes/variables, see https://docs.lammps.org/fix_ave_time.html
global_variables = ['c_shear_P[4]', 'v_Omega']
#global ave/time fix
setlist.append("fix Global_variables all ave/time 1 1 {} {} ave one".format(Delay, ' '.join(global_variables)))


#Create TTCF class 
ttcf = TTCF.TTCF(global_variables, profile_variables, Nsteps, Nbins, Nmappings)
data_global = np.zeros([Nsteps, len(global_variables)])
data_profile  = np.zeros([Nsteps, Nbins, len(profile_variables) + 2])

#Create random seed
if nprocs == 1:
    seed_v = str(12345)
else:
    np.random.seed(irank)
    seed_v = str(int(np.random.randint(1, 1e5 + 1)))

#Define LAMMPS object and initialise
args = ['-sc', 'none','-log', 'none','-var', 'rand_seed' , seed_v ,'-var', 'use_gpu', str(Use_GPU) ]
lmp = lammps(comm=MPI.COMM_SELF, cmdargs=args)

#Define a function
def collect_data(lammps_ptr):
    #We can't pass integers to LAMMPS fns but lists seem fine so use as counter
    t = len(count); count.append(1)
    lmp = lammps(ptr=lammps_ptr)
    data_profile[t, :, :] = utils.get_fix_data(lmp, "Profile_variables", profile_variables, Nbins)
    data_global[t, :] = utils.get_fix_data(lmp, "Global_variables", global_variables)
    
#Run equilibration  
lmp.file("system_setup.in")
lmp.command("timestep " + str(dt))
utils.run_mother_trajectory(lmp,Nsteps_Thermalization,Thermo_damp)

#Save snapshot to use for daughters
state = utils.save_state(lmp, "snapshot")

#Loop over all sets of daughters
for Nd in range(Ndaughters):

    print("Proc", irank+1, " with daughter =", Nd+1, " of ", Ndaughters,  flush=True)

    #Run mother starting from previous sample to generate the next sample
    utils.load_state(lmp, state)
    utils.run_mother_trajectory(lmp,Nsteps_Decorrelation,Thermo_damp)
    state = utils.save_state(lmp, "snapshot")

    #Branch off daughters for each mapping
    for Nm in range(Nmappings):

        #Load child state
        utils.load_state(lmp, state)

        #Apply mapping    
        utils.apply_mapping(lmp, Maps[Nm])

        #Apply forces and setup outputs
        utils.set_list(lmp, setlist)

        #Run zero to setup case
        lmp.command("run 0 pre yes post yes")

        #Extract profile and time averaged (global) data from LAMMPS
        data_profile[0, :, :] = utils.get_fix_data(lmp, "Profile_variables", profile_variables, Nbins)
        data_global[0, :] = utils.get_fix_data(lmp, "Global_variables", global_variables)
        omega = data_global[0, -1] 

        #Run over time collecting data using a python invoke called every Delay timesteps
        lmp.command('fix collect all python/invoke ' + str(Delay) +
                    ' end_of_step collect_data')
        count = [1] #A list used to keep track of number of records
        lmp.command("run " + str(Delay*(Nsteps-1)) + " pre yes post yes")
        lmp.command("unfix collect")

        #Turn off forces and outputs
        utils.unset_list(lmp, setlist)

        #Sum the mappings together
        ttcf.add_mappings(data_profile, data_global, omega)

    #Perform the integration
    ttcf.integrate(dt*Delay)

#Close lammps instance and plot time taken
lmp.close()
comm.Barrier()
t2 = MPI.Wtime()
if irank == root:
    print("Walltime =", t2 - t1, flush=True)

#This includes an MPI call so must be used by all processes
ttcf.finalise_output(irank, comm)

#Plot and output data
if showplots:
    ttcf.plot_data()
ttcf.save_data()

MPI.Finalize()
