#!/usr/bin/python3
import numpy as np
from lammps import lammps
from mpi4py import MPI
import time

from TTCF import utils
from TTCF import TTCF
from TTCF import bootstrapping as Bts

#This code is run using MPI - each processes will
#run this same bit code with its own memory
comm = MPI.COMM_WORLD
irank = comm.Get_rank()
nprocs = comm.Get_size()
t1 = MPI.Wtime()
root = 0
print("Proc {:d} out of {:d} procs".format(irank+1,nprocs), flush=True)

#Define lengths for all runs, number of Daughters, etc
Tot_Daughters         = 50000
Maps                  = [0]
Nsteps_Thermalization = 10000
Nsteps_Decorrelation  = 1000
Nsteps_Daughter       = 600
Delay                 = 1
Nbins                 = 50
dt                    = 0.0025
showplots             = False

# Define parameters for bootstrapping
totResamples = 10000
nResamples = int(np.ceil(totResamples/nprocs))
nGroups = nResamples

#Set the the GPU force calculation (1 GPU calculation, 0 no use of GPU).
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
profile_variables = ['vx', 'c_stress[4]']
#Define bin discretization
setlist.append("compute profile_layers all chunk/atom bin/1d y lower "+str(Bin_Width)+" units reduced")
#Define computes to get Omega
setlist.append("compute        shear_T all temp/deform")
setlist.append("compute        shear_P all pressure shear_T ")
setlist.append("variable       Omega equal -c_shear_P[4]*(xhi-xlo)*(yhi-ylo)*(zhi-zlo)*${srate}/(${k_B}*${T})")
#Profile (ave/chunk fix)
setlist.append("compute        stress all stress/atom shear_T")
setlist.append("fix Profile_variables all ave/chunk 1 1 {} profile_layers {} ave one".format(Delay, ' '.join(profile_variables)))
#And global (ave/time fix) variables, often custom computes/variables, see https://docs.lammps.org/fix_ave_time.html
global_variables = ['c_shear_P[4]', 'v_Omega']
bts_variables = ['c_shear_P[4]']
#global ave/time fix
setlist.append("fix Global_variables all ave/time 1 1 {} {} ave one".format(Delay, ' '.join(global_variables)))


#Create TTCF class 
ttcf = TTCF.TTCFnoMap(global_variables, profile_variables, Nsteps, Nbins, Nmappings)
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

#Run equilibration  
lmp.file("system_setup.in")
lmp.command("timestep " + str(dt))
utils.run_mother_trajectory(lmp,Nsteps_Thermalization,Thermo_damp)

#Save snapshot to use for daughters
state = utils.save_state(lmp, "snapshot")

# Create directories to save global variables for all trajectories
ttcf.createDirectories(irank)
comm.Barrier() # Maybe to remove

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

        #Run over time        
        for t in range(1, Nsteps):
            lmp.command("run " + str(Delay) + " pre yes post yes")
            data_profile[t, :, :] = utils.get_fix_data(lmp, "Profile_variables", profile_variables, Nbins)
            data_global[t, :] = utils.get_fix_data(lmp, "Global_variables", global_variables)

        #Turn off forces and outputs
        utils.unset_list(lmp, setlist)

        #Sum the mappings together
        ttcf.add_mappings(data_profile, data_global)

    #Perform the integration
    ttcf.integrate(dt*Delay)
    ttcf.writeTrajectories(irank)
    ttcf.updateMeanVar()

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

comm.Barrier()

t3 = MPI.Wtime()

for i in range(len(bts_variables)):
    bts = Bts.Bootstrap(variable=bts_variables[i], nResamples=nResamples,
                        nTrajectories=Tot_Daughters, nGroups=nGroups,
                        nTimesteps=Nsteps_Daughter)
    bts.concatenateTrajectories(comm)
    print('{}: Trajectories concatenated'.format(variables[i]))
    comm.Barrier()
    bts.readTrajectories(comm)
    print('Proc {}, {}: Trajectories read'.format(irank+1, variables[i]))
    comm.Barrier()
    bts.groupTrajectories()
    print('Proc {}, {}: Trajectories grouped'.format(irank+1, variables[i]))
    bts.averageWithin()
    print('Proc {}, {}: Trajectories averaged within'.format(irank+1, variables[i]))
    bts.resampleAvergeBetweenForMean()
    print('Proc {}, {}: Trajectories resampled and averaged between for the mean'.format(irank+1, variables[i]))
    bts.resampleAvergeBetweenForStd()
    print('Proc {}, {}: Trajectories resampled and averaged between for the standard deviation'.format(irank+1, variables[i]))
    bts.sumIntegrals()
    print('Proc {}, {}: B(t) mean and standard deviation created'.format(irank+1, variables[i]))
    
    comm.Barrier()
    
    bts.gather_over_MPI(comm, root=0)
    print('{}: mean and stardard deviation gathered by root'.format(variables[i]))
    if irank == 0:
        bts.meanForComparison()
        print('{}: Mean and its standard deviation computed'.format(variables[i]))
        bts.standardDeviation('')
        print('{}: Standard deviation computed'.format(variables[i]))
        bts.confidenceInterval()
        print('{}: Confidence interval saved'.format(variables[i]))
        bts.plotDistribution()
        print('{}: Distribution plot'.format(variables[i]))

t4 = MPI.Wtime()
if irank == root:
    print("Bootstrapping time =", t4 - t3, flush=True)
MPI.Finalize()
