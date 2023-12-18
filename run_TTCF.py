from mpi4py import MPI
from lammps import lammps
from lammps import LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR, LMP_TYPE_ARRAY

import matplotlib.pyplot as plt
import numpy as np

from utils import *

#This code is run using MPI - each processes will
#run this same bit code with its own memory
comm = MPI.COMM_WORLD
irank = comm.Get_rank()
nprocs = comm.Get_size()
t1 = MPI.Wtime()
root = 0
print("Proc {:d} out of {:d} procs".format(irank+1,nprocs), flush=True)

#Define lengths for all runs, number of Daughters, etc

Tot_Daughters= 1000
Ndaughters=int(np.ceil(Tot_Daughters/nprocs))

Maps=[0,7,36,35]
Nmappings=len(Maps)

Nsteps_Thermalization = 10000
Nsteps_Decorrelation  = 10000
Nsteps_Daughter       = 500

Delay=10
Nsteps_eff=int(Nsteps_Daughter/Delay)+1

Nbins=100
Bin_Width=1.0/float(Nbins)

dt = 0.0025

#Define profile quantities to compute
profile_variables = ['vx']
#Define bin discretization
computestr = "compute profile_layers all chunk/atom bin/1d y lower "+str(Bin_Width)+" units reduced"
#Profile (ave/chunk fix)
profilestr = "fix Profile_variables all ave/chunk 1 1 {} profile_layers {} ave one".format(Delay, ' '.join(profile_variables))
global_variables = ['c_shear_P[4]', 'v_Omega']
#And global (ave/time fix)
globalstr = "fix Global_variables all ave/time 1 1 {} {} ave one".format(Delay, ' '.join(global_variables))

avetime_ncol = len(global_variables)
avechunk_ncol = len(profile_variables) + 2

#Allocate empty arrays for data
DAV_global_mean  = np.zeros([Nsteps_eff, avetime_ncol])
DAV_profile_mean   = np.zeros([Nsteps_eff, Nbins, avechunk_ncol])

TTCF_global_mean  = np.zeros([Nsteps_eff, avetime_ncol])
TTCF_profile_mean   = np.zeros([Nsteps_eff, Nbins, avechunk_ncol])

DAV_global_var  = np.zeros([Nsteps_eff, avetime_ncol])
DAV_profile_var   = np.zeros([Nsteps_eff, Nbins, avechunk_ncol])

TTCF_global_var  = np.zeros([Nsteps_eff, avetime_ncol])
TTCF_profile_var   = np.zeros([Nsteps_eff, Nbins, avechunk_ncol])

data_global = np.zeros([Nsteps_eff, avetime_ncol])
data_profile  = np.zeros([Nsteps_eff, Nbins, avechunk_ncol])

TTCF_global_partial = np.zeros([Nsteps_eff, avetime_ncol])
TTCF_profile_partial= np.zeros([Nsteps_eff, Nbins, avechunk_ncol])

DAV_profile_partial= np.zeros([Nsteps_eff, Nbins, avechunk_ncol])
DAV_global_partial = np.zeros([Nsteps_eff, avetime_ncol])
        
integrand_profile_partial = np.zeros([Nsteps_eff, Nbins, avechunk_ncol])
integrand_global_partial  = np.zeros([Nsteps_eff, avetime_ncol])

#Create random seed
if nprocs == 1:
    seed_v = str(12345)
else:
    np.random.seed(irank)
    seed_v = str(int(np.random.randint(1, 1e5 + 1)))

#Define LAMMPS object and initialise
args = ['-sc', 'none','-log', 'none','-var', 'rand_seed' , seed_v]
lmp = lammps(comm=MPI.COMM_SELF, cmdargs=args)

#Run equilibration  
lmp.file("system_setup.in")
lmp.command("timestep " + str(dt))
lmp.command("variable Thermo_damp equal " +  str(10*dt))
lmp.command(" fix NVT_thermalization all nvt temp ${T} ${T} ${Thermo_damp} tchain 1")
lmp.command("run " + str(Nsteps_Thermalization))
lmp.command("unfix NVT_thermalization")

#Save snapshot to use for daughters
state = save_state(lmp, "snapshot")

Count = 0
for Nd in range(1,Ndaughters+1,1):

    print("Proc", irank+1, " with daughter =", Nd, " of ", Ndaughters,  flush=True)

    #Sampling of the daughters initial state
    load_state(lmp, state)
    lmp.command("fix NVT_sampling all nvt temp ${T} ${T} ${Thermo_damp} tchain 1")
    lmp.command("run " + str(Nsteps_Decorrelation))
    lmp.command("unfix NVT_sampling")
    state = save_state(lmp, "snapshot")

    DAV_profile_partial[:,:,:] = 0
    DAV_global_partial[:,:]    = 0
        
    integrand_profile_partial[:,:,:] = 0
    integrand_global_partial[:,:]    = 0

    for Nm in range(Nmappings):

        #Load child state
        load_state(lmp, state)

        #Apply mapping    
        lmp.command("variable map equal " + str(Maps[Nm]))
        lmp.command("variable Daughter_index equal " + str(Nd))
        lmp.command("include ./mappings.lmp")

        #Apply forces to system
        lmp.command("include ./set_daughter.lmp")

        #Setup all computes
        lmp.command(computestr)
        lmp.command(profilestr)
        lmp.command("compute		shear_T all temp/deform")     
        lmp.command("compute        shear_P all pressure shear_T ")
        lmp.command("variable       Omega equal -c_shear_P[4]*(xhi-xlo)*(yhi-ylo)*(zhi-zlo)*${srate}/(${k_B}*${T})")
        lmp.command(globalstr)

        #Run zero to setup case
        lmp.command("run 0 pre yes post yes")

        #Extract profile and time averaged (global) data from LAMMPS
        data_profile[0, :, :]= get_fix_data(lmp, "Profile_variables", profile_variables, Nbins)
        data_global[0, :] = get_fix_data(lmp, "Global_variables", global_variables)
        omega = data_global[0, -1] 

        #Run over time        
        for t in range(1 , Nsteps_eff , 1):
            lmp.command("run " + str(Delay) + " pre yes post no")
            data_profile[t, :, :]= get_fix_data(lmp, "Profile_variables", profile_variables, Nbins)
            data_global[t, :] = get_fix_data(lmp, "Global_variables", global_variables)

        #Turn off computes
        lmp.command("unfix Profile_variables")
        lmp.command("unfix Global_variables")
        lmp.command("uncompute profile_layers")
        lmp.command("uncompute shear_T")
        lmp.command("uncompute shear_P")
       
        lmp.command("include ./unset_daughter.lmp")

        #Sum the mappings together
        DAV_profile_partial  += data_profile[:,:,:]
        DAV_global_partial   += data_global[:,:]
        
        integrand_profile_partial += data_profile[:,:,:]*omega
        integrand_global_partial  += data_global[:,:]*omega
        
    #Perform the integration
    TTCF_profile_partial = TTCF_integration(integrand_profile_partial, dt*Delay)
    TTCF_global_partial  = TTCF_integration(integrand_global_partial, dt*Delay)

    #Add the initial value (t=0) 
    TTCF_profile_partial += DAV_profile_partial[0,:,:]
    TTCF_global_partial  += DAV_global_partial[0,:]

    #Average over the mappings and update the Count (# of children trajectories generated excluding the mappings)
    DAV_profile_partial  /= Nmappings   
    DAV_global_partial   /= Nmappings 
    TTCF_profile_partial /= Nmappings   
    TTCF_global_partial  /= Nmappings 
           
    Count += 1

    #Update all means and variances
    if Count >1:
    
        TTCF_profile_var= update_var(TTCF_profile_partial, TTCF_profile_mean, TTCF_profile_var, Count)      
        DAV_profile_var= update_var(DAV_profile_partial, DAV_profile_mean, DAV_profile_var, Count)
        TTCF_global_var= update_var(TTCF_global_partial, TTCF_global_mean, TTCF_global_var, Count)   
        DAV_global_var= update_var(DAV_global_partial, DAV_global_mean, DAV_global_var, Count)
      
    TTCF_profile_mean= update_mean(TTCF_profile_partial, TTCF_profile_mean, Count)     
    DAV_profile_mean= update_mean(DAV_profile_partial, DAV_profile_mean, Count)
    TTCF_global_mean= update_mean(TTCF_global_partial, TTCF_global_mean, Count)
    DAV_global_mean= update_mean(DAV_global_partial, DAV_global_mean, Count)
         


lmp.close()
t2 = MPI.Wtime()
if irank == root:
    print("Walltime =", t2 - t1, flush=True)

#Get FINAL COLUMN BECAUSE BY DEFAULT LAMMPS GIVE YOU ALSO THE USELESS INFO ABOUT THE BINS.
# For  N QUANTITIES, could TAKE THE LAST N ELEMENTS
TTCF_profile_mean = TTCF_profile_mean[:,:,-1]
DAV_profile_mean  = DAV_profile_mean[:,:,-1]

TTCF_profile_var = TTCF_profile_var[:,:,-1]
DAV_profile_var  = DAV_profile_var[:,:,-1]

TTCF_global_var/= float(Count)
DAV_global_var /= float(Count)
TTCF_profile_var /= float(Count)
DAV_profile_var  /= float(Count)

#Compute MEN AND VARIANCE OF BOTH DAV AND TTCF
TTCF_profile_mean_total = sum_over_MPI(TTCF_profile_mean, irank, comm)
DAV_profile_mean_total = sum_over_MPI(DAV_profile_mean, irank, comm)
TTCF_profile_var_total = sum_over_MPI(TTCF_profile_var, irank, comm)
DAV_profile_var_total = sum_over_MPI(DAV_profile_var, irank, comm)

TTCF_global_mean_total = sum_over_MPI(TTCF_global_mean, irank, comm)
DAV_global_mean_total = sum_over_MPI(DAV_global_mean, irank, comm)
TTCF_global_var_total = sum_over_MPI(TTCF_global_var, irank, comm)
DAV_global_var_total = sum_over_MPI(DAV_global_var, irank, comm)

#Total is None on everything but the root processor
if irank == root:
    TTCF_profile_mean_total = TTCF_profile_mean_total/float(nprocs)
    DAV_profile_mean_total  = DAV_profile_mean_total/float(nprocs)
    TTCF_profile_var_total  = TTCF_profile_var_total/np.sqrt(nprocs)
    DAV_profile_var_total   = DAV_profile_var_total/np.sqrt(nprocs)
    
    TTCF_global_mean_total = TTCF_global_mean_total/float(nprocs)
    DAV_global_mean_total  = DAV_global_mean_total/float(nprocs)
    TTCF_global_var_total  = TTCF_global_var_total/np.sqrt(nprocs)
    DAV_global_var_total   = DAV_global_var_total/np.sqrt(nprocs)
    
    TTCF_profile_SE_total  = np.sqrt(TTCF_profile_var_total)
    DAV_profile_SE_total   = np.sqrt(DAV_profile_var_total)
    TTCF_global_SE_total   = np.sqrt(TTCF_global_var_total)
    DAV_global_SE_total    = np.sqrt(DAV_global_var_total)
        
# This code animates the time history
#    plt.ion()
#    fig, ax = plt.subplots(1,1)
#    plt.show()
#    ft = True
#    for t in range(TTCF_profile_mean_total.shape[0]):
#        print(t)
#        l1, = ax.plot(DAV_profile_mean_total[t, :],'r-', label="DAV")
#        l2, = ax.plot(TTCF_profile_mean_total[t, :],'b-', label="TTCF")
#        if ft:
#            plt.legend()
#            ft=False
#        plt.pause(0.1)
#        l1.remove()
#        l2.remove()

    #This code plots the average over time
    plt.plot(np.mean(DAV_profile_mean_total[:, :],0),'r-', label="DAV")
    plt.plot(np.mean(TTCF_profile_mean_total[:, :],0),'b-', label="TTCF")
    plt.legend()
    plt.show()

    #Save variables at end of each batch in case of crash
    np.savetxt('profile_DAV.txt', DAV_profile_mean_total)
    np.savetxt('profile_TTCF.txt', TTCF_profile_mean_total)
    
    np.savetxt('profile_DAV_SE.txt', DAV_profile_SE_total)
    np.savetxt('profile_TTCF_SE.txt', TTCF_profile_SE_total)
    
    np.savetxt('global_DAV.txt', DAV_global_mean_total)
    np.savetxt('global_TTCF.txt', TTCF_global_mean_total)
    
    np.savetxt('global_DAV_SE.txt', DAV_global_SE_total)
    np.savetxt('global_TTCF_SE.txt', TTCF_global_SE_total)
    

MPI.Finalize()
