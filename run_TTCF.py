#### IN ORDER TO KEEP LAMMPS AND PYTHON VARIABLES SEPARATE, I REWROTE THE CODE A BIT. 
#### NOW ESSENTIALLY EVERY ACTION IS PERFOMED IN PYTHON VIA THE LMP.COMMAND() COMMAND.
#### THE INITIAL LAMMPS INPUT FILES CONTAINS THE PARAMETERS USED IN LAMMPS
#### AND THE SYSTEM SETUP (SET BOX SIZE, CREATE ATOMS, ETC...)
#### EACH SINGLE ACTION IS NOT RUN THROUGH PYTHON
#### THINGS TO DO: IMPLEMENT THE INTEGRATION PROCESS IN A SEPARATE FUNCTION
#### POSSIBLY, IMPLEMENT EACH SINGLE BLOCK (THERMALIZATION, SAMPLING, DAUGHTER) 
#### IN SEPARATE FUNCTIONS. NOTE THAT THE COMMAND LMP.COMMAND(""" .....  """) DOES NOT WORK
#### YOU NEED A SINGLE-LINE COMMAND (SINGLE LAMMPS INSTRUCTION)
#### SO CREATE SEPARATE FUNCTIONS FOR EACH BLOCK MIGHT BE USELESS.
#### LAST THING TO DO: CREATE FILE WITH PARAMETERS. 


from mpi4py import MPI
from lammps import lammps, PyLammps
from lammps import LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR, LMP_TYPE_ARRAY

import matplotlib.pyplot as plt

import numpy as np
import sys
import math


def sum_over_MPI(A, irank, root=0):
    #Put data into contiguous c arrays read to send through MPI
    sendbuf = np.ascontiguousarray(A)
    if irank == root:
        recvbuf = np.copy(np.ascontiguousarray(A))
    else:
        recvbuf = np.array(1)
    comm.Reduce([sendbuf, MPI.DOUBLE], [recvbuf, MPI.DOUBLE], op=MPI.SUM, root=root )
    #Summed arrays only exist on root process, unpack into variables
    if irank == root:
        return recvbuf
    else:
        return None

# Running tally of variance
def update_var(partial, mean, var, Count):
    return (Count-1)/float(Count)*var + ((Count-1)*((partial - mean)/float(Count))**2)

#Running tally of mean
def update_mean(partial, mean, Count):
    return ((Count-1)*mean + partial)/float(Count)


def get_globaldata(global_variables):
    Ncols = len(global_variables)
    out = np.empty(Ncols)
    for column in range(Ncols):
        out[column] = nlmp.extract_fix("Global_variables", LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR, column)

    return out

def get_globalDict(global_variables):
    """
        Otherwise we could return a dictonary 
        of named output variables as specified by the user
    """

    out = get_globaldata(global_variables)
    varDict = {}
    for i, pf in enumerate(global_variables):
        varDict[pf] = out[i]
    return varDict


def get_profiledata(profile_variables, Nbins):
    "A function for extracting profile fix information"
    Nrows = Nbins
    Ncols = len(profile_variables)+2
    out = np.empty([Nrows, Ncols])
    for row in range(Nrows):
        for column in range(Ncols):
            out[row, column] = nlmp.extract_fix("Profile_variables", LMP_STYLE_GLOBAL, 
                                                LMP_TYPE_ARRAY, row , column)
    return out

def get_profileDict(profile_variables, Nbins):
    """
        Otherwise we could return a dictonary 
        of named output variables as specified by the user
    """

    out = get_profiledata(profile_variables, Nbins)
    varDict = {}
    varDict['binNo'] = out[:, 0]
    varDict['Ncount'] = out[:, 1]
    for i, pf in enumerate(profile_variables):
        varDict[pf] = out[:, i]
    return varDict

def sum_prev_dt(A, t):
    return (   A[t-2,...] 
            +4*A[t-1,...] 
            +  A[t  ,...])/3.


def TTCF_integration_profile(f_profile, int_step, N, Nb , ncols ):

    #integral_profile = f_profile 
    integral_profile = np.zeros([N, Nb, ncols])

    for t in range(2 , N, 2):
    
        integral_profile[t,:,:]   = integral_profile[t-2,:,:] +  int_step/3*(   f_profile[t-2,:,:] + 4*f_profile[t-1,:,:] + f_profile[t,:,:] )
        #integral_profile[t-1,:,:] = integral_profile[t-2,:,:] + int_step/12*( 5*f_profile[t-2,:,:] + 8*f_profile[t-1,:,:] - f_profile[t,:,:] )
        integral_profile[t-1,:,:] = (integral_profile[t-2,:,:] + integral_profile[t,:,:])/2


    if (N % 2) == 0:

        integral_profile[-1,:,:] = integral_profile[-2,:,:] + int_step/2*f_profile[-1,:,:]

    return integral_profile

def TTCF_integration_global(f_global, int_step, N , ncols):

    #integral_global = f_global 
    integral_global = np.zeros([N,ncols])

    for t in range(2 , N, 2):
    
        integral_global[t,:]   = integral_global[t-2,:] +  int_step/3*(   f_global[t-2,:] + 4*f_global[t-1,:] + f_global[t,:] )
        #integral_global[t-1,:] = integral_global[t-2,:] + int_step/12*( 5*f_global[t-2,:] + 8*f_global[t-1,:] - f_global[t,:] )
        integral_global[t-1,:] = (integral_global[t-2,:] + integral_global[t,:])/2


    if (N % 2) == 0:

        integral_global[-1,:] = integral_global[-2,:] + int_step/2*f_global[-1,:]

    return integral_global

#This code is run using MPI - each processes will
#run this same bit code with its own memory
comm = MPI.COMM_WORLD
irank = comm.Get_rank()
nprocs = comm.Get_size()
t1 = MPI.Wtime()
root = 0
print("Proc {:d} out of {:d} procs".format(irank+1,nprocs))

#Define lengths for all runs, number of Daughters, etc

Tot_Daughters= 1000
Ndaughters=math.ceil(Tot_Daughters/nprocs)

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
np.random.seed(irank)
#seed_v = str(int(np.random.randint(1, 1e5 + 1)))
seed_v = str(12345)

#Define LAMMPS object and initialise
args = ['-sc', 'none','-log', 'none','-var', 'rand_seed' , seed_v]
lmp = lammps(comm=MPI.COMM_SELF, cmdargs=args)
L = PyLammps(ptr=lmp)
nlmp = lmp.numpy

#Run equilibration  
lmp.file("System_setup.in")
lmp.command("timestep " + str(dt))
lmp.command("variable Thermo_damp equal " +  str(10*dt))
lmp.command(" fix NVT_thermalization all nvt temp ${T} ${T} ${Thermo_damp} tchain 1")
lmp.command("run " + str(Nsteps_Thermalization))
lmp.command("unfix NVT_thermalization")

#Save snapshot to use for daughters
lmp.command("fix snapshot all store/state 0 x y z vx vy vz")

Count = 0
for Nd in range(1,Ndaughters+1,1):

    #Sampling of the daughters initial state
    lmp.command("include ./load_state.lmp")
    lmp.command("fix NVT_sampling all nvt temp ${T} ${T} ${Thermo_damp} tchain 1")
    lmp.command("run " + str(Nsteps_Decorrelation))
    lmp.command("unfix NVT_sampling")
    lmp.command("fix snapshot all store/state 0 x y z vx vy vz")

    DAV_profile_partial[:,:,:] = 0
    DAV_global_partial[:,:]    = 0
        
    integrand_profile_partial[:,:,:] = 0
    integrand_global_partial[:,:]    = 0

    for Nm in range(Nmappings):

        #Apply mapping    
        lmp.command("variable map equal " + str(Maps[Nm]))
        lmp.command("variable Daughter_index equal " + str(Nd))
        lmp.command("include ./load_state.lmp")
        lmp.command("include ./mappings.lmp")
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

        #Them functions data in a semi-automated way (Nbins, etc specified once)
        data_profile[0, :, :]= get_profiledata(profile_variables, Nbins)
        data_global[0, :] = get_globaldata(global_variables)

        omega = data_global[0, -1] 
        #TTCF_global_partial[0, :]=0
        #TTCF_profile_partial[0, :, :]=0
        #Run over time        
        for t in range(1 , Nsteps_eff , 1):
            lmp.command("run " + str(Delay) + " pre yes post no")
            data_profile[t, :, :]= get_profiledata(profile_variables, Nbins)
            data_global[t, :] = get_globaldata(global_variables)

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
    TTCF_profile_partial = TTCF_integration_profile(integrand_profile_partial, dt*Delay, Nsteps_eff , Nbins, avechunk_ncol )
    TTCF_global_partial  = TTCF_integration_global(integrand_global_partial , dt*Delay, Nsteps_eff , avetime_ncol )  
    
    
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
    TTCF_profile_var= update_var(TTCF_profile_partial, TTCF_profile_mean, TTCF_profile_var, Count)
    TTCF_profile_mean= update_mean(TTCF_profile_partial, TTCF_profile_mean, Count)
        
    DAV_profile_var= update_var(DAV_profile_partial, DAV_profile_mean, DAV_profile_var, Count)
    DAV_profile_mean= update_mean(DAV_profile_partial, DAV_profile_mean, Count)

    TTCF_global_var= update_var(TTCF_global_partial, TTCF_global_mean, TTCF_global_var, Count)
    TTCF_global_mean= update_mean(TTCF_global_partial, TTCF_global_mean, Count)
        
    DAV_global_var= update_var(DAV_global_partial, DAV_global_mean, DAV_global_var, Count)
    DAV_global_mean= update_mean(DAV_global_partial, DAV_global_mean, Count)
         


lmp.close()
t2 = MPI.Wtime()
if irank == root:
    print("Walltime =", t2 - t1, flush=True)

#I GET THE FINAL COLUMN BECAUSE BY DEFAULT LAMMPS GIVE YOU ALSO THE USELESS INFO ABOUT THE BINS. SINCE I HAVE ONLY ONE QUANTITY TO COMPUTE, I TAKE THE LAST.
# IF I HAD N QUANTITIES, I WOULD TAKE THE LAST N ELEMENTS
TTCF_profile_mean = TTCF_profile_mean[:,:,-1]
DAV_profile_mean  = DAV_profile_mean[:,:,-1]

TTCF_profile_var = TTCF_profile_var[:,:,-1]
DAV_profile_var  = DAV_profile_var[:,:,-1]

TTCF_global_var/= np.sqrt(Count)
DAV_global_var /= np.sqrt(Count)
TTCF_profile_var /= np.sqrt(Count)
DAV_profile_var  /= np.sqrt(Count)

### I HAVE COMPUTED MEN AND VARIANCE OF BOTH DAV AND TTCF WITHIN EACH SINGLE CORE, SO THAT IF YOU USE A SINGLE CORE YOU STILL HAVE AN ESTIMATE OF THE FLUCTUATIONS. 
### NOW WE NEED TO AVERAGE OVER THE DIFFERENT CORES. FOR THE MEAN, IT IS EASY, YOU SUM AND THE MEANS AND DIVIDE BY NCORES, FOR THE VARIANCE, YOU SUM THE VARIANCES AND DIVIDE BY THE SQRT(NCORES)
### ESSENTIALLY WE HAVE THE MEAN OF EACH CORE: M1,M2,M3,M4,... AND THE VARIANCE V1,V2,V3,V4,... 
### WHERE THE MEANS HAVE THE SUFFIX _mean AND THE VARIANCE _var, FOR EACH ARRAY, (TTCF_profile, TTCF_global, DAV_profile, DAV_global)
### SO THE TOTAL MEAN AND VARIANCE OVER THE CORES ARE TOT_MEAN=(M1+M2+M3+...)/NCORES AND TOT_VAR=(V1+V2+V3+...)/SQRT(NCORES)
TTCF_profile_mean_total = sum_over_MPI(TTCF_profile_mean, irank)
DAV_profile_mean_total = sum_over_MPI(DAV_profile_mean, irank)
TTCF_profile_var_total = sum_over_MPI(TTCF_profile_var, irank)
DAV_profile_var_total = sum_over_MPI(DAV_profile_var, irank)

TTCF_global_mean_total = sum_over_MPI(TTCF_global_mean, irank)
DAV_global_mean_total = sum_over_MPI(DAV_global_mean, irank)
TTCF_global_var_total = sum_over_MPI(TTCF_global_var, irank)
DAV_global_var_total = sum_over_MPI(DAV_global_var, irank)

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
    
    np.savetxt('profile_DAV_SE.txt', DAV_profile_var_total)
    np.savetxt('profile_TTCF_SE.txt', TTCF_profile_var_total)
    
    np.savetxt('global_DAV.txt', DAV_global_mean_total)
    np.savetxt('global_TTCF.txt', TTCF_global_mean_total)
    
    np.savetxt('global_DAV_SE.txt', DAV_global_var_total)
    np.savetxt('global_TTCF_SE.txt', TTCF_global_var_total)
    

MPI.Finalize()
