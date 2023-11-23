
from mpi4py import MPI
from lammps import lammps
from lammps import PyLammps
from lammps import LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR, LMP_TYPE_ARRAY
from lammps import OutputCapture
import numpy as np
from scipy import integrate
import math

import sys

Nsteps_Child=1000
Delay=10
Nsteps_eff=int(Nsteps_Child/Delay)+1
Nbins=100
Nchunks=100
Nchildren=10
Nmappings=4
avetime_ncol = 2
avechunk_ncol = 3
avechunks_nrows=1
stepsize_child = 1
shear_rate = 1
dt = 0.0025

maps=[0,7,36,35]


DAV_response_mean  = np.empty([Nsteps_eff, avetime_ncol])
DAV_profile_mean   = np.empty([Nsteps_eff, Nbins, avechunk_ncol])

TTCF_response_mean  = np.empty([Nsteps_eff, avetime_ncol])
TTCF_profile_mean   = np.empty([Nsteps_eff, Nbins, avechunk_ncol])

DAV_response_var  = np.empty([Nsteps_eff, avetime_ncol])
DAV_profile_var   = np.empty([Nsteps_eff, Nbins, avechunk_ncol])

TTCF_response_var  = np.empty([Nsteps_eff, avetime_ncol])
TTCF_profile_var   = np.empty([Nsteps_eff, Nbins, avechunk_ncol])

data_response = np.empty([Nmappings, Nsteps_eff, avetime_ncol])
data_profile  = np.empty([Nmappings, Nsteps_eff, Nbins, avechunk_ncol])
TTCF_response_partial = np.empty([Nsteps_eff, avetime_ncol])
TTCF_profile_partial= np.empty([Nsteps_eff, Nbins, avechunk_ncol])

args = ['-sc', 'none','-log', 'none','-var', 'perturbation_seed' , '12345']
lmp_script = lammps(comm=MPI.COMM_SELF, cmdargs=args)
L_script = PyLammps(ptr=lmp_script)
nlmp_script = lmp_script.numpy 

            
lmp_script.file("mother+daughter.in")
lmp_script.command("run " + str(1500))
            
lmp_script.command("unfix NVT_decorrelation")
lmp_script.command("fix snapshot all store/state 0 x y z vx vy vz")

Count = 0
        
for Nc in range(1,Nchildren+1,1):

                
    lmp_script.command("include ./load_state.lmp")
    lmp_script.command("fix NVT_sampling all nvt temp ${T} ${T} ${Thermo_damp} tchain 1")
    lmp_script.command("run " + str(1000))
    lmp_script.command("unfix NVT_sampling")
    lmp_script.command("fix snapshot all store/state 0 x y z vx vy vz")

    for Nm in range(Nmappings):
    
        TTCF_response_partial[0, :]=0
        TTCF_profile_partial[0, :, :]=0
    
        lmp_script.command("variable map equal " + str(maps[Nm]))
        lmp_script.command("variable child_index equal " + str(Nc))
        lmp_script.command("include ./load_state.lmp")
        lmp_script.command("include ./mappings.lmp")
        lmp_script.command("include ./set_daughter.lmp")
        

        #lmp_script.command("run " + str(0))
        
        for column in range(avetime_ncol):
          
            data_response[Nm ,0, column] = nlmp_script.extract_fix("Response", LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR, column)
          
        for y in range(Nbins):
            for column in range(avechunk_ncol):
                data_profile[Nm ,0, y , column] = nlmp_script.extract_fix("Profile", LMP_STYLE_GLOBAL, LMP_TYPE_ARRAY, y , column)
                
        omega = data_response[Nm ,0, -1] 
        
        for t in range(1 , Nsteps_eff , 1):
          
            lmp_script.command("run " + str(Delay))
        
            for column in range(avetime_ncol):
            
                data_response[Nm ,t, column] = nlmp_script.extract_fix("Response", LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR, column)
                
          
            for y in range(Nbins):
                for column in range(avechunk_ncol):
                   
                    data_profile[Nm ,t, y , column] = nlmp_script.extract_fix("Profile", LMP_STYLE_GLOBAL, LMP_TYPE_ARRAY, y , column)
            
            if (t % 2) == 0:
            
                TTCF_profile_partial[t,:,:]  = TTCF_profile_partial[t-2,:,:] + omega*Delay*dt/3*( data_profile[Nm,t-2,:,:] + 4*data_profile[Nm,t-1,:,:] + data_profile[Nm,t,:,:] )
                TTCF_response_partial[t,:]   = TTCF_response_partial[t-2,:]  + omega*Delay*dt/3*( data_response[Nm ,t-2 ,:] + 4*data_response[Nm ,t-1 ,:] + data_response[Nm ,t ,:])
            
                TTCF_profile_partial[t-1,:,:]  = ( TTCF_profile_partial[t-2,:,:] + TTCF_profile_partial[t,:,:] )/2
                TTCF_response_partial[t-1,:]   = ( TTCF_response_partial[t-2,:]  + TTCF_response_partial[t,:] )/2
        
        lmp_script.command("include ./unset_daughter.lmp")
        
   
        
        partial_profile  = data_profile[Nm,:,:,:]
        partial_response = data_response[Nm,:,:]
        Count += 1
        
        TTCF_profile_var+=((Count-1)/Count)*((TTCF_profile_partial - TTCF_profile_mean)**2)        
        TTCF_profile_mean=((Count-1)/Count)*TTCF_profile_mean + TTCF_profile_partial/Count
        
        TTCF_response_var+=((Count-1)/Count)*((TTCF_response_partial - TTCF_response_mean)**2)        
        TTCF_response_mean=((Count-1)/Count)*TTCF_response_mean + TTCF_response_partial/Count
        
        DAV_profile_var+=((Count-1)/Count)*((partial_profile - DAV_profile_mean)**2)        
        DAV_profile_mean=((Count-1)/Count)*DAV_profile_mean + partial_profile/Count
        
        DAV_response_var+=((Count-1)/Count)*((partial_response - DAV_response_mean)**2)        
        DAV_response_mean=((Count-1)/Count)*DAV_response_mean + partial_response/Count
 
lmp_script.close()



TTCF_profile_mean = TTCF_profile_mean[:,:,-1]
DAV_profile_mean  = DAV_profile_mean[:,:,-1]

TTCF_profile_var = TTCF_profile_var[:,:,-1]
DAV_profile_var  = DAV_profile_var[:,:,-1]

TTCF_response_var/= math.sqrt(Count)
DAV_response_var /= math.sqrt(Count)
TTCF_profile_var /= math.sqrt(Count)
DAV_profile_var  /= math.sqrt(Count)

### I HAVE COMPUTED MEN AND VARIANCE OF BOTH DAV AND TTCF WITHIN EACH SINGLE CORE, SO THAT IF YOU USE A SINGLE CORE YOU STILL HAVE AN ESTIMATE OF THE FLUCTUATIONS. 
### NOW WE NEED TO AVERAGE OVER THE DIFFERENT CORES. FOR THE MEAN, IT IS EASY, YOU SUM AND THE MEANS AND DIVIDE BY NCORES, FOR THE VARIANCE, YOU SUM THE VARIANCES AND DIVIDE BY THE SQRT(NCORES)
### ESSENTIALLY WE HAVE THE MEAN OF EACH CORE: M1,M2,M3,M4,... AND THE VARIANCE V1,V2,V3,V4,... 
### WHERE THE MEANS HAVE THE SUFFIX _mean AND THE VARIANCE _var, FOR EACH ARRAY, (TTCF_profile, TTCF_response, DAV_profile, DAV_response)
### SO THE TOTAL MEAN AND VARIANCE OVER THE CORES ARE TOT_MEAN=(M1+M2+M3+...)/NCORES AND TOT_VAR=(V1+V2+V3+...)/SQRT(NCORES)


    #Save variables at end of each batch in case of crash
#np.savetxt('response.txt', response)
#np.savetxt('DAV.txt', DAV_profile_mean)
#np.savetxt('TTCF.txt', TTCF_profile_mean)
MPI.Finalize()
