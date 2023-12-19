import numpy as np

from utils import *

class TTCF():

    def __init__(self, global_variables, profile_variables, 
                 Nsteps, Nbins, Nmappings):

        """
            Create a TTCF object to collect data

            inputs include
            global_variables - list - variables collected by ave/time
            profile_variables - list - variables collected by ave/chunk
            Nsteps - int - Number of steps daughter us run for when TTCF is collected
            Nbins - int - Number of bins in profiles
            Nmappings - int - Number of mappings used by duaghters
        """

        self.global_variables = global_variables
        self.profile_variables = profile_variables
        self.Nsteps = Nsteps
        self.Nbins = Nbins
        self.Nmappings = Nmappings
        self.Count = 0

        avetime_ncol = len(global_variables)
        avechunk_ncol = len(profile_variables) + 2

        #Allocate arrays to store data
        self.DAV_global_mean  = np.zeros([Nsteps, avetime_ncol])
        self.DAV_profile_mean   = np.zeros([Nsteps, Nbins, avechunk_ncol])

        self.TTCF_global_mean  = np.zeros([Nsteps, avetime_ncol])
        self.TTCF_profile_mean   = np.zeros([Nsteps, Nbins, avechunk_ncol])

        self.DAV_global_var  = np.zeros([Nsteps, avetime_ncol])
        self.DAV_profile_var   = np.zeros([Nsteps, Nbins, avechunk_ncol])

        self.TTCF_global_var  = np.zeros([Nsteps, avetime_ncol])
        self.TTCF_profile_var   = np.zeros([Nsteps, Nbins, avechunk_ncol])

        self.TTCF_global_partial = np.zeros([Nsteps, avetime_ncol])
        self.TTCF_profile_partial= np.zeros([Nsteps, Nbins, avechunk_ncol])

        self.DAV_profile_partial= np.zeros([Nsteps, Nbins, avechunk_ncol])
        self.DAV_global_partial = np.zeros([Nsteps, avetime_ncol])

        self.integrand_global_partial  = np.zeros([Nsteps, avetime_ncol])
        self.integrand_profile_partial = np.zeros([Nsteps, Nbins, avechunk_ncol])

    def add_mappings(self, data_profile, data_global, omega):

        #Sum the mappings together
        self.DAV_profile_partial  += data_profile[:,:,:]
        self.DAV_global_partial   += data_global[:,:]
        
        self.integrand_profile_partial += data_profile[:,:,:]*omega
        self.integrand_global_partial  += data_global[:,:]*omega

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

        #Update all means and variances
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

    def finalise_output(self, irank, comm, root=0):

        self.irank = irank
        self.comm = comm
        self.root = root
        self.nprocs = comm.Get_size()
        self.output_finalised = True

        #Get FINAL COLUMN BECAUSE BY DEFAULT LAMMPS GIVE YOU ALSO THE USELESS INFO ABOUT THE BINS.
        # For  N QUANTITIES, could TAKE THE LAST N ELEMENTS
        self.TTCF_profile_mean = self.TTCF_profile_mean[:,:,-1]
        self.DAV_profile_mean  = self.DAV_profile_mean[:,:,-1]

        self.TTCF_profile_var = self.TTCF_profile_var[:,:,-1]
        self.DAV_profile_var  = self.DAV_profile_var[:,:,-1]

        self.TTCF_global_var/= float(self.Count)
        self.DAV_global_var /= float(self.Count)
        self.TTCF_profile_var /= float(self.Count)
        self.DAV_profile_var  /= float(self.Count)

        #Compute MEN AND VARIANCE OF BOTH DAV AND TTCF
        self.TTCF_profile_mean_total = sum_over_MPI(self.TTCF_profile_mean, irank, comm)
        self.DAV_profile_mean_total = sum_over_MPI(self.DAV_profile_mean, irank, comm)
        self.TTCF_profile_var_total = sum_over_MPI(self.TTCF_profile_var, irank, comm)
        self.DAV_profile_var_total = sum_over_MPI(self.DAV_profile_var, irank, comm)

        self.TTCF_global_mean_total = sum_over_MPI(self.TTCF_global_mean, irank, comm)
        self.DAV_global_mean_total = sum_over_MPI(self.DAV_global_mean, irank, comm)
        self.TTCF_global_var_total = sum_over_MPI(self.TTCF_global_var, irank, comm)
        self.DAV_global_var_total = sum_over_MPI(self.DAV_global_var, irank, comm)

        #Total is None on everything but the root processor
        if irank == self.root:
            self.TTCF_profile_mean_total = self.TTCF_profile_mean_total/float(self.nprocs)
            self.DAV_profile_mean_total  = self.DAV_profile_mean_total/float(self.nprocs)
            self.TTCF_profile_var_total  = self.TTCF_profile_var_total/np.sqrt(self.nprocs)
            self.DAV_profile_var_total   = self.DAV_profile_var_total/np.sqrt(self.nprocs)
            
            self.TTCF_global_mean_total = self.TTCF_global_mean_total/float(self.nprocs)
            self.DAV_global_mean_total  = self.DAV_global_mean_total/float(self.nprocs)
            self.TTCF_global_var_total  = self.TTCF_global_var_total/np.sqrt(self.nprocs)
            self.DAV_global_var_total   = self.DAV_global_var_total/np.sqrt(self.nprocs)
            
            self.TTCF_profile_SE_total  = np.sqrt(self.TTCF_profile_var_total)
            self.DAV_profile_SE_total   = np.sqrt(self.DAV_profile_var_total)
            self.TTCF_global_SE_total   = np.sqrt(self.TTCF_global_var_total)
            self.DAV_global_SE_total    = np.sqrt(self.DAV_global_var_total)


    def plot_data(self, animated=False):

        if self.output_finalised and self.irank == self.root:

            import matplotlib.pyplot as plt

            if animated:
                # This code animates the time history
                plt.ion()
                fig, ax = plt.subplots(1,1)
                plt.show()
                ft = True
                for t in range(self.TTCF_profile_mean_total.shape[0]):
                    print(t)
                    l1, = ax.plot(self.DAV_profile_mean_total[t, :],'r-', label="DAV")
                    l2, = ax.plot(self.TTCF_profile_mean_total[t, :],'b-', label="TTCF")
                    if ft:
                        plt.legend()
                        ft=False
                    plt.pause(0.1)
                    l1.remove()
                    l2.remove()
            else:
                #This code plots the average over time
                plt.plot(np.mean(self.DAV_profile_mean_total[:, :],0),'r-', label="DAV")
                plt.plot(np.mean(self.TTCF_profile_mean_total[:, :],0),'b-', label="TTCF")
                plt.legend()
                plt.show()

    def save_data(self):

        if self.output_finalised and self.irank == self.root:
            #Save variables at end of each batch in case of crash
            np.savetxt('profile_DAV.txt', self.DAV_profile_mean_total)
            np.savetxt('profile_TTCF.txt', self.TTCF_profile_mean_total)
            
            np.savetxt('profile_DAV_SE.txt', self.DAV_profile_SE_total)
            np.savetxt('profile_TTCF_SE.txt', self.TTCF_profile_SE_total)
            
            np.savetxt('global_DAV.txt', self.DAV_global_mean_total)
            np.savetxt('global_TTCF.txt', self.TTCF_global_mean_total)
            
            np.savetxt('global_DAV_SE.txt', self.DAV_global_SE_total)
            np.savetxt('global_TTCF_SE.txt', self.TTCF_global_SE_total)
                 

