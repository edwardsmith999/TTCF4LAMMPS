from mpi4py import MPI
from lammps import lammps, PyLammps
from lammps import LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR, LMP_TYPE_ARRAY

import numpy as np

def sum_over_MPI(A, irank, comm, root=0):
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


def get_globaldata(global_variables, nlmp):
    Ncols = len(global_variables)
    out = np.empty(Ncols)
    for column in range(Ncols):
        out[column] = nlmp.extract_fix("Global_variables", LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR, column)

    return out

def get_globalDict(global_variables, nlmp):
    """
        Otherwise we could return a dictonary 
        of named output variables as specified by the user
    """

    out = get_globaldata(global_variables, nlmp)
    varDict = {}
    for i, pf in enumerate(global_variables):
        varDict[pf] = out[i]
    return varDict


def get_profiledata(profile_variables, Nbins, nlmp):
    "A function for extracting profile fix information"
    Nrows = Nbins
    Ncols = len(profile_variables)+2
    out = np.empty([Nrows, Ncols])
    for row in range(Nrows):
        for column in range(Ncols):
            out[row, column] = nlmp.extract_fix("Profile_variables", LMP_STYLE_GLOBAL, 
                                                LMP_TYPE_ARRAY, row , column)
    return out

def get_profileDict(profile_variables, Nbins, nlmp):
    """
        Otherwise we could return a dictonary 
        of named output variables as specified by the user
    """

    out = get_profiledata(profile_variables, Nbins, nlmp)
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

def TTCF_integration(A, int_step):
 
    integral = np.zeros(A.shape)
    N = A.shape[0]

    for t in range(2 , N, 2):
        integral[t,...]   = integral[t-2,...] +  int_step*sum_prev_dt(A, t)
        integral[t-1,...] = (integral[t-2,...] + integral[t,...])/2

    if (N % 2) == 0:
        integral[-1,...] = integral[-2,...] + int_step/2*A[-1,...]

    return integral

