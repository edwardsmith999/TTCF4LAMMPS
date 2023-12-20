from mpi4py import MPI
from lammps import lammps
from lammps import LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR, LMP_TYPE_ARRAY
import matplotlib.pyplot as plt
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

def update_var(partial, mean, var, Count):
    return (Count-2)/float(Count-1)*var + (partial - mean)**2/float(Count)

#Running tally of mean
def update_mean(partial, mean, Count):
    return ((Count-1)*mean + partial)/float(Count)


def  run_mother_trajectory(lmp,Nsteps,Thermo_damp):

    lmp.command("fix NVT_equilibrium all nvt temp ${T} ${T} " +  str(Thermo_damp) + " tchain 1")
    lmp.command("run " + str(Nsteps))
    lmp.command("unfix NVT_equilibrium")

    return None

def get_fix_data(lmp, fixname, variables, Nbins=None):

    nlmp = lmp.numpy
    if type(variables) is list:
        Ncols = len(variables)
    elif type(variables) is int:
        Ncols = variables
    if Nbins == None:
        out = np.empty(Ncols)
        for column in range(Ncols):
            out[column] = nlmp.extract_fix(fixname, LMP_STYLE_GLOBAL, 
                                            LMP_TYPE_VECTOR, column)
    else:
        #The extra 2 is to account for Cell index and Ncount in array types (e.g. chunks)
        Nrows = Nbins
        out = np.empty([Nbins, Ncols+2])
        for row in range(Nrows):
            for column in range(Ncols+2):
                out[row, column] = nlmp.extract_fix(fixname, LMP_STYLE_GLOBAL, 
                                                    LMP_TYPE_ARRAY, row , column)

    return out

def get_globaldata(lmp, global_variables):
    return get_fix_data(lmp, "Global_variables", global_variables)

def get_globalDict(lmp, global_variables):
    """
        Otherwise we could return a dictonary 
        of named output variables as specified by the user
    """
    nlmp = lmp.numpy
    out = get_globaldata(lmp, global_variables)
    varDict = {}
    for i, pf in enumerate(global_variables):
        varDict[pf] = out[i]
    return varDict

def get_profiledata(lmp, profile_variables, Nbins):
    "A function for extracting profile fix information"
    return get_fix_data(lmp, "Profile_variables", profile_variables, Nbins)

def get_profileDict(lmp, profile_variables, Nbins):
    """
        Otherwise we could return a dictonary 
        of named output variables as specified by the user
    """
    nlmp = lmp.numpy
    out = get_profiledata(lmp, profile_variables, Nbins)
    varDict = {}
    varDict['binNo'] = out[:, 0]
    varDict['Ncount'] = out[:, 1]
    for i, pf in enumerate(profile_variables):
        varDict[pf] = out[:, i]
    return varDict

def save_state(lmp, statename, save_variables=["x", "y", "z", "vx", "vy", "vz"]):

    state = {}
    state['name'] = statename
    state['save_variables'] = save_variables
    cmdstr = "fix " + statename + " all store/state 0 {}".format(' '.join(save_variables))
    lmp.command(cmdstr)

    return state

def load_state(lmp, state):
    
    cmdstr = "change_box all  xy final 0\n"
    for i, s in enumerate(state['save_variables']):
        varname = "p"+s 
        cmdstr += "variable " + varname + " atom f_"+state['name']+"["+str(i+1)+"]\n"
        cmdstr += "set             atom * " + s + " v_"+varname+"\n"

    for line in cmdstr.split("\n"):
        lmp.command(line)
    return None


def set_list(lmp, setlist):
    
    for s in setlist:
        lmp.command(s)

def unset_list(lmp, setlist):

    for s in setlist:
        w = s.split()
        if w[0] == "compute":
            lmp.command("uncompute " + w[1])
        elif w[0] == "fix":
            lmp.command("unfix " + w[1])
        else:
            #Not need to unset variables
            pass

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

