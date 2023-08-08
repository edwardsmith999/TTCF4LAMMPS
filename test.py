from mpi4py import MPI
from lammps import lammps
from lammps import PyLammps
from lammps import LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR, LMP_TYPE_ARRAY
from lammps import OutputCapture
import numpy as np

#This code is run using MPI - each processes will
#run this same bit code with its own memory
irank = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()
print("Proc {:d} out of {:d} procs has".format(irank,nprocs))

#Define variables (move to function?)
run_in_python = True
Child_sim=10
Child_No=nprocs
Shear_rate=0.01
with open("seed.txt", 'r') as f:
    seed_v = f.read()

# Run the mother on all processes to get done as fast as possible
# Note default with Lammps and mpi4py is to distribute run
# over all processes
args = ['-var', 'rseed', seed_v,
        '-var', 'Nchild', str(Child_No)]
lmp = lammps(comm=MPI.COMM_WORLD, cmdargs=args)
lmp.file("mother.in")
lmp.close()   #This deletes any data

#Run the children on own processes each doing own thing
i = irank+1
filename = "branch" + str(i) + ".dat"
print("Proc {:d} reading filename {}".format(irank, filename))

MPI.COMM_WORLD.Barrier()

multidata = []; multichunkdata = []
for j in [1,2,3,4]:
    args = ['-var', 'map', str(j), 
            '-var', 'child_index', str(i), 
            '-var', 'srate', str(Shear_rate)]
    lmp_child = lammps(comm=MPI.COMM_SELF, cmdargs=args)
    L = PyLammps(ptr=lmp_child)
    nlmp = lmp_child.numpy

    #File does everything except the run
    lmp_child.file("child.in")

    #Get metadata 
    Ly = lmp_child.extract_box()[1][1] - lmp_child.extract_box()[0][1]
    dy = lmp_child.extract_variable("Bin_Width")
    ncolumns = int(1./dy) #int(Ly/dy) if dy is not percentage of domain
    nrows = 6  #Number of variables out

    #Run step by step in Python getting output values each time 
    if run_in_python:
        data = []; chunkdata = []
        for t in range(Child_sim):

            #Run one step
            out=L.run("1 pre no post no")

            ##############################################
            #
            #   Get data from fixes programatically
            #
            ##############################################

            #Extract fxvx from fix
            fxvx = np.zeros([27])
            for column in range(27):
                fxvx[column] = nlmp.extract_fix("fxvx", LMP_STYLE_GLOBAL,
                                              LMP_TYPE_VECTOR, column)
            data.append(fxvx)

            #This can get all chunk data from a fix
            chunks = np.zeros([ncolumns,6])
            for column in range(ncolumns):
                for row in range(nrows):
                    chunks[column,row] = nlmp.extract_fix("tProf", LMP_STYLE_GLOBAL, 
                                                           LMP_TYPE_ARRAY, column, row)
            chunkdata.append(chunks)

            ##############################################
            #
            #   Get data from prints using stdout
            #
            ##############################################
            #This uses the pylammps interface
            #to get the stdout every step and extract keyword 
            #for required quanitities
            #for l in out:
            #    if ("childout" in l):
            #        data.append(l.replace("childout","").split())

            #This is a lower level version using pylammps              
            #OutputCaputure to get stdout instead
#            lmp_child.command("run 1 pre no post no")
#            with OutputCapture() as capture:
#                lmp_child.command("run 1 pre no post no")
#                lmp_child.flush_buffers()
#                out = capture.output
#            for l in out.split("\n"):
#                if ("childout" in l):
#                    data.append(l.replace("childout","").split())


    #Otherwise, run the whole thing in one go 
    #and collect all results at end from standardout
    else:
        #Then we run so output goes to out, convert to numpy array
        out=L.run(Child_sim)
        data = []
        for l in out:
            if ("childout" in l):
                data.append(l.replace("childout","").split())

    data = np.array(data, dtype=np.float64)
    multidata.append(data)
    multichunkdata.append(chunkdata)
    print("Proc {:d} mapping {:d} output data {:f}".format(irank, j, data[2,2]))

#An array of size [nmappings by Nsteps by Nvariables] on each rank
multidata = np.array(multidata)

#An array of chunk data in the format [nmappings by Nsteps bu Nchunks by Nvariables]
multichunkdata = np.array(multichunkdata)

#Finalise MPI
MPI.Finalize()

#Plot average over 4 mirrors and all timesteps
import matplotlib.pyplot as plt
plt.plot(np.mean(multichunkdata,(0,1)))
plt.show()

#DOES NOT WORK TO GET FIXES OR COMPUTES
#Attempt to get fix, only work for instant, not dynamically
#fix = lmp_child.extract_fix("8", LMP_STYLE_GLOBAL, LMP_TYPE_ARRAY)
#nlmp = lmp_child.numpy
#fix = nlmp.extract_fix("8",1,0)

#nlmp = lmp_child.numpy
#print("fix=", nlmp.extract_fix("tProf", LMP_STYLE_GLOBAL, LMP_TYPE_ARRAY))
#print("comput=", nlmp.extract_compute("tlayers",0,0))

