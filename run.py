#! /usr/bin/env python2.7
import numpy as np
import subprocess as sp
import sys
import os
import glob
import shutil
import cPickle as pickle

from branch_utils import phase_space_map, read_dissipation

# Import symwraplib
sys.path.insert(0, "./SimWrapPy/")
try:
    import simwraplib as swl
except ImportError:
    cmd = "git clone https://github.com/edwardsmith999/SimWrapPy.git ./SimWrapPy"
    downloadout = sp.check_output(cmd, shell=True)
    print(downloadout)
    sys.path.insert(0, "./SimWrapPy")
    import simwraplib as swl

def rm_wildcard(string):
    for fl in glob.glob(string):
        os.remove(fl)

def clean(basename="branch", folder=None):

    try:
        rm_wildcard(basename+"*.dat")
        rm_wildcard("mirror_"+basename+"*.dat")
        if folder != None:
            shutil.rmtree(folder)
    except OSError:
        print("Delete failed")

def run_mother(basename="branch", basedir=None,
               Nequil=100000, Nruns=1000, Ninloop=300,
               runcounter=0, rfiles = []): 
               #, rfiles=["disp.dat", "final_mother.dat"]):

    """
        Run the mother trajectory

        Inputs:
            basename -- the name of the mother outputs in the form basename{:d}.dat 
                        used as initial files for each child (default branch)
            basedir -- location to run parameter study (default current dir)
            rfiles -- files to make copies of with cumulative run count
    """
    #Directory which executable and input files are found
    if basedir == None:
        basedir = os.path.dirname(os.path.realpath(__file__)) + "/"


    #Run equilibration to create case
    branchfiles = []
    if Nequil != 0:

        inputchanges={"variable Nequil equal":Nequil}

        run = swl.LammpsRun(srcdir=None,
                            basedir=basedir,
                            rundir=basedir,
                            executable='./lmp', #Relative in basedir
                            inputfile='mother_equil.in', #Relative to basedir
                            outputfile='mother_equil.out',
                            inputchanges=inputchanges)

        study = swl.Study([[run]], 1)

    #Else, run a set of loops to create branch files
    elif Nruns != 0:

        inputchanges={"variable Ninloop equal":Ninloop,
                      "variable index loop":Nruns,
                      "    write_data INSTNCE1":basename+"*.dat"}

        run = swl.LammpsRun(srcdir=None,
                            basedir=basedir,
                            rundir=basedir,
                            executable='./lmp', #Relative in basedir
                            inputfile='mother_gen_branches.in', #Relative to basedir
                            outputfile='mother.out',
                            inputchanges=inputchanges)

        study = swl.Study([[run]], 1)

        #List of written branch files
        for r in range(Ninloop, (Nruns+1)*Ninloop, Ninloop):
            branchfiles.append(basename+str(r)+".dat")

    #Backup disp.dat of dissipation function
    if runcounter != 0:
        for rfile in rfiles:
            try:
                 shutil.copyfile(rfile, rfile+"to"+str(runcounter))
            except IOError:
                 print("Renamed file " + rfile + " not found")
    #increment run counter
    runcounter += Nequil+Ninloop*Nruns
    return runcounter, branchfiles


def run_children(ncpus=6, basename="branch", basedir=None, 
                 srcdir=None, use_all_files=True, 
                 studyfolder="study"):

    """
        Run a range of simulations

        Inputs:
            ncpus -- Number of cpus to use in parallel (default 6)
            basename -- the name of the mother outputs in the form basename{:d}.dat 
                        used as initial files for each child (default branch)
            basedir -- location to run parameter study (default current dir)
            srcdir -- location of lammps src code to create a copy for backup 
                        and used to compile executable
            use_all_files -- True or list/tuple with [first, last, step] which would,
                             for example, give basename000000.dat, basename000300.dat, 
                             basename000600.dat for [0,900,300] (default True uses all 
                             basename* files)
    """

    #Directory which executable and input files are found
    if basedir == None:
        basedir = os.path.dirname(os.path.realpath(__file__)) + "/" 

    #Setup changes to make to input file
    children = []
    
    if not use_all_files:
        restartfile = basename + "{:d}.dat"
        for i in range(use_all_files[0], use_all_files[1], use_all_files[2]):
            children.append(restartfile.format(i))
            children.append("mirror_" + restartfile.format(i))
    else:
        files = glob.glob(basename + "*.dat")
        files.sort()
        for i, filename in enumerate(files):
            children.append(filename)
            children.append("mirror_" + filename)

    #Get change dictonary to adjust input file names
    changeDict = swl.InputDict({'read_data': children})
    changes = changeDict.expand()
    filenames = changes.filenames()

    #Loop over all changes and assign each to a thread
    threadlist =[]
    for thread in range(0,len(changes)):

        rundir = (basedir + "ttcf" + filenames[thread].replace("readdata","")
                                                      .replace("branch","")
                                                      .replace("pdat",""))

        rfile = changes[thread]["read_data"]
        if (os.path.isfile(rfile)):
            #Create corresponding mirror file
            #if not "mirror" in rfile:
            #    phase_space_map(rfile,  maptype="reflectmom")

            #Mapping can be done in lammps input
            run = swl.LammpsRun(srcdir=None,
                                basedir=basedir,
                                rundir=rundir,
                                executable='./lmp', #Relative in basedir
                                inputfile='child.in', #Relative to basedir
                                outputfile='lammps.out', #Relative to basedir
                                restartfile=rfile,
                                deleteoutput=True)

            runlist = [run]
            threadlist.append(runlist)
            print('Run in directory '  + rundir + 
                   ' and dryrun is '  + str(run.dryrun))
        else:
            print("Restart file ", changes[thread]["read_data"], " not found")

    #Run the study which contains all threads
    study = swl.Study(threadlist, ncpus, studyfolder)


if __name__ == "__main__":

    from analyse_data import read_data
    
    Nequil=100000
    nbatches = 100
    ncpus = 4
    Nruns = 50 * ncpus
    studyfolder = "study"
    basename = "branch"
    outfile = "./outfile.npy"
    restart = True
    if restart:
         results = list(np.load(outfile))
    else:
        results = []
    #Run batches of mother and child trajectories to avoid
    #large numbers of files being created
    rc, _ = run_mother(basename=basename, Nequil=Nequil, Nruns=0, 
                       Ninloop=300, runcounter=0)
    for i in range(nbatches):

        #Delete previous batch of child files
        clean(basename=basename, folder=studyfolder)

        #Run mother trajectory for Nruns more steps and increment runcounter
        rc, branchfiles = run_mother(basename=basename, Nequil=0, Nruns=Nruns, 
                                     Ninloop=300, runcounter=rc)

        #Read dissipation from branch files generated by mother trajectory
        #No longer needed as output gives initial state 
        #since lammps on 12th Nov 2019 1955c57791b276f580f83eafe9eb5567ee3fab2d
        #d = []
        #for f in branchfiles:
        #    d.append(read_dissipation(f))
            #print("branch files = ", f, d)
        #dissipations.append(d)

        #Span off a bunch of child trajectores from each branch created
        #by the mother trajectory
        run_children(ncpus=ncpus, basename=basename, studyfolder=studyfolder)

        #We need to store every bit of trajectory data
        results.append(read_data(fdir=studyfolder))

        #Store Python pickle of data so far
        #pickle.dump(results, open("TTCF_run.p","w+"))
        #Use numpy save which is 3 time smaller and much faster
        np.save(outfile, np.array(results))

