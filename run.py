#! /usr/bin/env python2.7
import numpy as np
import subprocess as sp
import sys
import os
import glob
import shutil
import cPickle as pickle

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


def ms(s):
    return str(-float(s))

def phase_space_map(restartfile, maptype="flipymompos"):
    nl = 0; nlv = 0
    with open(restartfile, "r") as f:
        with open("mirror_"+restartfile, "w+") as g:
            for l in f:
                #Get number of atoms
                if "atoms" in l:
                    N = int(l.replace("atoms",""))
                #For Atoms, rewrite the next N records 
                if "Atoms" in l and "pos" in maptype:
                   nl = N
                #For velocities, rewrite the next N records 
                if "Velocities" in l:
                    nlv = N
                #Flip domain extents as well
                if "flipymompos" in maptype and "ylo" in l:
                    a = l.split()
                    g.write(   ms(a[1]) + " " + ms(a[0]) + " " 
                            + a[2] + " " + a[3] + "\n" )
                #If next line (nlv) is not zero, adapt/write this line
                elif nl != 0:
                    #Error handling here to skip any non-position records
                    try:
                        a = l.split()
                        if "flipymompos" in maptype:
                            g.write(  a[0] + " " + a[1]+ " " 
                                    + a[2] + " " + a[3]+ " " 
                                    + a[4]+ " " + ms(a[5]) + " " 
                                    + a[6] + " " + a[7] + " " 
                                    + a[8] + " " + a[9] + "\n")
                        nl -= 1
                    except IndexError:
                        g.write(l)

                #If next line (nlv) is not zero, adapt/write this line
                elif nlv != 0:
                    #Error handling here to skip any non-velocity records
                    try:
                        a = l.split()
                        if "reflectmom" in maptype:
                            g.write(a[0] + " " + ms(a[1]) 
                                         + " " + ms(a[2])
                                         + " " + ms(a[3]) + "\n")
                        elif "flipymompos" in maptype:
                            g.write(a[0] + " " + a[1]
                                         + " " + ms(a[3])
                                         + " " + a[2] + "\n")
                        nlv -= 1
                    except IndexError:
                        g.write(l)
                else:
                    g.write(l)

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
               runcounter=0,rfiles=["disp.dat", "final_mother.dat"]):

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

    inputchanges={"variable Nequil equal":Nequil, 
                  "variable Ninloop equal":Ninloop, 
                  "variable index loop":Nruns,
                  "    write_data INSTNCE1":basename+"*.dat"}

    run = swl.LammpsRun(srcdir=None,
                        basedir=basedir,
                        rundir=basedir,
                        executable='./lmp', #Relative in basedir
                        inputfile='mother_restart.in', #Relative to basedir
                        outputfile='mother.out',
                        inputchanges=inputchanges)

    study = swl.Study([[run]], 1)

    #Backup disp.dat of dissipation function
    if runcounter != 0:
        for rfile in rfiles:
            try:
                 shutil.copyfile(rfile, rfile+"to"+str(runcounter))
            except IOError:
                 print("Renamed file " + rfile + " not found")
    #increment run counter
    runcounter += Nequil+Ninloop*Nruns
    return runcounter


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
            if not "mirror" in rfile:
                phase_space_map(rfile,  maptype="reflectmom")

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

    ncpus = 4
    studyfolder = "study"
    results = []
    #Run batches of mother and child trajectories to avoid
    #large numbers of created files
    rc=run_mother(Nequil=10000, Nruns=0, Ninloop=0,runcounter=0)
    for i in range(3):
        print("runcounter = ", rc)
        rc=run_mother(Nequil=0, Nruns=ncpus, Ninloop=300, runcounter=rc)
        run_children(ncpus=ncpus, studyfolder=studyfolder)
        results.append(read_data(plot=False))
        clean(folder=studyfolder)
        pickle.dump(results, open("TTCF_run.p","w+"))



