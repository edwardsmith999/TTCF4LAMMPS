import numpy as np
import matplotlib.pyplot as plt

"""
    A set of tools to manipulate the LAMMPS restart files
    written by a master trajectory for children in TTCF.
    This includes making phase space mirrors, extracting 
    atom and bonds properties and getting the dissipation 
    function 
"""


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


def texttostr(bfile):
    #Read all data as text
    with open(bfile, "r") as f:
        t = f.read()
    return t

def getfilerec(f):
    return int(f.replace("branch","").replace(".dat",""))

def get_domain(t):
    #Get domain
    i = t.find("xlo")
    xlo = float(t[i-40:i].split("\n")[-1].split()[-2])
    xhi = float(t[i-40:i].split("\n")[-1].split()[-1])
    Lx = xhi - xlo
    i = t.find("ylo")
    ylo = float(t[i-60:i].split("\n")[-1].split()[-2])
    yhi = float(t[i-60:i].split("\n")[-1].split()[-1])
    Ly = yhi - ylo
    i = t.find("zlo")
    zlo = float(t[i-40:i].split("\n")[-1].split()[-2])
    zhi = float(t[i-40:i].split("\n")[-1].split()[-1])
    Lz = zhi - zlo
    return np.array([Lx, Ly, Lz])


def get_bond_coeff(t):
    i = t.find("Bond Coeffs")
    cline = t[i:].split("\n")[2]
    return float(cline.split()[1])


def get_atoms(t):

    #Get number of atoms
    i = t.find("atoms")
    Natoms = int(t[i-20:i].split("\n")[-1])

    #Find location of atoms
    i = t.find("Atoms")
    recs = t[i:].split("\n")[2:Natoms]
    atoms = np.zeros([Natoms,3])
    atomtype = np.zeros([Natoms])
    for r in recs:
        n, _, atype, _, x, y, z, _, _, _ = r.split()
        atoms[int(n)-1,:] = np.array([float(x), float(y), float(z)])
        atomtype[int(n)-1] = int(atype)

    return atoms, atomtype

def get_bonds(t, wrap_periodic=True, plotstuff=False, 
              tethersites=[4, 5], topbotflipsign=True):

    #Plot
    if plotstuff:
        fig, ax = plt.subplots(1,2)

    #Plot atomic positions    
    atoms, atomtype = get_atoms(t)
    if plotstuff:
        for n in range(atoms.shape[0]):
            if (atomtype[int(n)-1] in [4, 5]):
                ax[0].plot(atoms[int(n)-1,0], atoms[int(n)-1,1], 'rs')
                ax[1].plot(atoms[int(n)-1,2], atoms[int(n)-1,1], 'rs')
            else:
                ax[0].plot(atoms[int(n)-1,0], atoms[int(n)-1,1], 'bo', alpha=0.5)
                ax[1].plot(atoms[int(n)-1,2], atoms[int(n)-1,1], 'bo', alpha=0.5)

    #Get number of bonds
    i = t.find("bonds")
    Nbonds = int(t[i-20:i].split("\n")[-1])

    if wrap_periodic:
        domain = get_domain(t)
        halfdomain = 0.5*domain

    #Get location of bonds
    i = t.find("Bonds")
    recs = t[i:].split("\n")[2:Nbonds]
    bonds = np.zeros([Nbonds,3])
    for r in recs:
        n, _, i, j = r.split()

        if (atomtype[int(j)-1] in tethersites):
            bonds[int(n)-1,:] = atoms[int(i)-1,:] - atoms[int(j)-1,:]
        elif (atomtype[int(i)-1] in tethersites):
            bonds[int(n)-1,:] = atoms[int(j)-1,:] - atoms[int(i)-1,:]
        else:
            raise IOError("Read bond is not Tethered atom")

        if wrap_periodic:
            for ixyz in range(3):
                if (np.abs(bonds[int(n)-1,ixyz]) > halfdomain[ixyz]):
                    bonds[int(n)-1,ixyz] -= np.copysign(domain[ixyz], 
                                                        bonds[int(n)-1,ixyz])
        if plotstuff:
            ai = atoms[int(j)-1,:] + bonds[int(n)-1,:]

            ax[0].plot([ai[0], atoms[int(j)-1,0]],
                       [ai[1], atoms[int(j)-1,1]], '-r')

            ax[1].plot([ai[2], atoms[int(j)-1,2]],
                       [ai[1], atoms[int(j)-1,1]], '-r')

        #In calculating the dissipation, the bonds at the top are -ve
        #compared to the bonds at the bottom of the domain
        if topbotflipsign:
            bonds[int(n)-1,0] = np.copysign(bonds[int(n)-1,0], 
                                            atoms[int(i)-1,1]-halfdomain[1])

    if plotstuff:
        plt.show()

    return bonds

def get_force(bonds, coeff):
    F = coeff*np.sum(bonds,0)
    return F

def get_dissipation(F, beta, Uwall):
    return 0.5*beta*Uwall*F

def read_dissipation(fdir, T=1.0, Uwall=1.0, plotstuff=False):
    t = texttostr(fdir)
    bonds = get_bonds(t, plotstuff=plotstuff)
    coeff = get_bond_coeff(t)
    F = get_force(bonds, coeff)
    return get_dissipation(F[0], 1./T, Uwall)

if __name__ == "__main__":

    import glob
    files = glob.glob("branch*.dat")
    files.sort(key=getfilerec)

    for f in files:
        bonds = get_bonds(texttostr(f))
        disp = read_dissipation(f, plotstuff=True)
        print(f, np.sum(bonds,0), disp)
