import unittest
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp


def load_var(filename, var):
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('variable '+var):
                try:
                    out = float(line.split()[3])
                except ValueError:
                    print("Error: Unable to convert " + var + " to a float.")
                    out = None
                break
    return out

#Run TTCF code 
sp.call("mpiexec -n 4 python3 run_TTCF.py", cwd="../", shell=True)

#Load output profiles
DAV = np.loadtxt("../profile_DAV.txt")
TTCF = np.loadtxt("../profile_TTCF.txt")

#Analytical solution
filename = "../system_setup.in"
srate = load_var(filename, "srate")
rho = load_var(filename, "rho")
Npart = load_var(filename, "Npart")
print(rho, srate, Npart)
L = (Npart/rho)**(1.0/3.0)
x = np.linspace(0., L, TTCF.shape[1])
vx = srate*x

#Compare analytical to TTCF solution numerically


#Plot results
plt.plot(x, np.mean(DAV,0))
plt.plot(x, np.mean(TTCF,0))
plt.plot(x, vx, '-k')
plt.show()

