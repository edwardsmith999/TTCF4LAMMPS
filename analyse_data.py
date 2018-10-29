import numpy as np
import matplotlib.pyplot as plt
import glob

restartfile = "{:d}"
children = []
disp = 0.0; Pxy = 0.0
count = 0
dt = 0.005
#mother = np.genfromtxt("mother.txt")

folders = glob.glob("study/ttcfmirror*")
folders.sort()

for readfile in folders:

    try:
        print("Reading file ", readfile.replace("mirror","") , " and mirror")
        mirror = np.genfromtxt(readfile + "/output.txt")
        path = np.genfromtxt(readfile.replace("mirror","") + "/output.txt")

        Pxy += mirror[:,2]-mirror[:,3]
        Pxy += path[:,2]-path[:,3]
        disp += mirror[:,4]-mirror[:,5]
        disp += path[:,4]-path[:,5]
        count += 2
    except IOError:
        print(readfile + " fails")

plt.plot(dt*path[:,0], disp/count)
plt.plot(dt*path[:,0], Pxy/count)
plt.show()

