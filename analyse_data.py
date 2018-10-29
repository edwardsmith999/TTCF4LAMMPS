import numpy as np
import matplotlib.pyplot as plt

restartfile = "{:d}"
children = []
disp = 0.0; Pxy = 0.0
count = 0
dt = 0.005
mother = np.genfromtxt("mother.txt")

for i in range(100300, 199900+300, 300):
    readfile = restartfile.format(i)

    try:
        print("Reading file ", readfile)
        mirror = np.genfromtxt("study/ttcf"+readfile + "/output.txt")
        path = np.genfromtxt("study/ttcf"+"mirror"+readfile + "/output.txt")

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

