
import numpy as np
import matplotlib.pyplot as plt

with open("profile.wall.2d", "r") as f:
    filestr = f.read()

header = filestr.split("\n")[:3]
body = filestr.split("\n")[3:]
step, Ny, r = [int(a) for a in body[0].split()]

def read_rec(no):
    u = []
    for i in range(1+no*(Ny+1), Ny+1+no*(Ny+1)):
        u.append([float(a) for a in body[i].split()])
    u = np.array(u)
    return u

fig, ax = plt.subplots(1,1)
#plt.ion()
#plt.show()

for rec in range(10):
    u = read_rec(rec)
    # 0=Chunk 1=Coord1 2=Ncount 3=density/mass
    # 4=vx 5=vy 6=vz 7=temp 8=c_stress[1] 
    # 9=c_stress[2] 10=c_stress[3] 11=c_stress[4]
    plt.plot(u[:,1], u[:,3], label="Density")
    plt.plot(u[:,1], u[:,4], label="vx")
    plt.plot(u[:,1], u[:,5], label="vy")
    plt.plot(u[:,1], u[:,6], label="vz")
    plt.plot(u[:,1], u[:,7], label="T")
    plt.plot(u[:,1], u[:,8], label="Pxx")
    plt.plot(u[:,1], u[:,9], label="Pyy")
    plt.plot(u[:,1], u[:,10], label="Pzz")
    plt.plot(u[:,1], u[:,11], label="Pxy")

    plt.legend()
    #plt.ylim([-1.1, 2.1])
    #plt.pause(1.0)
    plt.show()

