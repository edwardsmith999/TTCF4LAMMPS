import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import glob as glob

def getfilerec(f):
    return int(f.replace("disp.datto",""))

def read_disp(filebase="disp.datto", step=300):
    files =glob.glob(filebase + "*")
    #k = [sortfn(f) for f in files]
    files.sort(key=getfilerec)
    nfiles = len(files)
    d = np.genfromtxt(files[0])
    nperfile = d[::step,:].shape[0] 
    nrecs = nperfile*nfiles
    disp = np.zeros([nrecs, d.shape[1]])
    for i, f in enumerate(files):
        print(i, i*step, f)
        d = np.genfromtxt(f)
        startrec = getfilerec(f)
        disp[nperfile*i:nperfile*(i+1),0] = startrec+d[::step,0]
        disp[nperfile*i:nperfile*(i+1),1:] = d[::step,1:]

    return disp

#Set constants
mu = 1.7
rho = 0.8442
U = 1
Lmin = 1.0
Lmax = 7.5
L = Lmax - Lmin
Re = rho * U * L / mu
gamma = U/L
Tinit = 1.0

#Load data
out = pickle.load(open("TTCF_run.p", "r"))
alldata = np.array(out)
data = np.mean(alldata, (0,1))
T = data[:,1]
MOPl = data[:,2]
MOPc = data[:,3]
MOPu = data[:,4]
beta = 1./Tinit
Pxy = 0.5*beta*U*data[:,5]
#displ = 0.5*beta*U*data[:,6]

#Plot data
fig, ax = plt.subplots(1,1)
#ax.plot(T, label="T")
ax.plot(MOPl, label="MOP lower")
ax.plot(MOPc, label="MOP centre")
ax.plot(MOPu, label="MOP upper")
plt.legend()
#ax2 = ax.twinx()
ax.plot(Pxy, 'k-', label="Pxy Virial")
#ax2.plot(displ, 'k--', label="Disp upper")
plt.legend()
plt.show()

# Get dissipation function of mother trajectory
# and correlate with initial value
disptb = read_disp()
ad = np.reshape(alldata,(disptb.shape[0],alldata.shape[-2],alldata.shape[-1]))
assert disptb.shape[0] == alldata.shape[0]
TTCF = np.zeros(ad.shape)
for i in range(disptb.shape[0]):
    #disp = disptb[i,1] + disptb[i,2]
    TTCF[i,:,:] = ad[i,:,:]*disptb[i,1]


######################################

# Attempt to plot analytical solution for stress as
# a function of time to check MOP output

def stress_at_loc(y, H=1., nmodes=20000):

    tau = 1.
    dt = 0.005
    t = 0. #dt*np.arange(1000)
    n = np.arange(nmodes)
    lam = n*np.pi*y/H 
    tau = np.sum(2* (-1)**n * np.exp(-mu*lam*t/rho)*np.cos(lam))
    tau = -(mu*U/H)*tau 


from CouetteAnalytical import CouetteAnalytical

CA = CouetteAnalytical(Re, U, Lmin, Lmax, nmodes=20000)
tautime = []
trng = 0.05*np.arange(60)
plot = True

if plot:
    fig, ax = plt.subplots(1,1)
    plt.ion()

for t in trng:

    print(t)
    #Get velocity
    y, u = CA.get_vprofile(t)
    y, uf = CA.get_vprofile(t, flip=True)
    u -= uf

    #Stress
    tau = mu*np.gradient(u, y)
    dy = np.gradient(y)[3]
    tautime.append(tau)

    if plot:
        ax.plot(y, u, 'k-')
        ax.plot(y, tau, 'r')
        plt.pause(0.001)
        plt.cla()

plt.clf()
plt.ioff()
fig, ax = plt.subplots(1,1)
tautime = np.array(tautime)
[ax.plot(trng, tautime[:,i], label="y="+ str(1.+i*dy)) for i in range(0,10,2)]
plt.legend()
plt.show()

scale = 3.
fig, ax = plt.subplots(1,1)
ax.plot(np.linspace(0, scale*trng.max(), MOPl.shape[0]),MOPl, label="MOP lower")
ax.plot(np.linspace(0, scale*trng.max(), MOPc.shape[0]),MOPc, label="MOP centre")
ax.plot(np.linspace(0, scale*trng.max(), MOPu.shape[0]), MOPu, label="MOP upper")
for i in [3,10,16]:
    ax.plot(trng, -tautime[:,i], label="y="+ str(1.+i*dy))
plt.legend()
ax.set_xlim(0., scale*trng.max())
plt.show()
#import scipy
#scale = tautime.shape[0]/float(MOPl.shape[0])
#ax.plot(trng, scipy.ndimage.zoom(MOPl, scale), label="MOP lower")
#ax.plot(trng, scipy.ndimage.zoom(MOPl, scale), label="MOP centre")
#ax.plot(trng, scipy.ndimage.zoom(MOPl, scale), label="MOP upper")

