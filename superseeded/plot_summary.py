import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import glob as glob
from scipy import integrate

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
U = 0.00025
Lmin = 0.3
Lmax = 5.3
L = Lmax - Lmin
Re = rho * U * L / mu
gamma = 2.*U/L
Tinit = 1.0
dt = 0.005
Lx = 30.
Lz = 30.
A = Lx*Lz

#Load data
# alldata shape = [batches, trajectories, time, quantities]
# The [batches] are batches of trajectories 
# [Trajectories] are pairs of mirrored child trajectories from a single
#                mother trajectory run over a number of steps 
# The [time] is the time histroy of the child trajectory
# The [quantities] are as follows:
# step, temperature, MOP_lower, MOP_center, MOP_upper, Virial_Pxy, 
# liquid_solid_fij_lower, liquid_solid_fij_upper, 
# sum tether_Fx_lower, sum tether_Fx_upper
# MOP_direct_lower MOP_direct_center MOP_direct_upper
# pyy fij_direct_lower fij_direct_upper
alldata = np.load("./outfile.npy") 

t = np.linspace(0.,alldata.shape[2]*dt,alldata.shape[2])

#Get average over all trajectories
data = np.mean(alldata,(0,1))
T = data[:,1]
MOPl = data[:,2]
MOPc = data[:,3]
MOPu = data[:,4]
Pxy = data[:,5]
fijl = data[:,6]
fiju = data[:,7]
displ = data[:,8]
dispu = data[:,9]

#Get error over all trajectories
error = np.std(alldata,(0,1))/np.sqrt(alldata.shape[0]*alldata.shape[1])
Te = error[:,1]
MOPle = error[:,2]
MOPce = error[:,3]
MOPue = error[:,4]
Pxye = error[:,5]
fijle = error[:,6]
fijue = error[:,7]
disple = error[:,8]
dispue = error[:,9]

dF = (alldata[:,:,:,7]-alldata[:,:,:,6])*L/A

#Split mirror and original
mp = alldata.reshape([alldata.shape[0],alldata.shape[1]/2,2,
                      alldata.shape[2],alldata.shape[3]])
del alldata

#Extract quantities
T = mp[:,:,:,:,1]
beta = 1./T
MOP_lower = mp[:,:,:,:,2]
MOP_center = mp[:,:,:,:,3]
MOP_upper = mp[:,:,:,:,4]
Virial_Pxy = mp[:,:,:,:,5]
liquid_solid_fij_lower = mp[:,:,:,:,6]/A
liquid_solid_fij_upper = mp[:,:,:,:,7]/A
liquid_solid_fij = liquid_solid_fij_upper - liquid_solid_fij_lower
disp_lower = -0.5*beta*gamma*L*mp[:,:,:,:,8]
disp_upper = 0.5*beta*gamma*L*mp[:,:,:,:,9]
disp = disp_upper+disp_lower
disp0 = disp[:,:,:,0]

# Get ttcf integrand, sum over batches [0], child trajectories [1] and mirror pairs [2]
integrand = np.einsum('ijm,ijmkl->kl', disp0, mp)/(mp.shape[0]*mp.shape[1]*mp.shape[2])

# Statistics improved by using
# <disp(0)*B(s)> - <disp(0)>*<B(s)>
integrand -= np.mean(disp0)*np.mean(mp,(0,1,2))

#Dissipation function used can be corresponding one for upper and lower surface if not average quantity
# Lower surface
for i in [2,6,8]: 
    integrand[:,i] = np.einsum('ijm,ijmk->k', disp_lower[:,:,:,0], mp[:,:,:,:,i])/(mp.shape[0]*mp.shape[1]*mp.shape[2])
    integrand[:,i] -= np.mean(disp_lower[:,:,:,0])*np.mean(mp[:,:,:,:,i],(0,1,2))
# Upper surface
for i in [4,7,9]:
    integrand[:,i] = np.einsum('ijm,ijmk->k', disp_upper[:,:,:,0], mp[:,:,:,:,i])/(mp.shape[0]*mp.shape[1]*mp.shape[2])
    integrand[:,i] -= np.mean(disp_upper[:,:,:,0])*np.mean(mp[:,:,:,:,i],(0,1,2))

#Get TTCF by integrating function
TTCF = np.empty(integrand.shape)
for i in range(integrand.shape[1]):
    TTCF[:-1,i] = integrate.cumtrapz(integrand[:,i], dx=dt)

del integrand

#Gives pretty much the same as a looped simpsons integral
#TTCFs = np.empty(integrand.shape)
#for i in range(integrand.shape[1]):
#    for j in range(1,integrand.shape[0]):
#        TTCFs[j,i] = integrate.simps(integrand[:j,i], dx=dt)

#Add mean of initial value (skip this as it just adds noise)
#TTCF += np.mean(mp[:,:,:,0,:],(0,1,2))


#Keep all trajectories
integrand_alldata = np.einsum('ijm,ijmkl->ijmkl', disp0, mp)
TTCF_alldata = np.empty(integrand_alldata.shape)
for i in range(integrand_alldata.shape[-1]):
    TTCF_alldata[:,:,:,:-1,i] = integrate.cumtrapz(integrand_alldata[:,:,:,:,i], dx=dt)

del integrand_alldata

#Save TTCF quantities with names
TTCF_data = np.mean(TTCF_alldata,(0,1,2))
TTTCF = TTCF_data[:,1]
MOPlTTCF = TTCF_data[:,2]
MOPcTTCF = TTCF_data[:,3]
MOPuTTCF = TTCF_data[:,4]
PxyTTCF = TTCF_data[:,5]
fijlTTCF = TTCF_data[:,6]
fijuTTCF = TTCF_data[:,7]
displTTCF = TTCF_data[:,8]
dispuTTCF = TTCF_data[:,9]

#Get TTCF error over all trajectories
TTCF_error = np.std(TTCF_alldata,(0,1,2))/np.sqrt(TTCF_alldata.shape[0]*TTCF_alldata.shape[1]*TTCF_alldata.shape[2])
TTTCFe = TTCF_error[:,1]
MOPlTTCFe = TTCF_error[:,2]
MOPcTTCFe = TTCF_error[:,3]
MOPuTTCFe = TTCF_error[:,4]
PxyTTCFe = TTCF_error[:,5]
fijlTTCFe = TTCF_error[:,6]
fijuTTCFe = TTCF_error[:,7]
displTTCFe = TTCF_error[:,8]
dispuTTCFe = TTCF_error[:,9]

del TTCF_alldata

#Plot virials
fig, ax = plt.subplots(1,1)
ax.errorbar(t, Pxy, Pxye, errorevery=50, color="b", ecolor='k', label="Virial DAV")
ax.errorbar(t, PxyTTCF, PxyTTCFe, errorevery=50, color="r", ecolor='k', label="Virial TTCF")
ax.legend()
plt.show()

#Plot data
fig, ax = plt.subplots(2,1)
ax[0].plot(t, MOPl, color='k', label="MOP lower")
ax[0].plot(t, MOPc, color='b', label="MOP centre")
ax[0].plot(t, MOPu, color='r', label="MOP upper")
ax[0].plot(t, Pxy, color='g', label="Pxy Virial")

ax[1].plot(t, MOPlTTCF, 'k--', label="TTCF MOP lower")
ax[1].plot(t, MOPcTTCF, 'b-', label="TTCFMOP centre")
ax[1].plot(t, MOPuTTCF, 'r-', label="TTCFMOP upper")
ax[1].plot(t, PxyTTCF, 'g-', label="TTCFPxy Virial")

ax[0].legend()
ax[1].legend()
plt.show()

fig, ax = plt.subplots(2,1)
ax[0].plot(t, fijl/A - fiju/A, 'k-', label="fij wall")
ax[0].plot(t, fijlTTCF/A - fijuTTCF/A, 'r-', label="TTCF fij wall")
ax[1].plot(t, displ/A - dispu/A, 'k-', label="disp")
ax[1].plot(t, displTTCF/A - dispuTTCF/A, 'r-', label="TTCF disp")

#ax[0].plot(t, fijl/A, 'k-', label="fijl")
#ax[0].plot(t, fiju/A, 'b-', label="fiju")
#ax[0].plot(t, fijlTTCF/A, 'r-', label="TTCF fijl")
#ax[0].plot(t, fijuTTCF/A, 'g-', label="TTCF fiju")

#ax[1].plot(t, displ/A, 'k-', label="displ")
#ax[1].plot(t, dispu/A, 'b-', label="dispu")
#ax[1].plot(t, displTTCF/A, 'r-', label="TTCF displ")
#ax[1].plot(t, dispuTTCF/A, 'g-', label="TTCF dispu")

ax[0].legend()
ax[1].legend()
plt.show()

# Get dissipation function of mother trajectory
# and correlate with initial value
#disptb = read_disp()
#ad = np.reshape(alldata,(disptb.shape[0],alldata.shape[-2],alldata.shape[-1]))
#assert disptb.shape[0] == alldata.shape[0]
#TTCF = np.zeros(ad.shape)
#for i in range(disptb.shape[0]):
#    #disp = disptb[i,1] + disptb[i,2]
#    TTCF[i,:,:] = ad[i,:,:]*disptb[i,1]


#Bernadi style plot of Direct AVeraging (DAV) data
fig, ax = plt.subplots(1,1)
d = np.mean(Virial_Pxy,(0,1,2))
s = np.std(Virial_Pxy,(0,1,2))/np.sqrt(Virial_Pxy.shape[0]*Virial_Pxy.shape[1])
plt.errorbar(t, d[:], s[:], errorevery=50, ecolor="k", capsize=2)

#dF is difference between force over top and bottom
# used for TTCF in 
# Delhommelle and Cummings (2005) PHYSICAL REVIEW B 72, 172201
#ax2 = ax.twinx()
a_dF0dF = np.einsum('ij,ijk->k', dF[:,:,0], dF[:,:,:])/(dF.shape[0]*dF.shape[1])
a_dF0 = np.mean(dF[:,:,0])
a_dF = np.mean(dF[:,:,:], (0,1))
inint = a_dF0dF - a_dF0*a_dF
ax.plot(t[:-1], -np.array([integrate.simps(inint[:i], dx=dt) for i in range(1,inint.shape[0])]),'r-')
plt.show()


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

