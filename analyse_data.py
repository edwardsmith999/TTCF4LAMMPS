import numpy as np
import glob

import numpy as np
import glob

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

def read_data(fdir="study", limit=None):

    #Get all folders
    folders = glob.glob(fdir+"/ttcfmirror*")
    folders.sort()

    if limit != None:
        folders = folders[:limit]

    data = []
    for readfile in folders:
        try:
            print("Reading file ", readfile.replace("mirror","") , " and mirror")
            mirror = np.genfromtxt(readfile + "/output.txt")
            data.append(mirror)
            path = np.genfromtxt(readfile.replace("mirror","") + "/output.txt")
            data.append(path)
        except IOError:
            print(readfile + " fails")
        except ValueError:
            print(readfile + " fails")

    return data



def average_data(fdir="study", dt=0.005, plot=True, limit=None):

    data = read_data(fdir, limit)

    count=0
    ft = True
    for d in data:

        if ft:
            firstpath = d
            ave = np.zeros(path.shape)
            ft=False

        #Add each mirrored pair together
        ave += d
        count += 2

    if plot:
        assert ft is False
        import matplotlib.pyplot as plt
        plt.plot(dt*firstpath[:,0], ave[:,5]/count, 'r-')
        plt.plot(dt*firstpath[:,0], ave[:,6]/count, 'r--')
        plt.plot(dt*firstpath[:,0], ave[:,2]/count, 'k-')
        plt.plot(dt*firstpath[:,0], ave[:,3]/count, 'b-')
        plt.plot(dt*firstpath[:,0], ave[:,4]/count, 'g-')
        plt.show()

    return ave/count


def correlate_time(A, rng=None, skip=1):

    if rng == None:
        rng = int(A.shape[0]/2.)

    autocorrel = np.zeros(rng)
    for shift in range(0, rng-1, skip): 
        autocorrel += A[shift]*A[shift:shift+rng]

    return autocorrel


if __name__ == "__main__":
    ave = average_data(plot=True)

#def read_data(fdir="study", dt=0.005, plot=True, limit=None):

#    #Get all folders
#    folders = glob.glob(fdir+"/ttcfmirror*")
#    folders.sort()

#    if limit != None:
#        folders = folders[:limit]

#    count=0
#    ft = True
#    for readfile in folders:

#        try:
#            print("Reading file ", readfile.replace("mirror","") , " and mirror")
#            mirror = np.genfromtxt(readfile + "/output.txt")
#            path = np.genfromtxt(readfile.replace("mirror","") + "/output.txt")

#            if ft:
#                firstpath = path 
#                ave = np.zeros(path.shape)
#                ft=False

#            #Add each mirrored pair together
#            ave += mirror+path
#            count += 2
#        except IOError:
#            print(readfile + " fails")
#        except ValueError:
#            print(readfile + " fails")

#    if plot:
#        import matplotlib.pyplot as plt
#        plt.plot(dt*firstpath[:,0], ave[:,5]/count, 'r-')
#        plt.plot(dt*firstpath[:,0], ave[:,6]/count, 'r--')
#        plt.plot(dt*firstpath[:,0], ave[:,2]/count, 'k-')
#        plt.plot(dt*firstpath[:,0], ave[:,3]/count, 'b-')
#        plt.plot(dt*firstpath[:,0], ave[:,4]/count, 'g-')
#        plt.show()

#    return ave/count

