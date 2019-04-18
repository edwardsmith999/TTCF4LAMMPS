import numpy as np
import glob


def read_data(fdir="study", dt=0.005, plot=True, limit=None):

    #Get all folders
    folders = glob.glob(fdir+"/ttcfmirror*")
    folders.sort()

    if limit != None:
        folders = folders[:limit]

    count=0
    ft = True
    for readfile in folders:

        try:
            print("Reading file ", readfile.replace("mirror","") , " and mirror")
            mirror = np.genfromtxt(readfile + "/output.txt")
            path = np.genfromtxt(readfile.replace("mirror","") + "/output.txt")

            if ft:
                firstpath = path 
                ave = np.zeros(path.shape)
                ft=False

            #Add each mirrored pair together
            ave += mirror+path
            count += 2
        except IOError:
            print(readfile + " fails")
        except ValueError:
            print(readfile + " fails")

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(dt*firstpath[:,0], ave[:,5]/count, 'r-')
        plt.plot(dt*firstpath[:,0], ave[:,6]/count, 'r--')
        plt.plot(dt*firstpath[:,0], ave[:,2]/count, 'k-')
        plt.plot(dt*firstpath[:,0], ave[:,3]/count, 'b-')
        plt.plot(dt*firstpath[:,0], ave[:,4]/count, 'g-')
        plt.show()

    return ave/count


if __name__ == "__main__":
    ave = read_data(plot=True)
