import os.path
import numpy as np
import warnings
import os
import glob
import shutil
import matplotlib.pyplot as plt


class Bootstrap():
    
    def __init__(self, variable, nResamples, nTrajectories, nGroups, nTimesteps,
                 intervalPercentage=95):
        '''
            Create an object to perform bootstrapping on a given variable,
            it can be used after collecting data using TTCF 

            Inputs
            variable: name of the variable, a 'variable_global' or 'variable_profile' directory must exist
            nResamples: number of resamples for the bootstrap
            nTrajectories: number of daughters simulations performed for the TTCF
            nGroups: number of groups in which trajectories results will be pre-averaged
            nTimesteps: number of timesteps performed for every daughter
            intervalPercentage: confidence interval, default is 95%

            Limitations
            It is possible to group the trajctories in chuncks, called groups,
            to make less intensive the resample step.
            The current implementation works only if I calculate the mean and
            its confidence interval, if we want to evaluate the standard
            deviation of the original distribution we need to impose nGroups
            equal to nTrajectories.
        '''

        self.variable = variable.replace('[', '')
        self.variable = self.variable.replace(']', '')
        self.nResamples = nResamples
        self.nTrajectories = nTrajectories
        self.nGroups = nGroups
        self.nTimesteps = nTimesteps + 1
        self.intervalPercentage = intervalPercentage
        self.directory = os.getcwd() + '/' + str(self.variable) + '_global'
        self.variable_OB = []
        self.variable_O = []
        self.variable_B = []
        self.averageBt = []
        self.stdFlag = False

        if self.nTrajectories % self.nGroups != 0:
            raise Exception('The number of trajectories must be a multiple of the number of groups')
        precision = (100-intervalPercentage)/2
        exponent = 0
        while (precision % 1) != 0:
            exponent -= 1
            precision *= 10
        if ((5 * 10**exponent) * nResamples) % 100 != 0:
            warnings.warn('A number of resamples per processor that is a mutiple of {} should be used'.format(int(100/(5 * 10**exponent))))


    def concatenateTrajectories(self, comm):
        """
            Function used to concatenate the results written on different files
            by different processes, in single a single file per variable of the
            TTCF equation.
        """
        if comm.Get_rank() == 0:
            for i in ['OB', 'O', 'B']:
                outFilename = self.directory + '/' + str(self.variable) + '_' + i +'_global.txt'
                inFilenames = glob.glob(self.directory + '/' + str(self.variable) + '_' + i + '_*.dat')
                if os.path.isfile(outFilename):
                    # warnings.warn('File {} already exists, skipping concatenation'.format(outFilename))
                    print('File {} already exists, skipping concatenation'.format(outFilename))
                else:
                    with open(outFilename, 'w+') as outF:
                        for filename in inFilenames:
                            with open(filename, 'r') as inF:
                                shutil.copyfileobj(inF, outF)
                            os.remove(filename)


    def readTrajectories(self, comm):
        if comm.Get_rank() == 0:
            filenames = [self. directory + '/' + self.variable + '_OB_global.txt',
                         self. directory + '/' + self.variable + '_O_global.txt',
                         self. directory + '/' + self.variable + '_B_global.txt']
            arrays = [[], [], []]
            for i in range(len(filenames)):
                with open(filenames[i]) as f:
                    for line in f:
                        line = line.strip()
                        line = line.split()
                        arrays[i].append([float(x) for x in line])
            self.variable_OB = np.array(arrays[0])
            self.variable_O = np.array(arrays[1])
            self.variable_B = np.array(arrays[2])
        else:
            self.variable_OB = np.empty((self.nTrajectories, self.nTimesteps))
            self.variable_O = np.empty((self.nTrajectories, self.nTimesteps))
            self.variable_B = np.empty((self.nTrajectories, self.nTimesteps))
        comm.Bcast(self.variable_OB, root=0)
        comm.Bcast(self.variable_O, root=0)
        comm.Bcast(self.variable_B, root=0)


    def groupTrajectories(self):
        """
            The function groups the trajctories in chuncks, called groups, to
            make less intensive the resample step.
            The current implementation works only if I calculate the mean and
            its confidence interval, if we want to evaluate the standard
            deviation of the original distribution we need to impose nGroups
            equal to nTrajectories.

            The arrays in output have shape
            (nGroups, nTimesteps, nTrajectories per group).
        """
        self.variable_OB = self.variable_OB.reshape((self.nGroups, int(self.nTrajectories/self.nGroups), self.nTimesteps))
        self.variable_O = self.variable_O.reshape((self.nGroups, int(self.nTrajectories/self.nGroups), self.nTimesteps))
        self.variable_B = self.variable_B.reshape((self.nGroups, int(self.nTrajectories/self.nGroups), self.nTimesteps))
        self.variable_OB = self.variable_OB.transpose((0, 2, 1))
        self.variable_O = self.variable_O.transpose((0, 2, 1))
        self.variable_B = self.variable_B.transpose((0, 2, 1))


    def averageWithin(self):
        """
            The function takes the previous arrays and overages on
            the trajctories in every group.

            The arrays in output have shape (nGroups, nTimesteps).
        """
        self.variable_OB = np.mean(self.variable_OB, axis=-1)
        self.variable_O = np.mean(self.variable_O, axis=-1)
        self.variable_B = np.mean(self.variable_B, axis=-1)


    def resampleAvergeBetweenForMean(self):
        """
            This function performs the resampling with replacement, classic of
            the bootstrapping technique.
            The purpose is to calculate and store the mean of every resample.
            
            The arrays in output have shape (nTimesteps, nResamples).
        """
        batches = np.zeros((self.nResamples, self.nTimesteps))
        for i in range(self.nResamples):
            newSample = np.random.randint(self.nGroups, size=(self.nGroups))
            batches[i] = np.mean(self.variable_OB[newSample, :], axis=0)
        self.variable_OB_mean= batches.T
        batches = np.zeros((self.nResamples, self.nTimesteps))
        for i in range(self.nResamples):
            newSample = np.random.randint(self.nGroups, size=(self.nGroups))
            batches[i] = np.mean(self.variable_O[newSample, :], axis=0)
        self.variable_O_mean= batches.T
        batches = np.zeros((self.nResamples, self.nTimesteps))
        for i in range(self.nResamples):
            newSample = np.random.randint(self.nGroups, size=(self.nGroups))
            batches[i] = np.mean(self.variable_B[newSample, :], axis=0)
        self.variable_B_mean = batches.T


    def resampleAvergeBetweenForStd(self):
        """
            This function performs the resampling with replacement, classic of
            the bootstrapping technique.
            The purpose is to calculate and store the standard deviation of
            every resample.
            
            The arrays in output have shape (nTimesteps, nResamples).
        """
        self.stdFlag = True
        batches = np.zeros((self.nResamples, self.nTimesteps))
        for i in range(self.nResamples):
            newSample = np.random.randint(self.nGroups, size=(self.nGroups))
            batches[i] = np.std(self.variable_OB[newSample, :], axis=0)
        self.variable_OB_std = batches.T
        batches = np.zeros((self.nResamples, self.nTimesteps))
        for i in range(self.nResamples):
            newSample = np.random.randint(self.nGroups, size=(self.nGroups))
            batches[i] = np.std(self.variable_O[newSample, :], axis=0)
        self.variable_O_std = batches.T
        batches = np.zeros((self.nResamples, self.nTimesteps))
        for i in range(self.nResamples):
            newSample = np.random.randint(self.nGroups, size=(self.nGroups))
            batches[i] = np.std(self.variable_B[newSample, :], axis=0)
        self.variable_B_std = batches.T


    def sumOriginalTrj(self):
        self.averageBt = self.variable_OB - self.variable_O*self.variable_B


    def sumIntegrals(self):
        self.averageBtMean = self.variable_OB_mean - self.variable_O_mean*self.variable_B_mean
        if self.stdFlag:
            self.averageBtStd = self.variable_OB_std - self.variable_O_std*self.variable_B_std

    def gather_over_MPI(self, comm, root=0):
        """
            Function used to gather from different MPI processes the results of
            the resampling procedure.
            Since we are gathering numpy arrays a send and a received buffer
            must be used.
            In addition, the array to be sent must be stored as contiguous.

            The gathered array are reshaped in order to have shape
            (nTimesteps, nResamples*nProcesses)
        """
        self.recvMean = None
        self.recvStd = None
        self.sendMean = np.ascontiguousarray(self.averageBtMean)
        if self.stdFlag:
            self.sendStd = np.ascontiguousarray(self.averageBtStd)
        if comm.Get_rank() == 0:
            recvSize = [comm.Get_size()] + [self.averageBtMean.shape[i] for i in range(len(self.averageBtMean.shape))]
            self.recvMean = np.empty(recvSize, dtype=self.sendMean.dtype)
            if self.stdFlag:
                self.recvStd = np.empty(recvSize, dtype=self.sendStd.dtype)
        comm.Gather(self.sendMean, self.recvMean, root=root)
        if self.stdFlag:
            comm.Gather(self.sendStd, self.recvStd, root=root)
        if comm.Get_rank() == 0:
            self.averageBtMean = self.recvMean
            self.averageBtMean = self.averageBtMean.transpose((1, 0, 2))
            self.averageBtMean = self.averageBtMean.reshape((self.nTimesteps, self.averageBtMean.shape[1]*self.averageBtMean.shape[2]))
            if self.stdFlag:
                self.averageBtStd = self.recvStd
                self.averageBtStd = self.averageBtStd.transpose((1, 0, 2))
                self.averageBtStd = self.averageBtStd.reshape((self.nTimesteps, self.averageBtStd.shape[1]*self.averageBtStd.shape[2]))

    def confidenceInterval(self):
        """
            Function used to compute and write to file the confidence interval
            of the mean from the resampling process.
        """
        lowPercentile = int((100-self.intervalPercentage)/2*self.averageBtMean.shape[1]/100)
        highPercentile = int((100+self.intervalPercentage)/2*self.averageBtMean.shape[1]/100)
        confInterval = np.zeros((self.nTimesteps, 2))
        for i in range(len(self.averageBtMean)):
            sortedArray = np.sort(self.averageBtMean[i])
            confInterval[i][0] = sortedArray[lowPercentile]
            confInterval[i][1] = sortedArray[highPercentile]
        np.savetxt(self.directory + '/' + self.variable + '_ci.txt', confInterval)


    def meanForComparison(self):
        """
            Function used to compute and write to file the mean and the
            standard deviation of the mean distribution, i.e. the standard
            error from the resampling process.

            Note: that there is no need to compute the mean in this way, since
            the TTCF already compute the correct mean, so this function is used
            as sanity check.
        """
        meanBt = np.mean(self.averageBtMean, axis=1)
        seBt = np.std(self.averageBtMean, axis=1)
        np.savetxt(self.directory + '/' + self.variable + '_mean.txt', meanBt)
        np.savetxt(self.directory + '/' + self.variable + '_meanStd.txt', seBt)


    def standardDeviation(self, identifier):
        """
            Function used to compute the standard deviation of the original
            distribution from the resampling process.
        """
        stdBt = np.mean(self.averageBtStd, axis=1)
        np.savetxt(self.directory + '/' + self.variable + '_std_' + identifier + '.txt', stdBt)


    def plotDistribution(self, timestep=-1, format='png'):
        """
            Function used to plot the distribution of the mean from the
            resampling process for a specific timestep.
            The default is the last timestep.
        """
        sortedArray = np.sort(self.averageBtMean[timestep])
        lowPercentile = int((100-self.intervalPercentage)/2*len(sortedArray)/100)
        highPercentile = int((100+self.intervalPercentage)/2*len(sortedArray)/100)
        confIntLow = sortedArray[lowPercentile]
        confIntHigh = sortedArray[highPercentile]
        centimeters = 1/2.54
        fig = plt.figure(figsize=(10*centimeters, 10*centimeters), constrained_layout=True)
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        ax.set_xlabel(r'$\langle B\rangle$')
        ax.set_ylabel(r'Count')
        ax.hist(self.averageBtMean[timestep], bins=100, color='tomato', alpha=.5)
        ax.vlines(np.mean(self.averageBtMean[timestep]), ymin=0, ymax=self.nResamples*3.5e-2, colors='tomato', label='Mean')
        ax.vlines([confIntLow, confIntHigh], ymin=0, ymax=self.nResamples*3.5e-2, colors='deepskyblue', label='95% Confidence interval')
        ax.legend()
        fig.savefig(self.directory + '/' + self.variable + '_averageDistribution.'+ format, dpi=300)
