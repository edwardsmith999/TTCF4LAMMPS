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
        '''

        self.variable = variable.replace('[', '')
        self.variable = self.variable.replace(']', '')
        self.nResamples = nResamples
        self.nTrajectories = nTrajectories
        self.nGroups = nGroups
        self.nTimesteps = nTimesteps + 1
        self.intervalPercentage = intervalPercentage
        self.directory = os.getcwd() + '/' + str(variable) + '_global'
        self.variable_OB = []
        self.variable_O = []
        self.variable_B = []
        self.averageBt = []


        if self.nTrajectories % self.nGroups != 0:
            raise Exception('The number of trajectories must be a multiple of the number of groups')
        precision = (100-intervalPercentage)/2
        exponent = 0
        while (precision % 1) != 0:
            exponent -= 1
            precision *= 10
        if ((5 * 10**exponent) * nResamples) % 100 != 0:
            warnings.warn('It would be better to use a number of resamples that is a mutiple of {}'.format(int(100/(5 * 10**exponent))))


    def concatenateTrajectories(self):
        for i in ['OB', 'O', 'B']:
            outFilename = self.directory + '/' + str(self.variable) + '_' + i +'_global.txt'
            inFilenames = glob.glob(self.directory + '/' + str(self.variable) + '_' + i + '_*.dat')
            with open(outFilename, 'w+') as outF:
                for filename in inFilenames:
                    with open(filename, 'r') as inF:
                        shutil.copyfileobj(inF, outF)
                    os.remove(filename)


    def readTrajectories(self):
        filenames = [self. directory + '/' + self.variable + '_OB_global.txt', self. directory + '/' + self.variable + '_O_global.txt', self. directory + '/' + self.variable + '_B_global.txt']
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
        print(self.variable_OB.shape)
        print(self.variable_O.shape)
        print(self.variable_B.shape)


    def groupTrajectories(self):
        self.variable_OB = self.variable_OB.reshape((self.nGroups, int(self.nTrajectories/self.nGroups), self.nTimesteps))
        self.variable_O = self.variable_O.reshape((self.nGroups, int(self.nTrajectories/self.nGroups), self.nTimesteps))
        self.variable_B = self.variable_B.reshape((self.nGroups, int(self.nTrajectories/self.nGroups), self.nTimesteps))
        self.variable_OB = self.variable_OB.transpose((0, 2, 1))
        self.variable_O = self.variable_O.transpose((0, 2, 1))
        self.variable_B = self.variable_B.transpose((0, 2, 1))
        print(self.variable_OB.shape)
        print(self.variable_O.shape)
        print(self.variable_B.shape)


    def averageWithin(self):
        self.variable_OB = np.mean(self.variable_OB, axis=-1)
        self.variable_O = np.mean(self.variable_O, axis=-1)
        self.variable_B = np.mean(self.variable_B, axis=-1)
        print(self.variable_OB.shape)
        print(self.variable_O.shape)
        print(self.variable_B.shape)


    # def resampleAvergeBetween(self):
    #     arrayShape = list(self.variable_OB.shape)
    #     arrayShape.insert(0, self.nResamples)
    #     arrayShape = tuple(arrayShape)
    #     batches = np.zeros(arrayShape)
    #     for i in range(self.nResamples):
    #         newSample = np.random.randint(arrayShape[1], size=(arrayShape[1]))
    #         batches[i] = self.variable_OB[newSample, :]
    #     self.variable_OB = np.mean(batches, axis=-2).T
    #     for i in range(self.nResamples):
    #         newSample = np.random.randint(arrayShape[1], size=(arrayShape[1]))
    #         batches[i] = self.variable_O[newSample, :]
    #     self.variable_O = np.mean(batches, axis=-2).T
    #     for i in range(self.nResamples):
    #         newSample = np.random.randint(arrayShape[1], size=(arrayShape[1]))
    #         batches[i] = self.variable_B[newSample, :]
    #     self.variable_B = np.mean(batches, axis=-2).T


    def resampleAvergeBetween(self):
        arrayShape = self.variable_OB.shape
        batches = np.zeros((self.nResamples, arrayShape[1]))
        for i in range(self.nResamples):
            newSample = np.random.randint(arrayShape[0], size=(arrayShape[0]))
            batches[i] = np.mean(self.variable_OB[newSample, :], axis=0)
        self.variable_OB = batches.T
        batches = np.zeros((self.nResamples, arrayShape[1]))
        for i in range(self.nResamples):
            newSample = np.random.randint(arrayShape[0], size=(arrayShape[0]))
            batches[i] = np.mean(self.variable_O[newSample, :], axis=0)
        self.variable_O = batches.T
        batches = np.zeros((self.nResamples, arrayShape[1]))
        for i in range(self.nResamples):
            newSample = np.random.randint(arrayShape[0], size=(arrayShape[0]))
            batches[i] = np.mean(self.variable_B[newSample, :], axis=0)
        self.variable_B = batches.T
        print(self.variable_OB.shape)
        print(self.variable_O.shape)
        print(self.variable_B.shape)


    # def averageBetween(self, batches):
    #     return np.mean(batches, axis=-2).T


    def sumIntegrals(self):
        self.averageBt = self.variable_OB - self.variable_O*self.variable_B
        print(self.averageBt)


    def confidenceInterval(self):
        lowPercentile = int((100-self.intervalPercentage)/2*self.nResamples/100)
        highPercentile = int((100+self.intervalPercentage)/2*self.nResamples/100)
        confInterval = np.zeros((self.nTimesteps, 2))
        for i in range(len(self.averageBt)):
            sortedArray = np.sort(self.averageBt[i])
            confInterval[i][0] = sortedArray[lowPercentile]
            confInterval[i][1] = sortedArray[highPercentile]
        np.savetxt(self.directory + '/' + self.variable + '_ci.txt', confInterval)


    def plotDistribution(self, timestep=-1, format='png'):
        sortedArray = np.sort(self.averageBt[timestep])
        lowPercentile = int((100-self.intervalPercentage)/2*self.nResamples/100)
        highPercentile = int((100+self.intervalPercentage)/2*self.nResamples/100)
        confIntLow = sortedArray[lowPercentile]
        confIntHigh = sortedArray[highPercentile]
        centimeters = 1/2.54
        fig = plt.figure(figsize=(10*centimeters, 10*centimeters), constrained_layout=True)
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        ax.set_xlabel(r'$\langle B\rangle$')
        ax.set_ylabel(r'Count')
        ax.hist(self.averageBt[timestep], bins=100, color='tomato', alpha=.5)
        ax.vlines(np.mean(self.averageBt[timestep]), ymin=0, ymax=self.nResamples*3.5e-2, colors='tomato', label='Mean')
        ax.vlines([confIntLow, confIntHigh], ymin=0, ymax=self.nResamples*3.5e-2, colors='deepskyblue', label='95% Confidence interval')
        ax.legend()
        fig.savefig(self.directory + '/' + self.variable + '_averageDistribution.'+ format, dpi=300)
