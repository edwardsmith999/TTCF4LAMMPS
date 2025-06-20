import unittest
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp
from scipy.optimize import curve_fit
import os 
import pytest

plotstuff = True

def fn(x, m):
    return m*x

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

# Fixture to run the command before tests
@pytest.fixture(scope="session", autouse=True)
def run_command_before_tests():
    command = "mpiexec -n 4 python3 run_TTCF.py"
    cwd = "../"
    sp.call(command, cwd=cwd, shell=True)

def test_files_exist():
    assert os.path.isfile("../profile_DAV.txt")
    assert os.path.isfile("../profile_TTCF.txt")

def test_files_contain_data():
    #Load output profiles
    DAV = np.loadtxt("../profile_DAV.txt")
    TTCF = np.loadtxt("../profile_TTCF.txt")

    assert TTCF.shape[0] > 0
    assert DAV.shape[0] > 0

def test_analytical_soln():
    #Load output profiles
    DAV = np.loadtxt("../profile_DAV.txt")
    TTCF = np.loadtxt("../profile_TTCF.txt")

    #Analytical solution
    filename = "../system_setup.in"
    srate = load_var(filename, "srate")
    rho = load_var(filename, "rho")
    Npart = load_var(filename, "Npart")
    L = (Npart/rho)**(1.0/3.0)
    x = np.linspace(0., L, TTCF.shape[1])
    vx = srate*x

    #Compare analytical to TTCF solution numerically
    paramsTTCF, covarianceTTCF = curve_fit(fn, x[1:-1], np.mean(TTCF[:,1:-1],0), p0=[0.])
    errorTTCF = abs(paramsTTCF[0]-srate)/srate

    paramsDAV, covarianceDAV = curve_fit(fn, x[1:-1], np.mean(DAV[:,1:-1],0), p0=[0.])
    errorDAV = abs(paramsDAV[0]-srate)/srate
    print("Fitted TTCF strainrate = ", paramsTTCF[0], 
          "Fitted DAV strainrate = ", paramsDAV[0],
          " enforced strainrate = ", srate, 
          " errorTTCF = ", errorTTCF, " errorDAV = ", errorDAV,
          "covarianceTTCF=", covarianceTTCF, "covarianceDAV=", covarianceDAV, )

    #Test error less than 20%
    assert errorTTCF < 0.2

    #If strain rate is low, error should be less
    # and covariance in fit
    if srate < 1e-5:
        assert errorTTCF < errorDAV
        assert covarianceTTCF[0] < covarianceDAV[0]

