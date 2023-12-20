import unittest
import numpy as np
from TTCF import TTCF

class TestTTCF(unittest.TestCase):
    def setUp(self):
        # Create a sample TTCF object for testing
        global_variables = ["var1", "var2"]
        profile_variables = ["var3", "var4"]
        Nsteps = 100
        Nbins = 10
        Nmappings = 5

        self.ttcf_obj = TTCF.TTCF(global_variables, profile_variables, Nsteps, Nbins, Nmappings)

    def test_initialization(self):
        # Test the initialization of the TTCF object
        self.assertEqual(self.ttcf_obj.Nsteps, 100)
        self.assertEqual(self.ttcf_obj.Nbins, 10)
        self.assertEqual(self.ttcf_obj.Nmappings, 5)
        self.assertEqual(self.ttcf_obj.Count, 0)

        # Ensure that arrays are initialized with the correct shape
        self.assertEqual(self.ttcf_obj.DAV_global_mean.shape, (100, 2))
        self.assertEqual(self.ttcf_obj.DAV_profile_mean.shape, (100, 10, 4))

    def test_add_mappings(self):
        # Test the add_mappings method

        # Create sample data for testing
        data_profile = np.ones((100, 10, 4))
        data_global = np.ones((100, 2))
        omega = 2.0

        # Add mappings to the TTCF object
        self.ttcf_obj.add_mappings(data_profile, data_global, omega)

        # Add assertions based on the expected behavior of your add_mappings method
        self.assertEqual(np.sum(self.ttcf_obj.DAV_profile_partial), 100 * 10 * 4)
        self.assertEqual(np.sum(self.ttcf_obj.DAV_global_partial), 100 * 2)

if __name__ == '__main__':
    unittest.main()
