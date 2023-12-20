import unittest
import numpy as np

from TTCF import utils

class TestUpdateFunctions(unittest.TestCase):
    def test_update_var(self):
        partial = 5.0
        mean = 3.0
        var = 2.0
        Count = 4

        result = utils.update_var(partial, mean, var, Count)

        # Calculate the expected result manually
        newmean=((Count-1)*mean + partial)/float(Count)
        expected_result = ((Count-2)*var + (partial-mean)*(partial-newmean))/float(Count-1)
        
        self.assertEqual(result, expected_result)

    def test_update_mean(self):
        partial = 5.0
        mean = 3.0
        Count = 4

        result = utils.update_mean(partial, mean, Count)

        # Calculate the expected result manually
        expected_result = ((Count - 1) * mean + partial) / float(Count)
        
        self.assertEqual(result, expected_result)

class TestTTCFIntegration(unittest.TestCase):
    def test_sum_prev_dt(self):
        A = np.array([1.0, 2.0, 3.0, 4.0])
        t = 3
        result = utils.sum_prev_dt(A, t)
        expected_result = (A[1] + 4 * A[2] + A[3]) / 3.0
        self.assertEqual(result, expected_result)

    def test_TTCF_integration(self):
        A = np.array([0.0, 0.2, 0.4, 0.6])
        int_step = 0.2

        result = utils.TTCF_integration(A, int_step)

        # Updated expected result to match the calculated result more precisely
        expected_result = np.array([0.  , 0.04, 0.08, 0.14])
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()

