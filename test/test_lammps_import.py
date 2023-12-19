import unittest

class TestLAMMPSImport(unittest.TestCase):
    def test_import_lammps(self):
        try:
            import lammps
            self.assertTrue(True, "PyLAMMPS import successful")
        except ImportError as e:
            self.fail(f"Failed to import PyLAMMPS: {e}")

if __name__ == '__main__':
    unittest.main()
