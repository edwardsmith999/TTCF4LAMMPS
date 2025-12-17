This release introduces the possibility to perform TTCF for viscosity calculation without using mappings.


The no mapping approach has the following features:
- Only **global variables** can be computed at the moment without the use of mappings
- The results from all trajectories are stored on disk
- The calculation of the method is performed on the complete results dataset, using bootstrapping

The following code as been added:
- `TTCFnoMap` class in `TTCF.py`, to perform TTCF without mappings
- `bootstrapping.py` file, that contains the code to perform standard error calculation
- An example script `run_TTCFnoMaps.py` for running TTCF without mappings
- An example script `run_TTCF_invoke.py` which uses a specific python function together with LAMMPS' `fix python/invoke` to collect data and avoid the use use of LAMMPS' `run` command inside a loop.

