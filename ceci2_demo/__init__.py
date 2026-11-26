"""Research replication for the CECI2 adaptive trading system."""

import os
import sys

# The macOS wheels of torch and XGBoost can load distinct OpenMP runtimes.
# A single OpenMP worker prevents crashes when the two libraries share a process.
if sys.platform == "darwin":
    os.environ["OMP_NUM_THREADS"] = "1"
