# This package contains baseline ranking algorithms for the TargetPanelBench.

# Importing these modules here allows consumers to do e.g.:
#
#     from baselines import simple_score_rank
#
# which can be helpful when using the notebook or writing custom scripts.

from .simple_score_rank import run_simple_baseline  # noqa: F401
from .cma_es import run_cma_es_baseline  # noqa: F401