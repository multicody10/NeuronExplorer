from __future__ import annotations

import numpy as np


class ActiveConceptProber:
    """
    Chooses which concept to emphasize next based on coverage.
    """
    def __init__(self, rng_seed: int = 1):
        self.rng = np.random.default_rng(rng_seed)

    def choose_focus_concept(self, labeler) -> int:
        coverage = labeler.concept_coverage()
        if coverage.size == 0:
            return 0
        min_cov = np.min(coverage)
        candidates = np.where(coverage <= min_cov + 1e-9)[0]
        return int(self.rng.choice(candidates))
