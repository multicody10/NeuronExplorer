from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class ConceptStreamConfig:
    n_concepts: int = 8
    n_noise: int = 8
    p_on: float = 0.2
    focus_boost: float = 0.6
    noise_scale: float = 1.0
    seed: int = 1


class ConceptStream:
    """
    Produces a stream of inputs with named latent concepts.
    x = [concepts, noise]
    """
    def __init__(self, cfg: ConceptStreamConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    @property
    def concept_names(self) -> List[str]:
        return [f"c{j}" for j in range(self.cfg.n_concepts)]

    def sample(self, batch: int = 256, focus_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        k = self.cfg.n_concepts
        p = np.full((batch, k), self.cfg.p_on, dtype=np.float32)
        if focus_idx is not None and 0 <= focus_idx < k:
            p[:, focus_idx] = min(0.95, self.cfg.p_on + self.cfg.focus_boost)
        c = (self.rng.random(size=(batch, k)) < p).astype(np.float32)
        noise = self.rng.normal(0.0, self.cfg.noise_scale, size=(batch, self.cfg.n_noise)).astype(np.float32)
        x = np.concatenate([c, noise], axis=1)
        return x, c
