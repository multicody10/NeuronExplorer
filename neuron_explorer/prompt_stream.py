from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class PromptStreamConfig:
    concepts: List[str]
    seed: int = 1
    p_combo: float = 0.3
    focus_boost: float = 0.6


class PromptStream:
    """Generate labeled text prompts for concept probing."""

    def __init__(self, cfg: PromptStreamConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._single_templates = [
            "Describe {concept} in one sentence.",
            "Give a simple example of {concept}.",
            "Explain {concept} to a beginner.",
            "Write a short note about {concept}.",
        ]
        self._combo_templates = [
            "Explain how {a} relates to {b}.",
            "Compare {a} and {b} with an example.",
            "Describe a scenario involving {a} and {b}.",
        ]

    @property
    def concept_names(self) -> List[str]:
        return list(self.cfg.concepts)

    def _choose_concepts(self, focus_idx: Optional[int]) -> List[int]:
        k = len(self.cfg.concepts)
        if k == 0:
            return []
        if self.rng.random() < self.cfg.p_combo and k >= 2:
            first = focus_idx if focus_idx is not None else int(self.rng.integers(0, k))
            second = int(self.rng.integers(0, k - 1))
            if second >= first:
                second += 1
            return [first, second]
        if focus_idx is not None and self.rng.random() < self.cfg.focus_boost:
            return [int(focus_idx)]
        return [int(self.rng.integers(0, k))]

    def sample(self, batch: int = 8, focus_idx: Optional[int] = None) -> Tuple[List[str], np.ndarray]:
        prompts: List[str] = []
        k = len(self.cfg.concepts)
        labels = np.zeros((batch, k), dtype=np.float32)
        for i in range(batch):
            chosen = self._choose_concepts(focus_idx)
            if not chosen:
                prompts.append("Write a short paragraph about an interesting topic.")
                continue
            if len(chosen) == 1:
                concept = self.cfg.concepts[chosen[0]]
                template = self.rng.choice(self._single_templates)
                prompts.append(template.format(concept=concept))
                labels[i, chosen[0]] = 1.0
            else:
                a, b = chosen
                ta = self.cfg.concepts[a]
                tb = self.cfg.concepts[b]
                template = self.rng.choice(self._combo_templates)
                prompts.append(template.format(a=ta, b=tb))
                labels[i, a] = 1.0
                labels[i, b] = 1.0
        return prompts, labels


@dataclass
class DatasetPromptStreamConfig:
    prompts: List[str]
    seed: int = 1
    shuffle: bool = True


class DatasetPromptStream:
    """Stream prompts from a fixed dataset; labels are empty."""

    def __init__(self, cfg: DatasetPromptStreamConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.prompts = [p.strip() for p in cfg.prompts if p and p.strip()]
        if cfg.shuffle:
            self.rng.shuffle(self.prompts)

    @property
    def concept_names(self) -> List[str]:
        return []

    def sample(self, batch: int = 8, focus_idx: Optional[int] = None) -> Tuple[List[str], np.ndarray]:
        if not self.prompts:
            return ["Write a short paragraph about an interesting topic."] * batch, np.zeros((batch, 0), dtype=np.float32)
        idx = self.rng.integers(0, len(self.prompts), size=batch)
        prompts = [self.prompts[int(i)] for i in idx]
        labels = np.zeros((batch, 0), dtype=np.float32)
        return prompts, labels
