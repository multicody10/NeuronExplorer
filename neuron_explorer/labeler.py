from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass
class AiLabelConfig:
    concept_names: List[str]
    eps: float = 1e-9


class OnlineAiNeuronLabeler:
    """
    Streaming labeling of each hidden neuron against a set of concepts.
    Maintains per neuron activation stats conditioned on concept present vs absent.
    """
    def __init__(self, cfg: AiLabelConfig, hidden: int):
        self.cfg = cfg
        self.hidden = hidden
        k = len(cfg.concept_names)

        self.n1 = np.zeros((hidden, k), dtype=np.float64)
        self.n0 = np.zeros((hidden, k), dtype=np.float64)
        self.s1 = np.zeros((hidden, k), dtype=np.float64)
        self.s0 = np.zeros((hidden, k), dtype=np.float64)

        self.pairs = [(i, i + 1) for i in range(0, min(k, 8), 2) if i + 1 < k]
        self.n11 = np.zeros((hidden, len(self.pairs)), dtype=np.float64)
        self.s11 = np.zeros((hidden, len(self.pairs)), dtype=np.float64)
        self.n10 = np.zeros((hidden, len(self.pairs)), dtype=np.float64)
        self.s10 = np.zeros((hidden, len(self.pairs)), dtype=np.float64)
        self.n01 = np.zeros((hidden, len(self.pairs)), dtype=np.float64)
        self.s01 = np.zeros((hidden, len(self.pairs)), dtype=np.float64)
        self.n00 = np.zeros((hidden, len(self.pairs)), dtype=np.float64)
        self.s00 = np.zeros((hidden, len(self.pairs)), dtype=np.float64)

        self.total_samples = 0

    def update(self, h_act: np.ndarray, concepts: np.ndarray) -> None:
        h_act = np.asarray(h_act, dtype=np.float32)
        concepts = np.asarray(concepts, dtype=np.float32)
        b = h_act.shape[0]
        k = concepts.shape[1]
        assert k == len(self.cfg.concept_names)

        for j in range(k):
            mask1 = concepts[:, j] > 0.5
            mask0 = ~mask1
            if mask1.any():
                self.n1[:, j] += mask1.sum()
                self.s1[:, j] += h_act[mask1].sum(axis=0)
            if mask0.any():
                self.n0[:, j] += mask0.sum()
                self.s0[:, j] += h_act[mask0].sum(axis=0)

        for pidx, (a, b_) in enumerate(self.pairs):
            A = concepts[:, a] > 0.5
            B = concepts[:, b_] > 0.5
            m11 = A & B
            m10 = A & (~B)
            m01 = (~A) & B
            m00 = (~A) & (~B)

            if m11.any():
                self.n11[:, pidx] += m11.sum()
                self.s11[:, pidx] += h_act[m11].sum(axis=0)
            if m10.any():
                self.n10[:, pidx] += m10.sum()
                self.s10[:, pidx] += h_act[m10].sum(axis=0)
            if m01.any():
                self.n01[:, pidx] += m01.sum()
                self.s01[:, pidx] += h_act[m01].sum(axis=0)
            if m00.any():
                self.n00[:, pidx] += m00.sum()
                self.s00[:, pidx] += h_act[m00].sum(axis=0)

        self.total_samples += b

    def _mean_act(self) -> np.ndarray:
        eps = self.cfg.eps
        return (self.s1.sum(axis=1) + self.s0.sum(axis=1)) / (self.n1.sum(axis=1) + self.n0.sum(axis=1) + eps)

    def effect_matrix(self) -> np.ndarray:
        eps = self.cfg.eps
        m1 = self.s1 / (self.n1 + eps)
        m0 = self.s0 / (self.n0 + eps)
        return m1 - m0

    def concept_coverage(self) -> np.ndarray:
        return (self.n1 + self.n0).mean(axis=0)

    def snapshot(self, top_n: int = 40) -> pd.DataFrame:
        eps = self.cfg.eps
        k = len(self.cfg.concept_names)
        rows: List[Dict[str, object]] = []

        mean_all = self._mean_act()
        diff_all = self.effect_matrix()

        for i in range(self.hidden):
            diff = diff_all[i]
            best = int(np.argmax(diff)) if k > 0 else 0
            conf = float(diff[best] / (np.abs(diff).sum() + eps)) if k > 0 else 0.0

            if self.pairs:
                pair_scores = []
                for pidx, (a, b_) in enumerate(self.pairs):
                    m11 = self.s11[i, pidx] / (self.n11[i, pidx] + eps)
                    m10 = self.s10[i, pidx] / (self.n10[i, pidx] + eps)
                    m01 = self.s01[i, pidx] / (self.n01[i, pidx] + eps)
                    m00 = self.s00[i, pidx] / (self.n00[i, pidx] + eps)
                    score = m11 - max(m10, m01, m00)
                    pair_scores.append(score)
                pair_best = int(np.argmax(pair_scores))
                pair_conf = float(pair_scores[pair_best] / (np.abs(pair_scores).sum() + eps))
                pair_name = f"{self.cfg.concept_names[self.pairs[pair_best][0]]}+{self.cfg.concept_names[self.pairs[pair_best][1]]}"
            else:
                pair_name = "n/a"
                pair_conf = 0.0

            label_single = self.cfg.concept_names[best] if k > 0 else "n/a"
            guess = self._guess_label(
                label_single=label_single,
                conf_single=conf,
                pair_name=pair_name,
                pair_conf=pair_conf,
            )
            guess_detail = self._guess_detail(
                diff=diff,
                label_single=label_single,
                conf_single=conf,
                pair_name=pair_name,
                pair_conf=pair_conf,
            )
            top_concepts = self._top_concepts(diff, top_n=3)

            rows.append({
                "unit": i,
                "mean_act": float(mean_all[i]),
                "label_single": self.cfg.concept_names[best] if k > 0 else "n/a",
                "conf_single": conf,
                "label_pair": pair_name,
                "conf_pair": pair_conf,
                "guess": guess,
                "guess_detail": guess_detail,
                "top_concepts": top_concepts,
            })

        df = pd.DataFrame(rows)
        df = df.sort_values(["mean_act"], ascending=False).head(top_n).reset_index(drop=True)
        return df

    def _guess_label(self, label_single: str, conf_single: float, pair_name: str, pair_conf: float) -> str:
        if pair_name != "n/a" and pair_conf > max(conf_single, 0.35):
            return f"sees {pair_name}"
        if label_single != "n/a":
            return f"sees {label_single}"
        return "unknown"

    def _guess_detail(
        self,
        diff: np.ndarray,
        label_single: str,
        conf_single: float,
        pair_name: str,
        pair_conf: float,
    ) -> str:
        if diff.size == 0:
            return "unknown"
        pos_idx = int(np.argmax(diff))
        neg_idx = int(np.argmin(diff))
        pos_val = float(diff[pos_idx])
        neg_val = float(diff[neg_idx])

        if pair_name != "n/a" and pair_conf >= max(conf_single, 0.35):
            base = f"sees the combo {pair_name}"
        elif conf_single >= 0.45:
            base = f"strongly sees {label_single}"
        elif conf_single >= 0.25:
            base = f"weakly sees {label_single}"
        else:
            base = "mixed / unclear"

        if abs(neg_val) > abs(pos_val) * 1.1:
            base = f"{base}; avoids {self.cfg.concept_names[neg_idx]}"
        return base

    def _top_concepts(self, diff: np.ndarray, top_n: int = 3) -> str:
        if diff.size == 0:
            return "n/a"
        idx = np.argsort(-np.abs(diff))[: min(top_n, diff.size)]
        labels = []
        for i in idx:
            sign = "+" if diff[i] >= 0 else "-"
            labels.append(f"{self.cfg.concept_names[int(i)]}{sign}")
        return ", ".join(labels)

    def unit_profile(self, unit: int) -> Dict[str, np.ndarray]:
        eps = self.cfg.eps
        m1 = self.s1[unit] / (self.n1[unit] + eps)
        m0 = self.s0[unit] / (self.n0[unit] + eps)
        diff = m1 - m0
        return {"m1": m1, "m0": m0, "diff": diff}

    def guess_for_unit(self, unit: int) -> Dict[str, object]:
        eps = self.cfg.eps
        k = len(self.cfg.concept_names)
        diff = self.effect_matrix()[unit]
        best = int(np.argmax(diff)) if k > 0 else 0
        conf_single = float(diff[best] / (np.abs(diff).sum() + eps)) if k > 0 else 0.0
        label_single = self.cfg.concept_names[best] if k > 0 else "n/a"

        pair_name = "n/a"
        pair_conf = 0.0
        if self.pairs:
            pair_scores = []
            for pidx, (a, b_) in enumerate(self.pairs):
                m11 = self.s11[unit, pidx] / (self.n11[unit, pidx] + eps)
                m10 = self.s10[unit, pidx] / (self.n10[unit, pidx] + eps)
                m01 = self.s01[unit, pidx] / (self.n01[unit, pidx] + eps)
                m00 = self.s00[unit, pidx] / (self.n00[unit, pidx] + eps)
                score = m11 - max(m10, m01, m00)
                pair_scores.append(score)
            pair_best = int(np.argmax(pair_scores))
            pair_conf = float(pair_scores[pair_best] / (np.abs(pair_scores).sum() + eps))
            pair_name = f"{self.cfg.concept_names[self.pairs[pair_best][0]]}+{self.cfg.concept_names[self.pairs[pair_best][1]]}"

        guess = self._guess_label(label_single, conf_single, pair_name, pair_conf)
        guess_detail = self._guess_detail(diff, label_single, conf_single, pair_name, pair_conf)
        top_concepts = self._top_concepts(diff, top_n=3)
        return {
            "guess": guess,
            "guess_detail": guess_detail,
            "top_concepts": top_concepts,
            "label_single": label_single,
            "conf_single": conf_single,
            "label_pair": pair_name,
            "conf_pair": pair_conf,
        }
