from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import io
import os
import tempfile
import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

try:
    from safetensors.torch import load_file as safetensors_load_file

    SAFETENSORS_AVAILABLE = True
except Exception:
    safetensors_load_file = None
    SAFETENSORS_AVAILABLE = False


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


@dataclass
class SAEWeights:
    W_enc: np.ndarray
    b_enc: np.ndarray
    source: str
    meta: Dict[str, object]


def random_sae_weights(input_dim: int, n_features: int, seed: int) -> SAEWeights:
    rng = np.random.default_rng(int(seed))
    scale = 1.0 / np.sqrt(max(1, int(input_dim)))
    W = rng.normal(
        loc=0.0,
        scale=scale,
        size=(int(input_dim), int(n_features)),
    ).astype(np.float32)
    b = np.zeros((int(n_features),), dtype=np.float32)
    return SAEWeights(W_enc=W, b_enc=b, source="random", meta={"seed": int(seed)})


def load_sae_weights_npz(data: bytes) -> Tuple[SAEWeights, List[str]]:
    warnings: List[str] = []
    with np.load(io.BytesIO(data)) as npz:
        keys = list(npz.files)
        W = None
        for key in ("W_enc", "encoder", "W", "W_in"):
            if key in npz:
                W = npz[key]
                break
        if W is None:
            raise ValueError("Missing W_enc in Sparse Autoencoder .npz.")
        b = None
        for key in ("b_enc", "bias", "b"):
            if key in npz:
                b = npz[key]
                break
        if b is None:
            b = np.zeros((W.shape[1],), dtype=np.float32)
            warnings.append("Missing Sparse Autoencoder bias; using zeros.")
    weights = SAEWeights(W_enc=W, b_enc=b, source="npz", meta={"keys": keys})
    return weights, warnings


def _extract_sae_from_state_dict(state: Dict[str, object], source: str) -> Tuple[SAEWeights, List[str]]:
    warnings: List[str] = []
    keys = list(state.keys())

    def find_key(candidates: List[str]) -> str | None:
        for cand in candidates:
            for key in keys:
                if key == cand or key.endswith(cand):
                    return key
        return None

    w_key = find_key(["W_enc", "encoder.weight", "W_in", "W"])
    b_key = find_key(["b_enc", "encoder.bias", "b_in", "b"])

    def to_numpy(val: object) -> np.ndarray:
        if hasattr(val, "detach"):
            return val.detach().cpu().numpy()
        return np.asarray(val)

    if w_key is None:
        for key in keys:
            arr = state[key]
            if hasattr(arr, "ndim") and getattr(arr, "ndim", 0) == 2:
                w_key = key
                warnings.append(f"Guessed encoder weight key: {key}")
                break
    if w_key is None:
        raise ValueError("Could not find a 2D encoder weight in Sparse Autoencoder file.")

    W = to_numpy(state[w_key]).astype(np.float32)
    if b_key is None:
        b = np.zeros((W.shape[1],), dtype=np.float32)
        warnings.append("Missing encoder bias; using zeros.")
    else:
        b = to_numpy(state[b_key]).astype(np.float32)

    weights = SAEWeights(W_enc=W, b_enc=b, source=source, meta={"keys": keys, "W_key": w_key, "b_key": b_key})
    return weights, warnings


def load_sae_weights_torch(data: bytes) -> Tuple[SAEWeights, List[str]]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to load .pt Sparse Autoencoder weights.")
    obj = torch.load(io.BytesIO(data), map_location="cpu")
    if hasattr(obj, "state_dict"):
        obj = obj.state_dict()
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise ValueError("Unsupported Sparse Autoencoder .pt format; expected a state dict.")
    return _extract_sae_from_state_dict(obj, "pt")


def load_sae_weights_safetensors(data: bytes) -> Tuple[SAEWeights, List[str]]:
    if not SAFETENSORS_AVAILABLE:
        raise RuntimeError("safetensors is required to load .safetensors Sparse Autoencoder weights.")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        state = safetensors_load_file(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
    return _extract_sae_from_state_dict(state, "safetensors")


class UnitAdapter:
    name = "Raw neurons"
    unit_label = "Neuron"
    unit_label_plural = "Neurons"

    def __init__(self, input_dim: int):
        self.input_dim = int(input_dim)

    @property
    def unit_dim(self) -> int:
        return int(self.input_dim)

    def encode(self, h: np.ndarray) -> np.ndarray:
        return np.asarray(h, dtype=np.float32)

    def metadata(self) -> Dict[str, object]:
        return {
            "type": "raw",
            "input_dim": self.input_dim,
            "unit_dim": self.unit_dim,
        }


class IdentityUnitAdapter(UnitAdapter):
    pass


class SAEFeatureAdapter(UnitAdapter):
    name = "Sparse Autoencoder features"
    unit_label = "Feature"
    unit_label_plural = "Features"

    def __init__(self, input_dim: int, weights: SAEWeights):
        super().__init__(input_dim)
        W = np.asarray(weights.W_enc, dtype=np.float32)
        b = np.asarray(weights.b_enc, dtype=np.float32)
        if W.ndim != 2:
            raise ValueError("W_enc must be a 2D array.")
        if W.shape[0] != self.input_dim:
            if W.shape[1] == self.input_dim:
                W = W.T
                weights.meta = dict(weights.meta or {})
                weights.meta["transposed"] = True
            else:
                raise ValueError(
                    f"SAE input dim mismatch: model hidden {self.input_dim}, W_enc {W.shape}"
                )
        if b.ndim != 1 or b.shape[0] != W.shape[1]:
            b = np.zeros((W.shape[1],), dtype=np.float32)
            weights.meta = dict(weights.meta or {})
            weights.meta["bias_fallback"] = True
        self.weights = weights
        self.W_enc = W
        self.b_enc = b

    @property
    def unit_dim(self) -> int:
        return int(self.W_enc.shape[1])

    def encode(self, h: np.ndarray) -> np.ndarray:
        h = np.asarray(h, dtype=np.float32)
        return relu(h @ self.W_enc + self.b_enc)

    def metadata(self) -> Dict[str, object]:
        base = super().metadata()
        base.update(
            {
                "type": "sae",
                "source": self.weights.source,
                "unit_dim": self.unit_dim,
            }
        )
        base.update(self.weights.meta or {})
        return base
