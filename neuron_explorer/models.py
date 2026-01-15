from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional
from contextlib import nullcontext
import io
import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TRANSFORMERS_AVAILABLE = False


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


@dataclass
class ModelConfig:
    n_concepts: int
    n_noise: int
    hidden_dim: int
    seed: int = 1
    device: str = "cpu"

    @property
    def input_dim(self) -> int:
        return int(self.n_concepts + self.n_noise)


class ModelAdapter:
    name: str
    description: str
    input_mode: str = "vector"

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    @property
    def input_dim(self) -> int:
        return self.cfg.input_dim

    @property
    def hidden_dim(self) -> int:
        raise NotImplementedError

    def forward_hidden(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def forward_hidden_text(self, prompts: List[str]) -> np.ndarray:
        raise NotImplementedError


class ToyConceptModel(ModelAdapter):
    name = "Toy Concept MLP"
    description = "Interpretable hidden units wired to single concepts and simple pairs."

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        rng = np.random.default_rng(cfg.seed)
        d_in = cfg.input_dim
        h = cfg.hidden_dim

        W = np.zeros((d_in, h), dtype=np.float32)
        b = np.zeros((h,), dtype=np.float32)

        # Units 0..(n_concepts-1): single concept detectors
        for j in range(cfg.n_concepts):
            if j >= h:
                break
            W[j, j] = 3.0
            b[j] = 1.5

        # Next few units: pair interactions (AND-like)
        pairs = [(i, i + 1) for i in range(0, min(cfg.n_concepts, h - 1), 2)]
        for i, (a, b_) in enumerate(pairs, start=min(cfg.n_concepts, h)):
            if i >= h:
                break
            W[a, i] = 2.0
            W[b_, i] = 2.0
            b[i] = 2.5

        # Remaining units: nuisance responses to noise dims
        for i in range(min(cfg.n_concepts + len(pairs), h), h):
            nd = cfg.n_concepts + (i - (cfg.n_concepts + len(pairs))) % max(cfg.n_noise, 1)
            if nd < d_in:
                W[nd, i] = 1.5
                b[i] = 0.2

        # Add a tiny random background to make units less brittle
        W += 0.05 * rng.normal(size=W.shape).astype(np.float32)

        self.W = W
        self.b = b

    @property
    def hidden_dim(self) -> int:
        return int(self.W.shape[1])

    def forward_hidden(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.W + self.b
        return relu(z)


class TorchMlpModel(ModelAdapter):
    name = "Torch MLP"
    description = "Simple PyTorch MLP, useful as a template for real models."

    def __init__(self, cfg: ModelConfig):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available.")
        super().__init__(cfg)
        torch.manual_seed(cfg.seed)
        self.net = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        self.device = torch.device(cfg.device)
        self.net.to(self.device)

    @property
    def hidden_dim(self) -> int:
        return int(self.cfg.hidden_dim)

    def forward_hidden(self, x: np.ndarray) -> np.ndarray:
        xt = torch.from_numpy(x).float().to(self.device)
        with torch.no_grad():
            # Capture the first hidden layer as the "neurons" to explore
            h = self.net[0](xt)
            h = torch.relu(h)
        return h.detach().cpu().numpy()


class TorchScriptAdapter(ModelAdapter):
    name = "Custom TorchScript"
    description = "Loads a TorchScript .pt/.pth and uses its forward output as hidden activations."

    def __init__(self, cfg: ModelConfig, script_bytes: bytes):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available.")
        super().__init__(cfg)
        self._hidden_dim: Optional[int] = None
        buffer = io.BytesIO(script_bytes)
        self.device = torch.device(cfg.device)
        self.model = torch.jit.load(buffer, map_location=self.device)
        self.model.eval()
        try:
            self.model.to(self.device)
        except Exception:
            pass
        self._hidden_dim = self._infer_hidden_dim()

    def _extract_hidden(self, output):
        if isinstance(output, (tuple, list)) and output:
            return output[0]
        if isinstance(output, dict) and output:
            if "hidden" in output:
                return output["hidden"]
            return next(iter(output.values()))
        return output

    def _infer_hidden_dim(self) -> int:
        dummy = torch.zeros(1, self.cfg.input_dim)
        with torch.no_grad():
            out = self.model(dummy)
        h = self._extract_hidden(out)
        if hasattr(h, "shape") and len(h.shape) >= 2:
            return int(h.shape[1])
        raise RuntimeError("Could not infer hidden dimension from TorchScript output.")

    @property
    def hidden_dim(self) -> int:
        if self._hidden_dim is None:
            raise RuntimeError("Hidden dimension not initialized.")
        return self._hidden_dim

    def forward_hidden(self, x: np.ndarray) -> np.ndarray:
        xt = torch.from_numpy(x).float().to(self.device)
        with torch.no_grad():
            out = self.model(xt)
        h = self._extract_hidden(out)
        if not hasattr(h, "detach"):
            raise RuntimeError("TorchScript output is not a tensor.")
        return h.detach().cpu().numpy()


class TransformersCausalLMAdapter(ModelAdapter):
    name = "Transformers LM (.safetensors)"
    description = "Loads a Transformers CausalLM (safetensors) and probes a chosen layer."
    input_mode = "text"

    def __init__(
        self,
        cfg: ModelConfig,
        model_id: str,
        device: str,
        dtype: str,
        layer_idx: int,
        max_length: int,
        use_chat_template: bool,
        trust_remote_code: bool,
        attn_implementation: Optional[str] = None,
    ):
        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            raise RuntimeError("Transformers and PyTorch are required.")
        super().__init__(cfg)
        self.device = torch.device(device)
        model_dtype = None
        if dtype == "float16":
            model_dtype = torch.float16
        elif dtype == "bfloat16":
            model_dtype = torch.bfloat16
        elif dtype == "float32":
            model_dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=bool(trust_remote_code),
        )
        added_pad_token = False
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                added_pad_token = True
        low_cpu_mem_usage = self.device.type == "cpu"
        model_kwargs = dict(
            trust_remote_code=bool(trust_remote_code),
            use_safetensors=True,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        if isinstance(model_dtype, torch.dtype):
            model_kwargs["dtype"] = model_dtype
        if attn_implementation and attn_implementation != "auto":
            model_kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.model.eval()
        if self.device.type != "cpu":
            try:
                self.model.to(self.device)
            except NotImplementedError:
                # Reload without meta tensors when moving to GPU
                model_kwargs["low_cpu_mem_usage"] = False
                self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
                self.model.eval()
                self.model.to(self.device)
        if added_pad_token:
            self.model.resize_token_embeddings(len(self.tokenizer))
        if isinstance(model_dtype, torch.dtype):
            try:
                self.model.to(self.device, dtype=model_dtype)
            except Exception:
                pass

        self.max_length = int(max_length)
        self.layer_idx = int(layer_idx)
        self.use_chat_template = bool(use_chat_template)
        self.model_dtype = model_dtype

        self.num_layers = int(getattr(self.model.config, "num_hidden_layers", 0) or 0)
        self._hidden_dim = int(
            getattr(self.model.config, "hidden_size", 0)
            or getattr(self.model.config, "n_embd", 0)
            or 0
        )
        if self._hidden_dim <= 0:
            raise RuntimeError("Could not infer hidden size from model config.")

    @property
    def hidden_dim(self) -> int:
        return int(self._hidden_dim)

    @property
    def hidden_state_count(self) -> int:
        if self.num_layers > 0:
            return int(self.num_layers + 1)
        return 1

    def _format_prompts(self, prompts: List[str]) -> List[str]:
        if not self.use_chat_template or not hasattr(self.tokenizer, "apply_chat_template"):
            return prompts
        formatted = []
        for prompt in prompts:
            msgs = [{"role": "user", "content": prompt}]
            formatted.append(
                self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
            )
        return formatted

    def _resolve_layer_index(self, total_layers: int) -> int:
        idx = int(self.layer_idx)
        if idx < 0:
            idx = total_layers + idx
        idx = max(0, min(idx, total_layers - 1))
        return idx

    def _autocast_context(self):
        if self.device.type != "cuda":
            return nullcontext()
        enabled = isinstance(self.model_dtype, torch.dtype)
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(
                device_type="cuda",
                enabled=enabled,
                dtype=self.model_dtype if enabled else None,
            )
        return torch.cuda.amp.autocast(
            enabled=enabled,
            dtype=self.model_dtype if enabled else None,
        )

    def forward_hidden_text(self, prompts: List[str]) -> np.ndarray:
        if not prompts:
            return np.zeros((0, self.hidden_dim), dtype=np.float32)
        texts = self._format_prompts(prompts)
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.max_length),
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad(), self._autocast_context():
            outputs = self.model(**encoded, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states
        if not hidden_states:
            raise RuntimeError("Model did not return hidden states.")
        layer = self._resolve_layer_index(len(hidden_states))
        h = hidden_states[layer]
        attn = encoded.get("attention_mask")
        if attn is None:
            pooled = h.mean(dim=1)
        else:
            attn = attn.unsqueeze(-1)
            pooled = (h * attn).sum(dim=1) / attn.sum(dim=1).clamp_min(1.0)
        return pooled.detach().cpu().numpy().astype(np.float32)

    def forward_hidden_text_all_layers(self, prompts: List[str]) -> np.ndarray:
        if not prompts:
            return np.zeros((0, 0, self.hidden_dim), dtype=np.float32)
        texts = self._format_prompts(prompts)
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.max_length),
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad(), self._autocast_context():
            outputs = self.model(**encoded, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states
        if not hidden_states:
            raise RuntimeError("Model did not return hidden states.")
        attn = encoded.get("attention_mask")
        pooled_layers = []
        for h in hidden_states:
            if attn is None:
                pooled = h.mean(dim=1)
            else:
                attn_ = attn.unsqueeze(-1)
                pooled = (h * attn_).sum(dim=1) / attn_.sum(dim=1).clamp_min(1.0)
            pooled_layers.append(pooled.detach().cpu().numpy().astype(np.float32))
        return np.stack(pooled_layers, axis=0)




@dataclass
class ModelOption:
    key: str
    name: str
    description: str
    build: Callable[[ModelConfig], ModelAdapter]


def list_model_options() -> List[ModelOption]:
    options = [
        ModelOption(
            key="toy",
            name=ToyConceptModel.name,
            description=ToyConceptModel.description,
            build=lambda cfg: ToyConceptModel(cfg),
        ),
    ]
    if TORCH_AVAILABLE:
        options.append(
            ModelOption(
                key="torch_mlp",
                name=TorchMlpModel.name,
                description=TorchMlpModel.description,
                build=lambda cfg: TorchMlpModel(cfg),
            )
        )
    return options
