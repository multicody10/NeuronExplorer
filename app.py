from __future__ import annotations

import hashlib
import json
import time
import contextlib
import os
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st

try:
    import torch
except Exception:
    torch = None

from neuron_explorer.labeler import AiLabelConfig, OnlineAiNeuronLabeler
from neuron_explorer.models import (
    ModelConfig,
    TORCH_AVAILABLE,
    TRANSFORMERS_AVAILABLE,
    TorchScriptAdapter,
    TransformersCausalLMAdapter,
    list_model_options,
)
from neuron_explorer.unit_sources import (
    IdentityUnitAdapter,
    SAEFeatureAdapter,
    load_sae_weights_npz,
    load_sae_weights_safetensors,
    load_sae_weights_torch,
    random_sae_weights,
)
from neuron_explorer.prober import ActiveConceptProber
from neuron_explorer.prompt_stream import (
    DatasetPromptStream,
    DatasetPromptStreamConfig,
    PromptStream,
    PromptStreamConfig,
)
from neuron_explorer.stream import ConceptStream, ConceptStreamConfig
try:
    from token_sae_decode import (
        ActivationSensor,
        build_indices,
        load_model as load_token_model,
        load_sae as load_token_sae,
    )
    TOKEN_SAE_AVAILABLE = True
except Exception:
    ActivationSensor = None
    build_indices = None
    load_token_model = None
    load_token_sae = None
    TOKEN_SAE_AVAILABLE = False

SETTINGS_FILE = "settings.json"

def _load_settings() -> dict:
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_settings(data: dict) -> None:
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

CUDA_AVAILABLE = False
CUDA_CAPABILITY = None
CUDA_NAME = None
if torch is not None and torch.cuda.is_available():
    CUDA_AVAILABLE = True
    CUDA_CAPABILITY = torch.cuda.get_device_capability()
    CUDA_NAME = torch.cuda.get_device_name(0)
try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except Exception:
    go = None
    PLOTLY_AVAILABLE = False

def _do_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def _clean_path(value: str) -> str:
    return value.replace("\ufeff", "").strip()


def _load_prompt_dataset(uploaded_file, max_prompts: int, jsonl_field: str) -> tuple[list[str], list[str]]:
    if uploaded_file is None:
        return [], []
    warnings = []
    raw = uploaded_file.getvalue()
    text = raw.decode("utf-8", errors="ignore")
    ext = os.path.splitext(uploaded_file.name or "")[1].lower()
    prompts: list[str] = []

    def add_prompt(val: str) -> None:
        s = val.strip()
        if s:
            prompts.append(s)

    if ext == ".jsonl":
        field_candidates = [jsonl_field, "text", "prompt", "content"]
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, str):
                add_prompt(obj)
                continue
            if isinstance(obj, dict):
                found = False
                for key in field_candidates:
                    if key and key in obj and isinstance(obj[key], str):
                        add_prompt(obj[key])
                        found = True
                        break
                if not found:
                    warnings.append("Some JSONL lines lacked a text field.")
    elif ext == ".json":
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, str):
                    add_prompt(item)
                elif isinstance(item, dict):
                    for key in (jsonl_field, "text", "prompt", "content"):
                        if key in item and isinstance(item[key], str):
                            add_prompt(item[key])
                            break
        elif isinstance(obj, dict):
            data = obj.get("data") if isinstance(obj.get("data"), list) else []
            for item in data:
                if isinstance(item, str):
                    add_prompt(item)
                elif isinstance(item, dict):
                    for key in (jsonl_field, "text", "prompt", "content"):
                        if key in item and isinstance(item[key], str):
                            add_prompt(item[key])
                            break
        else:
            warnings.append("JSON file did not contain a list of prompts.")
    else:
        for line in text.splitlines():
            add_prompt(line)

    if max_prompts > 0:
        prompts = prompts[: int(max_prompts)]
    if not prompts:
        warnings.append("No prompts found in dataset.")
    return prompts, warnings


def _load_prompt_dataset_from_path(path: str, max_prompts: int, jsonl_field: str) -> tuple[list[str], list[str]]:
    if not path or not os.path.exists(path):
        return [], ["Dataset path not found."]
    class _Shim:
        def __init__(self, name: str, data: bytes):
            self.name = name
            self._data = data
        def getvalue(self):
            return self._data
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception as exc:
        return [], [f"Failed to read dataset: {exc}"]
    shim = _Shim(os.path.basename(path), data)
    return _load_prompt_dataset(shim, max_prompts, jsonl_field)


def _normalize_device_dtype(device: str, dtype: str) -> tuple[str, str]:
    if device == "cuda" and torch is not None and not torch.cuda.is_available():
        device = "cpu"
    if device == "cpu" and dtype in ("float16", "bfloat16"):
        dtype = "float32"
    return device, dtype


def _ensure_bytes_path(data: bytes, name: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    digest = hashlib.sha256(data).hexdigest()[:12]
    base = f"{os.path.splitext(name)[0] or 'sae'}-{digest}{os.path.splitext(name)[1]}"
    path = os.path.join(cache_dir, base)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data)
    return path


def _get_token_model(model_id: str, device: str, dtype: str, trust_remote_code: bool):
    device, dtype = _normalize_device_dtype(device, dtype)
    key = (model_id, device, dtype, bool(trust_remote_code))
    cache_key = st.session_state.get("token_model_key")
    if cache_key != key:
        st.session_state["token_model_key"] = key
        st.session_state["token_model"] = None
        st.session_state["token_tokenizer"] = None
    if st.session_state.get("token_model") is None:
        model, tokenizer = load_token_model(
            model_id,
            device=device,
            dtype=dtype,
            trust_remote_code=bool(trust_remote_code),
        )
        st.session_state["token_model"] = model
        st.session_state["token_tokenizer"] = tokenizer
    return st.session_state["token_model"], st.session_state["token_tokenizer"]


def _get_token_sae(sae_path: str, device: str, dtype: str):
    device, dtype = _normalize_device_dtype(device, dtype)
    key = (sae_path, device, dtype)
    cache_key = st.session_state.get("token_sae_key")
    if cache_key != key:
        st.session_state["token_sae_key"] = key
        st.session_state["token_sae"] = None
    if st.session_state.get("token_sae") is None:
        st.session_state["token_sae"] = load_token_sae(sae_path, device=device, dtype=dtype)
    return st.session_state["token_sae"]


def _open_token_db(out_dir: str) -> sqlite3.Connection:
    os.makedirs(out_dir, exist_ok=True)
    db_path = os.path.join(out_dir, "decoded.sqlite")
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS token_activations (
            sample_id INTEGER,
            position INTEGER,
            token_id INTEGER,
            token_str TEXT,
            topk_ids TEXT,
            topk_vals TEXT,
            context TEXT
        )
        """
    )
    return conn


def _purge_token_db(out_dir: str) -> list[str]:
    removed = []
    db_path = os.path.join(out_dir, "decoded.sqlite")
    for suffix in ("", "-wal", "-shm", "-journal"):
        path = f"{db_path}{suffix}"
        if os.path.exists(path):
            os.remove(path)
            removed.append(os.path.basename(path))
    return removed


def _purge_token_indices(out_dir: str) -> None:
    db_path = os.path.join(out_dir, "decoded.sqlite")
    if not os.path.exists(db_path):
        return
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS feature_contexts")
    cur.execute("DROP TABLE IF EXISTS span_features")
    cur.execute("DROP TABLE IF EXISTS feature_cooccurrence")
    conn.commit()
    conn.close()


def _amp_context(device: str, dtype: str):
    if device == "cuda" and torch is not None:
        if dtype == "float16":
            return torch.cuda.amp.autocast(dtype=torch.float16)
        if dtype == "bfloat16":
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _run_token_offline_decode(
    prompts: list[str],
    model,
    tokenizer,
    sae,
    layer: int,
    hookpoint: str,
    topk: int,
    max_len: int,
    batch_size: int,
    out_dir: str,
    device: str,
    dtype: str,
    ctx_window: int,
    progress_cb=None,
) -> tuple[int, str]:
    if not prompts:
        return 0, os.path.join(out_dir, "decoded.sqlite")
    sensor = ActivationSensor(model, layer_idx=int(layer), hookpoint=hookpoint)
    conn = _open_token_db(out_dir)
    insert_sql = (
        "INSERT INTO token_activations "
        "(sample_id, position, token_id, token_str, topk_ids, topk_vals, context) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)"
    )
    buffer: list[tuple] = []
    sample_id = 0
    total = len(prompts)
    model_device = next(model.parameters()).device

    def flush():
        nonlocal buffer
        if buffer:
            conn.executemany(insert_sql, buffer)
            conn.commit()
            buffer = []

    for start in range(0, total, batch_size):
        batch = prompts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(max_len),
        )
        input_ids = enc["input_ids"].to(model_device)
        attention_mask = enc["attention_mask"].to(model_device)
        with torch.no_grad(), _amp_context(device, dtype):
            _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        h = sensor.pop()
        h = h.to(sae.W.dtype)
        B, T, D = h.shape
        flat = h.reshape(B * T, D)
        scores = sae.encode(flat)
        k_eff = min(int(topk), scores.shape[-1])
        top_vals, top_ids = torch.topk(scores, k=k_eff, dim=-1)
        tokens = input_ids.detach().cpu().numpy()
        attn = attention_mask.detach().cpu().numpy()
        top_vals = top_vals.detach().cpu().numpy()
        top_ids = top_ids.detach().cpu().numpy()
        for bi in range(B):
            token_list = tokenizer.convert_ids_to_tokens(tokens[bi].tolist())
            for ti in range(T):
                if attn[bi, ti] == 0:
                    continue
                tok_id = int(tokens[bi, ti])
                tok_str = token_list[ti]
                ctx_tokens = token_list[
                    max(0, ti - int(ctx_window)) : min(T, ti + int(ctx_window) + 1)
                ]
                record = (
                    sample_id + bi,
                    int(ti),
                    tok_id,
                    tok_str,
                    json.dumps(top_ids[bi * T + ti].tolist()),
                    json.dumps(top_vals[bi * T + ti].tolist()),
                    json.dumps(ctx_tokens),
                )
                buffer.append(record)
        sample_id += B
        flush()
        if progress_cb is not None:
            progress_cb(min(1.0, (start + len(batch)) / max(1, total)))

    sensor.close()
    conn.close()
    return total, os.path.join(out_dir, "decoded.sqlite")


def _run_token_realtime_decode(
    prompt: str,
    model,
    tokenizer,
    sae,
    layer: int,
    hookpoint: str,
    topk: int,
    max_new_tokens: int,
    device: str,
    dtype: str,
) -> tuple[str, list[dict]]:
    sensor = ActivationSensor(model, layer_idx=int(layer), hookpoint=hookpoint)
    model_device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(model_device)
    attention_mask = enc["attention_mask"].to(model_device)
    outputs = []
    with torch.no_grad(), _amp_context(device, dtype):
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    past = out.past_key_values
    h = sensor.pop()
    last_h = h[:, -1:, :].to(sae.W.dtype)
    scores = sae.encode(last_h.squeeze(1))
    k_eff = min(int(topk), scores.shape[-1])
    vals, ids = torch.topk(scores, k=k_eff, dim=-1)
    token_id = int(input_ids[0, -1].item())
    token_str = tokenizer.convert_ids_to_tokens([token_id])[0]
    outputs.append(
        {
            "step": 0,
            "token": token_str,
            "token_id": token_id,
            "topk_ids": ids[0].detach().cpu().numpy().tolist(),
            "topk_vals": vals[0].detach().cpu().numpy().tolist(),
        }
    )
    cur_ids = input_ids
    for step in range(1, int(max_new_tokens) + 1):
        with torch.no_grad(), _amp_context(device, dtype):
            logits = out.logits[:, -1, :]
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            out = model(input_ids=next_id, use_cache=True, past_key_values=past)
        past = out.past_key_values
        h = sensor.pop()
        last_h = h[:, -1:, :].to(sae.W.dtype)
        scores = sae.encode(last_h.squeeze(1))
        k_eff = min(int(topk), scores.shape[-1])
        vals, ids = torch.topk(scores, k=k_eff, dim=-1)
        token_id = int(next_id.item())
        token_str = tokenizer.convert_ids_to_tokens([token_id])[0]
        outputs.append(
            {
                "step": step,
                "token": token_str,
                "token_id": token_id,
                "topk_ids": ids[0].detach().cpu().numpy().tolist(),
                "topk_vals": vals[0].detach().cpu().numpy().tolist(),
            }
        )
        cur_ids = torch.cat([cur_ids, next_id], dim=1)
    sensor.close()
    decoded_text = tokenizer.decode(cur_ids[0], skip_special_tokens=True)
    return decoded_text, outputs


def _reset_mapping_state(state: dict) -> dict:
    unit_dim = int(getattr(state.get("unit_adapter"), "unit_dim", state["labeler"].hidden))
    concept_names = state["stream"].concept_names
    cfg = AiLabelConfig(concept_names=concept_names)
    if state.get("layer_mode") == "all":
        layer_count = len(state.get("labelers", [])) or int(
            getattr(state.get("model"), "hidden_state_count", 1)
        )
        labelers = [OnlineAiNeuronLabeler(cfg, hidden=unit_dim) for _ in range(layer_count)]
        state["labelers"] = labelers
        active_layer = int(state.get("active_layer", 0))
        active_layer = max(0, min(active_layer, len(labelers) - 1))
        state["labeler"] = labelers[active_layer]
        state["feature_dicts"] = [init_feature_dict(unit_dim) for _ in range(layer_count)]
        state["feature_dict"] = state["feature_dicts"][active_layer]
    else:
        state["labeler"] = OnlineAiNeuronLabeler(cfg, hidden=unit_dim)
        if "feature_dict" in state:
            state["feature_dict"] = init_feature_dict(unit_dim)
    state["history"] = []
    state["step"] = 0
    state["last_focus"] = None
    state["auto_conf_hist"] = []
    st.session_state.pop("map_cache", None)
    st.session_state["map_epoch"] = int(st.session_state.get("map_epoch", 0)) + 1
    return state

st.set_page_config(page_title="Neuron Explorer", layout="wide")

SETTINGS = _load_settings()

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
:root {
  --bg: #0c1117;
  --panel: #121a23;
  --panel-2: #0f1620;
  --border: #223044;
  --ink: #e6edf3;
  --muted: #9aa4b2;
  --accent: #f5a524;
  --accent-strong: #db8a1f;
  --accent-soft: rgba(245, 165, 36, 0.15);
}
html, body, [class*="css"] {
  font-family: "Space Grotesk", "Segoe UI", sans-serif;
}
:root {
  color-scheme: dark;
}
.stApp {
  background:
    radial-gradient(1400px circle at 10% 0%, #1b2430 0%, #0c1117 52%, #0a0f14 100%),
    linear-gradient(180deg, #0c1117 0%, #0a0f14 100%);
  color: var(--ink);
}
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #101823 0%, #0b1118 100%);
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
  letter-spacing: 0.02em;
}
[data-testid="stMetricValue"] {
  color: var(--accent);
}
.small-muted {
  color: var(--muted);
  font-size: 0.9rem;
}
.stButton > button {
  background: var(--accent-strong);
  color: #1b1206;
  border: 1px solid #ffb94a;
  border-radius: 12px;
  font-weight: 700;
  letter-spacing: 0.01em;
}
.stButton > button:hover {
  background: var(--accent);
  color: #1b1206;
  border-color: var(--accent);
}
.stButton > button:disabled {
  background: #202833;
  color: #7f8b85;
  border-color: #2a3441;
}
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea,
div[data-baseweb="select"] > div {
  background-color: var(--panel);
  color: var(--ink);
  border-color: var(--border);
}
div[data-baseweb="input"] input::placeholder,
div[data-baseweb="textarea"] textarea::placeholder {
  color: #6f7b8a;
}
div[data-testid="stDataFrame"] {
  background: var(--panel-2);
  border-radius: 14px;
  border: 1px solid var(--border);
}
.hero {
  background: linear-gradient(135deg, rgba(245, 165, 36, 0.16), rgba(39, 47, 59, 0.9));
  border: 1px solid rgba(245, 165, 36, 0.25);
  border-radius: 18px;
  padding: 18px 20px;
  box-shadow: 0 12px 30px rgba(7, 10, 14, 0.4);
}
.hero-title {
  font-size: 2.1rem;
  font-weight: 700;
  margin: 0;
}
.hero-sub {
  color: var(--muted);
  margin-top: 4px;
}
.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 0.85rem;
  color: var(--muted);
}
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 14px 16px;
}
.panel h4 {
  margin-top: 0.2rem;
}
.mono {
  font-family: "JetBrains Mono", "SF Mono", monospace;
  font-size: 0.9rem;
}
</style>
""",
    unsafe_allow_html=True,
)

gpu_label = "GPU ready" if CUDA_AVAILABLE else "CPU only"
st.markdown(
    f"""
<div class="hero">
  <div class="hero-title">Neuron Explorer</div>
  <div style="height: 6px;"></div>
  <div class="hero-sub">
    Streaming concept probes, Sparse Autoencoder feature swaps, and token-level decoding in one cockpit.
  </div>
  <div style="margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px;">
    <span class="badge">⚡ {gpu_label}</span>
    <span class="badge">Model: {SETTINGS.get("model_choice", "n/a")}</span>
    <span class="badge">Mode: live + offline</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

FEATURES_SMALL = [
    "dog",
    "cat",
    "car",
    "tree",
    "person",
    "music",
    "food",
    "city",
    "science",
    "art",
    "history",
    "emotions",
]

FEATURES_MEDIUM = [
    "dog",
    "cat",
    "bird",
    "fish",
    "horse",
    "car",
    "truck",
    "train",
    "plane",
    "tree",
    "flower",
    "mountain",
    "river",
    "ocean",
    "city",
    "village",
    "kitchen",
    "bedroom",
    "office",
    "school",
    "doctor",
    "teacher",
    "artist",
    "athlete",
    "police",
    "music",
    "painting",
    "dance",
    "movie",
    "sports",
    "soccer",
    "basketball",
    "baseball",
    "science",
    "physics",
    "chemistry",
    "biology",
    "math",
    "algebra",
    "geometry",
    "history",
    "war",
    "politics",
    "economics",
    "finance",
    "cooking",
    "baking",
    "travel",
    "hotel",
    "beach",
    "mountain",
    "emotions",
    "happy",
    "sad",
    "angry",
    "fear",
    "surprise",
    "love",
    "friendship",
    "conflict",
]

FEATURES_LARGE = [
    "dog",
    "cat",
    "bird",
    "fish",
    "horse",
    "cow",
    "sheep",
    "elephant",
    "tiger",
    "lion",
    "bear",
    "wolf",
    "rabbit",
    "mouse",
    "insect",
    "snake",
    "frog",
    "car",
    "truck",
    "train",
    "plane",
    "boat",
    "bicycle",
    "motorcycle",
    "bus",
    "tree",
    "flower",
    "forest",
    "mountain",
    "river",
    "lake",
    "ocean",
    "desert",
    "island",
    "city",
    "village",
    "house",
    "apartment",
    "kitchen",
    "bedroom",
    "bathroom",
    "office",
    "school",
    "hospital",
    "restaurant",
    "market",
    "person",
    "child",
    "adult",
    "elder",
    "doctor",
    "teacher",
    "artist",
    "athlete",
    "engineer",
    "soldier",
    "police",
    "music",
    "painting",
    "dance",
    "movie",
    "theater",
    "poetry",
    "sports",
    "soccer",
    "basketball",
    "baseball",
    "tennis",
    "science",
    "physics",
    "chemistry",
    "biology",
    "astronomy",
    "geology",
    "math",
    "algebra",
    "geometry",
    "calculus",
    "statistics",
    "history",
    "ancient",
    "medieval",
    "modern",
    "war",
    "politics",
    "economics",
    "finance",
    "market",
    "trade",
    "law",
    "ethics",
    "philosophy",
    "religion",
    "myth",
    "cooking",
    "baking",
    "grilling",
    "coffee",
    "tea",
    "travel",
    "airport",
    "hotel",
    "beach",
    "mountain",
    "camping",
    "emotions",
    "happy",
    "sad",
    "angry",
    "fear",
    "surprise",
    "love",
    "friendship",
    "conflict",
    "family",
    "work",
    "education",
    "health",
    "fitness",
    "medicine",
    "sleep",
    "dream",
    "technology",
    "computer",
    "internet",
    "security",
    "robot",
    "code",
    "bug",
    "database",
    "network",
    "design",
    "architecture",
    "fashion",
    "color",
    "texture",
    "shape",
    "size",
    "speed",
    "time",
    "space",
    "light",
    "sound",
    "taste",
    "smell",
    "touch",
    "cause",
    "effect",
    "reason",
    "plan",
    "goal",
    "risk",
    "safety",
    "order",
    "chaos",
    "truth",
    "lie",
    "question",
    "answer",
    "problem",
    "solution",
]


def generate_concepts(size: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    nouns = [
        "dog",
        "cat",
        "bird",
        "fish",
        "horse",
        "car",
        "train",
        "plane",
        "tree",
        "river",
        "city",
        "forest",
        "mountain",
        "kitchen",
        "office",
        "school",
        "doctor",
        "artist",
        "athlete",
        "music",
        "painting",
        "science",
        "history",
        "emotion",
        "memory",
        "attention",
        "reasoning",
        "logic",
        "pattern",
        "motion",
        "color",
        "texture",
        "shape",
        "time",
        "space",
        "risk",
        "safety",
        "order",
        "chaos",
    ]
    verbs = [
        "sees",
        "describes",
        "compares",
        "explains",
        "predicts",
        "avoids",
        "prefers",
        "summarizes",
        "transforms",
        "represents",
    ]
    adjectives = [
        "small",
        "large",
        "fast",
        "slow",
        "bright",
        "dark",
        "noisy",
        "quiet",
        "abstract",
        "concrete",
        "visual",
        "verbal",
        "social",
        "emotional",
        "scientific",
        "historical",
        "causal",
        "spatial",
        "temporal",
    ]
    relations = [
        "and",
        "vs",
        "with",
        "without",
        "before",
        "after",
        "because",
        "therefore",
        "causes",
        "implies",
    ]

    concepts: set[str] = set()
    while len(concepts) < size:
        template = rng.integers(0, 4)
        a = rng.choice(nouns)
        b = rng.choice(nouns)
        adj = rng.choice(adjectives)
        verb = rng.choice(verbs)
        rel = rng.choice(relations)
        if template == 0:
            concepts.add(f"{adj} {a}")
        elif template == 1:
            concepts.add(f"{a} {rel} {b}")
        elif template == 2:
            concepts.add(f"{verb} {a}")
        else:
            concepts.add(f"{adj} {a} {rel} {b}")
        if len(concepts) >= size:
            break
    return sorted(concepts)


def init_feature_dict(hidden_dim: int, top_k: int = 3) -> dict:
    return {
        "top_k": int(top_k),
        "top_values": np.full((hidden_dim, top_k), -np.inf, dtype=np.float32),
        "top_prompts": [[""] * top_k for _ in range(hidden_dim)],
        "sum": np.zeros(hidden_dim, dtype=np.float64),
        "sumsq": np.zeros(hidden_dim, dtype=np.float64),
        "count": 0,
    }


def update_feature_dict(feature_dict: dict, prompts: list[str], activations: np.ndarray) -> None:
    if activations.size == 0:
        return
    vals = np.asarray(activations, dtype=np.float32)
    vals64 = vals.astype(np.float64, copy=False)
    feature_dict["sum"] += vals64.sum(axis=0)
    feature_dict["sumsq"] += np.square(vals64).sum(axis=0)
    feature_dict["count"] += vals.shape[0]

    top_values = feature_dict["top_values"]
    top_prompts = feature_dict["top_prompts"]
    for i, prompt in enumerate(prompts):
        v = vals[i]
        min_vals = top_values[:, -1]
        idxs = np.where(v > min_vals)[0]
        for d in idxs.tolist():
            cand = float(v[d])
            row = top_values[d]
            if cand <= row[-1]:
                continue
            insert_at = int(np.searchsorted(-row, -cand, side="left"))
            if insert_at < 0:
                insert_at = 0
            if insert_at < row.size:
                row[insert_at + 1 :] = row[insert_at:-1]
                row[insert_at] = cand
                tp = top_prompts[d]
                tp[insert_at + 1 :] = tp[insert_at:-1]
                tp[insert_at] = prompt
model_options = list_model_options()
model_labels = [opt.name for opt in model_options]
model_lookup = {opt.name: opt for opt in model_options}
custom_label = "Custom file (TorchScript .pt/.pth)"
hf_label = "Transformers (.safetensors)"
model_choices = model_labels + [custom_label, hf_label]

with st.sidebar:
    st.header("Model")
    default_choice = SETTINGS.get("model_choice", model_choices[0])
    default_index = model_choices.index(default_choice) if default_choice in model_choices else 0
    model_choice = st.selectbox(
        "Model",
        model_choices,
        index=default_index,
        help="Choose the model backend to explore.",
    )
    uploaded_file = None
    if model_choice == custom_label:
        uploaded_file = st.file_uploader(
            "Model file",
            type=["pt", "pth"],
            help="Upload a TorchScript model file.",
        )

    hf_model_id = ""
    hf_device = "cpu"
    hf_dtype = "auto"
    hf_effective_dtype = "auto"
    hf_layer_idx = -1
    hf_layer_mode = "Single layer"
    hf_active_layer = -1
    hf_max_length = 256
    hf_use_chat_template = True
    hf_trust_remote_code = False
    concept_names_text = ""
    torch_device = "cuda" if (TORCH_AVAILABLE and torch is not None and torch.cuda.is_available()) else "cpu"
    hf_attn_impl = "auto"
    concept_source = "Manual list"
    concept_size = 256
    prompt_source = "Concept prompts"
    dataset_file = None
    dataset_text_field = "text"
    dataset_max_prompts = 50000
    dataset_shuffle = True
    auto_run = False
    auto_interval_ms = 1000
    auto_steps = 5
    hf_local_path = ""
    hf_local_file = ""

    if model_choice == hf_label:
        st.subheader("Transformers settings")
        hf_model_id = _clean_path(st.text_input(
            "Model id or local path (folder with safetensors)",
            value=SETTINGS.get("hf_model_id", "Qwen/Qwen3-0.6B"),
            help="Hugging Face model id or a local folder containing safetensors + config/tokenizer.",
        ))
        prompt_source = st.selectbox(
            "Prompt source",
            ["Concept prompts", "Dataset file"],
            index=0 if SETTINGS.get("prompt_source") != "Dataset file" else 1,
            help="Use synthetic concept prompts or supply your own dataset.",
        )
        if prompt_source == "Dataset file":
            dataset_file = st.file_uploader(
                "Prompt dataset (.txt/.json/.jsonl)",
                type=["txt", "json", "jsonl"],
                help="One prompt per line (txt) or a JSON/JSONL list of prompt strings.",
            )
            dataset_text_field = st.text_input(
                "JSONL text field",
                value=SETTINGS.get("dataset_text_field", "text"),
                help="Field name to read from JSONL/JSON dict entries.",
            )
            dataset_max_prompts = st.number_input(
                "Max prompts to load",
                min_value=100,
                max_value=500000,
                value=int(SETTINGS.get("dataset_max_prompts", 50000)),
                step=100,
                help="Limit to avoid loading huge datasets into memory.",
            )
            dataset_shuffle = st.toggle(
                "Shuffle dataset",
                value=bool(SETTINGS.get("dataset_shuffle", True)),
                help="Shuffle the dataset after loading.",
            )
            concept_names_text = ""
        else:
            concept_source = st.selectbox(
                "Concept source",
                ["Manual list", "Generated (large)"],
                index=0 if SETTINGS.get("concept_source") != "Generated (large)" else 1,
                help="Manual list uses your concepts. Generated builds a large synthetic feature list.",
            )
            if concept_source == "Manual list":
                if "hf_concepts" not in st.session_state:
                    st.session_state["hf_concepts"] = SETTINGS.get("hf_concepts", ", ".join(FEATURES_MEDIUM))
                preset = st.selectbox(
                    "Concept preset",
                    ["Current", "Small", "Medium", "Large"],
                    help="Quickly switch to a curated concept list.",
                )
                if st.button("Apply preset", help="Replace the concept list with the chosen preset."):
                    if preset == "Small":
                        st.session_state["hf_concepts"] = ", ".join(FEATURES_SMALL)
                    elif preset == "Medium":
                        st.session_state["hf_concepts"] = ", ".join(FEATURES_MEDIUM)
                    elif preset == "Large":
                        st.session_state["hf_concepts"] = ", ".join(FEATURES_LARGE)
                concept_names_text = st.text_area(
                    "Concepts (comma-separated)",
                    key="hf_concepts",
                    height=160,
                    help="Comma-separated list of concepts to probe.",
                )
            else:
                concept_size = st.slider(
                    "Generated concept count",
                    128,
                    2000,
                    int(SETTINGS.get("concept_size", 512)),
                    64,
                    help="Larger lists provide broader coverage but run slower.",
                )
                st.caption("Generated concepts are synthetic, Sparse Autoencoder-style feature hints.")
                concept_names_text = ""
        hf_layer_mode = st.selectbox(
            "Layer mode",
            ["Single layer", "All layers"],
            index=0 if SETTINGS.get("hf_layer_mode") != "All layers" else 1,
            help="Single layer probes one layer; all layers collects stats for every layer.",
        )
        if hf_layer_mode == "Single layer":
            hf_layer_idx = st.number_input(
                "Layer index (-1 = last)",
                min_value=-200,
                max_value=200,
                value=int(SETTINGS.get("hf_layer_idx", -1)),
                step=1,
                help="Which hidden layer to probe. -1 means last.",
            )
        else:
            hf_active_layer = st.number_input(
                "Active layer for views (-1 = last)",
                min_value=-200,
                max_value=200,
                value=int(SETTINGS.get("hf_active_layer", -1)),
                step=1,
                help="Which layer to show in the UI when all layers are collected.",
            )
        hf_max_length = st.number_input(
            "Max tokens",
            min_value=32,
            max_value=4096,
            value=int(SETTINGS.get("hf_max_length", 256)),
            step=16,
            help="Maximum tokens per prompt batch.",
        )
        hf_use_chat_template = st.toggle(
            "Use chat template",
            value=bool(SETTINGS.get("hf_use_chat_template", True)),
            help="Use the model's chat template when available.",
        )
        hf_trust_remote_code = st.toggle(
            "Trust remote code",
            value=bool(SETTINGS.get("hf_trust_remote_code", False)),
            help="Enable if the model requires custom code from its repo.",
        )
        if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
            hf_device = st.selectbox(
                "Device",
                ["cuda", "cpu"],
                index=0 if SETTINGS.get("hf_device", "cuda") == "cuda" else 1,
                help="Run on GPU if available.",
            )
        else:
            hf_device = st.selectbox("Device", ["cpu"], index=0, help="GPU not available.")
        hf_dtype = st.selectbox(
            "Torch dtype",
            ["auto", "float16", "bfloat16", "float32"],
            index=["auto", "float16", "bfloat16", "float32"].index(SETTINGS.get("hf_dtype", "auto")),
            help="Precision for model weights.",
        )
        if st.session_state.get("force_cpu") and hf_device == "cuda":
            st.warning("CUDA error detected; forcing CPU for safety. Clear in Run settings if you want to retry GPU.")
            hf_device = "cpu"
        hf_effective_dtype = hf_dtype
        if hf_device == "cuda" and hf_dtype == "bfloat16":
            if torch is None or not torch.cuda.is_bf16_supported():
                hf_effective_dtype = "float16"
                st.caption("bf16 not supported on this GPU; using float16 instead.")
        if hf_device == "cpu" and hf_dtype in ("float16", "bfloat16"):
            hf_effective_dtype = "float32"
            st.caption("CPU does not support this dtype; using float32 instead.")
        hf_attn_impl = st.selectbox(
            "Attention implementation",
            ["auto", "eager", "sdpa", "flash_attention_2"],
            index=["auto", "eager", "sdpa", "flash_attention_2"].index(SETTINGS.get("hf_attn_impl", "auto")),
            help="Use eager on older GPUs (e.g., GTX 10-series).",
        )
        if hf_device == "cuda" and CUDA_CAPABILITY is not None and CUDA_CAPABILITY[0] < 8 and hf_attn_impl == "auto":
            st.caption(f"GPU {CUDA_NAME} (cc {CUDA_CAPABILITY[0]}.{CUDA_CAPABILITY[1]}) detected: auto will use eager.")
        n_concepts = 0
        n_noise = 0
        hidden_dim = 0
    else:
        st.subheader("Toy model settings")
        n_concepts = st.number_input(
            "Concept count", min_value=2, max_value=32, value=8, step=1, help="Number of concepts."
        )
        n_noise = st.number_input(
            "Noise dims", min_value=0, max_value=32, value=8, step=1, help="Extra noisy dimensions."
        )
        hidden_dim = st.number_input(
            "Hidden units", min_value=4, max_value=256, value=16, step=1, help="Hidden neurons to explore."
        )
        if model_choice == custom_label or model_choice == "Torch MLP":
            if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
                torch_device = st.selectbox("Device", ["cuda", "cpu"], help="Run on GPU if available.")
            else:
                torch_device = st.selectbox("Device", ["cpu"], help="GPU not available.")
    seed = st.number_input("Seed", min_value=1, max_value=9999, value=7, step=1, help="Random seed.")

    st.divider()
    st.subheader("Unit source")
    unit_source = st.selectbox(
        "Units",
        ["Raw neurons", "Sparse Autoencoder features"],
        help="Choose whether to inspect raw hidden neurons or Sparse Autoencoder-derived feature units.",
    )
    unit_label_sidebar = "neurons" if unit_source == "Raw neurons" else "features"
    sae_mode = "Random (prototype)"
    sae_seed = 1
    sae_n_features = 512
    sae_file = None
    if unit_source == "Sparse Autoencoder features":
        sae_mode = st.selectbox(
            "Sparse Autoencoder weights",
            ["Random (prototype)", "Load .npz", "Load .pt/.safetensors"],
            help="Load real Sparse Autoencoder encoder weights or use a random prototype for quick demos.",
        )
        if sae_mode == "Random (prototype)":
            sae_n_features = st.number_input(
                "Sparse Autoencoder feature count",
                min_value=32,
                max_value=8192,
                value=512,
                step=32,
                help="Number of Sparse Autoencoder features to simulate.",
            )
            sae_seed = st.number_input(
                "Sparse Autoencoder seed",
                min_value=1,
                max_value=9999,
                value=1,
                step=1,
                help="Random seed for prototype Sparse Autoencoder weights.",
            )
        elif sae_mode == "Load .npz":
            sae_file = st.file_uploader(
                "Sparse Autoencoder weights (.npz)",
                type=["npz"],
                help="Upload a Sparse Autoencoder encoder npz with W_enc (+ optional b_enc).",
            )
        else:
            sae_file = st.file_uploader(
                "Sparse Autoencoder weights (.pt/.safetensors)",
                type=["pt", "pth", "safetensors"],
                help="Upload a real Sparse Autoencoder encoder state dict (.pt/.pth) or .safetensors.",
            )

    st.divider()
    with st.expander("Stream controls", expanded=False):
        if model_choice == hf_label:
            p_combo = st.slider(
                "Pair prompt rate", 0.0, 0.8, 0.3, 0.05, help="How often to generate paired prompts."
            )
            focus_boost = st.slider(
                "Active probe boost",
                0.0,
                0.8,
                0.6,
                0.05,
                help="How strongly the prober favors uncertain concepts.",
            )
            p_on = 0.2
            noise_scale = 1.0
        else:
            p_on = st.slider(
                "Base concept probability", 0.05, 0.6, 0.2, 0.05, help="Base activation rate."
            )
            focus_boost = st.slider(
                "Active probe boost",
                0.0,
                0.8,
                0.6,
                0.05,
                help="How strongly the prober favors uncertain concepts.",
            )
            noise_scale = st.slider("Noise scale", 0.2, 2.0, 1.0, 0.1, help="Noise magnitude.")
            p_combo = 0.3
        active_probe = st.toggle(
            "Active probing", value=True, help="Selects stimuli to reduce uncertainty."
        )

    st.divider()
    with st.expander("Run settings", expanded=True):
        if st.session_state.get("force_cpu"):
            st.info("CUDA failover active: running on CPU. Use 'Clear CUDA failover' to retry GPU.")
            if st.button("Clear CUDA failover", help="Allow GPU again on the next run."):
                st.session_state.pop("force_cpu", None)
        if model_choice == hf_label:
            batch_size = st.number_input(
                "Batch size", min_value=1, max_value=64, value=4, step=1, help="Prompts per batch."
            )
        else:
            batch_size = st.number_input(
                "Batch size", min_value=32, max_value=2048, value=256, step=32, help="Samples per batch."
            )
        if "autorun_active" not in st.session_state:
            st.session_state["autorun_active"] = False
        auto_cols = st.columns([2, 1, 1])
        with auto_cols[0]:
            if st.session_state["autorun_active"]:
                if st.button("Stop auto‑explore", type="primary", help="Stop continuous exploration."):
                    st.session_state["autorun_active"] = False
            else:
                if st.button("Auto‑explore", type="primary", help="Run continuously until mapping converges."):
                    st.session_state["autorun_active"] = True
                    st.session_state["auto_conf_hist"] = []
                    st.session_state["map_stats"] = {"last_t": None, "last_step": 0, "speed": 0.0}
                    st.session_state["autorun_armed"] = True
        with auto_cols[1]:
            if "auto_interval_ms" not in st.session_state:
                st.session_state["auto_interval_ms"] = int(SETTINGS.get("auto_interval_ms", 400))
            auto_interval_ms = st.slider(
                "Update interval (ms)",
                100,
                2000,
                int(st.session_state["auto_interval_ms"]),
                50,
                help="How often the UI refreshes during auto‑explore.",
            )
        with auto_cols[2]:
            if "auto_steps" not in st.session_state:
                st.session_state["auto_steps"] = int(
                    SETTINGS.get("auto_steps", 15 if model_choice == hf_label else 30)
                )
            auto_steps = st.slider(
                "Steps per update",
                1,
                300,
                int(st.session_state["auto_steps"]),
                1,
                help="How many steps to run each refresh.",
            )
        with st.expander("Auto‑explore settings", expanded=False):
            if "auto_max_steps" not in st.session_state:
                st.session_state["auto_max_steps"] = int(
                    SETTINGS.get("auto_max_steps", 5000 if model_choice == hf_label else 2000)
                )
            if "auto_min_conf" not in st.session_state:
                st.session_state["auto_min_conf"] = float(SETTINGS.get("auto_min_conf", 0.45))
            if "auto_min_steps" not in st.session_state:
                st.session_state["auto_min_steps"] = int(
                    SETTINGS.get("auto_min_steps", 500 if model_choice == hf_label else 200)
                )
            if "auto_min_samples" not in st.session_state:
                st.session_state["auto_min_samples"] = int(
                    SETTINGS.get("auto_min_samples", 2000 if model_choice == hf_label else 500)
                )
            if "auto_patience" not in st.session_state:
                st.session_state["auto_patience"] = int(SETTINGS.get("auto_patience", 8))
            if "auto_min_delta" not in st.session_state:
                st.session_state["auto_min_delta"] = float(SETTINGS.get("auto_min_delta", 0.01))
            auto_max_steps = st.slider(
                "Max steps",
                200,
                20000,
                int(st.session_state["auto_max_steps"]),
                100,
                help="Hard cap for auto‑explore steps.",
            )
            auto_min_steps = st.slider(
                "Min steps before stop",
                50,
                10000,
                int(st.session_state["auto_min_steps"]),
                50,
                help="Minimum steps before auto‑explore can stop.",
            )
            auto_min_conf = st.slider(
                "Min mean confidence",
                0.1,
                0.9,
                float(st.session_state["auto_min_conf"]),
                0.01,
                help="Target confidence level.",
            )
            auto_min_samples = st.slider(
                "Min feature samples",
                0,
                20000,
                int(st.session_state["auto_min_samples"]),
                100,
                help="Minimum samples for feature dictionary.",
            )
            auto_patience = st.slider(
                "Stability window",
                3,
                20,
                int(st.session_state["auto_patience"]),
                1,
                help="Number of recent updates to check for stability.",
            )
            auto_min_delta = st.slider(
                "Min improvement",
                0.0,
                0.2,
                float(st.session_state["auto_min_delta"]),
                0.005,
                help="Minimum improvement across the stability window.",
            )
        auto_run = st.session_state["autorun_active"]
    if "steps_per_run" not in st.session_state:
        st.session_state["steps_per_run"] = int(
            SETTINGS.get("steps_per_run", 50 if model_choice == hf_label else 200)
        )
    steps_per_run = st.number_input(
        "Steps per run",
        min_value=1,
        max_value=2000,
        value=int(st.session_state["steps_per_run"]),
        step=10,
        help="Used by the single Run button.",
    )
    record_every = st.number_input(
        "Record every N steps", min_value=1, max_value=200, value=10, step=1, help="History sampling rate."
    )
    top_n = st.number_input(
        f"Top {unit_label_sidebar}",
        min_value=5,
        max_value=100,
        value=40,
        step=5,
        help="How many units to show in tables.",
    )


model_ready = True
dataset_prompts = []
dataset_warnings = []
dataset_hash = None
if model_choice == custom_label:
    model_ready = uploaded_file is not None and TORCH_AVAILABLE
    if uploaded_file is None:
        st.sidebar.info("Upload a TorchScript .pt/.pth file to enable running.")
    elif not TORCH_AVAILABLE:
        st.sidebar.error("PyTorch is not installed. Install it to use custom models.")
if model_choice == hf_label:
    if prompt_source == "Dataset file":
        if dataset_file is not None:
            dataset_bytes = dataset_file.getvalue()
            dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()[:12]
            dataset_prompts, dataset_warnings = _load_prompt_dataset(
                dataset_file,
                int(dataset_max_prompts),
                dataset_text_field,
            )
        concept_list = []
        model_ready = (
            bool(hf_model_id)
            and TORCH_AVAILABLE
            and TRANSFORMERS_AVAILABLE
            and len(dataset_prompts) > 0
        )
    else:
        if concept_source == "Generated (large)":
            concept_list = generate_concepts(int(concept_size), int(seed))
        else:
            concept_list = [c.strip() for c in concept_names_text.split(",") if c.strip()]
        model_ready = bool(hf_model_id) and TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE and len(concept_list) > 0
    if not hf_model_id:
        st.sidebar.info("Enter a Hugging Face model id or local path to enable running.")
    elif not TRANSFORMERS_AVAILABLE:
        st.sidebar.error("Transformers is not installed. Install HF support to use this model.")
    elif not TORCH_AVAILABLE:
        st.sidebar.error("PyTorch is not installed. Install HF support to use this model.")
    elif prompt_source == "Dataset file" and len(dataset_prompts) == 0:
        st.sidebar.info("Upload a dataset with at least one prompt.")
    elif prompt_source != "Dataset file" and len(concept_list) == 0:
        st.sidebar.info("Add at least one concept to probe.")
else:
    concept_list = []

if not model_ready and st.session_state.get("autorun_active"):
    st.session_state["autorun_active"] = False

if model_choice == hf_label and prompt_source == "Dataset file":
    if dataset_prompts:
        st.sidebar.caption(f"Loaded {len(dataset_prompts)} prompts.")
    if dataset_warnings:
        st.sidebar.warning(" ".join(dataset_warnings))

sae_weights = None
sae_warnings = []
sae_error = None
sae_hash = None
sae_feature_count = int(sae_n_features)
if unit_source == "Sparse Autoencoder features":
    if sae_mode == "Load .npz":
        if sae_file is None:
            sae_error = "Upload a .npz with Sparse Autoencoder weights."
        else:
            sae_bytes = sae_file.getvalue()
            sae_hash = hashlib.sha256(sae_bytes).hexdigest()[:12]
            try:
                sae_weights, sae_warnings = load_sae_weights_npz(sae_bytes)
                sae_shape = np.asarray(sae_weights.W_enc).shape
                sae_feature_count = int(max(sae_shape)) if len(sae_shape) == 2 else int(sae_n_features)
            except Exception as exc:
                sae_error = f"Failed to load Sparse Autoencoder weights: {exc}"
    elif sae_mode == "Load .pt/.safetensors":
        if sae_file is None:
            sae_error = "Upload a .pt or .safetensors Sparse Autoencoder file."
        else:
            sae_bytes = sae_file.getvalue()
            sae_hash = hashlib.sha256(sae_bytes).hexdigest()[:12]
            try:
                name = (sae_file.name or "").lower()
                if name.endswith(".safetensors"):
                    sae_weights, sae_warnings = load_sae_weights_safetensors(sae_bytes)
                else:
                    sae_weights, sae_warnings = load_sae_weights_torch(sae_bytes)
                sae_shape = np.asarray(sae_weights.W_enc).shape
                sae_feature_count = int(max(sae_shape)) if len(sae_shape) == 2 else int(sae_n_features)
            except Exception as exc:
                sae_error = f"Failed to load Sparse Autoencoder weights: {exc}"
    else:
        sae_feature_count = int(sae_n_features)

if unit_source == "Sparse Autoencoder features":
    if sae_mode in ("Load .npz", "Load .pt/.safetensors"):
        if sae_file is None:
            model_ready = False
            st.sidebar.info("Upload a Sparse Autoencoder file to enable Sparse Autoencoder features.")
        elif sae_error is not None:
            model_ready = False
            st.sidebar.error(sae_error)
    if sae_warnings:
        st.sidebar.warning(" ".join(sae_warnings))

custom_hash = None
if uploaded_file is not None:
    data = uploaded_file.getvalue()
    custom_hash = hashlib.sha256(data).hexdigest()[:12]

hf_hash = None
if hf_model_id:
    hf_hash = hashlib.sha256(hf_model_id.encode("utf-8")).hexdigest()[:12]

state_key = (
    model_choice,
    custom_hash,
    hf_hash,
    tuple(concept_list),
    concept_source,
    int(concept_size),
    prompt_source,
    dataset_hash,
    int(dataset_max_prompts),
    str(dataset_text_field),
    bool(dataset_shuffle),
    int(hf_layer_idx),
    hf_layer_mode,
    int(hf_active_layer),
    int(hf_max_length),
    hf_device,
    hf_effective_dtype,
    bool(hf_use_chat_template),
    bool(hf_trust_remote_code),
    float(p_combo),
    torch_device,
    hf_attn_impl,
    int(auto_interval_ms),
    int(auto_steps),
    int(auto_max_steps),
    int(auto_min_steps),
    float(auto_min_conf),
    int(auto_min_samples),
    int(auto_patience),
    float(auto_min_delta),
    int(n_concepts),
    int(n_noise),
    int(hidden_dim),
    int(seed),
    float(p_on),
    float(focus_boost),
    float(noise_scale),
    unit_source,
    sae_mode,
    int(sae_feature_count),
    int(sae_seed),
    sae_hash,
)


def init_state() -> dict:
    def build_unit_adapter(input_dim: int):
        if unit_source == "Sparse Autoencoder features":
            if sae_mode in ("Load .npz", "Load .pt/.safetensors"):
                if sae_weights is None:
                    raise RuntimeError("No Sparse Autoencoder weights loaded.")
                weights = sae_weights
            else:
                weights = random_sae_weights(
                    int(input_dim), int(sae_feature_count), int(sae_seed)
                )
            adapter = SAEFeatureAdapter(int(input_dim), weights)
            warnings = list(sae_warnings)
            meta = weights.meta or {}
            if meta.get("transposed"):
                warnings.append("Transposed W_enc to match model hidden size.")
            if meta.get("bias_fallback"):
                warnings.append("Bias shape mismatch; using zeros.")
            return adapter, warnings
        adapter = IdentityUnitAdapter(int(input_dim))
        return adapter, []

    if model_choice == hf_label:
        if prompt_source == "Dataset file":
            stream_cfg = DatasetPromptStreamConfig(
                prompts=list(dataset_prompts),
                seed=int(seed),
                shuffle=bool(dataset_shuffle),
            )
            stream = DatasetPromptStream(stream_cfg)
        else:
            stream_cfg = PromptStreamConfig(
                concepts=concept_list,
                seed=int(seed),
                p_combo=float(p_combo),
                focus_boost=float(focus_boost),
            )
            stream = PromptStream(stream_cfg)
        model_cfg = ModelConfig(
            n_concepts=len(concept_list),
            n_noise=0,
            hidden_dim=1,
            seed=int(seed),
            device=hf_device,
        )
        attn_impl = hf_attn_impl
        if (
            hf_device == "cuda"
            and CUDA_CAPABILITY is not None
            and CUDA_CAPABILITY[0] < 8
            and hf_attn_impl == "auto"
        ):
            attn_impl = "eager"
        device_for_model = hf_device
        dtype_for_model = hf_effective_dtype
        try:
            model = TransformersCausalLMAdapter(
                model_cfg,
                model_id=hf_model_id,
                device=device_for_model,
                dtype=dtype_for_model,
                layer_idx=hf_layer_idx,
                max_length=hf_max_length,
                use_chat_template=hf_use_chat_template,
                trust_remote_code=hf_trust_remote_code,
                attn_implementation=attn_impl,
            )
        except RuntimeError as exc:
            msg = str(exc)
            if (
                device_for_model == "cuda"
                and ("CUDA error" in msg or "illegal memory access" in msg)
            ):
                st.warning(
                    "CUDA error while loading the model. Falling back to CPU for this run. "
                    "Try Device=CPU, Attention=eager, or a smaller batch size."
                )
                device_for_model = "cpu"
                dtype_for_model = "float32"
                model_cfg = ModelConfig(
                    n_concepts=len(concept_list),
                    n_noise=0,
                    hidden_dim=1,
                    seed=int(seed),
                    device=device_for_model,
                )
                model = TransformersCausalLMAdapter(
                    model_cfg,
                    model_id=hf_model_id,
                    device=device_for_model,
                    dtype=dtype_for_model,
                    layer_idx=hf_layer_idx,
                    max_length=hf_max_length,
                    use_chat_template=hf_use_chat_template,
                    trust_remote_code=hf_trust_remote_code,
                    attn_implementation=attn_impl,
                )
            else:
                raise
        unit_adapter, unit_warnings = build_unit_adapter(model.hidden_dim)
        unit_dim = unit_adapter.unit_dim
        if hf_layer_mode == "All layers":
            layer_count = model.hidden_state_count
            labelers = [
                OnlineAiNeuronLabeler(AiLabelConfig(concept_names=stream.concept_names), hidden=unit_dim)
                for _ in range(layer_count)
            ]
            feature_dicts = [init_feature_dict(unit_dim) for _ in range(layer_count)]
            active_layer = int(hf_active_layer)
            if active_layer < 0:
                active_layer = layer_count + active_layer
            active_layer = max(0, min(active_layer, layer_count - 1))
            return {
                "stream": stream,
                "model": model,
                "labelers": labelers,
                "labeler": labelers[active_layer],
                "feature_dicts": feature_dicts,
                "feature_dict": feature_dicts[active_layer],
                "prober": ActiveConceptProber(rng_seed=int(seed)),
                "history": [],
                "step": 0,
                "last_focus": None,
                "layer_mode": "all",
                "active_layer": active_layer,
                "unit_adapter": unit_adapter,
                "unit_label": unit_adapter.unit_label,
                "unit_label_plural": unit_adapter.unit_label_plural,
                "unit_source": unit_source,
                "unit_meta": unit_adapter.metadata(),
                "unit_warnings": unit_warnings,
            }
        labeler = OnlineAiNeuronLabeler(
            AiLabelConfig(concept_names=stream.concept_names), hidden=unit_dim
        )
        return {
            "stream": stream,
            "model": model,
            "labeler": labeler,
            "feature_dict": init_feature_dict(unit_dim),
            "prober": ActiveConceptProber(rng_seed=int(seed)),
            "history": [],
            "step": 0,
            "last_focus": None,
            "layer_mode": "single",
            "unit_adapter": unit_adapter,
            "unit_label": unit_adapter.unit_label,
            "unit_label_plural": unit_adapter.unit_label_plural,
            "unit_source": unit_source,
            "unit_meta": unit_adapter.metadata(),
            "unit_warnings": unit_warnings,
        }
    else:
        stream_cfg = ConceptStreamConfig(
            n_concepts=int(n_concepts),
            n_noise=int(n_noise),
            p_on=float(p_on),
            focus_boost=float(focus_boost),
            noise_scale=float(noise_scale),
            seed=int(seed),
        )
        stream = ConceptStream(stream_cfg)
        model_cfg = ModelConfig(
            n_concepts=int(n_concepts),
            n_noise=int(n_noise),
            hidden_dim=int(hidden_dim),
            seed=int(seed),
            device=torch_device,
        )
        if model_choice == custom_label:
            if uploaded_file is None:
                raise RuntimeError("No model file uploaded.")
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch is required for custom TorchScript models.")
            model = TorchScriptAdapter(model_cfg, uploaded_file.getvalue())
        else:
            model = model_lookup[model_choice].build(model_cfg)
    unit_adapter, unit_warnings = build_unit_adapter(model.hidden_dim)
    unit_dim = unit_adapter.unit_dim
    labeler = OnlineAiNeuronLabeler(
        AiLabelConfig(concept_names=stream.concept_names), hidden=unit_dim
    )
    prober = ActiveConceptProber(rng_seed=int(seed))
    return {
        "stream": stream,
        "model": model,
        "labeler": labeler,
        "prober": prober,
        "history": [],
        "step": 0,
        "last_focus": None,
        "unit_adapter": unit_adapter,
        "unit_label": unit_adapter.unit_label,
        "unit_label_plural": unit_adapter.unit_label_plural,
        "unit_source": unit_source,
        "unit_meta": unit_adapter.metadata(),
        "unit_warnings": unit_warnings,
    }


if "state" not in st.session_state or st.session_state.get("state_key") != state_key:
    if not model_ready:
        st.session_state.pop("map_cache", None)
        st.warning("Complete the model settings to continue.")
        st.stop()
    st.session_state.pop("map_cache", None)
    st.session_state["map_epoch"] = int(st.session_state.get("map_epoch", 0)) + 1
    st.session_state["state"] = init_state()
    st.session_state["state_key"] = state_key

state = st.session_state["state"]

unit_label = state.get("unit_label", "Neuron")
unit_label_plural = state.get("unit_label_plural", "Neurons")
unit_label_lower = unit_label.lower()
unit_label_plural_lower = unit_label_plural.lower()


def _encode_units(h: np.ndarray) -> np.ndarray:
    adapter = state.get("unit_adapter")
    if adapter is None:
        return h
    return adapter.encode(h)


def _pca_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    if x.shape[1] == 1:
        x = np.concatenate([x, np.zeros_like(x)], axis=1)
    try:
        u, s, _ = np.linalg.svd(x, full_matrices=False)
        coords = u[:, :2] * s[:2]
        if not np.all(np.isfinite(coords)):
            raise np.linalg.LinAlgError("non-finite coords")
        return coords.astype(np.float32)
    except np.linalg.LinAlgError:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        cov = x.T @ x
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vecs = vecs[:, order[:2]]
        coords = x @ vecs
        return coords.astype(np.float32)


def _pca_3d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    if x.shape[1] == 1:
        x = np.concatenate([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
    if x.shape[1] == 2:
        x = np.concatenate([x, np.zeros((x.shape[0], 1), dtype=x.dtype)], axis=1)
    try:
        u, s, _ = np.linalg.svd(x, full_matrices=False)
        coords = u[:, :3] * s[:3]
        if not np.all(np.isfinite(coords)):
            raise np.linalg.LinAlgError("non-finite coords")
        return coords.astype(np.float32)
    except np.linalg.LinAlgError:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        cov = x.T @ x
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vecs = vecs[:, order[:3]]
        coords = x @ vecs
        return coords.astype(np.float32)


def _build_similarity_edges(coords: np.ndarray, feats: np.ndarray, k: int, min_sim: float):
    if k <= 0 or feats.shape[0] == 0:
        return []
    f = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    sim = f @ f.T
    edges = set()
    for i in range(sim.shape[0]):
        idx = np.argsort(sim[i])[::-1]
        count = 0
        for j in idx:
            if i == j:
                continue
            if sim[i, j] < min_sim:
                break
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))
            count += 1
            if count >= k:
                break
    return list(edges)

quick_run = None
control_bar = st.container()
with control_bar:
    col_run, col_step, col_reset, col_note = st.columns([1, 1, 1, 5], gap="small")
    run_clicked = col_run.button(
        f"Run ({int(steps_per_run)})",
        disabled=not model_ready,
        help="Run exactly the number of steps set in the sidebar.",
    )
    step_clicked = col_step.button("Single step", disabled=not model_ready)
    reset_clicked = col_reset.button("Reset")
    col_note.markdown(
        "<div class='small-muted'>Tip: keep active probing on to watch confidence rise faster.</div>",
        unsafe_allow_html=True,
    )

if reset_clicked:
    st.session_state["state"] = init_state()
    st.session_state["state_key"] = state_key
    st.session_state.pop("map_cache", None)
    st.session_state["map_epoch"] = int(st.session_state.get("map_epoch", 0)) + 1
    state = st.session_state["state"]


def run_steps(num_steps: int) -> None:
    progress = None
    if num_steps > 1:
        progress = st.progress(0, text="Running...")
    update_every = max(1, num_steps // 20)
    for i in range(num_steps):
        focus_idx = None
        if active_probe:
            focus_idx = state["prober"].choose_focus_concept(state["labeler"])
        try:
            if getattr(state["model"], "input_mode", "vector") == "text":
                prompts, c = state["stream"].sample(batch=int(batch_size), focus_idx=focus_idx)
                state["last_prompts"] = prompts[: min(10, len(prompts))]
                if state.get("layer_mode") == "all" and hasattr(state["model"], "forward_hidden_text_all_layers"):
                    h_all = state["model"].forward_hidden_text_all_layers(prompts)
                    for idx, labeler in enumerate(state["labelers"]):
                        if idx >= h_all.shape[0]:
                            break
                        units = _encode_units(h_all[idx])
                        labeler.update(units, c)
                        if "feature_dicts" in state:
                            update_feature_dict(state["feature_dicts"][idx], prompts, units)
                    state["labeler"] = state["labelers"][state["active_layer"]]
                    if "feature_dicts" in state:
                        state["feature_dict"] = state["feature_dicts"][state["active_layer"]]
                else:
                    h = state["model"].forward_hidden_text(prompts)
                    units = _encode_units(h)
                    state["labeler"].update(units, c)
                    if "feature_dict" in state:
                        update_feature_dict(state["feature_dict"], prompts, units)
            else:
                x, c = state["stream"].sample(batch=int(batch_size), focus_idx=focus_idx)
                h = state["model"].forward_hidden(x)
                units = _encode_units(h)
                state["labeler"].update(units, c)
        except RuntimeError as exc:
            msg = str(exc)
            if "CUDA error" in msg or "illegal memory access" in msg:
                st.error(
                    "CUDA kernel error. Try: switch Device to CPU, set Attention to eager, "
                    "lower batch size / max tokens, or disable All layers mode."
                )
                st.session_state["autorun_active"] = False
                st.session_state["force_cpu"] = True
                st.stop()
            raise
        state["step"] += 1
        state["last_focus"] = focus_idx

        if state["step"] % int(record_every) == 0:
            snap = state["labeler"].snapshot(top_n=int(top_n))
            if not snap.empty:
                state["history"].append(
                    {
                        "step": state["step"],
                        "mean_conf": float(snap["conf_single"].mean()),
                        "mean_act": float(snap["mean_act"].mean()),
                    }
                )
        if progress is not None and (i % update_every == 0 or i == num_steps - 1):
            progress.progress((i + 1) / num_steps, text=f"Running step {i + 1} / {num_steps}")
    if progress is not None:
        progress.empty()


def _autorun_should_stop(
    state: dict,
    mean_conf: float,
    min_conf: float,
    min_steps: int,
    min_samples: int,
    patience: int,
    min_delta: float,
    max_steps: int,
) -> bool:
    hist = state.setdefault("auto_conf_hist", [])
    hist.append(float(mean_conf))
    if len(hist) > int(patience):
        hist[:] = hist[-int(patience) :]
    if state["step"] >= int(max_steps):
        return True
    if state["step"] < int(min_steps):
        return False
    if mean_conf < float(min_conf):
        return False
    if "feature_dict" in state and state["feature_dict"].get("count", 0) < int(min_samples):
        return False
    if len(hist) < int(patience):
        return False
    return (max(hist) - min(hist)) < float(min_delta)


if auto_run and model_ready:
    if st.session_state.pop("autorun_armed", False):
        run_steps(int(auto_steps))
    else:
        run_steps(int(auto_steps))
    if st.session_state.get("autorun_active"):
        time.sleep(auto_interval_ms / 1000.0)
        _do_rerun()
elif run_clicked or step_clicked:
    run_steps(1 if step_clicked else int(steps_per_run))

_save_settings(
    {
        "model_choice": model_choice,
        "hf_model_id": hf_model_id,
        "hf_layer_mode": hf_layer_mode,
        "hf_layer_idx": hf_layer_idx,
        "hf_active_layer": hf_active_layer,
        "hf_max_length": hf_max_length,
        "hf_use_chat_template": hf_use_chat_template,
        "hf_trust_remote_code": hf_trust_remote_code,
        "hf_device": hf_device,
        "hf_dtype": hf_dtype,
        "hf_attn_impl": hf_attn_impl,
        "prompt_source": prompt_source,
        "dataset_text_field": dataset_text_field,
        "dataset_max_prompts": dataset_max_prompts,
        "dataset_shuffle": dataset_shuffle,
        "token_model_id": st.session_state.get("token_model_id", ""),
        "token_device": st.session_state.get("token_device", "cuda"),
        "token_dtype": st.session_state.get("token_dtype", "float16"),
        "token_layer": st.session_state.get("token_layer", -1),
        "token_hookpoint": st.session_state.get("token_hookpoint", "resid_post"),
        "token_topk": st.session_state.get("token_topk", 16),
        "token_max_len": st.session_state.get("token_max_len", 128),
        "token_batch_size": st.session_state.get("token_batch_size", 1),
        "token_ctx_window": st.session_state.get("token_ctx_window", 4),
        "token_out_dir": st.session_state.get("token_out_dir", "token_sae_out"),
        "token_dataset_path": st.session_state.get("token_dataset_path", ""),
        "token_dataset_field": st.session_state.get("token_dataset_field", "text"),
        "token_dataset_max": st.session_state.get("token_dataset_max", 20000),
        "token_dataset_shuffle": st.session_state.get("token_dataset_shuffle", True),
        "token_sae_path": st.session_state.get("token_sae_path", ""),
        "token_trust_remote": st.session_state.get("token_trust_remote", False),
        "token_realtime_max": st.session_state.get("token_realtime_max", 32),
        "concept_source": concept_source,
        "concept_size": concept_size,
        "hf_concepts": st.session_state.get("hf_concepts", ""),
        "auto_interval_ms": auto_interval_ms,
        "auto_steps": auto_steps,
        "auto_max_steps": auto_max_steps,
        "auto_min_steps": auto_min_steps,
        "auto_min_conf": auto_min_conf,
        "auto_min_samples": auto_min_samples,
        "auto_patience": auto_patience,
        "auto_min_delta": auto_min_delta,
        "steps_per_run": steps_per_run,
    }
)

snap = state["labeler"].snapshot(top_n=int(top_n))

if "map_stats" not in st.session_state:
    st.session_state["map_stats"] = {"last_t": None, "last_step": 0, "speed": 0.0}

if st.session_state.get("autorun_active") and model_ready:
    mean_conf = float(snap["conf_single"].mean()) if not snap.empty else 0.0
    if _autorun_should_stop(
        state,
        mean_conf=mean_conf,
        min_conf=auto_min_conf,
        min_steps=auto_min_steps,
        min_samples=auto_min_samples,
        patience=auto_patience,
        min_delta=auto_min_delta,
        max_steps=auto_max_steps,
    ):
        st.session_state["autorun_active"] = False
        st.info("Auto‑explore complete: mapping stabilized.")

metric_cols = st.columns(4)
metric_cols[0].metric("Steps", state["step"])
metric_cols[1].metric("Total samples", state["labeler"].total_samples)
if state["last_focus"] is not None and state["stream"].concept_names:
    active_name = state["stream"].concept_names[state["last_focus"]]
else:
    active_name = "n/a"
metric_cols[2].metric("Active concept", active_name)
metric_cols[3].metric("Mean conf (top units)", f"{snap['conf_single'].mean():.3f}" if not snap.empty else "0.000")

if st.session_state.get("autorun_active"):
    now = time.time()
    stats = st.session_state["map_stats"]
    if stats["last_t"] is not None:
        dt = now - stats["last_t"]
        ds = state["step"] - stats["last_step"]
        if dt > 0:
            stats["speed"] = 0.7 * stats["speed"] + 0.3 * (ds / dt)
    stats["last_t"] = now
    stats["last_step"] = state["step"]
    st.caption(f"Auto‑exploring… {stats['speed']:.1f} steps/sec")
    pulse = (now * 0.6) % 1.0
    st.progress(pulse, text="Auto‑exploring…")

tab_map, tab_cards, tab_trends, tab_inspect, tab_features, tab_token, tab_storage = st.tabs(
    [
        f"{unit_label} map",
        f"{unit_label} cards",
        "Trends",
        f"{unit_label} inspection",
        "Feature explorer",
        "Token Sparse Autoencoder",
        "Storage",
    ]
)
has_concepts = bool(state["stream"].concept_names)

with tab_map:
    if not has_concepts:
        st.info("Concept labels are disabled for dataset prompts; map view is unavailable.")
    elif not PLOTLY_AVAILABLE:
        st.info(f"Install plotly to enable the {unit_label_lower} map view.")
    else:
        with st.expander("Map settings", expanded=False):
            map_edges = st.slider(
                f"Connections per {unit_label_lower}",
                min_value=0,
                max_value=8,
                value=2,
                step=1,
                help="How many nearest neighbors to connect per unit.",
            )
            map_min_sim = st.slider(
                "Min similarity",
                min_value=-1.0,
                max_value=1.0,
                value=0.4,
                step=0.05,
                help="Ignore edges weaker than this similarity threshold.",
            )
            map_show_labels = st.toggle(
                "Show labels",
                value=False,
                help="Show text labels for unit guesses on the map.",
            )
            map_spread = st.slider(
                "Spread",
                min_value=0.5,
                max_value=6.0,
                value=2.5,
                step=0.1,
                help="Expand or contract the 2D layout.",
            )
            map_jitter = st.slider(
                "Jitter",
                min_value=0.0,
                max_value=1.0,
                value=0.15,
                step=0.05,
                help="Add small noise to separate overlapping points.",
            )
            map_norm = st.toggle(
                "Normalize features",
                value=True,
                help="Standardize features before PCA for a cleaner layout.",
            )
            map_hide_empty = st.toggle(
                "Hide empty units",
                value=True,
                help="Remove units with near-zero effect magnitude.",
            )
            map_filter_pct = st.slider(
                "Filter lowest effect percentile",
                0,
                90,
                20,
                5,
                help="Drop the lowest-activity units to reduce clutter.",
            )
            map_refresh = st.slider(
                "Refresh every N steps",
                1,
                200,
                10,
                1,
                help="Reduce recomputation by updating the map every N steps.",
            )
            map_btn_cols = st.columns([1, 1])
            if map_btn_cols[0].button("Clear map cache", help="Force the map to recompute from current data."):
                st.session_state.pop("map_cache", None)
                st.session_state["map_epoch"] = int(st.session_state.get("map_epoch", 0)) + 1
            if map_btn_cols[1].button("Reset map data", help="Clear label stats and start the map fresh."):
                st.session_state["state"] = _reset_mapping_state(state)
                state = st.session_state["state"]

        prompt_supported = getattr(state.get("model"), "input_mode", "vector") == "text"
        with st.expander("Prompt mapper", expanded=False):
            prompt_map_enabled = st.toggle(
                "Enable realtime prompt mapper",
                value=bool(st.session_state.get("prompt_map_enabled", False)),
                key="prompt_map_enabled",
                help="Project a custom prompt onto the map for visual intuition.",
            )
            if not prompt_supported:
                st.info("Prompt mapping is available only for text models.")
            prompt_live = st.toggle(
                "Live update",
                value=bool(st.session_state.get("prompt_map_live", False)),
                key="prompt_map_live",
                disabled=not prompt_map_enabled or not prompt_supported,
                help="Recompute on every prompt edit (more GPU/CPU use).",
            )
            prompt_top_pct = st.slider(
                "Use top activations (%)",
                5,
                100,
                int(st.session_state.get("prompt_map_top_pct", 30)),
                5,
                disabled=not prompt_map_enabled or not prompt_supported,
                help="Only keep the most active units when positioning the prompt.",
            )
            st.session_state["prompt_map_top_pct"] = int(prompt_top_pct)
            prompt_text = st.text_area(
                "Prompt",
                value=st.session_state.get("prompt_map_text", ""),
                key="prompt_map_text",
                height=100,
                disabled=not prompt_map_enabled or not prompt_supported,
                help="Type a prompt to project onto the map.",
            )
            map_prompt_clicked = st.button(
                "Map prompt",
                disabled=not prompt_map_enabled or not prompt_supported or not prompt_text,
                help="Project the prompt onto the map once.",
            )

        spinner = st.spinner("Updating map…") if st.session_state.get("autorun_active") else contextlib.nullcontext()
        with spinner:
            cache = st.session_state.setdefault("map_cache", {})
            step_bucket = int(state["step"] // max(1, int(map_refresh)))
            cache_key = (
                int(st.session_state.get("map_epoch", 0)),
                step_bucket,
                int(state["labeler"].total_samples),
                state.get("active_layer"),
                int(state["labeler"].hidden),
                int(map_edges),
                float(map_min_sim),
                bool(map_show_labels),
                float(map_spread),
                float(map_jitter),
                bool(map_norm),
                bool(map_hide_empty),
                int(map_filter_pct),
                int(map_refresh),
            )
            if cache.get("key") != cache_key:
                feats = state["labeler"].effect_matrix()
                feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
                if map_norm:
                    std = feats.std(axis=0, keepdims=True)
                    std[std < 1e-6] = 1.0
                    feats = (feats - feats.mean(axis=0, keepdims=True)) / std
                effect_norm = np.linalg.norm(feats, axis=1)
                has_activity = int(state["labeler"].total_samples) > 0
                keep_idx = np.arange(feats.shape[0])
                if has_activity and map_hide_empty and keep_idx.size:
                    keep_idx = keep_idx[effect_norm > 1e-8]
                if has_activity and map_filter_pct > 0 and keep_idx.size:
                    thresh = np.quantile(effect_norm[keep_idx], float(map_filter_pct) / 100.0)
                    keep_idx = keep_idx[effect_norm > float(thresh)]
                if keep_idx.size == 0 and feats.shape[0] > 0 and has_activity:
                    keep_idx = np.arange(feats.shape[0])
                if not has_activity:
                    keep_idx = np.array([], dtype=np.int64)
                feats = feats[keep_idx]
                map_is_3d = state.get("layer_mode") == "all"
                coords = _pca_3d(feats) if map_is_3d else _pca_2d(feats)
                if coords.shape[0] > 0:
                    coords = coords * float(map_spread)
                    if map_jitter > 0:
                        rng = np.random.default_rng(0)
                        coords = coords + rng.normal(
                            scale=float(map_jitter), size=coords.shape
                        ).astype(np.float32)

                diff = feats
                denom = np.sum(np.abs(diff), axis=1, keepdims=True) + 1e-9
                confs = (np.max(diff, axis=1, keepdims=True) / denom).flatten()
                guesses = [state["labeler"].guess_for_unit(int(i))["guess"] for i in keep_idx]

                edges = _build_similarity_edges(coords, feats, int(map_edges), float(map_min_sim))
                if not edges and int(map_edges) > 0 and feats.shape[0] > 1:
                    edges = _build_similarity_edges(coords, feats, int(map_edges), -1.0)
                cache.update(
                    {
                        "key": cache_key,
                        "coords": coords,
                        "edges": edges,
                        "guesses": guesses,
                        "confs": confs,
                        "keep_idx": keep_idx,
                    }
                )

            coords = cache.get("coords", np.zeros((0, 2), dtype=np.float32))
            edges = cache.get("edges", [])
            guesses = cache.get("guesses", [])
            confs = cache.get("confs", np.zeros((coords.shape[0],), dtype=np.float32))
            keep_idx = cache.get("keep_idx", np.arange(coords.shape[0]))
            prompt_coord = None
            prompt_error = None
            prompt_top_units = None
            if int(state["labeler"].total_samples) == 0:
                st.info("Run a few steps to populate the map.")

            if (
                prompt_supported
                and prompt_map_enabled
                and prompt_text
                and (prompt_live or map_prompt_clicked)
            ):
                try:
                    if state.get("layer_mode") == "all" and hasattr(
                        state["model"], "forward_hidden_text_all_layers"
                    ):
                        h_all = state["model"].forward_hidden_text_all_layers([prompt_text])
                        active = int(state.get("active_layer", 0))
                        if h_all.shape[0] > active:
                            h = h_all[active : active + 1]
                        else:
                            h = h_all[:1]
                    else:
                        h = state["model"].forward_hidden_text([prompt_text])
                    units = _encode_units(h)
                    if units.shape[0] > 0 and units.shape[1] == int(state["labeler"].hidden):
                        weights = np.abs(units[0])[keep_idx]
                        if prompt_top_pct < 100:
                            thresh = np.quantile(weights, 1.0 - float(prompt_top_pct) / 100.0)
                            weights = np.where(weights >= thresh, weights, 0.0)
                        if weights.sum() > 0 and coords.shape[0] == weights.shape[0]:
                            prompt_coord = (coords * weights[:, None]).sum(axis=0) / weights.sum()
                            top_idx = np.argsort(weights)[::-1][:5]
                            prompt_top_units = keep_idx[top_idx]
                except RuntimeError as exc:
                    prompt_error = str(exc)

            map_is_3d = state.get("layer_mode") == "all"
            fig = go.Figure()
            if map_is_3d:
                line_x, line_y, line_z = [], [], []
                for a, b in edges:
                    line_x.extend([coords[a, 0], coords[b, 0], None])
                    line_y.extend([coords[a, 1], coords[b, 1], None])
                    line_z.extend([coords[a, 2], coords[b, 2], None])
                if edges:
                    fig.add_trace(
                        go.Scatter3d(
                            x=line_x,
                            y=line_y,
                            z=line_z,
                            mode="lines",
                            line=dict(color="rgba(120, 160, 200, 0.25)", width=2),
                            hoverinfo="skip",
                        )
                    )
                fig.add_trace(
                    go.Scatter3d(
                        x=coords[:, 0],
                        y=coords[:, 1],
                        z=coords[:, 2],
                        mode="markers+text" if map_show_labels else "markers",
                        text=[f"{i}" for i in keep_idx],
                        textposition="top center",
                        marker=dict(
                            size=4,
                            color=confs,
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="Conf"),
                            line=dict(color="#0b0f14", width=0.5),
                        ),
                        hovertemplate=(
                            f"{unit_label_lower} %{{customdata[0]}}<br>"
                            "guess %{customdata[1]}<br>"
                            "conf %{customdata[2]:.2f}<extra></extra>"
                        ),
                        customdata=np.column_stack(
                            [
                                np.array(keep_idx, dtype=np.int64),
                                np.array(guesses, dtype=object),
                                confs,
                            ]
                        )
                        if len(guesses)
                        else None,
                    )
                )
                if prompt_coord is not None:
                    fig.add_trace(
                        go.Scatter3d(
                            x=[prompt_coord[0]],
                            y=[prompt_coord[1]],
                            z=[prompt_coord[2]],
                            mode="markers+text",
                            text=["prompt"],
                            textposition="bottom center",
                            marker=dict(
                                size=10,
                                color="#ff6b6b",
                                symbol="diamond",
                                line=dict(color="#f8fffb", width=1),
                            ),
                            hovertemplate="Prompt projection<extra></extra>",
                        )
                    )
                fig.update_layout(
                    height=560,
                    margin=dict(l=10, r=10, t=10, b=10),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    scene=dict(
                        xaxis=dict(showgrid=False, zeroline=False, visible=False),
                        yaxis=dict(showgrid=False, zeroline=False, visible=False),
                        zaxis=dict(showgrid=False, zeroline=False, visible=False),
                    ),
                )
            else:
                line_x = []
                line_y = []
                for a, b in edges:
                    line_x.extend([coords[a, 0], coords[b, 0], None])
                    line_y.extend([coords[a, 1], coords[b, 1], None])
                if edges:
                    fig.add_trace(
                        go.Scatter(
                            x=line_x,
                            y=line_y,
                            mode="lines",
                            line=dict(color="rgba(120, 160, 200, 0.25)", width=1),
                            hoverinfo="skip",
                        )
                    )
                fig.add_trace(
                    go.Scatter(
                        x=coords[:, 0],
                        y=coords[:, 1],
                        mode="markers+text" if map_show_labels else "markers",
                        text=[f"{i}" for i in keep_idx],
                        textposition="top center",
                        marker=dict(
                            size=8,
                            color=confs,
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="Conf"),
                            line=dict(color="#0b0f14", width=0.5),
                        ),
                        hovertemplate=(
                            f"{unit_label_lower} %{{customdata[0]}}<br>"
                            "guess %{customdata[1]}<br>"
                            "conf %{customdata[2]:.2f}<extra></extra>"
                        ),
                        customdata=np.column_stack(
                            [
                                np.array(keep_idx, dtype=np.int64),
                                np.array(guesses, dtype=object),
                                confs,
                            ]
                        )
                        if len(guesses)
                        else None,
                    )
                )
                if prompt_coord is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=[prompt_coord[0]],
                            y=[prompt_coord[1]],
                            mode="markers+text",
                            text=["prompt"],
                            textposition="bottom center",
                            marker=dict(
                                size=16,
                                color="#ff6b6b",
                                symbol="diamond",
                                line=dict(color="#f8fffb", width=1),
                            ),
                            hovertemplate="Prompt projection<extra></extra>",
                        )
                    )
                fig.update_layout(
                    height=560,
                    margin=dict(l=10, r=10, t=10, b=10),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, zeroline=False, visible=False),
                    yaxis=dict(showgrid=False, zeroline=False, visible=False),
                    dragmode="pan",
                )
            st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})
            if prompt_error:
                st.warning(f"Prompt mapper error: {prompt_error}")
            elif prompt_coord is not None and prompt_top_units is not None:
                top_units = ", ".join(str(int(i)) for i in prompt_top_units)
                st.caption(f"Top active {unit_label_lower}s for prompt: {top_units}")

with tab_cards:
    if not has_concepts:
        st.info("Concept labels are disabled for dataset prompts; use Feature explorer to decode.")
    else:
        spinner = st.spinner("Updating cards…") if st.session_state.get("autorun_active") else contextlib.nullcontext()
        with spinner:
            st.dataframe(snap, width="stretch", hide_index=True)

with tab_trends:
    if not has_concepts:
        st.info("Trends rely on concept labels; dataset mode focuses on feature decoding.")
    else:
        spinner = st.spinner("Updating trends…") if st.session_state.get("autorun_active") else contextlib.nullcontext()
        with spinner:
            st.subheader("Live prompt feed")
            last_prompts = state.get("last_prompts", [])
            if last_prompts:
                st.caption("Most recent prompts (sample)")
                st.code("\n".join([f"- {p}" for p in last_prompts]), language="markdown")
            else:
                st.write("Run a few steps to see generated prompts.")
            if state.get("layer_mode") == "all":
                rows = []
                for idx, labeler in enumerate(state["labelers"]):
                    snap_l = labeler.snapshot(top_n=min(10, int(top_n)))
                    mean_conf = float(snap_l["conf_single"].mean()) if not snap_l.empty else 0.0
                    rows.append({"layer": idx, "mean_conf": mean_conf})
                layer_df = pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)
                st.subheader("Layer summary")
                st.dataframe(layer_df, width="stretch", hide_index=True)
                st.caption("Layer 0 is the embedding output; last layer is the final hidden state.")
            if state["history"]:
                hist_df = pd.DataFrame(state["history"])
                st.line_chart(hist_df.set_index("step")["mean_conf"])
            else:
                st.write("Run a few steps to populate the trend.")

with tab_inspect:
    if not has_concepts:
        st.info("Unit inspection depends on concept labels; use Feature explorer instead.")
    else:
        spinner = st.spinner("Updating inspection…") if st.session_state.get("autorun_active") else contextlib.nullcontext()
        with spinner:
            unit_idx = st.number_input(
                f"{unit_label} index",
                min_value=0,
                max_value=max(0, state["labeler"].hidden - 1),
                value=0,
                step=1,
                help="Inspect a specific unit.",
            )
            guess = state["labeler"].guess_for_unit(int(unit_idx))
            st.metric("Guess", guess["guess"])
            st.caption(f"Detail: {guess['guess_detail']}")
            st.caption(f"Top effects: {guess['top_concepts']}")
            profile = state["labeler"].unit_profile(int(unit_idx))
            concept_df = pd.DataFrame(
                {
                    "concept": state["stream"].concept_names,
                    "effect": profile["diff"],
                    "mean_on": profile["m1"],
                    "mean_off": profile["m0"],
                }
            )
            st.bar_chart(concept_df.set_index("concept")["effect"])
            st.dataframe(concept_df, width="stretch", hide_index=True)

            st.subheader("Concept coverage")
            coverage = state["labeler"].concept_coverage()
            cov_df = pd.DataFrame({"concept": state["stream"].concept_names, "coverage": coverage})
            st.bar_chart(cov_df.set_index("concept")["coverage"])

with tab_features:
    spinner = st.spinner("Updating feature explorer…") if st.session_state.get("autorun_active") else contextlib.nullcontext()
    with spinner:
        st.subheader("Unit space")
        unit_meta = state.get("unit_meta", {})
        meta_cols = st.columns(3)
        meta_cols[0].metric("Unit source", state.get("unit_source", "Raw neurons"))
        meta_cols[1].metric(
            "Input dim",
            int(unit_meta.get("input_dim", state["labeler"].hidden)),
        )
        meta_cols[2].metric(
            "Unit dim",
            int(unit_meta.get("unit_dim", state["labeler"].hidden)),
        )
        if unit_meta.get("type") == "sae":
            source = unit_meta.get("source", "unknown")
            st.caption(
                f"Sparse Autoencoder source: {source}. Decoder views, feature traces, and token attributions can plug in here."
            )
        if state.get("unit_warnings"):
            st.warning(" ".join(state["unit_warnings"]))

        st.subheader("Activation dictionary")
        if "feature_dict" not in state:
            st.info("Activation dictionary is available for Transformers models.")
        else:
            fd = state["feature_dict"]
            count = fd.get("count", 0)
            df = pd.DataFrame()
            if count == 0:
                st.write("Run a few steps to populate the activation dictionary.")
            else:
                mean = fd["sum"] / max(1, count)
                var = fd["sumsq"] / max(1, count) - mean ** 2
                std = np.sqrt(np.maximum(var, 0.0))
                score_mode = st.selectbox("Rank by", ["Std", "Mean"], index=0)
                if score_mode == "Mean":
                    score = mean
                else:
                    score = std
                top_features = st.slider(
                    f"Top {unit_label_plural_lower}", 10, 200, 50, 10
                )
                idxs = np.argsort(score)[::-1][: int(top_features)]
                rows = []
                for d in idxs.tolist():
                    prompts = fd["top_prompts"][d]
                    vals = fd["top_values"][d]
                    rows.append(
                        {
                            "unit": d,
                            "mean": float(mean[d]),
                            "std": float(std[d]),
                            "top_prompts": " | ".join([p for p in prompts if p]),
                            "top_values": ", ".join([f"{v:.3f}" for v in vals]),
                        }
                    )
                df = pd.DataFrame(rows)
                st.dataframe(df, width="stretch", hide_index=True)

            st.subheader("Export / Import (.neuronmap)")
            cards = snap.to_dict(orient="records") if not snap.empty else []
            meta = {
                "model_choice": model_choice,
                "hf_model_id": hf_model_id,
                "layer_mode": state.get("layer_mode", "single"),
                "active_layer": state.get("active_layer", None),
                "concepts": state["stream"].concept_names,
                "steps": state["step"],
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "unit_source": state.get("unit_source", "Raw neurons"),
            }
            payload = {
                "format": "neuronmap-v1",
                "meta": meta,
                "cards": cards,
                "units": df.to_dict(orient="records"),
            }
            serialized = json.dumps(payload, indent=2)
            st.download_button(
                "Download .neuronmap",
                data=serialized,
                file_name="neuron_map.neuronmap",
                mime="application/json",
            )
            upload = st.file_uploader("Load .neuronmap", type=["neuronmap", "json"])
            if upload is not None:
                try:
                    data = json.loads(upload.getvalue().decode("utf-8"))
                    if data.get("format") != "neuronmap-v1":
                        st.error("Unknown format.")
                    else:
                        units = data.get("units", [])
                        st.write(
                            f"Loaded {len(units)} units from {data.get('meta', {}).get('model_choice', 'unknown')}"
                        )
                        st.dataframe(pd.DataFrame(units), width="stretch", hide_index=True)
                except Exception as exc:
                    st.error(f"Failed to load file: {exc}")

with tab_token:
    st.subheader("Token-level Sparse Autoencoder decoding")
    st.caption("Decode activations per token with a real Sparse Autoencoder. Offline mode builds indices; realtime mode decodes the newest token only.")

    if not TOKEN_SAE_AVAILABLE:
        st.error("Token-level decoding requires transformers + torch. Install requirements-hf.txt.")
    token_cols = st.columns([2, 1])
    with token_cols[0]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Offline mapping")
        token_dataset_path = st.text_input(
            "Dataset path (optional)",
            value=SETTINGS.get("token_dataset_path", ""),
            help="Path to a .txt/.json/.jsonl file on disk.",
            key="token_dataset_path",
        )
        token_dataset_file = st.file_uploader(
            "Or upload dataset",
            type=["txt", "json", "jsonl"],
            key="token_dataset_file",
        )
        token_dataset_field = st.text_input(
            "JSONL field",
            value=SETTINGS.get("token_dataset_field", "text"),
            key="token_dataset_field",
        )
        token_dataset_max = st.number_input(
            "Max prompts to load",
            min_value=100,
            max_value=500000,
            value=int(SETTINGS.get("token_dataset_max", 20000)),
            step=100,
            key="token_dataset_max",
        )
        token_dataset_shuffle = st.toggle(
            "Shuffle dataset",
            value=bool(SETTINGS.get("token_dataset_shuffle", True)),
            key="token_dataset_shuffle",
        )
        st.divider()
        token_batch_size = st.number_input(
            "Batch size",
            min_value=1,
            max_value=32,
            value=int(SETTINGS.get("token_batch_size", 1)),
            step=1,
            key="token_batch_size",
        )
        token_max_len = st.number_input(
            "Max tokens",
            min_value=32,
            max_value=2048,
            value=int(SETTINGS.get("token_max_len", 128)),
            step=16,
            key="token_max_len",
        )
        token_ctx_window = st.slider(
            "Context window (tokens)",
            min_value=2,
            max_value=20,
            value=int(SETTINGS.get("token_ctx_window", 4)),
            step=1,
            key="token_ctx_window",
        )
        token_topk = st.slider(
            "Top-K features per token",
            min_value=4,
            max_value=128,
            value=int(SETTINGS.get("token_topk", 16)),
            step=4,
            key="token_topk",
        )
        token_out_dir = st.text_input(
            "Output directory",
            value=SETTINGS.get("token_out_dir", "token_sae_out"),
            key="token_out_dir",
        )
        build_indices_after = st.toggle(
            "Build indices after decode",
            value=True,
            key="token_build_indices",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with token_cols[1]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Model + Sparse Autoencoder")
        token_model_id = st.text_input(
            "Model id",
            value=SETTINGS.get("token_model_id", hf_model_id or "Qwen/Qwen3-0.6B"),
            key="token_model_id",
        )
        token_device = st.selectbox(
            "Device",
            ["cuda", "cpu"],
            index=0 if SETTINGS.get("token_device", "cuda") == "cuda" and CUDA_AVAILABLE else 1,
            key="token_device",
        )
        token_dtype = st.selectbox(
            "Torch dtype",
            ["float16", "float32", "bfloat16"],
            index=["float16", "float32", "bfloat16"].index(SETTINGS.get("token_dtype", "float16"))
            if SETTINGS.get("token_dtype", "float16") in ("float16", "float32", "bfloat16")
            else 0,
            key="token_dtype",
        )
        token_layer = st.number_input(
            "Layer index",
            min_value=-200,
            max_value=200,
            value=int(SETTINGS.get("token_layer", -1)),
            step=1,
            key="token_layer",
        )
        token_hookpoint = st.selectbox(
            "Hookpoint",
            ["resid_post", "resid_pre"],
            index=0 if SETTINGS.get("token_hookpoint", "resid_post") == "resid_post" else 1,
            key="token_hookpoint",
        )
        token_trust_remote = st.toggle(
            "Trust remote code",
            value=bool(SETTINGS.get("token_trust_remote", False)),
            key="token_trust_remote",
        )
        token_sae_path_input = st.text_input(
            "Sparse Autoencoder path (optional)",
            value=SETTINGS.get("token_sae_path", ""),
            key="token_sae_path",
        )
        token_sae_file = st.file_uploader(
            "Or upload Sparse Autoencoder (.pt/.npz)",
            type=["pt", "pth", "npz"],
            key="token_sae_file",
        )
        token_cache_dir = os.path.join(".cache", "token_sae")
        token_sae_path = token_sae_path_input
        token_sae_hash = None
        if token_sae_file is not None:
            token_sae_bytes = token_sae_file.getvalue()
            token_sae_hash = hashlib.sha256(token_sae_bytes).hexdigest()[:12]
            token_sae_path = _ensure_bytes_path(token_sae_bytes, token_sae_file.name, token_cache_dir)
        st.caption(f"Resolved Sparse Autoencoder: `{token_sae_path or 'n/a'}`")
        if token_sae_hash:
            st.caption(f"Sparse Autoencoder hash: {token_sae_hash}")
        token_clear_cache = st.button("Clear token model cache", key="token_clear_cache")
        if token_clear_cache:
            st.session_state.pop("token_model", None)
            st.session_state.pop("token_tokenizer", None)
            st.session_state.pop("token_model_key", None)
            st.session_state.pop("token_sae", None)
            st.session_state.pop("token_sae_key", None)
            st.success("Token-level model cache cleared.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Realtime decoding")
    token_realtime_prompt = st.text_area(
        "Prompt",
        value=st.session_state.get("token_realtime_prompt", ""),
        height=120,
        key="token_realtime_prompt",
    )
    token_realtime_max = st.number_input(
        "Max new tokens",
        min_value=1,
        max_value=256,
        value=int(SETTINGS.get("token_realtime_max", 32)),
        step=1,
        key="token_realtime_max",
    )
    st.divider()
    run_cols = st.columns([1, 1, 2])
    run_offline = run_cols[0].button("Run offline decode", type="primary", disabled=not TOKEN_SAE_AVAILABLE)
    run_realtime = run_cols[1].button("Run realtime decode", disabled=not TOKEN_SAE_AVAILABLE)
    preview_dataset = run_cols[2].toggle("Preview dataset", value=False)

    if preview_dataset and TOKEN_SAE_AVAILABLE:
        if token_dataset_path:
            preview_prompts, preview_warn = _load_prompt_dataset_from_path(
                token_dataset_path, int(token_dataset_max), token_dataset_field
            )
        else:
            preview_prompts, preview_warn = _load_prompt_dataset(
                token_dataset_file, int(token_dataset_max), token_dataset_field
            )
        if preview_warn:
            st.warning(" ".join(preview_warn))
        if preview_prompts:
            st.caption(f"Previewing {min(3, len(preview_prompts))} prompts (of {len(preview_prompts)})")
            st.code("\n".join(preview_prompts[:3]), language="markdown")

    if (run_offline or run_realtime) and TOKEN_SAE_AVAILABLE:
        if not token_model_id:
            st.error("Provide a model id.")
        elif not token_sae_path:
            st.error("Provide a Sparse Autoencoder path or upload a file.")
        else:
            try:
                model, tokenizer = _get_token_model(
                    token_model_id, token_device, token_dtype, token_trust_remote
                )
                sae = _get_token_sae(token_sae_path, token_device, token_dtype)
                if run_offline:
                    if token_dataset_path:
                        prompts, warnings = _load_prompt_dataset_from_path(
                            token_dataset_path, int(token_dataset_max), token_dataset_field
                        )
                    else:
                        prompts, warnings = _load_prompt_dataset(
                            token_dataset_file, int(token_dataset_max), token_dataset_field
                        )
                    if warnings:
                        st.warning(" ".join(warnings))
                    if not prompts:
                        st.error("Dataset is empty. Add prompts and try again.")
                    else:
                        if token_dataset_shuffle:
                            rng = np.random.default_rng(0)
                            rng.shuffle(prompts)
                        prog = st.progress(0.0, text="Decoding tokens…")
                        total, db_path = _run_token_offline_decode(
                            prompts=prompts,
                            model=model,
                            tokenizer=tokenizer,
                            sae=sae,
                            layer=token_layer,
                            hookpoint=token_hookpoint,
                            topk=token_topk,
                            max_len=token_max_len,
                            batch_size=token_batch_size,
                            out_dir=token_out_dir,
                            device=token_device,
                            dtype=token_dtype,
                            ctx_window=token_ctx_window,
                            progress_cb=lambda p: prog.progress(p, text=f"Decoding {int(p*100)}%"),
                        )
                        prog.empty()
                        st.success(f"Decoded {total} prompts to {db_path}.")
                        if build_indices_after:
                            build_indices(token_out_dir, token_out_dir)
                            st.success("Indices built.")
                if run_realtime:
                    if not token_realtime_prompt:
                        st.error("Add a realtime prompt first.")
                    else:
                        decoded_text, rows = _run_token_realtime_decode(
                            token_realtime_prompt,
                            model=model,
                            tokenizer=tokenizer,
                            sae=sae,
                            layer=token_layer,
                            hookpoint=token_hookpoint,
                            topk=token_topk,
                            max_new_tokens=token_realtime_max,
                            device=token_device,
                            dtype=token_dtype,
                        )
                        st.session_state["token_realtime_rows"] = rows
                        st.session_state["token_realtime_text"] = decoded_text
            except Exception as exc:
                st.error(f"Token Sparse Autoencoder decode error: {exc}")

    if st.session_state.get("token_realtime_text"):
        st.markdown("##### Generated text")
        st.write(st.session_state.get("token_realtime_text"))
    if st.session_state.get("token_realtime_rows"):
        df = pd.DataFrame(st.session_state["token_realtime_rows"])
        df["topk_vals"] = df["topk_vals"].apply(lambda xs: ", ".join([f"{v:.3f}" for v in xs]))
        df["topk_ids"] = df["topk_ids"].apply(lambda xs: ", ".join([str(v) for v in xs]))
        st.dataframe(df, width="stretch", hide_index=True)

with tab_storage:
    st.subheader("Storage & cleanup")
    token_out_dir = st.session_state.get("token_out_dir", SETTINGS.get("token_out_dir", "token_sae_out"))
    db_path = os.path.join(token_out_dir, "decoded.sqlite")
    st.markdown(
        f"<div class='panel'><strong>Token Sparse Autoencoder output</strong><br><span class='mono'>{db_path}</span></div>",
        unsafe_allow_html=True,
    )
    db_exists = os.path.exists(db_path)
    if db_exists:
        st.caption(f"Decoded DB size: {os.path.getsize(db_path) / (1024 * 1024):.2f} MB")
    purge_cols = st.columns([1, 1, 2])
    if purge_cols[0].button("Purge decoded DB"):
        removed = _purge_token_db(token_out_dir)
        if removed:
            st.success(f"Removed {', '.join(removed)}")
        else:
            st.info("Nothing to remove.")
    if purge_cols[1].button("Purge indices"):
        _purge_token_indices(token_out_dir)
        st.success("Index tables removed.")
    if purge_cols[2].button("Rebuild indices", disabled=not TOKEN_SAE_AVAILABLE):
        if not db_exists:
            st.error("No decoded DB found.")
        elif not TOKEN_SAE_AVAILABLE or build_indices is None:
            st.error("Token Sparse Autoencoder tools are unavailable.")
        else:
            build_indices(token_out_dir, token_out_dir)
            st.success("Indices rebuilt.")

    if db_exists:
        st.divider()
        st.markdown("#### Quick lookup")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        feature_id = st.number_input("Feature id", min_value=0, max_value=1000000, value=0, step=1)
        if st.button("Lookup feature contexts"):
            try:
                cur.execute("SELECT contexts FROM feature_contexts WHERE feature_id = ?", (int(feature_id),))
                row = cur.fetchone()
                if row:
                    payload = json.loads(row[0])
                    st.dataframe(pd.DataFrame(payload), width="stretch", hide_index=True)
                else:
                    st.info("No contexts found. Build indices first.")
            except sqlite3.Error:
                st.info("Index tables missing. Build indices first.")
        span_query = st.text_input("Span text contains")
        if st.button("Lookup span features"):
            try:
                cur.execute("SELECT span, features FROM span_features WHERE span LIKE ?", (f"%{span_query}%",))
                rows = cur.fetchmany(5)
                if rows:
                    data = []
                    for span, feats in rows:
                        data.append({"span": span, "features": feats})
                    st.dataframe(pd.DataFrame(data), width="stretch", hide_index=True)
                else:
                    st.info("No span matches.")
            except sqlite3.Error:
                st.info("Index tables missing. Build indices first.")
        conn.close()
