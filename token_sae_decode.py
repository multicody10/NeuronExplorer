from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class SAE:
    W: torch.Tensor
    b: torch.Tensor
    w_layout: str  # "FD", "DF", or "AUTO"

    def _resolve_layout(self, h: torch.Tensor) -> str:
        if self.w_layout in ("FD", "DF"):
            return self.w_layout
        d = h.shape[-1]
        if self.W.shape[1] == d:
            self.w_layout = "FD"
        elif self.W.shape[0] == d:
            self.w_layout = "DF"
        else:
            raise RuntimeError(f"Sparse Autoencoder W_enc shape {tuple(self.W.shape)} does not match hidden dim {d}")
        return self.w_layout

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        layout = self._resolve_layout(h)
        if layout == "FD":
            out = h @ self.W.T
        else:
            out = h @ self.W
        if self.b is not None:
            out = out + self.b
        return torch.relu(out)


class ActivationSensor:
    def __init__(self, model, layer_idx: int, hookpoint: str = "resid_post"):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.hookpoint = hookpoint
        self.last_hidden: Optional[torch.Tensor] = None
        self.handle = None
        self._install()

    def _resolve_layers(self):
        candidates = [
            "model.layers",
            "model.model.layers",
            "transformer.h",
            "gpt_neox.layers",
            "transformer.layers",
            "decoder.layers",
        ]
        for path in candidates:
            cur = self.model
            ok = True
            for part in path.split("."):
                if not hasattr(cur, part):
                    ok = False
                    break
                cur = getattr(cur, part)
            if ok:
                return cur
        raise RuntimeError("Could not find transformer layers on model.")

    def _install(self):
        layers = self._resolve_layers()
        if self.layer_idx < 0:
            self.layer_idx = len(layers) + self.layer_idx
        self.layer_idx = max(0, min(self.layer_idx, len(layers) - 1))
        module = layers[self.layer_idx]
        self.handle = module.register_forward_hook(self._hook)

    def _extract_tensor(self, obj):
        if isinstance(obj, (tuple, list)) and obj:
            return obj[0]
        return obj

    def _hook(self, module, inputs, output):
        if self.hookpoint == "resid_pre":
            h = inputs[0]
        else:
            h = self._extract_tensor(output)
        self.last_hidden = h

    def pop(self) -> torch.Tensor:
        if self.last_hidden is None:
            raise RuntimeError("No activations captured.")
        h = self.last_hidden
        self.last_hidden = None
        return h

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def load_model(model_name_or_path: str, device: str, dtype: str, trust_remote_code: bool = False):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if device == "cpu" and dtype in ("float16", "bfloat16"):
        dtype = "float32"
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype, None)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        trust_remote_code=bool(trust_remote_code),
    )
    added_pad = False
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            added_pad = True
    kwargs = {"dtype": torch_dtype} if torch_dtype is not None else {}
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=bool(trust_remote_code),
        **kwargs,
    )
    model.eval()
    model.to(torch.device(device))
    if added_pad:
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def _load_sae_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path) as npz:
        keys = list(npz.files)
        W = None
        for key in ("W_enc", "encoder", "W", "W_in"):
            if key in npz:
                W = npz[key]
                break
        if W is None:
            for key in keys:
                if npz[key].ndim == 2:
                    W = npz[key]
                    break
        if W is None:
            raise ValueError("Could not find W_enc in Sparse Autoencoder npz.")
        b = None
        for key in ("b_enc", "bias", "b"):
            if key in npz:
                b = npz[key]
                break
        if b is None:
            b = np.zeros((W.shape[1],), dtype=np.float32)
    return W, b


def _load_sae_pt(path: str) -> Tuple[np.ndarray, np.ndarray]:
    obj = torch.load(path, map_location="cpu")
    if hasattr(obj, "state_dict"):
        obj = obj.state_dict()
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise ValueError("Unsupported Sparse Autoencoder pt format; expected a state dict.")
    keys = list(obj.keys())

    def find_key(candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            for key in keys:
                if key == cand or key.endswith(cand):
                    return key
        return None

    w_key = find_key(["W_enc", "encoder.weight", "W_in", "W"])
    b_key = find_key(["b_enc", "encoder.bias", "b_in", "b"])

    if w_key is None:
        for key in keys:
            arr = obj[key]
            if hasattr(arr, "ndim") and getattr(arr, "ndim", 0) == 2:
                w_key = key
                break
    if w_key is None:
        raise ValueError("Could not find 2D W_enc in Sparse Autoencoder pt.")

    W = obj[w_key].detach().cpu().numpy().astype(np.float32)
    if b_key is None:
        b = np.zeros((W.shape[1],), dtype=np.float32)
    else:
        b = obj[b_key].detach().cpu().numpy().astype(np.float32)
    return W, b


def load_sae(path: str, device: str, dtype: str) -> SAE:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if device == "cpu" and dtype in ("float16", "bfloat16"):
        dtype = "float32"
    if path.endswith(".npz"):
        W, b = _load_sae_npz(path)
    else:
        W, b = _load_sae_pt(path)

    W = np.asarray(W, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32) if b is not None else None

    if W.ndim != 2:
        raise ValueError("W_enc must be 2D.")
    w_layout = "AUTO"
    if b is not None and b.ndim == 1:
        if W.shape[0] == b.shape[0]:
            w_layout = "FD"
        elif W.shape[1] == b.shape[0]:
            w_layout = "DF"
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype, torch.float32)
    W_t = torch.tensor(W, device=device, dtype=torch_dtype)
    b_t = torch.tensor(b, device=device, dtype=torch_dtype) if b is not None else None
    return SAE(W=W_t, b=b_t, w_layout=w_layout)


def _iter_dataset(path: str) -> Iterable[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    for key in ("text", "prompt"):
                        if key in obj and isinstance(obj[key], str):
                            yield obj[key]
                            break
                elif isinstance(obj, str):
                    yield obj
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, str):
                    yield item
                elif isinstance(item, dict):
                    for key in ("text", "prompt"):
                        if key in item and isinstance(item[key], str):
                            yield item[key]
                            break
        elif isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            for item in obj["data"]:
                if isinstance(item, str):
                    yield item
                elif isinstance(item, dict):
                    for key in ("text", "prompt"):
                        if key in item and isinstance(item[key], str):
                            yield item[key]
                            break
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def _open_db(out_dir: str) -> sqlite3.Connection:
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


def run_offline(
    dataset_path: str,
    batch_size: int,
    max_len: int,
    layer: int,
    hookpoint: str,
    topK: int,
    out_dir: str,
    model_name_or_path: str = "Qwen/Qwen3-0.6B",
    sae_path: Optional[str] = None,
    device: str = "cuda",
    dtype: str = "float16",
):
    model, tokenizer = load_model(model_name_or_path, device=device, dtype=dtype)
    if sae_path is None:
        raise ValueError("sae_path is required for offline decoding.")
    sae = load_sae(sae_path, device=device, dtype=dtype)
    sensor = ActivationSensor(model, layer_idx=layer, hookpoint=hookpoint)

    conn = _open_db(out_dir)
    insert_sql = (
        "INSERT INTO token_activations "
        "(sample_id, position, token_id, token_str, topk_ids, topk_vals, context) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)"
    )
    buffer: List[Tuple] = []
    sample_id = 0

    def flush():
        nonlocal buffer
        if buffer:
            conn.executemany(insert_sql, buffer)
            conn.commit()
            buffer = []

    with torch.no_grad():
        batch: List[str] = []
        for text in _iter_dataset(dataset_path):
            batch.append(text)
            if len(batch) >= batch_size:
                sample_id = _process_batch(
                    batch,
                    model,
                    tokenizer,
                    sae,
                    sensor,
                    max_len,
                    topK,
                    sample_id,
                    buffer,
                )
                flush()
                batch = []
        if batch:
            sample_id = _process_batch(
                batch,
                model,
                tokenizer,
                sae,
                sensor,
                max_len,
                topK,
                sample_id,
                buffer,
            )
            flush()
    sensor.close()
    conn.close()
    return os.path.join(out_dir, "decoded.sqlite")


def _process_batch(
    batch: List[str],
    model,
    tokenizer,
    sae: SAE,
    sensor: ActivationSensor,
    max_len: int,
    topK: int,
    sample_id: int,
    buffer: List[Tuple],
):
    enc = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    model_device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(model_device)
    attention_mask = enc["attention_mask"].to(model_device)
    _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    h = sensor.pop()  # [B, T, D]
    h = h.to(sae.W.dtype)
    B, T, D = h.shape
    flat = h.reshape(B * T, D)
    scores = sae.encode(flat)  # [B*T, F]
    top_vals, top_ids = torch.topk(scores, k=topK, dim=-1)

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
            ctx_tokens = token_list[max(0, ti - 4) : min(T, ti + 5)]
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
    return sample_id + B


def build_indices(decoded_records: str, out_dir: str, top_contexts: int = 20, top_features: int = 10):
    db_path = decoded_records
    if os.path.isdir(decoded_records):
        db_path = os.path.join(decoded_records, "decoded.sqlite")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS feature_contexts (feature_id INTEGER PRIMARY KEY, contexts TEXT)"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS span_features (span TEXT PRIMARY KEY, features TEXT)"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS feature_cooccurrence (feature_i INTEGER, feature_j INTEGER, count INTEGER, PRIMARY KEY(feature_i, feature_j))"
    )

    feature_top: Dict[int, List[Tuple[float, str]]] = {}
    span_feats: Dict[str, Dict[int, float]] = {}
    co_counts: Dict[Tuple[int, int], int] = {}

    for row in cur.execute("SELECT token_str, topk_ids, topk_vals, context FROM token_activations"):
        _, topk_ids, topk_vals, context = row
        ids = json.loads(topk_ids)
        vals = json.loads(topk_vals)
        ctx_tokens = json.loads(context)
        ctx_text = " ".join(ctx_tokens)
        for fid, val in zip(ids, vals):
            fid = int(fid)
            val = float(val)
            bucket = feature_top.setdefault(fid, [])
            bucket.append((val, ctx_text))
        span_bucket = span_feats.setdefault(ctx_text, {})
        for fid, val in zip(ids, vals):
            fid = int(fid)
            val = float(val)
            span_bucket[fid] = max(span_bucket.get(fid, 0.0), val)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = int(ids[i])
                b = int(ids[j])
                key = (a, b) if a < b else (b, a)
                co_counts[key] = co_counts.get(key, 0) + 1

    cur.execute("DELETE FROM feature_contexts")
    for fid, ctxs in feature_top.items():
        ctxs = sorted(ctxs, key=lambda x: x[0], reverse=True)[:top_contexts]
        payload = [{"value": v, "context": c} for v, c in ctxs]
        cur.execute("INSERT OR REPLACE INTO feature_contexts VALUES (?, ?)", (fid, json.dumps(payload)))

    cur.execute("DELETE FROM span_features")
    for span, feats in span_feats.items():
        top = sorted(feats.items(), key=lambda x: x[1], reverse=True)[:top_features]
        payload = [{"feature": int(fid), "value": float(val)} for fid, val in top]
        cur.execute("INSERT OR REPLACE INTO span_features VALUES (?, ?)", (span, json.dumps(payload)))

    cur.execute("DELETE FROM feature_cooccurrence")
    for (a, b), cnt in co_counts.items():
        cur.execute("INSERT OR REPLACE INTO feature_cooccurrence VALUES (?, ?, ?)", (a, b, int(cnt)))

    conn.commit()
    conn.close()
    return db_path


def run_realtime(
    prompt: str,
    max_new_tokens: int,
    layer: int,
    hookpoint: str,
    topK: int,
    model_name_or_path: str = "Qwen/Qwen3-0.6B",
    sae_path: Optional[str] = None,
    device: str = "cuda",
    dtype: str = "float16",
):
    model, tokenizer = load_model(model_name_or_path, device=device, dtype=dtype)
    if sae_path is None:
        raise ValueError("sae_path is required for realtime decoding.")
    sae = load_sae(sae_path, device=device, dtype=dtype)
    sensor = ActivationSensor(model, layer_idx=layer, hookpoint=hookpoint)

    enc = tokenizer(prompt, return_tensors="pt")
    model_device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(model_device)
    attention_mask = enc["attention_mask"].to(model_device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        past = out.past_key_values
        h = sensor.pop()
        last_h = h[:, -1:, :]
        _print_topk(last_h, sae, tokenizer, input_ids[:, -1], topK)
        cur_ids = input_ids
        for _ in range(max_new_tokens):
            logits = out.logits[:, -1, :]
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            out = model(input_ids=next_id, use_cache=True, past_key_values=past)
            past = out.past_key_values
            h = sensor.pop()
            last_h = h[:, -1:, :]
            _print_topk(last_h, sae, tokenizer, next_id.squeeze(1), topK)
            cur_ids = torch.cat([cur_ids, next_id], dim=1)
    sensor.close()


def _print_topk(h: torch.Tensor, sae: SAE, tokenizer, token_id: torch.Tensor, topK: int):
    h = h.squeeze(1).to(sae.W.dtype)
    scores = sae.encode(h)
    vals, ids = torch.topk(scores, k=topK, dim=-1)
    vals = vals.detach().cpu().numpy()[0].tolist()
    ids = ids.detach().cpu().numpy()[0].tolist()
    tok = tokenizer.convert_ids_to_tokens([int(token_id.item())])[0]
    print(f"token={tok} topk={list(zip(ids, [round(v, 4) for v in vals]))}")


def main():
    parser = argparse.ArgumentParser(description="Token-level Sparse Autoencoder decoding")
    sub = parser.add_subparsers(dest="cmd", required=True)

    offline = sub.add_parser("offline", help="Run offline dataset decoding")
    offline.add_argument("--model", default="Qwen/Qwen3-0.6B")
    offline.add_argument("--sae", required=True)
    offline.add_argument("--dataset", required=True)
    offline.add_argument("--batch-size", type=int, default=4)
    offline.add_argument("--max-len", type=int, default=256)
    offline.add_argument("--layer", type=int, default=-1)
    offline.add_argument("--hookpoint", default="resid_post")
    offline.add_argument("--topk", type=int, default=16)
    offline.add_argument("--out-dir", default="sae_out")
    offline.add_argument("--device", default="cuda")
    offline.add_argument("--dtype", default="float16")

    realtime = sub.add_parser("realtime", help="Run realtime decoding")
    realtime.add_argument("--model", default="Qwen/Qwen3-0.6B")
    realtime.add_argument("--sae", required=True)
    realtime.add_argument("--prompt", required=True)
    realtime.add_argument("--max-new-tokens", type=int, default=32)
    realtime.add_argument("--layer", type=int, default=-1)
    realtime.add_argument("--hookpoint", default="resid_post")
    realtime.add_argument("--topk", type=int, default=16)
    realtime.add_argument("--device", default="cuda")
    realtime.add_argument("--dtype", default="float16")

    build = sub.add_parser("build-indices", help="Build indices from decoded sqlite")
    build.add_argument("--decoded", required=True)
    build.add_argument("--out-dir", default="sae_out")
    build.add_argument("--top-contexts", type=int, default=20)
    build.add_argument("--top-features", type=int, default=10)

    args = parser.parse_args()
    if args.cmd == "offline":
        path = run_offline(
            dataset_path=args.dataset,
            batch_size=args.batch_size,
            max_len=args.max_len,
            layer=args.layer,
            hookpoint=args.hookpoint,
            topK=args.topk,
            out_dir=args.out_dir,
            model_name_or_path=args.model,
            sae_path=args.sae,
            device=args.device,
            dtype=args.dtype,
        )
        print(f"decoded sqlite: {path}")
        build_indices(path, args.out_dir, top_contexts=20, top_features=10)
        print("indices built")
    elif args.cmd == "realtime":
        run_realtime(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            layer=args.layer,
            hookpoint=args.hookpoint,
            topK=args.topk,
            model_name_or_path=args.model,
            sae_path=args.sae,
            device=args.device,
            dtype=args.dtype,
        )
    elif args.cmd == "build-indices":
        build_indices(args.decoded, args.out_dir, args.top_contexts, args.top_features)
        print("indices built")


if __name__ == "__main__":
    main()
