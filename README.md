# Neuron Explorer

Streamlit GUI for exploring neuron activations with a streaming concept labeler, active probing, and a realtime map view inspired by the real‑time neuron labeling demo.

![UI](https://i.imgur.com/mRHtMP8.png)

## Features
- Live neuron labeling with confidence tracking, trends, and auto‑explore
- Active probing to focus on under‑sampled concepts
- Unit inspection + activation dictionary (top prompts per unit)
- Interactive map with prompt projection; 3D map in **All layers** mode
- Token‑level Sparse Autoencoder decoding (offline + realtime)
- Storage tools: rebuild indices, purge decoded databases
- Works with a built‑in toy model, TorchScript models, and Transformers safetensors models

## Requirements
- Windows 10
- Python 3.10+

## Quick start (Windows 10)
Use the bundled runner (installs requirements and launches the app):
```powershell
.\run.bat
```

Manual setup:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-hf.txt
streamlit run app.py
```

Optional TorchScript support only:
```powershell
pip install -r requirements-torch.txt
```

## Configuration
All settings live in the Streamlit sidebar and are saved to `settings.json` on exit.
To publish, keep `settings.json` out of Git and copy defaults from `settings.example.json`.

- Model selection, device, dtype, and attention implementation
- Concept list or generated concept dictionary
- Active probing and auto-explore controls
- Map settings and prompt mapper options
- Token‑level Sparse Autoencoder decoding options and storage paths

## Run and development
```powershell
streamlit run app.py
```

## Models
Transformers: enter a Hugging Face model id or a local folder path containing `.safetensors`, tokenizer, and config files.
Local model folders are ignored in Git via `.gitignore`.

## Token‑level Sparse Autoencoder decoding
Use the **Token Sparse Autoencoder** tab to:
- Run offline dataset passes (batch decode, top‑K per token, sqlite output)
- Run realtime decoding during generation (newest token only)
- Build indices for feature → contexts and span → features

Dataset formats:
- `.jsonl` with `{ "text": "..." }` or `{ "prompt": "..." }`
- `.txt` with one prompt per line

Decoded data lives under `token_out_dir/decoded.sqlite` and can be purged from the **Storage** tab.

## Troubleshooting
Start by reading the Streamlit console output.

- `HFValidationError: Repo id must use alphanumeric...`  
  Hidden characters or an invalid local path. Re-type the model path and delete `settings.json` if it persists.
- `CUDA error: an illegal memory access was encountered`  
  Switch Device to CPU, set Attention to eager, lower batch size / max tokens, or disable All layers.
- `expected mat1 and mat2 to have the same dtype`  
  Set Torch dtype to `float16` or `float32` and keep it consistent with the device.
- `TypeError: set_autocast_dtype(): ... must be torch.dtype, not str`  
  Ensure Torch dtype is set to `auto`, `float16`, `bfloat16`, or `float32` in the UI.
- `SVD did not converge` (map)  
  Reduce map jitter/spread or run a few more steps to stabilize activations.
- Map/graphs empty  
  Click **Run** to generate data; the map updates after steps are collected.

## Research
  <details>
<summary><strong>TokenSensor (Experimental)</strong></summary>

TokenSensor is an experimental interpretability mode that treats a language model like a running system and exposes pertoken telemetry.

Every token that flows through the model produces hidden states at each layer. TokenSensor attaches a small recorder at a chosen hook point (usually a residual stream) and translates each token hidden state into a sparse, humaninspectable feature readout using a pretrained Sparse Autoencoder (SAE). This reframes mapping as building an index over token events, more like profiling and debugging than afterthefact inspection.

## What it records

For each token position, TokenSensor stores a compact record:

- token id and decoded token string
- position in the sequence
- hook point (layer and stream)
- top K SAE features and activation values

From this stream you can build:

- feature to top activating contexts
- prompt span to dominant features
- feature cooccurrence graphs
- feature exemplars for labeling and verification

## How it works

At a chosen hook point, for each token hidden vector `h` with size `D`:

1. Run the model to obtain `h`
2. Decode SAE activations `a = f(W_enc · h + b)`
3. Keep only top K activations for that token
4. Store `(token, pos, topK feature ids, topK values)`

The point is sparsity. You do not store everything, you store the most informative slice.

## Operating modes

### Offline mapping mode (best signal, fastest overall)
- Run a dataset of prompts in batches
- Collect hidden states for all tokens
- Decode SAE with a single large batched matmul
- Build indices for browsing and labeling

### Realtime sensing mode (best interactivity)
- During generation, hook only the newest token position
- Decode SAE for that one token
- Display top K features live with a rolling context window

## Speed notes

TokenSensor cannot be free. You must run tokens to get hidden states, and SAE decode adds compute. Overhead stays bounded by design:

- decode only one or a few layers
- batch decode whenever possible
- keep only top K per token
- prefer GPU half precision for SAE decode when available
- in realtime decode only the newest token

## Labeling and truth

Feature names are hypotheses, not truth. Treat labels as metadata backed by evidence:

- label from many top contexts, not a single example
- look for clusters inside a feature’s contexts
- collect counterexamples where the label should apply but does not
- use causal tests (feature ablation or steering) when it matters

## Coaxing features for exploration

TokenSensor supports targeted exploration, with a strict evidence mindset:

- **Probe search**: find where a feature fires in real datasets and read contexts (evidence)
- **Feature targeted generation**: bias sampling toward continuations that increase a chosen feature (microscope, not proof)

## Data contract

### Datasets
- `.jsonl` with `{ "text": "..." }` or `{ "prompt": "..." }`
- `.txt` with one prompt per line

### SAEs
- pretrained SAE weights compatible with the chosen hook point
- config describing `D`, number of features, nonlinearity, and target layer

### Outputs
- per token top K feature records
- feature and span indices for fast browsing and search

</details>
