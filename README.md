# Neuron Explorer

Streamlit GUI for exploring neuron activations with a streaming concept labeler, active probing, and a realtime map view inspired by the real‑time neuron labeling demo.

![Neuron Explorer UI](docs/ui.png)

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

## Screenshot
Place the attached UI image at `docs/ui.png` so the README preview renders it.
