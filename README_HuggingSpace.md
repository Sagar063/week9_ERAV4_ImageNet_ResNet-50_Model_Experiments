# 🧠 Deploying ResNet‑50 (ImageNet‑1k) to Hugging Face Spaces — Cookbook Guide

This document is a **complete, beginner‑friendly cookbook** to take you from a *training checkpoint* to a working **Hugging Face Space** that runs entirely on **CPU** using **Gradio**.  
It’s written from real‑world deployment experience with both **local RTX 4060 Ti** and **AWS A10G (g5.xlarge)** ImageNet trainings.

---

## 1️⃣ Overview

We trained two ResNet‑50 ImageNet‑1k models (local + AWS) and then deployed them for public interactive inference.  
This guide explains the **end‑to‑end process** so anyone can reproduce the same deployment for any PyTorch model.

---

## 2️⃣ Prerequisites

Before you start:
- You already have a **saved training checkpoint** — for example:
  ```bash
  checkpoints/r50_imagenet1k_onecycle_amp_bs64_ep150/best.pth
  checkpoints/imagenet1kfull_g5x_1gpu_dali_nvme_lr0p125_bs256_e150_work6/best_acc_epoch193.pth
  ```
- The checkpoint dict should contain your model state under `"model"` key (typical ERA / DDP style).  
- You know the training mean / std values (these will be stored in `meta.json`).

Optional but recommended:
- A Hugging Face account (`https://huggingface.co/join`)
- Installed CLI:
  ```bash
  pip install huggingface_hub
  huggingface-cli login
  ```

---

## 3️⃣ Step‑by‑Step Deployment Recipe

### 🧩 Step 1 – Convert your training checkpoint to CPU weights

We’ll remove optimizer/scheduler/AMP states and save a clean fp32 state_dict.

```python
# convert_to_cpu.py
import json, torch
from torchvision.models import resnet50

CKPT_PATH = r"/path/to/your/best_or_last_checkpoint.pth"
OUT_FP32  = "model_cpu_fp32.pth"
OUT_META  = "meta.json"

def strip_prefixes(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            out[k[7:]] = v
        elif k.startswith("model."):
            out[k[6:]] = v
        else:
            out[k] = v
    return out

def main():
    obj = torch.load(CKPT_PATH, map_location="cpu")
    sd  = strip_prefixes(obj["model"])
    for k, v in list(sd.items()):
        if torch.is_tensor(v):
            sd[k] = v.float().cpu()

    m = resnet50(weights=None)
    m.load_state_dict(sd, strict=False)

    torch.save(m.state_dict(), OUT_FP32)
    mean = obj.get("mean", [0.485, 0.456, 0.406])
    std  = obj.get("std",  [0.229, 0.224, 0.225])
    json.dump({"mean": [float(x) for x in mean],
               "std":  [float(x) for x in std],
               "image_size": 224}, open(OUT_META, "w"))
    print("✅ Saved:", OUT_FP32, OUT_META)

if __name__ == "__main__":
    main()
```

You’ll now have:
```
model_cpu_fp32.pth
meta.json
```

---

### 🧮 Step 2 – Verify locally

```bash
python -m venv .venv && . .venv/Scripts/activate
pip install torch torchvision pillow requests numpy gradio
python app.py
```
Open http://127.0.0.1:7860 — test image upload + URL prediction.

If predictions look wrong for every image → check console:  
it should print `[info] using meta normalization: mean=[...], std=[...]`.

---

### 🗂️ Step 3 – Upload weights to Hugging Face Model Hub

Create a new model repo:
```bash
huggingface-cli repo create my-resnet50-cpu --type model
```
Push your two files:
```bash
git clone https://huggingface.co/<user>/my-resnet50-cpu
cd my-resnet50-cpu
cp ../model_cpu_fp32.pth ../meta.json .
git add . && git commit -m "Add CPU model + meta" && git push
```

---

### ⚙️ Step 4 – Build your Space

Create a new **Space** at <https://huggingface.co/new-space>  
Set **SDK = Gradio**, and **Hardware = CPU Basic**.

Then upload these files:

| File | Purpose |
|------|----------|
| `app.py` | Gradio UI definition |
| `inference.py` | Model loader + predictor |
| `requirements.txt` | Pinned packages |
| `runtime.txt` | Python version (e.g., `python‑3.10`) |
| `utils/imagenet_class_index.json` | Human labels (optional) |
| `README.md` | Space front‑matter metadata |

**Important fields inside `inference.py`:**
```python
HF_MODEL_REPO = "<user>/my-resnet50-cpu"
HF_MODEL_FILE = "model_cpu_fp32.pth"
```
Save → Commit → “Restart Space”.  
Within ~2 minutes your app builds and shows `Running`.

---

## 4️⃣ Test Your Space

Once it runs, open:
```
https://huggingface.co/spaces/<user>/<space-name>
```
Upload an image or paste an image URL → you’ll see Top‑K predictions with confidence.

> 💡 Hint: You can check build logs via *Settings → Logs* if it’s stuck on “Building” or “Error starting server”.

---

## 5️⃣ How Others Can Access Your Work

### Clone the Space (code + UI)
```bash
git lfs install
git clone https://huggingface.co/spaces/<user>/<space-name>
```

### Download model snapshot (weights + meta)
```bash
pip install huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="<user>/my-resnet50-cpu", local_dir="./model_download")
PY
```

### Or download a single file (raw URL)
In the Hub UI → **Files** → click on `model_cpu_fp32.pth` → **Raw** → copy URL.  
Then:
```bash
curl -L -o model_cpu_fp32.pth "<raw-url>"
```

---

## 6️⃣ Common Issues & Fixes

| Problem | Cause / Fix |
|----------|-------------|
| 🔁 Package conflicts (`gradio_client`) | Pin Gradio e.g., `gradio==4.44.1`; remove manual `gradio_client` pin. |
| ⚠️ Everything predicts same class | Missing meta.json → wrong normalization. Ensure you see console log for mean/std. |
| ❌ Checkpoint fails to load | Head mismatch → check that `fc.weight` shape = [1000, 2048]. |
| ⏳ Space stuck on Building | Restart Space / Clear cache / Rebuild Factory (under Settings). |
| 💥 Internal Server Error | Keep Gradio ≤ 4.44 and use simple components (no BarPlot schema bugs). |

---

## 7️⃣ Security Tips

✅ Safe to publish:
```
app.py
inference.py
requirements.txt
runtime.txt
utils/*
meta.json
README.md
```
🚫 Do NOT commit:
```
.env
.token*
.netrc
*.pt
*.pth
*.ckpt
__pycache__/
```

If you ever need secrets, store them under **Space → Settings → Repository Secrets** and access via `os.environ`.

---

## 8️⃣ Live Demo Placeholders

| Model Type | Description | Space Link |
|-------------|--------------|------------|
| 🖥️ Local CPU | Trained on RTX 4060 Ti (OneCycleLR + AMP) | [🔗 Live Demo](https://huggingface.co/spaces/<user>/<local-space>) |
| ☁️ AWS Model | Trained on A10G (g5.xlarge) with DALI pipeline | [🔗 Live Demo](https://huggingface.co/spaces/<user>/<aws-space>) |

---

## 9️⃣ Key Takeaways

- Converting to a **clean CPU fp32** checkpoint is critical for portable deployment.  
- `meta.json` ensures identical preprocessing (mean/std/image size).  
- Pin versions to avoid resolver conflicts.  
- You can replicate this pipeline for any PyTorch model — just swap `resnet50()` with your own architecture.

---

> 🧾 **Note:** This README is the complete deployment cookbook.  
> Pair it with your main project README section “Model Deployment and Inferencing” for a perfect submission.
