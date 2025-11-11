# � Deploying ResNet‑50 (ImageNet‑1k) to Hugging Face Spaces — Cookbook Guide

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
- A Hugging Face account (https://huggingface.co/join)
- Installed CLI:
  ```bash
  pip install huggingface_hub
  huggingface-cli login
  ```

---

## 3️⃣ Step‑by‑Step Deployment Recipe

### � Step 1 – Convert your training checkpoint to CPU weights

We’ll remove optimizer/scheduler/AMP states and save a clean fp32 state_dict.

```python
# scripts/convert_to_cpu.py
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
    print("✅ Saved:", OUT_FP32, OUT_META)

if __name__ == "__main__":
    main()
```

You’ll now have:
```
model_cpu_fp32.pth
meta.json
```

---

### � Step 2 – Verify locally

```bash
python -m venv .venv && . .venv/Scripts/activate    # or: source .venv/bin/activate (Linux/Mac)
pip install torch torchvision pillow requests numpy gradio
python app.py
```
Open http://127.0.0.1:7860 — test image upload + URL prediction.

If predictions look wrong for every image → check console:  
it should print `[info] using meta normalization: mean=[...], std=[...]`.

---

### �️ Step 3 – Upload weights to Hugging Face Model Hub (Sagar’s repos)

You can push to either/both of these public model repos:

- **Local‑CPU model repo:** `Sunny063/r50-imagenet1k-cpu-weights_ERAV4_Assignment_AWS`  
- **AWS‑trained model repo:** `Sunny063/my-new-resnet50-aws`

Create or clone and push:
```bash
# Example for AWS-trained model repo
git lfs install
git clone https://huggingface.co/Sunny063/my-new-resnet50-aws
cd my-new-resnet50-aws
cp ../model_cpu_fp32.pth ../meta.json .
git add . && git commit -m "Add CPU model + meta" && git push
```

---

### ⚙️ Step 4 – Build your Space

Create a new **Space** at https://huggingface.co/new-space  
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
HF_MODEL_REPO = "Sunny063/my-new-resnet50-aws"     # or: Sunny063/r50-imagenet1k-cpu-weights_ERAV4_Assignment_AWS
HF_MODEL_FILE = "model_cpu_fp32.pth"
```
Save → Commit → **Restart Space**. You should see `Running` after the image is built.

---

## 4️⃣ Test Your Space

Once it runs, open:
```
# Local-CPU model Space
https://huggingface.co/spaces/Sunny063/ResNet50-Imagenet-CPU-Demo-ERAV4_CPU_Model

# AWS-trained model Space
https://huggingface.co/spaces/Sunny063/ResNet50-Imagenet-AWS-Demo-ERAV4
```
Upload an image or paste an image URL → you’ll see Top‑K predictions with confidence.

> � Hint: You can check build logs via *Settings → Logs* if it’s stuck on “Building” or “Error starting server”.

---

## 5️⃣ How Others Can Access Your Work

### Clone the Space (code + UI)
```bash
git lfs install
git clone https://huggingface.co/spaces/Sunny063/ResNet50-Imagenet-AWS-Demo-ERAV4
git clone https://huggingface.co/spaces/Sunny063/ResNet50-Imagenet-CPU-Demo-ERAV4_CPU_Model
```

### Download model snapshot (weights + meta)
```bash
pip install huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download
# CPU model:
snapshot_download(repo_id="Sunny063/r50-imagenet1k-cpu-weights_ERAV4_Assignment_AWS", local_dir="./cpu_model_download")
# AWS model:
snapshot_download(repo_id="Sunny063/my-new-resnet50-aws", local_dir="./aws_model_download")
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
| � Package conflicts (`gradio_client`) | Pin Gradio e.g., `gradio==4.44.1` or `5.49.1`; don’t pin `gradio_client` yourself. |
| ⚠️ Everything predicts same class | Missing `meta.json` → wrong normalization. Ensure console prints mean/std on startup. |
| ❌ Checkpoint fails to load | Head mismatch → check `fc.weight` shape is `[1000, 2048]`. |
| ⏳ Space stuck on Building | Restart Space / Clear cache / Rebuild factory (Settings → Factory reboot). |
| � Internal Server Error | Keep Gradio simple; avoid fancy components that introduce schema bugs in some versions. |
| � **AttributeError: Can't get attribute 'FrozenLRSched' ...** | Your checkpoint references a training‑time class. **Fix:** use the converter below that stubs the class *or* uses `torch.serialization.add_safe_globals` / `weights_only=True` fallback (see next section). |

### � Quick Fix — AWS checkpoint with `FrozenLRSched`

Use this safer converter when your checkpoint contains custom classes:

```python
# scripts/convert_to_cpu_for_Aws_model.py
# convert_to_cpu.py
import json, torch
from torchvision.models import resnet50

# <<< EDIT THIS to your source checkpoint >>>
CKPT_PATH = r"D:\ERA\week9\ImageNet-ResNet50-CNN_HuggingSpace_ERAV4\imagenet1kfull_g5x_1gpu_dali_nvme_lr0p125_bs256_e150_work6\last_epoch194.pth"
# CKPT_PATH = r"D:\ERA\week9\ImageNet-ResNet50-CNN_HuggingSpace_ERAV4\imagenet1kfull_g5x_1gpu_dali_nvme_lr0p125_bs256_e150_work6\checkpoint.pth"

OUT_FP32 = "aws_model_cpu_fp32.pth"
OUT_META = "aws_meta.json"

# ---- STUB any custom classes that might appear inside the checkpoint ----
# (If you see new "Can't get attribute 'XYZ'" errors, add another stub: class XYZ: pass)
class FrozenLRSched:
    pass

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

def load_checkpoint_anyhow(path):
    """
    1) Try weights_only=True (safe; avoids unpickling scheduler objects).
    2) If that fails, allow our stubs and fall back to regular load.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"[info] weights_only load failed: {type(e).__name__}: {e}")
        print("[info] Falling back to regular torch.load with stubbed classes...")
        # allow our stubs explicitly
        try:
            from torch.serialization import add_safe_globals
            add_safe_globals([FrozenLRSched])
        except Exception:
            # older torch may not have add_safe_globals; regular load will still work
            pass
        return torch.load(path, map_location="cpu")  # do NOT use on untrusted files

def main():
    obj = load_checkpoint_anyhow(CKPT_PATH)

    if not isinstance(obj, dict) or "model" not in obj:
        # Help debug unexpected layouts
        top = list(obj.keys())[:20] if isinstance(obj, dict) else type(obj)
        raise ValueError(
            f"Expected a dict checkpoint with key 'model'. Got: {top}"
        )

    # 1) Extract weights and clean keys
    sd = obj["model"]
    if not isinstance(sd, dict):
        raise ValueError("obj['model'] is not a dict-like state_dict.")
    sd = strip_prefixes(sd)

    # ensure FP32 on CPU
    for k, v in list(sd.items()):
        if torch.is_tensor(v):
            sd[k] = v.float().cpu()

    # 2) Build vanilla torchvision ResNet-50 (1000 classes) and load
    m = resnet50(weights=None)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print("missing:", missing)
    print("unexpected:", unexpected)

    # 3) Save clean fp32 weights
    torch.save(m.state_dict(), OUT_FP32)
    print("✅ Saved:", OUT_FP32)

    # 4) Save meta with normalization (fallback to IMNet defaults if absent)
    mean = obj.get("mean", [0.485, 0.456, 0.406])
    std  = obj.get("std",  [0.229, 0.224, 0.225])
    meta = {"mean": [float(x) for x in mean], "std": [float(x) for x in std], "image_size": 224}
    with open(OUT_META, "w") as f:
        json.dump(meta, f)
    print("✅ Saved:", OUT_META, meta)

if __name__ == "__main__":
    main()

```

> Keep converter scripts under `scripts/` so they don’t affect Space builds.

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
scripts/ (converter scripts — optional but okay to publish)
```
� Do NOT commit:
```
.env
.token*
.netrc
*.pt
*.pth
*.ckpt   # (except the small CPU fp32 model you intend to publish)
__pycache__/
```

If you ever need secrets, store them under **Space → Settings → Repository Secrets** and access via `os.environ`.

---

## 8️⃣ Live Demos (Sagar)

| Model Type | Description | Space Link |
|-------------|--------------|------------|
| �️ Local CPU | Trained on RTX 4060 Ti (OneCycleLR + AMP) | https://huggingface.co/spaces/Sunny063/ResNet50-Imagenet-CPU-Demo-ERAV4_CPU_Model |
| ☁️ AWS Model | Trained on A10G (g5.xlarge) with DALI pipeline | https://huggingface.co/spaces/Sunny063/ResNet50-Imagenet-AWS-Demo-ERAV4 |

---

## 9️⃣ Key Takeaways

- Converting to a **clean CPU fp32** checkpoint is critical for portable deployment.  
- `meta.json` ensures identical preprocessing (mean/std/image size).  
- Pin versions to avoid resolver conflicts.  
- You can replicate this pipeline for any PyTorch model — just swap `resnet50()` with your architecture.

---

> � **Note:** This README is the complete deployment cookbook.  

