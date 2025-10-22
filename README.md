[![made-with-pytorch](https://img.shields.io/badge/Made%20with-PyTorch-red.svg)](https://pytorch.org/)
[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)]()

# ResNet-50 from scratch on ImageNet-Mini (OneCycleLR + AMP)

This repository trains **ResNet-50 from scratch** on **ImageNet-Mini** using **OneCycleLR** and **mixed precision (AMP)** on a **single GPU (RTX 4060 Ti, 16 GB)**.  
Outputs include TensorBoard logs, structured CSV and Markdown training logs for each epoch, model summary, and an optional classification report

> **Learning Rate Finder (LR-Finder)**  
> We first run LR-Finder to discover a good `max_lr`. LR-Finder explores a rising LR schedule and records loss vs LR, helping pick a stable yet fast starting LR.  
> **Why**: avoids blind grid-search; **What we got**: a “sweet-spot” LR range shown below; **How it’s used**: we feed the chosen `--max-lr` into OneCycleLR for full training.

![LR Finder](lr_finder_plots/lr_finder_20251020_020141_start1e-07_end1.0_iter100.png)

---

## 1. Overview

- **Task**: Image classification on **ImageNet-Mini (~4 GB, 1k classes)**  
- **Backbone**: ResNet-50 (from scratch, no pretrained weights)  
- **Policy**: OneCycleLR (per-batch), label smoothing, SGD momentum, WD  
- **Precision**: AMP (fp16 autocast + GradScaler)  
- **Device**: **RTX 4060 Ti (16 GB VRAM)**  
- **Monitoring**: TensorBoard + CIFAR-style CSV/Markdown logs  
- **Checkpoints**: `checkpoints/r50_onecycle_amp/{checkpoint,best}.pth`
-**TensorBoard** (recommended while training)  
> ```bash
> tensorboard --logdir runs
> ```  
> This shows scalars like `train/loss_step`, `train/lr`, `train/top1`, etc., updated live.

---

## 2. Quickstart

### 2.1 Clone & setup environment
```bash
git clone https://github.com/Sagar063/week9_ERAV4_ImageNet_ResNet-50_Model_Experiments.git
cd week9_ERAV4_ImageNet_ResNet-50_Model_Experiments
python -m venv .venv && source .venv/bin/activate      # Linux/Mac
# or
# .\.venv\Scripts\Activate.ps1                       # Windows PowerShell

pip install -r requirements.txt
```

### 2.2 Train from scratch (ImageNet-Mini)
Download Dataset (Kaggle): https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000  
Place it as:
```
data/imagenet-mini/
  ├─ train/
  └─ val/
```

**Run LR-Finder (recommended)**
```bash
python lr_finder.py find_lr --num_iter 100 --end_lr 1.0 --batch_size 64
```

**Run training (single-GPU)**
```bash
python train.py --name r50_onecycle_amp --epochs 20 --batch-size 64 \
  --max-lr 0.1 --workers 8 --img-size 224 --reports
```

### 2.3 Resume from checkpoint
```bash
python train.py --name r50_onecycle_amp --resume
```

### 2.4 Key arguments
| Arg | Default | Meaning |
|---|---:|---|
| `--data-root` | `data/imagenet-mini` | Root containing `train/` and `val/` (ImageFolder) |
| `--name` | `r50_onecycle_amp` | Run/experiment name used for all output folders |
| `--epochs` | `20` | Number of epochs |
| `--batch-size` | `64` | Global batch size (single-GPU) |
| `--workers` | `8` | DataLoader workers |
| `--img-size` | `224` | Input image size |
| `--max-lr` | `None` | Peak LR (if None, uses linear scaling rule `0.1 * (batch/256)`) |
| `--pct-start` | `0.3` | Fraction of steps for LR warm-up (OneCycleLR) |
| `--div-factor` | `25.0` | Initial LR = `max_lr/div_factor` |
| `--final-div-factor` | `1e4` | Final LR = `max_lr/final_div_factor` |
| `--no-amp` | `False` | Disable AMP if set |
| `--use-class-style-aug` | `False` | Alternate augmentation style |
| `--resume` | `False` | Resume from `checkpoints/<name>/checkpoint.pth` |
| `--reports` | `False` | Generate classification report & save curves |

### 2.5 Repository layout
```
week8_ERAV4_CIFAR_100_ResNetModel_Experiments/
├─ train.py
├─ model.py
├─ lr_finder.py
├─ dataset/
│ └─ imagenet_mini.py
├─ lr_finder_plots/                  # latest LR plot is embedded above
├─ runs/r50_onecycle_amp/            # TensorBoard event files
└─ out/
│   ├─ r50_onecycle_amp/
│      ├─ train_log.csv
│      ├─ logs.md
│      └─ out/
└─ reports/
│   ├─ r50_onecycle_amp/
│       ├─ accuracy_curve.png
│       ├─ classification_report.txt
│       └─ confusion_matrix.csv
│       ├─ loss_curve.png
│       └─ model_summary.txt
└─ images/
│   ├─ {imagenet_samples.png,resnet50_arch.png}  # OPTIONAL:  images
├─ update_readme.py
├─ README.md

```

---

## 3. About ImageNet

**ImageNet** is one of the most influential datasets in computer vision research.  
It contains over **14 million labeled images** organized into more than **22,000 categories**, and has become the foundation for evaluating and benchmarking deep neural networks for image classification and object recognition.

For this iteration, we use **ImageNet-Mini**, a curated **1,000-class subset (~4 GB)** derived from the ImageNet-1K dataset.  
It maintains the same structure and class diversity but is dramatically smaller, making it ideal for **rapid experimentation**, **debugging pipelines**, and **prototyping architectures** locally before scaling to full ImageNet-1K.

**Key Highlights:**
- 📚 **Standard Benchmark:** Widely used for assessing model accuracy and robustness.  
- ⚙️ **Lightweight & Scalable:** Enables faster iteration on consumer GPUs.  
- 🎯 **Diverse Classes:** Includes animals, vehicles, natural scenes, and household objects.  
- 🧩 **Transfer Learning Hub:** Models pretrained on ImageNet form the backbone of countless computer-vision systems.

### Sample Classes and Images

![ImageNet Samples](images/imagenet_images.png)  
*ImageNet-Mini sample classes and images.*

---

## 4. About ResNet-50

**ResNet-50** (He et al., 2015) is a deep convolutional neural network consisting of **50 layers** built on the concept of *residual learning*.  
Residual connections (skip connections) allow gradients to flow more effectively through very deep networks, mitigating the **vanishing-gradient problem** and enabling the successful training of extremely deep CNNs.

**Key Features:**
- 🧩 **Residual Blocks:** Learn identity mappings that help deeper networks converge faster.  
- ⚙️ **Bottleneck Design:** Uses 1×1, 3×3, and 1×1 convolutions to balance accuracy and computation.  
- 🧠 **Depth:** 48 convolutional + 1 max-pool + 1 average-pool + 1 fully-connected layer (≈ 25.6 M parameters).  
- 🎯 **Input:** 224 × 224 × 3 images  **Output:** 1000 classes (ImageNet-1K).  
- 🚀 **Impact:** ResNet architectures revolutionized deep learning and remain a standard backbone for modern vision models.

### Architecture Diagram

![ResNet-50 Architecture](images/ResNet_50_architecture.png)  
*Residual Networks (ResNet-50) architecture.*

---

## 5. Learning Rate Finder (LR-Finder)

Before starting full training, we run a **Learning Rate Finder (LR-Finder)** to determine an optimal `--max-lr` value for the **OneCycleLR** policy.  
This ensures faster and more stable convergence by selecting a learning rate that is high enough to accelerate training but low enough to avoid divergence.

---

### 🔍 Why we use LR-Finder

- 🚀 **Eliminates guesswork:** Automatically finds the ideal learning-rate range.  
- ⚖️ **Improves efficiency:** Prevents wasting epochs on suboptimal LRs.  
- 📈 **Optimizes OneCycleLR:** The discovered LR becomes the peak (`max_lr`) in the OneCycle schedule.  
- 💡 **Enhances reproducibility:** The LR-Finder curve can be regenerated anytime before training.

---

### ⚙️ How it works

The script [`lr_finder.py`](lr_finder.py) performs a **learning-rate range test** using `torch_lr_finder.LRFinder`.  
You can run it as:

```bash
python lr_finder.py find_lr \
  --start_lr 1e-7 \
  --end_lr 1.0 \
  --num_iter 100 \
  --batch_size 64

```
---

### ⚙️ What happens internally
- Initializes a ResNet-50 model (no pretrained weights) and builds the ImageNet-Mini training DataLoader.
- Starts from a learning rate of 1e-7 and increases it exponentially up to 1.0 over 100 iterations.
- Tracks the instantaneous training loss for each mini-batch.
- Plots Loss vs Learning Rate and saves the curve to lr_finder_plots/.
---

### 📈 Observation — My Experiment

Below is the LR-Finder curve obtained from my ImageNet-Mini run (`iter: 100`):

![Learning Rate Finder](lr_finder_plots/2025-10-22_17h04_35.png)

**Interpretation:**
- The x-axis shows the **learning rate**, and the y-axis shows the **training loss**.  
- For very small learning rates (< 1e-5), the loss stays almost flat — learning is too slow.
- Between 1e-4 and 3e-3, the loss starts to drop steadily — the network begins learning efficiently.
- After ≈ 1e-2, the loss shoots up — indicating instability and divergence.

🟢 Optimal LR range: Between 1e-3 and 3e-2.
🟢 Suggested LR: 4.64 × 10⁻³ (steepest descent region), which is automatically reported by the script (lr_finder.suggestion()).
This value is selected as the --max-lr for the OneCycleLR schedule.


---

### 🧩 Using the result
The suggested LR is used as the peak learning rate (--max-lr) for the OneCycleLR scheduler in the main training script.
During training, OneCycleLR starts below this LR, ramps up to it, and then gradually decays — forming a smooth, single-cycle learning-rate curve across all epochs.

Final Training Command:
```bash
python train.py --name r50_onecycle_amp \
  --epochs 20 \
  --batch-size 64 \
  --max-lr 0.0046 \
  --workers 8 \
  --img-size 224 \
  --reports
```
The LR-Finder ensures that the OneCycleLR schedule begins with a well-calibrated peak learning rate — leading to faster convergence, better stability, and improved final accuracy.
---

## 6. Results Summary — `r50_onecycle_amp`

**Curves**
- Loss: `reports/r50_onecycle_amp/loss_curve.png`  
- Accuracy (Top-1 / Top-5): `reports/r50_onecycle_amp/accuracy_curve.png`

![Loss](reports/r50_onecycle_amp/loss_curve.png)
![Accuracy](reports/r50_onecycle_amp/accuracy_curve.png)

**Metrics** (from `out/r50_onecycle_amp/train_log.csv`)
| Split | Top-1 (%) | Top-5 (%) | Loss | Throughput (img/s) |
|---|---:|---:|---:|---:|
| Val (best) | 15.1415 | 32.5006 | 5.456362 | 827.05 |
| Val (final) | 14.8101 | 33.0614 | 5.446987 | 947.88 |

> Exact values are auto-filled by `update_readme.py`.

---

## 6. Detailed Results & Observations

### 6.1 Training logs (markdown)
<details><summary>Show logs.md</summary>

```text
# Training Logs (terminal-like)
… (inserted from out/r50_onecycle_amp/logs.md) …
```
</details>

### 6.2 Model summary
<details><summary>Show model_summary.txt</summary>

```text
… (inserted from reports/r50_onecycle_amp/model_summary.txt) …
```
</details>

### 6.3 Classification report (Top-1)
We include **overall accuracy** and the **macro/weighted averages** from:
`reports/r50_onecycle_amp/classification_report.txt`.

```
accuracy …  
macro avg …  
weighted avg …
```

> (Confusion matrix intentionally omitted.)

---

## 7. Notes
- OneCycleLR is stepped **per batch**, producing a single continuous rise-and-fall LR curve across all epochs.
- AMP keeps memory and compute efficient on the 4060 Ti while maintaining model quality.

## 8. Conclusion — Training on Local Machine
- The 4060 Ti can comfortably handle ImageNet-Mini with batch sizes around 64 (AMP on), giving quick iteration on augmentations and scheduler settings.
- Use TensorBoard (`tensorboard --logdir runs`) for live LR and loss monitoring.

## 9. Next Steps — EC2
- Scale to ImageNet-1k on EC2 with multi-GPU (DDP), reusing the same `--data-root`/`--name` structure.
- Re-run LR-Finder on the new hardware/batch size, then train with OneCycleLR.
