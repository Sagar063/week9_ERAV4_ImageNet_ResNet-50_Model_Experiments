# ResNet-50 training from scratch on ImageNet1k (OneCycleLR + AMP) using AWS

This is a PyTorch implementation of training a ResNet-50 model **from scratch** on the **ImageNet-1k** dataset on an AWS EC2 instance, as part of *The School of AI – ERA-V4* course.

This repository trains ResNet-50 from scratch on **ImageNet-Mini** (for pipeline validation) and on **full ImageNet-1k**, using **OneCycleLR** and mixed precision (**AMP**):
- **Local**: single GPU (RTX 4060 Ti, 16 GB)
- **AWS**: spot instance **g5.xlarge** (NVIDIA A10G 24 GB)

Outputs include **TensorBoard** logs (local), **Weights & Biases** (AWS), plus structured **CSV** and **Markdown** training logs, model summary, and classification report.

---

## 1) Overview

**Task**  
- Dataset: **ImageNet-1k** (~1.28M train, 50k val, **1000** classes).  
- Train/val folder layout is standard `train/` and `val/` class-folders.

**Backbone**  
- **ResNet-50** from scratch (no pretrained weights).

**Optimization / Policy**  
- Optimizer: **SGD** (momentum, weight decay)  
- Schedule: **OneCycleLR** (per-batch) with `--max-lr` determined via LR-Finder  
- **Label smoothing**  
- Precision: **AMP** (fp16 autocast + GradScaler)

**Devices**  
- **Local**: RTX 4060 Ti (16 GB VRAM)  
- **AWS**: g5.xlarge (4 vCPU, 16 GiB RAM) with **NVIDIA A10G 24 GB**

**Monitoring**  
- **Local**: TensorBoard + CSV/Markdown logs  
- **AWS**: W&B (`imagenet1k_runs`) + CSV/Markdown logs

**Checkpoints (paths you can reuse)**
- **Local**  
  - `checkpoints\r50_imagenet1k_onecycle_amp_bs64_ep150\best.pth` → best validation (deploy/HF)  
  - `checkpoints\r50_imagenet1k_onecycle_amp_bs64_ep150\checkpoint.pth` → resume  
- **AWS**  
  - `/mnt/imagenet1k/checkpoints/imagenet1kfull_g5x_1gpu_dali_nvme_lr0p125_bs256_e150_work6/last_epoch.pth` → resume  
  - `/mnt/imagenet1k/checkpoints/imagenet1kfull_g5x_1gpu_dali_nvme_lr0p125_bs256_e150_work6/best_acc_epoch141.pth` → best validation (deploy/HF)

**TensorBoard (local)**  
```bash
tensorboard --logdir runs
```
Shows `train/loss`, `train/lr`, `train/top1`, etc., live.

**W&B (AWS)**  
- Project: `imagenet1k_runs`  
- Run name: `imagenet1kfull_g5x_1gpu_dali_nvme_lr0p125_bs256_e150_work6` (example)

> **Auto-filled results (from `out/<RUN>/train_log.csv`):**  
> Best Val Top-1: **77.82%** • Best Val Top-5: **93.82%** • Best Epoch: **228**

---

## 2) Quickstart

### 2.1 Clone & setup environment

```bash
git clone https://github.com/Sagar063/week9_ERAV4_ImageNet_ResNet-50_Model_Experiments.git
cd week9_ERAV4_ImageNet_ResNet-50_Model_Experiments
```

**Local (Windows)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .
equirements.txt
```

**AWS**
1) **Mount your ImageNet EBS volume at `/mnt/imagenet1k`**
```bash
# See disks
lsblk -f

# Create mountpoint (idempotent)
sudo mkdir -p /mnt/imagenet1k

# Mount your 400-GB EBS volume (replace if your data disk isn't /dev/nvme1n1)
sudo mount -o defaults,noatime /dev/nvme1n1 /mnt/imagenet1k

# Give yourself ownership
sudo chown -R ubuntu:ubuntu /mnt/imagenet1k

# Verify
df -h /mnt/imagenet1k && ls -lah /mnt/imagenet1k
```

2) **(Ephemeral) NVMe workspace + virtual environment**
```bash
# The DLAMI ephemeral NVMe is typically at /opt/dlami/nvme
df -h /opt/dlami/nvme || sudo mkdir -p /opt/dlami/nvme

# Make a fresh venv (idempotent)
sudo mkdir -p /opt/dlami/nvme/envs
sudo chown -R ubuntu:ubuntu /opt/dlami/nvme
python3 -m venv /opt/dlami/nvme/envs/imagenet1k_venv

# Activate and install deps from your repo
source /opt/dlami/nvme/envs/imagenet1k_venv/bin/activate
cd ~/week9_ERAV4_ImageNet_ResNet-50_Model_Experiments
pip install --upgrade pip
pip install -r requirements_aws.txt
```

### 2.2 Train from scratch

#### ImageNet-1k :  download for training in local machine
  - Size: **156 GB**
  - Source 1: https://www.kaggle.com/c/imagenet-object-localization-challenge/
    - Download via Kaggle CLI:
    ```bash
    kaggle competitions download -c imagenet-object-localization-challenge
    ```
**For dataset download in AWS and AWS setup, see the [ImageNet Dataset Setup on AWS instructions](./README_AWS.md).**

#### Once the dataset is downloaded place as:
```
data/imagenet/
  ├─ train/
  └─ val/
  └─ test/
```

**Run LR-Finder (recommended)**
```bash
python lr_finder.py find_lr --num_iter 100 --end_lr 1.0 --batch_size 64
```

**Run training (Local)**
```bash
python train_full_ImageNet1k_SingleGPU.py --data-root data/imagenet  --batch-size 64 --epochs 150 --max-lr 0.0125  --pct-start 0.1 --workers 8 --reports --use-best-for-reports --name r50_imagenet1k_onecycle_amp_bs64_ep150
```

**Run training (AWS)**
```bash
tmux new -s imagenet1k_full -n train

# Inside tmux
source /opt/dlami/nvme/envs/imagenet1k_venv/bin/activate
cd ~/week9_ERAV4_ImageNet_ResNet-50_Model_Experiments

# (Optional) start W&B session automatically when resuming
unset RESET_SCHED
unset FREEZE_LR
unset FREEZE_LR_VALUE

bash scripts/launch_single_gpu.sh /mnt/imagenet1k 150 256 6 \
  --max-lr 0.125 \
  --stats-file data_stats/imagenet_1k_aws_stats.json \
  --show-progress --amp --channels-last \
  --out-dir imagenet1kfull_g5x_1gpu_dali_nvme_lr0p125_bs256_e150_work6 \
  --wandb --wandb-project imagenet1k_runs \
  --wandb-tags imagenet1k_full,dali,1gpu,nvme,lr0p125,bs256,e150,work6
```

### 2.3 Resume from checkpoint

**Local**
```bash
python train_full_ImageNet1k_SingleGPU.py   --data-root data/imagenet   --batch-size 64   --epochs 235   --max-lr 0.0125   --pct-start 0.1   --workers 8   --reports   --use-best-for-reports   --name r50_imagenet1k_onecycle_amp_bs64_ep150   --resume
```


**AWS** Refer README_AWS.md for more instructions
```bash
tmux new -s imagenet1k_full -n train # tmux attach -t imagenet1k_full  if tmux is running

# Inside tmux
source /opt/dlami/nvme/envs/imagenet1k_venv/bin/activate
cd ~/week9_ERAV4_ImageNet_ResNet-50_Model_Experiments

# (Optional) start W&B session automatically when resuming
unset RESET_SCHED
unset FREEZE_LR
unset FREEZE_LR_VALUE

bash scripts/launch_single_gpu.sh /mnt/imagenet1k 160 256 6 \
  --max-lr 0.125 \ # Put value we got by executing lr_finder.py
  --stats-file data_stats/imagenet_1k_aws_stats.json \
  --show-progress \
  --amp --channels-last \
  --resume /mnt/imagenet1k/checkpoints/imagenet1kfull_g5x_1gpu_dali_nvme_lr0p125_bs256_e150_work6/last_epoch119.pth \
  --out-dir imagenet1kfull_g5x_1gpu_dali_nvme_lr0p125_bs256_e150_work6 \
  --wandb --wandb-project imagenet1k_runs \
  --wandb-tags imagenet1k_full,dali,1gpu,onecycle_reset,bs256,ep120to160,resumed_e120
```
**For more details on diffrent resume approaches, see the [Resuming trainings on AWS instructions](./README_AWS.md).**

### 2.4 Key arguments
#### For training in local machine
| Arg | Default | Meaning |
|---|---:|---|
| `--data-root` | `data/imagenet` | Root containing `train/` and `val/` (ImageFolder) | --> dataset path for training train_full_ImageNet1k_SingleGPU.py
| `--name` | `r50_imagenet1k_onecycle_amp_bs64_ep150` | Run/experiment name used for all output folders |
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

#### For training in AWS machine

| Arg | Default | Meaning |
|---|---:|---|
| `--data` | **required** | Root that contains `train/` and `val/` (ImageFolder) |
| `--out-dir` | `./` | Run name (used to create `/mnt/imagenet1k/{out,runs,reports,checkpoints}/<out-dir>`) |
| `--epochs` | `90` | Total epochs (used to align/extend scheduler) |
| `--batch-size` | `256` | **Per-GPU** batch size (global = `batch × world_size`) |
| `--eval-batch-size` | `256` | Validation batch size |
| `--workers` | `min(8, cpu_count)` | DataLoader workers |
| `--crop-size` | `224` | Input crop size |
| `--loader` | `dali` | `dali` (fast) or `albumentations` (PyTorch DataLoader path) |
| `--stats-file` | `None` | JSON with mean/std (used by Albumentations). Example: `data_stats/imagenet_1k_aws_stats.json` |
| `--stats-samples` | `50000` | Samples to compute mean/std (albumentations path only) |
| `--use-class-style` | `False` | Enable class-style augmentation (albumentations only) |
| `--amp` | off by default | Enable Automatic Mixed Precision |
| `--channels-last` | off by default | Use NHWC memory format |
| `--resume` | `''` | Path to checkpoint (`.../last_epochXYZ.pth` or `best_acc_epochXYZ.pth`) |
| `--num-classes` | `1000` | Number of classes |
| `--pretrained` | off | Start from torchvision weights (for finetune experiments) |
| `--seed` | `42` | Seed (with cudnn.benchmark = True) |
| `--use-tb` | off | TensorBoard logging → `runs/<out-dir>` |
| `--show-progress` | off | Show per-batch `tqdm` bars (rank-0 only) |
| `--do-report` | off | Save `classification_report.txt` and `confusion_matrix.csv` |
| `--max-lr` | `None` | Peak LR (if None → linear scaling `0.1 × (global_bsz / 256)`) |
| `--pct-start` | `0.3` | OneCycle warm-up fraction |
| `--div-factor` | `25.0` | OneCycle initial LR = `max_lr / div_factor` |
| `--final-div-factor` | `1e4` | OneCycle final LR = `max_lr / final_div_factor` |
| `--wandb` | off | Enable Weights & Biases |
| `--wandb-project` | `imagenet1k_runs` | W&B project name |
| `--wandb-entity` | `None` | W&B entity/org (optional) |
| `--wandb-tags` | `''` | Comma-separated tags |
| `--wandb-offline` | off | Log offline and sync later |

---

Environment Variables (affect resume/freeze behavior)

| Env Var | Effect | When to Use |
|---|---|---|
| `RESET_SCHED=1` | Rebuilds and realigns scheduler when resuming with a different `--epochs`. | Extend or shrink total epochs on resume. |
| `FREEZE_LR=1` | Replaces scheduler with constant-LR scheduler. | Keep LR fixed for entire (resumed) run. |
| `FREEZE_LR_VALUE=<float>` | Explicit LR value if freezing. | Optional override; else uses last used LR. |

---

Example Combinations

| Scenario | Command / Env Vars | Result |
|---|---|---|
| **Fresh run with fixed LR** | `FREEZE_LR=1`, no resume | Constant LR entire run |
| **Resume normally** | (no env vars) | Scheduler continues smoothly |
| **Resume + extend epochs** | `export RESET_SCHED=1` | Scheduler realigned, OneCycle continues |
| **Resume + freeze LR** | `export FREEZE_LR=1` | LR frozen to last used value |
| **Resume + extend + freeze** | `export RESET_SCHED=1; export FREEZE_LR=1` | Scheduler rebuilt then frozen immediately |


### 2.5 Repository layout
```
week9_ERAV4_ImageNet_ResNet-50_Model_Experiments/
├─ train.py # Train script for ImageNet-Mini (subset) experiments
├─ train_full_ImageNet1k_SingleGPU.py # Train full ImageNet-1k on local machine (single GPU)
├─ train_full_ImageNet_AWS.py # Train full ImageNet-1k on AWS (DALI, mixed precision)
├─ model.py # ResNet-50 architecture definition
├─ lr_finder.py # Learning-rate finder (plots LR vs loss)
│
├─ dataset/
│ ├─ imagenet.py # TorchVision-style ImageNet loader (for albumentations)
│ ├─ imagenet_dali.py # NVIDIA DALI-based high-performance ImageNet loader
│ └─ imagenet_mini.py # Lightweight ImageNet-Mini loader for debugging
│
├─ data/ # Training data (local mini version)
├─ data_stats/
│ └─ imagenet_1k_aws_stats.json # Cached channel mean/std stats used for normalization
│
├─ lr_finder_plots/
│ ├─ lr_finder_plots_imagenette/ # LR finder results for Imagenette
│ └─ lr_finder_plots_imagenet1k_AWS/ # LR finder results for AWS DALI pipeline
│
├─ checkpoints/
│ ├─ imagenet1kfull_g5x_1gpu_dali_nvme_lr0p125_bs256_e150_work6/ # AWS (full ImageNet) runs
│ │ ├─ best_acc_epochXXX.pth
│ │ ├─ last_epochXXX.pth
│ │ └─ last_epoch.pth
│ ├─ r50_imagenet1k_onecycle_amp_bs64_ep150/ # Local full ImageNet run
│ │ ├─ best.pth
│ │ └─ checkpoint.pth
│ └─ r50_onecycle_amp/ # Imagenet-Mini baseline
│ └─ best.pth
│
├─ out/
│ ├─ imagenet1kfull_g5x_1gpu_dali_nvme_lr0p125_bs256_e150_work6/ # AWS training logs
│ │ ├─ train_log.csv
│ │ ├─ logs.md
│ │ └─ metrics.csv (optional)
│ ├─ r50_imagenet1k_onecycle_amp_bs64_ep150/ # Local full ImageNet logs
│ │ ├─ train_log.csv
│ │ └─ logs.md
│ └─ r50_onecycle_amp/ # Imagenet-Mini logs
│ ├─ train_log.csv
│ └─ logs.md
│
├─ reports/
│ ├─ imagenet1kfull_g5x_1gpu_dali_nvme_lr0p125_bs256_e150_work6/ # AWS training reports
│ │ ├─ accuracy_curve.png
│ │ ├─ loss_curve.png
│ │ ├─ classification_report.txt
│ │ ├─ confusion_matrix.csv
│ │ └─ model_summary.txt
│ ├─ r50_imagenet1k_onecycle_amp_bs64_ep150/ # Local full ImageNet reports
│ │ ├─ accuracy_curve.png
│ │ ├─ loss_curve.png
│ │ ├─ classification_report.txt
│ │ ├─ confusion_matrix.csv
│ │ └─ model_summary.txt
│ └─ r50_onecycle_amp/ # Imagenet-Mini reports
│ ├─ accuracy_curve.png
│ ├─ loss_curve.png
│ ├─ classification_report.txt
│ ├─ confusion_matrix.csv
│ └─ model_summary.txt
│
├─ runs/
│ ├─ imagenet1kfull_g5x_1gpu_dali_nvme_lr0p125_bs256_e150_work6/ # AWS TensorBoard runs
│ ├─ r50_imagenet1k_onecycle_amp_bs64_ep150/ # Local TensorBoard logs
│ └─ r50_onecycle_amp/ # Imagenet-Mini TensorBoard logs
│
├─ images/
│ ├─ resnet50_arch.png # Model architecture visualization
│ └─ imagenet_samples.png # Sample input images
│
├─ scripts/ # Bash utilities (launch, resume, etc.)
│ └─ fix_imagenet_val.py
│ └─ launch_lr_finder.sh
│ └─ launch_single_gpu.sh
│ └─ launch_multi_gpu.sh
│ └─ make_debug_data.py
│ └─ rehumanize_imagenet_reports.py   # To change classification report and confusion matrix from class ID to Class names
│
├─ utils/
│ └─ imagenet_class_index.json # Synset → human-readable name mapping
│
├─ update_readme.py # Auto-update README metadata and results tables
├─ requirements.txt # Local dependencies
├─ requirements.aws.txt # AWS dependencies (DALI, etc.)
├─ README.md # Project overview and results summary
├─ README_AWS.md # AWS Setup instructions
└─ .gitignore / .gitattributes # Git configuration
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

## 5) Learning Rate Finder (LR-Finder)

Before starting full training, we run a **Learning Rate Finder** to determine an optimal `--max-lr` for OneCycleLR. This reduces guesswork and stabilizes convergence.

**Why LR-Finder**
- 🚀 Eliminates guesswork (finds a good LR range)
- ⚖️ Improves efficiency (avoid suboptimal LRs)
- 📈 Optimizes OneCycleLR (use the discovered LR as `max_lr`)
- 💡 Reproducible (curve can be regenerated anytime)

**How it works**  
`lr_finder.py` performs a range test (`start_lr → end_lr` over N iterations), recording loss vs LR and saving a plot to `lr_finder_plots/`.

**🧮 How to Decide --num_iter Based on Batch Size**

When running the Learning Rate Finder, each batch = one iteration.
For a dataset with N samples and batch size B:

iterations per epoch = 𝑁/𝐵
But we don’t need a full epoch to see the LR vs. loss pattern —
only about 1/10th of an epoch, capped at ~1000 iterations.

So the rule of thumb is:

✅ Run at most one full epoch worth of iterations, but never more than 1000.

| Dataset       | #Samples | Batch Size | 1 Epoch ≈ (iters) | Recommended `--num_iter` |
| ------------- | -------- | ---------- | ----------------- | ------------------------ |
| CIFAR-10      | 50 000   | 128        | 390               | 300 – 400                |
| ImageNet-Mini | 100 000  | 128        | 780               | 700 – 800                |
| ImageNet-1k   | 1.28 M   | 64         | 20 000            | 800 – 1000  (cap)        |
| ImageNet-1k   | 1.28 M   | 128        | 10 000            | 800 – 1000               |
| ImageNet-1k   | 1.28 M   | 256        | 5 000             | 500 – 800                |
| ImageNet-1k   | 1.28 M   | 512        | 2 500             | 300 – 500                |

**Run (Local)**
```bash
python lr_finder.py find_lr --data-root data/imagenet --batch-size 64 --num-iter 1000
```

**Run (AWS)**
```bash
bash scripts/launch_lr_finder.sh /mnt/imagenet1k 256 2000 lr_finder_plots_imagenet1k_AWS
```
**🎯 How to Pick the Final max_lr for Training**

If you’re using OneCycleLR, the max_lr is the peak learning rate during training.
You can estimate it either from the LR Finder or by scaling with batch size.

🔹 Option 1: Use LR Finder Result

Pick the LR value just before the loss starts to blow up (the lowest point before the spike).

🔹 Option 2: Scale Linearly with Batch Size

Use the linear scaling rule (starting from baseline 0.1 for batch = 256):

max_lr = 0.1 × batch size/256

| Batch Size | Scaled `max_lr` | Comment                              |
| ---------- | --------------- | ------------------------------------ |
| 64         | 0.025           | good for small GPUs                  |
| 128        | 0.05            | standard choice                      |
| 256        | 0.1             | baseline for ResNet-50               |
| 512        | 0.2             | aggressive; requires stable training |
| 1024       | 0.4             | multi-GPU large-batch setup          |

**Plot (AWS ImageNet-1k, iter=2000)**  
`lr_finder_plots_imagenet1k_AWS/lr_finder_20251031_203333_start1e-07_end1.0_iter2000.png`

![LR-Finder (AWS)](lr_finder_plots_imagenet1k_AWS/lr_finder_20251031_203333_start1e-07_end1.0_iter2000.png)

**Interpretation**  
- Loss decreases smoothly up to **~0.10–0.12**, then trends upward.  
- We selected **`--max-lr 0.125`** for OneCycleLR on AWS (used in training commands above).  
- For local runs (smaller batch), a slightly lower `max_lr` (e.g., **0.1**) is reasonable.


> **Auto-filled (from runs):** For run **`r50_imagenet1k_onecycle_amp_bs64_ep150`**, best Val Top-1 = **77.82%**, Top-5 = **93.82%** at epoch **228**.

---
## 6) Training and Evaluation Summary

This section consolidates the training results and performance comparison between **Local (RTX 4060 Ti)** and **AWS (g5.xlarge A10G)** runs.

---

### 🖥️ A) Single-GPU Training (Local)

**Infrastructure**

| Component | Specification |
|------------|----------------|
| **GPU** | NVIDIA RTX 4060 Ti (16 GB VRAM) |
| **CPU / RAM** | 32 GB System RAM  •  1 TB SSD |
| **OS / Runtime** | Windows 11 + PyTorch 2.x (AMP enabled) |

**Training Profile**

| Parameter | Value |
|------------|--------|
| **Epochs** | ≈ 250  (1 hr / epoch → ~240 hrs total) |
| **Batch Size** | 64 |
| **Max LR** | 0.0125 |
| **Scheduler** | OneCycleLR |
| **Precision** | Automatic Mixed Precision (AMP) |
| **Optimizations** | Checkpointing • Pin Memory • Multi-worker DataLoader • Label Smoothing • Standard Augmentations |

**Auto-filled Metrics** (from `out/r50_imagenet1k_onecycle_amp_bs64_ep150/train_log.csv`)

| Metric | Value |
|--------|--------|
| Training Top-1 | 82.63% |
| Training Top-5 | 94.05% |
| Validation Top-1 | 77.82% |
| Validation Top-5 | 93.82% |


**Visual Logs**

**CLI snapshot**
![Local Training CLI](images/Local_Training_Image5.png)

**Epoch progress**
![Epoch 1–12](images/Local_Training_Image1.png)
![Epoch 9–22](images/Local_Training_Image2.png)
![Epoch 72-79](images/Local_Training_Image3.png)
![Epoch 225-235](images/Local_Training_Image4.png)

**TensorBoard metrics**
*(added TensorBoard screenshots from `runs/r50_imagenet1k_onecycle_amp_bs64_ep150`)*
![Trainining Progress](images/Local_Training_Image6_TrainTensorBoard.png)
![Evaluation](images/Local_Valdiation_Image7_TrainTensorBoard.png)

<details>
<summary><strong>6.A.1 Training logs (markdown)</strong></summary>

```markdown
# Training Logs (terminal-like)

```
[Val] Epoch 000 | loss 42.9541 | top1 0.10% | top5 0.51% | lr 0.000500 | ips 782.2
[Train] Epoch 000 | loss 6.2441 | top1 2.86% | top5 8.83% | lr 0.000631 | ips 381.9
[Val] Epoch 000 | loss 5.5784 | top1 7.23% | top5 19.87% | lr 0.000631 | ips 790.9
[Train] Epoch 001 | loss 5.3519 | top1 10.11% | top5 24.93% | lr 0.001019 | ips 386.6
[Val] Epoch 001 | loss 4.7754 | top1 16.92% | top5 37.59% | lr 0.001019 | ips 472.5
[Train] Epoch 002 | loss 4.7556 | top1 18.20% | top5 38.42% | lr 0.001646 | ips 389.6
[Val] Epoch 002 | loss 4.2375 | top1 25.79% | top5 49.95% | lr 0.001646 | ips 653.5
[Train] Epoch 003 | loss 4.2885 | top1 25.84% | top5 48.94% | lr 0.002485 | ips 386.4
[Val] Epoch 003 | loss 3.8422 | top1 33.14% | top5 58.81% | lr 0.002485 | ips 695.8
[Train] Epoch 004 | loss 3.9358 | top1 32.17% | top5 56.66% | lr 0.003500 | ips 386.1
[Val] Epoch 004 | loss 3.5427 | top1 38.67% | top5 65.16% | lr 0.003500 | ips 626.0
[Train] Epoch 005 | loss 3.6863 | top1 37.08% | top5 62.00% | lr 0.004646 | ips 383.9
[Val] Epoch 005 | loss 3.2951 | top1 44.29% | top5 70.58% | lr 0.004646 | ips 562.5
[Train] Epoch 006 | loss 3.5014 | top1 40.81% | top5 65.78% | lr 0.005873 | ips 386.5
[Val] Epoch 006 | loss 3.1841 | top1 46.67% | top5 72.80% | lr 0.005873 | ips 1058.8
[Train] Epoch 007 | loss 3.3714 | top1 43.53% | top5 68.33% | lr 0.007127 | ips 386.1
[Val] Epoch 007 | loss 3.0176 | top1 50.40% | top5 75.55% | lr 0.007127 | ips 657.0
[Train] Epoch 008 | loss 3.2714 | top1 45.66% | top5 70.28% | lr 0.008354 | ips 386.8
[Val] Epoch 008 | loss 2.9634 | top1 51.47% | top5 76.97% | lr 0.008354 | ips 1064.5
[Train] Epoch 009 | loss 3.1916 | top1 47.37% | top5 71.84% | lr 0.009500 | ips 389.5
[Val] Epoch 009 | loss 2.8974 | top1 52.69% | top5 77.96% | lr 0.009500 | ips 688.8
[Train] Epoch 010 | loss 3.1293 | top1 48.71% | top5 72.99% | lr 0.010515 | ips 390.1
[Val] Epoch 010 | loss 2.8075 | top1 54.73% | top5 79.49% | lr 0.010515 | ips 697.7
[Train] Epoch 011 | loss 3.0766 | top1 49.82% | top5 73.93% | lr 0.011354 | ips 389.9
[Val] Epoch 011 | loss 2.7683 | top1 55.86% | top5 80.26% | lr 0.011354 | ips 1012.4
[Train] Epoch 012 | loss 3.0349 | top1 50.73% | top5 74.73% | lr 0.011981 | ips 387.1
[Val] Epoch 012 | loss 2.7037 | top1 57.18% | top5 81.19% | lr 0.011981 | ips 1125.8
[Train] Epoch 013 | loss 2.9953 | top1 51.57% | top5 75.47% | lr 0.012369 | ips 390.1
[Val] Epoch 013 | loss 2.7174 | top1 56.37% | top5 81.03% | lr 0.012369 | ips 685.3
[Train] Epoch 014 | loss 2.9633 | top1 52.28% | top5 76.04% | lr 0.012500 | ips 393.6
[Val] Epoch 014 | loss 2.7241 | top1 56.19% | top5 80.79% | lr 0.012500 | ips 675.1
[Train] Epoch 015 | loss 2.9319 | top1 52.95% | top5 76.60% | lr 0.012498 | ips 390.0
[Val] Epoch 015 | loss 2.6234 | top1 58.98% | top5 82.85% | lr 0.012498 | ips 477.9
[Train] Epoch 016 | loss 2.9056 | top1 53.52% | top5 77.03% | lr 0.012493 | ips 390.1
[Val] Epoch 016 | loss 2.6460 | top1 58.33% | top5 81.95% | lr 0.012493 | ips 686.4
[Train] Epoch 017 | loss 2.8835 | top1 54.01% | top5 77.41% | lr 0.012485 | ips 393.6
[Val] Epoch 017 | loss 2.6042 | top1 59.44% | top5 83.16% | lr 0.012485 | ips 700.2
[Train] Epoch 018 | loss 2.8615 | top1 54.53% | top5 77.79% | lr 0.012473 | ips 393.5
[Val] Epoch 018 | loss 2.5824 | top1 59.51% | top5 83.26% | lr 0.012473 | ips 715.9
[Train] Epoch 019 | loss 2.8407 | top1 54.96% | top5 78.19% | lr 0.012458 | ips 390.2
[Val] Epoch 019 | loss 2.5676 | top1 60.37% | top5 83.91% | lr 0.012458 | ips 1121.3
[Train] Epoch 020 | loss 2.8210 | top1 55.41% | top5 78.47% | lr 0.012439 | ips 390.1
[Val] Epoch 020 | loss 2.5162 | top1 61.27% | top5 84.56% | lr 0.012439 | ips 634.2
[Train] Epoch 021 | loss 2.8035 | top1 55.79% | top5 78.77% | lr 0.012417 | ips 389.9
[Val] Epoch 021 | loss 2.5507 | top1 60.52% | top5 83.91% | lr 0.012417 | ips 498.9
[Train] Epoch 022 | loss 2.7912 | top1 56.10% | top5 79.00% | lr 0.012392 | ips 393.7
[Val] Epoch 022 | loss 2.5018 | top1 61.89% | top5 84.95% | lr 0.012392 | ips 678.4
[Train] Epoch 023 | loss 2.7776 | top1 56.38% | top5 79.21% | lr 0.012363 | ips 386.9
[Val] Epoch 023 | loss 2.4850 | top1 62.08% | top5 84.76% | lr 0.012363 | ips 1080.7
[Train] Epoch 024 | loss 2.7639 | top1 56.69% | top5 79.42% | lr 0.012332 | ips 388.0
[Val] Epoch 024 | loss 2.4913 | top1 61.92% | top5 84.86% | lr 0.012332 | ips 551.1
[Train] Epoch 025 | loss 2.7524 | top1 56.90% | top5 79.64% | lr 0.012296 | ips 386.8
[Val] Epoch 025 | loss 2.4813 | top1 61.90% | top5 85.05% | lr 0.012296 | ips 631.1
[Train] Epoch 026 | loss 2.7423 | top1 57.15% | top5 79.82% | lr 0.012258 | ips 389.2
[Val] Epoch 026 | loss 2.4692 | top1 62.45% | top5 85.14% | lr 0.012258 | ips 667.8
[Train] Epoch 027 | loss 2.7311 | top1 57.42% | top5 79.98% | lr 0.012216 | ips 382.9
[Val] Epoch 027 | loss 2.4284 | top1 63.24% | top5 85.83% | lr 0.012216 | ips 1061.4
[Train] Epoch 028 | loss 2.7207 | top1 57.65% | top5 80.17% | lr 0.012171 | ips 380.9
[Val] Epoch 028 | loss 2.4288 | top1 63.72% | top5 85.75% | lr 0.012171 | ips 637.9
[Train] Epoch 029 | loss 2.7124 | top1 57.83% | top5 80.27% | lr 0.012123 | ips 387.8
[Val] Epoch 029 | loss 2.4414 | top1 63.10% | top5 85.60% | lr 0.012123 | ips 1047.4
[Train] Epoch 030 | loss 2.7014 | top1 58.08% | top5 80.46% | lr 0.012072 | ips 385.3
[Val] Epoch 030 | loss 2.4470 | top1 62.98% | top5 85.62% | lr 0.012072 | ips 646.1
[Train] Epoch 031 | loss 2.6944 | top1 58.25% | top5 80.58% | lr 0.012017 | ips 386.2
[Val] Epoch 031 | loss 2.4361 | top1 63.15% | top5 85.81% | lr 0.012017 | ips 668.2
[Train] Epoch 032 | loss 2.6877 | top1 58.33% | top5 80.69% | lr 0.011960 | ips 388.5
[Val] Epoch 032 | loss 2.4291 | top1 63.15% | top5 85.96% | lr 0.011960 | ips 559.9
[Train] Epoch 033 | loss 2.6801 | top1 58.52% | top5 80.81% | lr 0.011899 | ips 385.7
[Val] Epoch 033 | loss 2.4334 | top1 63.41% | top5 85.65% | lr 0.011899 | ips 654.3
[Train] Epoch 034 | loss 2.6731 | top1 58.72% | top5 80.90% | lr 0.011835 | ips 388.4
[Val] Epoch 034 | loss 2.4451 | top1 62.88% | top5 85.62% | lr 0.011835 | ips 679.0
[Train] Epoch 035 | loss 2.6681 | top1 58.87% | top5 81.04% | lr 0.011768 | ips 386.0
[Val] Epoch 035 | loss 2.4190 | top1 63.39% | top5 86.06% | lr 0.011768 | ips 686.2
[Train] Epoch 036 | loss 2.6601 | top1 58.99% | top5 81.12% | lr 0.011699 | ips 386.8
[Val] Epoch 036 | loss 2.3881 | top1 64.49% | top5 86.61% | lr 0.011699 | ips 1056.6
[Train] Epoch 037 | loss 2.6544 | top1 59.12% | top5 81.22% | lr 0.011626 | ips 388.0
[Val] Epoch 037 | loss 2.3741 | top1 64.73% | top5 86.46% | lr 0.011626 | ips 547.1
[Train] Epoch 038 | loss 2.6517 | top1 59.22% | top5 81.27% | lr 0.011550 | ips 391.0
[Val] Epoch 038 | loss 2.3876 | top1 64.36% | top5 86.41% | lr 0.011550 | ips 666.7
[Train] Epoch 039 | loss 2.6448 | top1 59.34% | top5 81.36% | lr 0.011472 | ips 387.5
[Val] Epoch 039 | loss 2.3970 | top1 63.96% | top5 86.25% | lr 0.011472 | ips 1097.7
[Train] Epoch 040 | loss 2.6389 | top1 59.45% | top5 81.48% | lr 0.011390 | ips 387.8
[Val] Epoch 040 | loss 2.3753 | top1 64.49% | top5 86.66% | lr 0.011390 | ips 677.8
[Train] Epoch 041 | loss 2.6314 | top1 59.61% | top5 81.62% | lr 0.011306 | ips 390.7
[Val] Epoch 041 | loss 2.3672 | top1 64.76% | top5 86.75% | lr 0.011306 | ips 512.4
[Train] Epoch 042 | loss 2.6298 | top1 59.69% | top5 81.61% | lr 0.011220 | ips 390.6
[Val] Epoch 042 | loss 2.3878 | top1 64.47% | top5 86.41% | lr 0.011220 | ips 687.0
[Train] Epoch 043 | loss 2.6235 | top1 59.85% | top5 81.68% | lr 0.011130 | ips 387.4
[Val] Epoch 043 | loss 2.3642 | top1 65.03% | top5 86.74% | lr 0.011130 | ips 803.7
[Train] Epoch 044 | loss 2.6209 | top1 59.91% | top5 81.74% | lr 0.011038 | ips 387.8
[Val] Epoch 044 | loss 2.3550 | top1 64.94% | top5 87.02% | lr 0.011038 | ips 655.6
[Train] Epoch 045 | loss 2.6168 | top1 60.00% | top5 81.83% | lr 0.010943 | ips 390.7
[Val] Epoch 045 | loss 2.3585 | top1 64.84% | top5 86.72% | lr 0.010943 | ips 690.8
[Train] Epoch 046 | loss 2.6115 | top1 60.11% | top5 81.89% | lr 0.010846 | ips 387.7
[Val] Epoch 046 | loss 2.3589 | top1 64.92% | top5 86.90% | lr 0.010846 | ips 1070.3
[Train] Epoch 047 | loss 2.6055 | top1 60.25% | top5 81.95% | lr 0.010746 | ips 387.2
[Val] Epoch 047 | loss 2.3893 | top1 64.05% | top5 86.34% | lr 0.010746 | ips 670.8
[Train] Epoch 048 | loss 2.6050 | top1 60.24% | top5 81.98% | lr 0.010644 | ips 385.5
[Val] Epoch 048 | loss 2.3598 | top1 65.32% | top5 86.83% | lr 0.010644 | ips 795.7
[Train] Epoch 049 | loss 2.6010 | top1 60.33% | top5 82.04% | lr 0.010539 | ips 385.5
[Val] Epoch 049 | loss 2.3452 | top1 65.28% | top5 87.12% | lr 0.010539 | ips 645.8
[Train] Epoch 050 | loss 2.5943 | top1 60.51% | top5 82.13% | lr 0.010432 | ips 389.4
[Val] Epoch 050 | loss 2.3395 | top1 65.44% | top5 87.23% | lr 0.010432 | ips 670.8
[Train] Epoch 051 | loss 2.5883 | top1 60.62% | top5 82.26% | lr 0.010323 | ips 389.9
[Val] Epoch 051 | loss 2.3355 | top1 65.25% | top5 87.22% | lr 0.010323 | ips 698.6
[Train] Epoch 052 | loss 2.5851 | top1 60.70% | top5 82.28% | lr 0.010211 | ips 388.0
[Val] Epoch 052 | loss 2.3316 | top1 65.25% | top5 87.25% | lr 0.010211 | ips 540.6
[Train] Epoch 053 | loss 2.5812 | top1 60.77% | top5 82.37% | lr 0.010098 | ips 384.4
[Val] Epoch 053 | loss 2.3139 | top1 66.06% | top5 87.47% | lr 0.010098 | ips 1041.5
[Train] Epoch 054 | loss 2.5797 | top1 60.81% | top5 82.38% | lr 0.009982 | ips 382.2
[Val] Epoch 054 | loss 2.3183 | top1 65.72% | top5 87.42% | lr 0.009982 | ips 656.4
[Train] Epoch 055 | loss 2.5732 | top1 61.00% | top5 82.49% | lr 0.009865 | ips 389.6
[Val] Epoch 055 | loss 2.3166 | top1 65.77% | top5 87.42% | lr 0.009865 | ips 690.6
[Train] Epoch 056 | loss 2.5708 | top1 61.04% | top5 82.51% | lr 0.009745 | ips 387.3
[Val] Epoch 056 | loss 2.3258 | top1 65.62% | top5 87.32% | lr 0.009745 | ips 989.8
[Train] Epoch 057 | loss 2.5657 | top1 61.14% | top5 82.64% | lr 0.009623 | ips 387.5
[Val] Epoch 057 | loss 2.3252 | top1 65.64% | top5 87.52% | lr 0.009623 | ips 683.6
[Train] Epoch 058 | loss 2.5654 | top1 61.13% | top5 82.58% | lr 0.009500 | ips 390.7
[Val] Epoch 058 | loss 2.3046 | top1 66.25% | top5 87.67% | lr 0.009500 | ips 660.0
[Train] Epoch 059 | loss 2.5596 | top1 61.26% | top5 82.70% | lr 0.009375 | ips 385.5
[Val] Epoch 059 | loss 2.3289 | top1 65.67% | top5 87.33% | lr 0.009375 | ips 1070.9
[Train] Epoch 060 | loss 2.5567 | top1 61.33% | top5 82.74% | lr 0.009248 | ips 385.1
[Val] Epoch 060 | loss 2.3124 | top1 66.08% | top5 87.57% | lr 0.009248 | ips 412.8
[Train] Epoch 061 | loss 2.5527 | top1 61.42% | top5 82.80% | lr 0.009120 | ips 385.5
[Val] Epoch 061 | loss 2.3055 | top1 66.02% | top5 87.58% | lr 0.009120 | ips 646.3
[Train] Epoch 062 | loss 2.5502 | top1 61.45% | top5 82.84% | lr 0.008990 | ips 391.1
[Val] Epoch 062 | loss 2.2901 | top1 66.57% | top5 87.82% | lr 0.008990 | ips 701.3
[Train] Epoch 063 | loss 2.5461 | top1 61.61% | top5 82.90% | lr 0.008858 | ips 389.8
[Val] Epoch 063 | loss 2.3057 | top1 66.42% | top5 87.46% | lr 0.008858 | ips 687.2
[Train] Epoch 064 | loss 2.5424 | top1 61.67% | top5 82.97% | lr 0.008726 | ips 393.3
[Val] Epoch 064 | loss 2.2845 | top1 66.73% | top5 88.06% | lr 0.008726 | ips 708.6
[Train] Epoch 065 | loss 2.5403 | top1 61.73% | top5 83.00% | lr 0.008591 | ips 390.7
[Val] Epoch 065 | loss 2.3083 | top1 66.11% | top5 87.61% | lr 0.008591 | ips 643.6
[Train] Epoch 066 | loss 2.5319 | top1 61.94% | top5 83.12% | lr 0.008456 | ips 394.2
[Val] Epoch 066 | loss 2.2802 | top1 66.96% | top5 88.05% | lr 0.008456 | ips 687.5
[Train] Epoch 067 | loss 2.5302 | top1 61.94% | top5 83.14% | lr 0.008319 | ips 388.2
[Val] Epoch 067 | loss 2.2705 | top1 66.99% | top5 88.19% | lr 0.008319 | ips 1127.3
[Train] Epoch 068 | loss 2.5264 | top1 62.02% | top5 83.19% | lr 0.008181 | ips 390.8
[Val] Epoch 068 | loss 2.2558 | top1 67.61% | top5 88.45% | lr 0.008181 | ips 692.4
[Train] Epoch 069 | loss 2.5243 | top1 62.08% | top5 83.20% | lr 0.008043 | ips 394.1
[Val] Epoch 069 | loss 2.2680 | top1 67.00% | top5 88.17% | lr 0.008043 | ips 691.4
[Train] Epoch 070 | loss 2.5189 | top1 62.17% | top5 83.34% | lr 0.007903 | ips 390.8
[Val] Epoch 070 | loss 2.2936 | top1 66.37% | top5 87.64% | lr 0.007903 | ips 683.9
[Train] Epoch 071 | loss 2.5144 | top1 62.31% | top5 83.40% | lr 0.007762 | ips 394.3
[Val] Epoch 071 | loss 2.2477 | top1 67.76% | top5 88.35% | lr 0.007762 | ips 650.0
[Train] Epoch 072 | loss 2.5142 | top1 62.26% | top5 83.41% | lr 0.007621 | ips 390.7
[Val] Epoch 072 | loss 2.2680 | top1 67.12% | top5 88.28% | lr 0.007621 | ips 685.2
[Train] Epoch 073 | loss 2.5110 | top1 62.44% | top5 83.44% | lr 0.007478 | ips 394.2
[Val] Epoch 073 | loss 2.2676 | top1 67.30% | top5 88.25% | lr 0.007478 | ips 711.1
[Train] Epoch 074 | loss 2.5044 | top1 62.58% | top5 83.55% | lr 0.007335 | ips 388.8
[Val] Epoch 074 | loss 2.2647 | top1 67.02% | top5 88.31% | lr 0.007335 | ips 1128.3
[Train] Epoch 075 | loss 2.5007 | top1 62.63% | top5 83.61% | lr 0.007192 | ips 391.0
[Val] Epoch 075 | loss 2.2558 | top1 67.39% | top5 88.43% | lr 0.007192 | ips 1137.6
[Train] Epoch 076 | loss 2.4973 | top1 62.71% | top5 83.63% | lr 0.007048 | ips 382.3
[Val] Epoch 076 | loss 2.2538 | top1 67.47% | top5 88.40% | lr 0.007048 | ips 786.3
[Train] Epoch 077 | loss 2.4924 | top1 62.83% | top5 83.72% | lr 0.006903 | ips 384.6
[Val] Epoch 077 | loss 2.2529 | top1 67.40% | top5 88.35% | lr 0.006903 | ips 690.5
[Train] Epoch 078 | loss 2.4894 | top1 62.90% | top5 83.77% | lr 0.006759 | ips 384.6
[Val] Epoch 078 | loss 2.2456 | top1 67.61% | top5 88.44% | lr 0.006759 | ips 538.8
[Train] Epoch 079 | loss 2.4850 | top1 63.01% | top5 83.84% | lr 0.006613 | ips 383.4
[Val] Epoch 079 | loss 2.2246 | top1 67.99% | top5 88.80% | lr 0.006613 | ips 661.3
[Train] Epoch 080 | loss 2.4773 | top1 63.12% | top5 83.94% | lr 0.006468 | ips 386.5
[Val] Epoch 080 | loss 2.2257 | top1 68.24% | top5 88.84% | lr 0.006468 | ips 664.3
[Train] Epoch 081 | loss 2.4756 | top1 63.22% | top5 83.97% | lr 0.006323 | ips 388.9
[Val] Epoch 081 | loss 2.2362 | top1 67.82% | top5 88.68% | lr 0.006323 | ips 716.7
[Train] Epoch 082 | loss 2.4703 | top1 63.31% | top5 84.07% | lr 0.006177 | ips 389.5
[Val] Epoch 082 | loss 2.2091 | top1 68.77% | top5 89.04% | lr 0.006177 | ips 632.5
[Train] Epoch 083 | loss 2.4637 | top1 63.48% | top5 84.14% | lr 0.006032 | ips 389.2
[Val] Epoch 083 | loss 2.2509 | top1 67.49% | top5 88.29% | lr 0.006032 | ips 1113.1
[Train] Epoch 084 | loss 2.4621 | top1 63.54% | top5 84.20% | lr 0.005887 | ips 389.0
[Val] Epoch 084 | loss 2.2177 | top1 68.04% | top5 88.83% | lr 0.005887 | ips 635.9
[Train] Epoch 085 | loss 2.4538 | top1 63.71% | top5 84.35% | lr 0.005742 | ips 388.6
[Val] Epoch 085 | loss 2.2194 | top1 68.16% | top5 88.76% | lr 0.005742 | ips 649.8
[Train] Epoch 086 | loss 2.4538 | top1 63.75% | top5 84.28% | lr 0.005597 | ips 390.8
[Val] Epoch 086 | loss 2.2243 | top1 68.27% | top5 88.83% | lr 0.005597 | ips 674.6
[Train] Epoch 087 | loss 2.4457 | top1 63.94% | top5 84.41% | lr 0.005452 | ips 390.7
[Val] Epoch 087 | loss 2.2306 | top1 67.96% | top5 88.83% | lr 0.005452 | ips 504.8
[Train] Epoch 088 | loss 2.4415 | top1 63.98% | top5 84.52% | lr 0.005308 | ips 388.0
[Val] Epoch 088 | loss 2.2111 | top1 68.59% | top5 88.96% | lr 0.005308 | ips 1012.9
[Train] Epoch 089 | loss 2.4347 | top1 64.14% | top5 84.58% | lr 0.005165 | ips 388.8
[Val] Epoch 089 | loss 2.2078 | top1 68.76% | top5 88.99% | lr 0.005165 | ips 663.8
[Train] Epoch 090 | loss 2.4317 | top1 64.23% | top5 84.67% | lr 0.005022 | ips 389.2
[Val] Epoch 090 | loss 2.2117 | top1 68.35% | top5 89.02% | lr 0.005022 | ips 1119.8
[Train] Epoch 091 | loss 2.4215 | top1 64.46% | top5 84.83% | lr 0.004880 | ips 389.5
[Val] Epoch 091 | loss 2.2087 | top1 68.57% | top5 89.02% | lr 0.004880 | ips 698.0
[Train] Epoch 092 | loss 2.4197 | top1 64.53% | top5 84.84% | lr 0.004738 | ips 392.5
[Val] Epoch 092 | loss 2.1832 | top1 69.34% | top5 89.44% | lr 0.004738 | ips 683.2
[Train] Epoch 093 | loss 2.4135 | top1 64.71% | top5 84.88% | lr 0.004597 | ips 389.0
[Val] Epoch 093 | loss 2.1803 | top1 68.82% | top5 89.35% | lr 0.004597 | ips 649.5
[Train] Epoch 094 | loss 2.4067 | top1 64.86% | top5 85.05% | lr 0.004458 | ips 389.5
[Val] Epoch 094 | loss 2.1841 | top1 69.03% | top5 89.34% | lr 0.004458 | ips 710.1
[Train] Epoch 095 | loss 2.3995 | top1 64.96% | top5 85.14% | lr 0.004319 | ips 388.8
[Val] Epoch 095 | loss 2.1717 | top1 69.43% | top5 89.58% | lr 0.004319 | ips 652.3
[Train] Epoch 096 | loss 2.3926 | top1 65.17% | top5 85.25% | lr 0.004181 | ips 389.1
[Val] Epoch 096 | loss 2.1740 | top1 69.53% | top5 89.38% | lr 0.004181 | ips 1120.1
[Train] Epoch 097 | loss 2.3910 | top1 65.26% | top5 85.23% | lr 0.004044 | ips 389.1
[Val] Epoch 097 | loss 2.1665 | top1 69.28% | top5 89.64% | lr 0.004044 | ips 693.5
[Train] Epoch 098 | loss 2.3836 | top1 65.35% | top5 85.38% | lr 0.003909 | ips 392.5
[Val] Epoch 098 | loss 2.1874 | top1 68.93% | top5 89.13% | lr 0.003909 | ips 685.0
[Train] Epoch 099 | loss 2.3796 | top1 65.53% | top5 85.42% | lr 0.003775 | ips 388.9
[Val] Epoch 099 | loss 2.1421 | top1 70.22% | top5 89.92% | lr 0.003775 | ips 1015.3
[Train] Epoch 100 | loss 2.3697 | top1 65.69% | top5 85.60% | lr 0.003642 | ips 386.2
[Val] Epoch 100 | loss 2.1417 | top1 69.94% | top5 89.91% | lr 0.003642 | ips 673.7
[Train] Epoch 101 | loss 2.3618 | top1 65.87% | top5 85.73% | lr 0.003510 | ips 392.5
[Val] Epoch 101 | loss 2.1784 | top1 69.36% | top5 89.37% | lr 0.003510 | ips 702.2
[Train] Epoch 102 | loss 2.3527 | top1 66.07% | top5 85.85% | lr 0.003380 | ips 388.7
[Val] Epoch 102 | loss 2.1287 | top1 70.26% | top5 90.00% | lr 0.003380 | ips 708.9
[Train] Epoch 103 | loss 2.3474 | top1 66.24% | top5 85.96% | lr 0.003252 | ips 375.1
[Val] Epoch 103 | loss 2.1370 | top1 70.31% | top5 90.11% | lr 0.003252 | ips 1123.7
[Train] Epoch 104 | loss 2.3377 | top1 66.48% | top5 86.08% | lr 0.003125 | ips 388.3
[Val] Epoch 104 | loss 2.1337 | top1 70.30% | top5 90.18% | lr 0.003125 | ips 551.6
[Train] Epoch 105 | loss 2.3314 | top1 66.62% | top5 86.16% | lr 0.003000 | ips 386.1
[Val] Epoch 105 | loss 2.1114 | top1 70.73% | top5 90.35% | lr 0.003000 | ips 1069.1
[Train] Epoch 106 | loss 2.3245 | top1 66.83% | top5 86.23% | lr 0.002877 | ips 386.3
[Val] Epoch 106 | loss 2.1188 | top1 70.86% | top5 90.06% | lr 0.002877 | ips 431.8
[Train] Epoch 107 | loss 2.3183 | top1 66.90% | top5 86.35% | lr 0.002755 | ips 386.1
[Val] Epoch 107 | loss 2.0962 | top1 71.31% | top5 90.66% | lr 0.002755 | ips 1060.4
[Train] Epoch 108 | loss 2.3057 | top1 67.26% | top5 86.52% | lr 0.002635 | ips 386.2
[Val] Epoch 108 | loss 2.1086 | top1 70.95% | top5 90.44% | lr 0.002635 | ips 1064.6
[Train] Epoch 109 | loss 2.2945 | top1 67.55% | top5 86.68% | lr 0.002518 | ips 385.9
[Val] Epoch 109 | loss 2.1024 | top1 71.30% | top5 90.51% | lr 0.002518 | ips 667.0
[Train] Epoch 110 | loss 2.2876 | top1 67.72% | top5 86.77% | lr 0.002402 | ips 386.2
[Val] Epoch 110 | loss 2.0957 | top1 71.36% | top5 90.58% | lr 0.002402 | ips 802.4
[Train] Epoch 111 | loss 2.2805 | top1 67.89% | top5 86.88% | lr 0.002289 | ips 386.0
[Val] Epoch 111 | loss 2.0846 | top1 71.74% | top5 90.49% | lr 0.002289 | ips 651.7
[Train] Epoch 112 | loss 2.2672 | top1 68.19% | top5 87.09% | lr 0.002177 | ips 389.1
[Val] Epoch 112 | loss 2.0782 | top1 71.87% | top5 90.74% | lr 0.002177 | ips 680.8
[Train] Epoch 113 | loss 2.2551 | top1 68.47% | top5 87.22% | lr 0.002068 | ips 386.9
[Val] Epoch 113 | loss 2.0753 | top1 71.92% | top5 90.86% | lr 0.002068 | ips 699.1
[Train] Epoch 114 | loss 2.2466 | top1 68.68% | top5 87.38% | lr 0.001961 | ips 386.2
[Val] Epoch 114 | loss 2.0616 | top1 72.06% | top5 91.01% | lr 0.001961 | ips 557.1
[Train] Epoch 115 | loss 2.2346 | top1 68.95% | top5 87.56% | lr 0.001856 | ips 386.7
[Val] Epoch 115 | loss 2.0542 | top1 72.33% | top5 91.09% | lr 0.001856 | ips 1067.1
[Train] Epoch 116 | loss 2.2251 | top1 69.21% | top5 87.68% | lr 0.001754 | ips 387.2
[Val] Epoch 116 | loss 2.0518 | top1 72.60% | top5 91.06% | lr 0.001754 | ips 683.1
[Train] Epoch 117 | loss 2.2126 | top1 69.49% | top5 87.89% | lr 0.001654 | ips 389.5
[Val] Epoch 117 | loss 2.0453 | top1 72.51% | top5 91.15% | lr 0.001654 | ips 675.8
[Train] Epoch 118 | loss 2.2011 | top1 69.80% | top5 88.05% | lr 0.001557 | ips 388.9
[Val] Epoch 118 | loss 2.0351 | top1 72.96% | top5 91.31% | lr 0.001557 | ips 677.5
[Train] Epoch 119 | loss 2.1891 | top1 70.13% | top5 88.22% | lr 0.001462 | ips 383.3
[Val] Epoch 119 | loss 2.0264 | top1 72.96% | top5 91.52% | lr 0.001462 | ips 529.8
[Train] Epoch 120 | loss 2.1760 | top1 70.42% | top5 88.40% | lr 0.001370 | ips 390.7
[Val] Epoch 120 | loss 2.0263 | top1 73.04% | top5 91.47% | lr 0.001370 | ips 667.6
[Train] Epoch 121 | loss 2.1603 | top1 70.84% | top5 88.57% | lr 0.001281 | ips 387.4
[Val] Epoch 121 | loss 2.0164 | top1 73.28% | top5 91.66% | lr 0.001281 | ips 1096.3
[Train] Epoch 122 | loss 2.1440 | top1 71.16% | top5 88.84% | lr 0.001194 | ips 386.9
[Val] Epoch 122 | loss 2.0074 | top1 73.55% | top5 91.73% | lr 0.001194 | ips 681.2
[Train] Epoch 123 | loss 2.1353 | top1 71.47% | top5 88.92% | lr 0.001110 | ips 390.7
[Val] Epoch 123 | loss 2.0015 | top1 73.75% | top5 91.78% | lr 0.001110 | ips 565.4
[Train] Epoch 124 | loss 2.1181 | top1 71.86% | top5 89.18% | lr 0.001028 | ips 387.1
[Val] Epoch 124 | loss 1.9861 | top1 73.93% | top5 92.13% | lr 0.001028 | ips 657.8
[Train] Epoch 125 | loss 2.1044 | top1 72.22% | top5 89.38% | lr 0.000950 | ips 390.3
[Val] Epoch 125 | loss 1.9803 | top1 74.18% | top5 92.06% | lr 0.000950 | ips 537.3
[Train] Epoch 126 | loss 2.0903 | top1 72.58% | top5 89.55% | lr 0.000874 | ips 387.4
[Val] Epoch 126 | loss 1.9765 | top1 74.24% | top5 92.06% | lr 0.000874 | ips 1052.9
[Train] Epoch 127 | loss 2.0714 | top1 73.05% | top5 89.81% | lr 0.000801 | ips 384.1
[Val] Epoch 127 | loss 1.9735 | top1 74.27% | top5 92.17% | lr 0.000801 | ips 543.8
[Train] Epoch 128 | loss 2.0558 | top1 73.42% | top5 90.03% | lr 0.000732 | ips 383.2
[Val] Epoch 128 | loss 1.9509 | top1 74.79% | top5 92.40% | lr 0.000732 | ips 1047.5
[Train] Epoch 129 | loss 2.0444 | top1 73.76% | top5 90.14% | lr 0.000665 | ips 385.9
[Val] Epoch 129 | loss 1.9494 | top1 75.00% | top5 92.46% | lr 0.000665 | ips 547.3
[Train] Epoch 130 | loss 2.0215 | top1 74.33% | top5 90.46% | lr 0.000601 | ips 385.9
[Val] Epoch 130 | loss 1.9412 | top1 75.18% | top5 92.67% | lr 0.000601 | ips 1071.8
[Train] Epoch 131 | loss 2.0053 | top1 74.76% | top5 90.69% | lr 0.000540 | ips 387.0
[Val] Epoch 131 | loss 1.9329 | top1 75.11% | top5 92.65% | lr 0.000540 | ips 572.5
[Train] Epoch 132 | loss 1.9898 | top1 75.16% | top5 90.87% | lr 0.000483 | ips 386.7
[Val] Epoch 132 | loss 1.9179 | top1 75.62% | top5 92.82% | lr 0.000483 | ips 690.6
[Train] Epoch 133 | loss 1.9709 | top1 75.67% | top5 91.12% | lr 0.000428 | ips 392.3
[Val] Epoch 133 | loss 1.9109 | top1 75.83% | top5 92.91% | lr 0.000428 | ips 692.2
[Train] Epoch 134 | loss 1.9529 | top1 76.09% | top5 91.35% | lr 0.000377 | ips 389.1
[Val] Epoch 134 | loss 1.9040 | top1 76.04% | top5 93.00% | lr 0.000377 | ips 1130.7
[Train] Epoch 135 | loss 1.9345 | top1 76.61% | top5 91.59% | lr 0.000329 | ips 389.3
[Val] Epoch 135 | loss 1.8941 | top1 76.32% | top5 93.18% | lr 0.000329 | ips 693.4
[Train] Epoch 136 | loss 1.9148 | top1 77.12% | top5 91.83% | lr 0.000284 | ips 388.8
[Val] Epoch 136 | loss 1.8905 | top1 76.42% | top5 93.11% | lr 0.000284 | ips 1002.0
[Train] Epoch 137 | loss 1.9009 | top1 77.54% | top5 91.99% | lr 0.000242 | ips 388.9
[Val] Epoch 137 | loss 1.8874 | top1 76.56% | top5 93.25% | lr 0.000242 | ips 674.4
[Train] Epoch 138 | loss 1.8805 | top1 78.05% | top5 92.24% | lr 0.000204 | ips 386.6
[Val] Epoch 138 | loss 1.8745 | top1 76.95% | top5 93.29% | lr 0.000204 | ips 1012.5
[Train] Epoch 139 | loss 1.8645 | top1 78.50% | top5 92.46% | lr 0.000169 | ips 390.4
[Val] Epoch 139 | loss 1.8693 | top1 77.16% | top5 93.39% | lr 0.000169 | ips 683.8
[Train] Epoch 140 | loss 1.8502 | top1 78.88% | top5 92.61% | lr 0.000137 | ips 394.4
[Val] Epoch 140 | loss 1.8649 | top1 77.22% | top5 93.50% | lr 0.000137 | ips 696.1
[Train] Epoch 141 | loss 1.8342 | top1 79.32% | top5 92.80% | lr 0.000108 | ips 391.1
[Val] Epoch 141 | loss 1.8613 | top1 76.99% | top5 93.54% | lr 0.000108 | ips 1134.5
[Train] Epoch 142 | loss 1.8199 | top1 79.72% | top5 92.97% | lr 0.000083 | ips 391.0
[Val] Epoch 142 | loss 1.8547 | top1 77.26% | top5 93.60% | lr 0.000083 | ips 621.8
[Train] Epoch 143 | loss 1.8099 | top1 80.01% | top5 93.07% | lr 0.000061 | ips 387.5
[Val] Epoch 143 | loss 1.8541 | top1 77.44% | top5 93.68% | lr 0.000061 | ips 657.1
[Train] Epoch 144 | loss 1.7984 | top1 80.29% | top5 93.22% | lr 0.000042 | ips 388.2
[Val] Epoch 144 | loss 1.8485 | top1 77.53% | top5 93.72% | lr 0.000042 | ips 530.3
[Train] Epoch 145 | loss 1.7911 | top1 80.50% | top5 93.30% | lr 0.000027 | ips 387.5
[Val] Epoch 145 | loss 1.8455 | top1 77.50% | top5 93.68% | lr 0.000027 | ips 1071.8
[Train] Epoch 146 | loss 1.7862 | top1 80.63% | top5 93.34% | lr 0.000015 | ips 387.5
[Val] Epoch 146 | loss 1.8482 | top1 77.62% | top5 93.74% | lr 0.000015 | ips 1079.0
[Train] Epoch 147 | loss 1.7824 | top1 80.74% | top5 93.39% | lr 0.000007 | ips 386.1
[Val] Epoch 147 | loss 1.8432 | top1 77.65% | top5 93.74% | lr 0.000007 | ips 664.7
[Train] Epoch 148 | loss 1.7791 | top1 80.85% | top5 93.39% | lr 0.000002 | ips 389.1
[Val] Epoch 148 | loss 1.8439 | top1 77.58% | top5 93.71% | lr 0.000002 | ips 658.6
[Train] Epoch 149 | loss 1.7797 | top1 80.83% | top5 93.39% | lr 0.000000 | ips 386.1
[Val] Epoch 149 | loss 1.8430 | top1 77.57% | top5 93.76% | lr 0.000000 | ips 1077.1
```
```
[Val] Epoch 150 | loss 1.8430 | top1 77.57% | top5 93.76% | lr 0.000000 | ips 385.1
```
[Val] Epoch 150 | loss 1.8430 | top1 77.57% | top5 93.76% | lr 0.000500 | ips 717.9
[Train] Epoch 150 | loss 1.9516 | top1 76.13% | top5 91.40% | lr 0.000470 | ips 380.4
[Val] Epoch 150 | loss 1.9378 | top1 75.21% | top5 92.48% | lr 0.000470 | ips 661.8
[Train] Epoch 151 | loss 1.9541 | top1 76.08% | top5 91.35% | lr 0.000422 | ips 390.3
[Val] Epoch 151 | loss 1.9288 | top1 75.43% | top5 92.65% | lr 0.000422 | ips 717.6
[Train] Epoch 152 | loss 1.9399 | top1 76.46% | top5 91.55% | lr 0.000377 | ips 393.9
[Val] Epoch 152 | loss 1.9174 | top1 75.82% | top5 92.82% | lr 0.000377 | ips 750.1
[Train] Epoch 153 | loss 1.9250 | top1 76.82% | top5 91.72% | lr 0.000334 | ips 388.1
[Val] Epoch 153 | loss 1.9091 | top1 76.12% | top5 92.92% | lr 0.000334 | ips 1114.1
[Train] Epoch 154 | loss 1.9075 | top1 77.29% | top5 91.94% | lr 0.000294 | ips 390.9
[Val] Epoch 154 | loss 1.9051 | top1 76.18% | top5 92.90% | lr 0.000294 | ips 1135.0
[Train] Epoch 155 | loss 1.8899 | top1 77.81% | top5 92.17% | lr 0.000257 | ips 391.0
[Val] Epoch 155 | loss 1.8946 | top1 76.29% | top5 93.13% | lr 0.000257 | ips 687.2
[Train] Epoch 156 | loss 1.8739 | top1 78.25% | top5 92.35% | lr 0.000221 | ips 389.9
[Val] Epoch 156 | loss 1.8882 | top1 76.54% | top5 93.20% | lr 0.000221 | ips 500.0
[Train] Epoch 157 | loss 1.8586 | top1 78.68% | top5 92.49% | lr 0.000189 | ips 390.3
[Val] Epoch 157 | loss 1.8788 | top1 76.56% | top5 93.31% | lr 0.000189 | ips 682.7
[Train] Epoch 158 | loss 1.8403 | top1 79.17% | top5 92.71% | lr 0.000159 | ips 390.6
[Val] Epoch 158 | loss 1.8728 | top1 76.87% | top5 93.44% | lr 0.000159 | ips 712.3
[Train] Epoch 159 | loss 1.8253 | top1 79.60% | top5 92.89% | lr 0.000131 | ips 391.0
[Val] Epoch 159 | loss 1.8683 | top1 76.96% | top5 93.47% | lr 0.000131 | ips 727.4
[Train] Epoch 160 | loss 1.8103 | top1 80.00% | top5 93.06% | lr 0.000106 | ips 391.1
[Val] Epoch 160 | loss 1.8658 | top1 77.14% | top5 93.44% | lr 0.000106 | ips 522.6
[Train] Epoch 161 | loss 1.7956 | top1 80.41% | top5 93.22% | lr 0.000084 | ips 388.4
[Val] Epoch 161 | loss 1.8632 | top1 77.21% | top5 93.52% | lr 0.000084 | ips 1074.8
[Train] Epoch 162 | loss 1.7820 | top1 80.80% | top5 93.37% | lr 0.000064 | ips 387.5
[Val] Epoch 162 | loss 1.8556 | top1 77.29% | top5 93.69% | lr 0.000064 | ips 1059.4
[Train] Epoch 163 | loss 1.7752 | top1 80.95% | top5 93.44% | lr 0.000047 | ips 386.7
[Val] Epoch 163 | loss 1.8523 | top1 77.35% | top5 93.65% | lr 0.000047 | ips 538.3
[Train] Epoch 164 | loss 1.7635 | top1 81.31% | top5 93.59% | lr 0.000033 | ips 389.7
[Val] Epoch 164 | loss 1.8478 | top1 77.48% | top5 93.64% | lr 0.000033 | ips 712.4
[Train] Epoch 165 | loss 1.7596 | top1 81.44% | top5 93.60% | lr 0.000021 | ips 383.9
[Val] Epoch 165 | loss 1.8485 | top1 77.62% | top5 93.64% | lr 0.000021 | ips 800.2
[Train] Epoch 166 | loss 1.7515 | top1 81.66% | top5 93.70% | lr 0.000012 | ips 386.8
[Val] Epoch 166 | loss 1.8487 | top1 77.66% | top5 93.75% | lr 0.000012 | ips 705.2
[Train] Epoch 167 | loss 1.7472 | top1 81.80% | top5 93.75% | lr 0.000005 | ips 386.5
[Val] Epoch 167 | loss 1.8458 | top1 77.66% | top5 93.79% | lr 0.000005 | ips 801.6
[Train] Epoch 168 | loss 1.7420 | top1 81.92% | top5 93.81% | lr 0.000001 | ips 387.0
[Val] Epoch 168 | loss 1.8468 | top1 77.75% | top5 93.72% | lr 0.000001 | ips 666.2
[Train] Epoch 169 | loss 1.7423 | top1 81.91% | top5 93.81% | lr 0.000000 | ips 389.2
[Val] Epoch 169 | loss 1.8474 | top1 77.71% | top5 93.70% | lr 0.000000 | ips 708.7
```
```
[Val] Epoch 170 | loss 1.8474 | top1 77.71% | top5 93.70% | lr 0.000500 | ips 785.0
[Train] Epoch 170 | loss 2.3794 | top1 65.20% | top5 85.54% | lr 0.001796 | ips 380.7
[Val] Epoch 170 | loss 2.1547 | top1 69.84% | top5 89.72% | lr 0.001796 | ips 601.1
[Train] Epoch 171 | loss 2.3178 | top1 66.75% | top5 86.37% | lr 0.001727 | ips 387.3
[Val] Epoch 171 | loss 2.1296 | top1 70.29% | top5 90.07% | lr 0.001727 | ips 698.9
[Train] Epoch 172 | loss 2.2795 | top1 67.76% | top5 86.94% | lr 0.001659 | ips 389.3
[Val] Epoch 172 | loss 2.0969 | top1 71.29% | top5 90.52% | lr 0.001659 | ips 576.7
[Train] Epoch 173 | loss 2.2492 | top1 68.53% | top5 87.37% | lr 0.001592 | ips 386.5
[Val] Epoch 173 | loss 2.0753 | top1 71.82% | top5 90.76% | lr 0.001592 | ips 675.7
[Train] Epoch 174 | loss 2.2270 | top1 69.11% | top5 87.68% | lr 0.001527 | ips 389.1
[Val] Epoch 174 | loss 2.0671 | top1 71.98% | top5 90.82% | lr 0.001527 | ips 577.7
[Train] Epoch 175 | loss 2.2079 | top1 69.54% | top5 87.95% | lr 0.001462 | ips 386.6
[Val] Epoch 175 | loss 2.0489 | top1 72.43% | top5 91.12% | lr 0.001462 | ips 1063.3
[Train] Epoch 176 | loss 2.1906 | top1 69.97% | top5 88.17% | lr 0.001399 | ips 383.0
[Val] Epoch 176 | loss 2.0440 | top1 72.62% | top5 91.23% | lr 0.001399 | ips 715.4
[Train] Epoch 177 | loss 2.1735 | top1 70.39% | top5 88.42% | lr 0.001337 | ips 390.3
[Val] Epoch 177 | loss 2.0161 | top1 73.23% | top5 91.61% | lr 0.001337 | ips 1124.4
[Train] Epoch 178 | loss 2.1630 | top1 70.67% | top5 88.54% | lr 0.001277 | ips 390.5
[Val] Epoch 178 | loss 2.0298 | top1 72.91% | top5 91.47% | lr 0.001277 | ips 702.2
[Train] Epoch 179 | loss 2.1460 | top1 71.13% | top5 88.81% | lr 0.001217 | ips 390.0
[Val] Epoch 179 | loss 2.0307 | top1 73.09% | top5 91.46% | lr 0.001217 | ips 1015.7
[Train] Epoch 180 | loss 2.1363 | top1 71.35% | top5 88.93% | lr 0.001159 | ips 390.7
[Val] Epoch 180 | loss 2.0090 | top1 73.45% | top5 91.82% | lr 0.001159 | ips 707.1
[Train] Epoch 181 | loss 2.1243 | top1 71.63% | top5 89.09% | lr 0.001102 | ips 392.9
[Val] Epoch 181 | loss 2.0039 | top1 73.62% | top5 91.74% | lr 0.001102 | ips 714.3
[Train] Epoch 182 | loss 2.1109 | top1 72.00% | top5 89.28% | lr 0.001046 | ips 393.0
[Val] Epoch 182 | loss 2.0010 | top1 73.57% | top5 91.81% | lr 0.001046 | ips 603.7
[Train] Epoch 183 | loss 2.0989 | top1 72.27% | top5 89.45% | lr 0.000992 | ips 385.3
[Val] Epoch 183 | loss 1.9919 | top1 73.76% | top5 91.81% | lr 0.000992 | ips 651.4
[Train] Epoch 184 | loss 2.0867 | top1 72.60% | top5 89.62% | lr 0.000939 | ips 392.9
[Val] Epoch 184 | loss 1.9782 | top1 74.03% | top5 92.18% | lr 0.000939 | ips 740.9
[Train] Epoch 185 | loss 2.0750 | top1 72.90% | top5 89.78% | lr 0.000888 | ips 390.0
[Val] Epoch 185 | loss 1.9785 | top1 74.23% | top5 92.16% | lr 0.000888 | ips 1017.5
[Train] Epoch 186 | loss 2.0624 | top1 73.20% | top5 89.95% | lr 0.000837 | ips 389.5
[Val] Epoch 186 | loss 1.9747 | top1 74.25% | top5 92.16% | lr 0.000837 | ips 744.1
[Train] Epoch 187 | loss 2.0502 | top1 73.54% | top5 90.06% | lr 0.000788 | ips 390.2
[Val] Epoch 187 | loss 1.9603 | top1 74.57% | top5 92.30% | lr 0.000788 | ips 1105.4
[Train] Epoch 188 | loss 2.0362 | top1 73.90% | top5 90.30% | lr 0.000741 | ips 390.3
[Val] Epoch 188 | loss 1.9537 | top1 74.80% | top5 92.41% | lr 0.000741 | ips 1134.4
[Train] Epoch 189 | loss 2.0220 | top1 74.27% | top5 90.48% | lr 0.000695 | ips 387.1
[Val] Epoch 189 | loss 1.9559 | top1 74.80% | top5 92.40% | lr 0.000695 | ips 655.5
[Train] Epoch 190 | loss 2.0113 | top1 74.57% | top5 90.62% | lr 0.000650 | ips 392.4
[Val] Epoch 190 | loss 1.9439 | top1 75.10% | top5 92.54% | lr 0.000650 | ips 715.5
[Train] Epoch 191 | loss 1.9996 | top1 74.83% | top5 90.77% | lr 0.000607 | ips 385.1
[Val] Epoch 191 | loss 1.9457 | top1 75.24% | top5 92.50% | lr 0.000607 | ips 678.4
[Train] Epoch 192 | loss 1.9860 | top1 75.22% | top5 90.94% | lr 0.000565 | ips 382.9
[Val] Epoch 192 | loss 1.9298 | top1 75.42% | top5 92.73% | lr 0.000565 | ips 575.1
[Train] Epoch 193 | loss 1.9735 | top1 75.58% | top5 91.08% | lr 0.000524 | ips 388.1
[Val] Epoch 193 | loss 1.9273 | top1 75.45% | top5 92.66% | lr 0.000524 | ips 740.0
[Train] Epoch 194 | loss 1.9593 | top1 75.92% | top5 91.25% | lr 0.000485 | ips 385.1
[Val] Epoch 194 | loss 1.9270 | top1 75.51% | top5 92.78% | lr 0.000485 | ips 695.8
[Train] Epoch 195 | loss 1.9438 | top1 76.34% | top5 91.45% | lr 0.000448 | ips 389.4
[Val] Epoch 195 | loss 1.9174 | top1 75.64% | top5 92.78% | lr 0.000448 | ips 731.0
[Train] Epoch 196 | loss 1.9307 | top1 76.67% | top5 91.61% | lr 0.000412 | ips 383.8
[Val] Epoch 196 | loss 1.9130 | top1 75.73% | top5 92.98% | lr 0.000412 | ips 570.8
[Train] Epoch 197 | loss 1.9183 | top1 77.01% | top5 91.77% | lr 0.000377 | ips 385.7
[Val] Epoch 197 | loss 1.9042 | top1 75.88% | top5 93.02% | lr 0.000377 | ips 694.3
[Train] Epoch 198 | loss 1.9041 | top1 77.34% | top5 91.95% | lr 0.000344 | ips 385.3
[Val] Epoch 198 | loss 1.9008 | top1 76.22% | top5 93.13% | lr 0.000344 | ips 561.8
[Train] Epoch 199 | loss 1.8897 | top1 77.76% | top5 92.12% | lr 0.000312 | ips 388.4
[Val] Epoch 199 | loss 1.8923 | top1 76.32% | top5 93.14% | lr 0.000312 | ips 730.7
[Train] Epoch 200 | loss 1.8753 | top1 78.15% | top5 92.32% | lr 0.000282 | ips 384.9
[Val] Epoch 200 | loss 1.8902 | top1 76.56% | top5 93.14% | lr 0.000282 | ips 1076.8
[Train] Epoch 201 | loss 1.8620 | top1 78.52% | top5 92.45% | lr 0.000253 | ips 385.2
[Val] Epoch 201 | loss 1.8840 | top1 76.54% | top5 93.26% | lr 0.000253 | ips 1069.9
[Train] Epoch 202 | loss 1.8473 | top1 78.95% | top5 92.61% | lr 0.000226 | ips 384.6
[Val] Epoch 202 | loss 1.8846 | top1 76.76% | top5 93.29% | lr 0.000226 | ips 451.9
[Train] Epoch 203 | loss 1.8322 | top1 79.31% | top5 92.79% | lr 0.000200 | ips 384.5
[Val] Epoch 203 | loss 1.8744 | top1 76.82% | top5 93.33% | lr 0.000200 | ips 686.7
[Train] Epoch 204 | loss 1.8233 | top1 79.61% | top5 92.91% | lr 0.000176 | ips 386.8
[Val] Epoch 204 | loss 1.8695 | top1 77.07% | top5 93.47% | lr 0.000176 | ips 605.7
[Train] Epoch 205 | loss 1.8078 | top1 80.01% | top5 93.07% | lr 0.000154 | ips 384.8
[Val] Epoch 205 | loss 1.8677 | top1 77.13% | top5 93.48% | lr 0.000154 | ips 733.6
[Train] Epoch 206 | loss 1.7951 | top1 80.38% | top5 93.24% | lr 0.000133 | ips 384.9
[Val] Epoch 206 | loss 1.8627 | top1 77.40% | top5 93.47% | lr 0.000133 | ips 1070.2
[Train] Epoch 207 | loss 1.7854 | top1 80.63% | top5 93.32% | lr 0.000113 | ips 384.6
[Val] Epoch 207 | loss 1.8601 | top1 77.21% | top5 93.52% | lr 0.000113 | ips 690.1
[Train] Epoch 208 | loss 1.7776 | top1 80.85% | top5 93.39% | lr 0.000095 | ips 387.0
[Val] Epoch 208 | loss 1.8586 | top1 77.30% | top5 93.53% | lr 0.000095 | ips 681.0
[Train] Epoch 209 | loss 1.7620 | top1 81.28% | top5 93.60% | lr 0.000079 | ips 384.9
[Val] Epoch 209 | loss 1.8536 | top1 77.47% | top5 93.57% | lr 0.000079 | ips 455.6
[Train] Epoch 210 | loss 1.7507 | top1 81.59% | top5 93.75% | lr 0.000064 | ips 387.1
[Val] Epoch 210 | loss 1.8503 | top1 77.53% | top5 93.68% | lr 0.000064 | ips 706.8
[Train] Epoch 211 | loss 1.7440 | top1 81.83% | top5 93.78% | lr 0.000050 | ips 385.1
[Val] Epoch 211 | loss 1.8518 | top1 77.48% | top5 93.69% | lr 0.000050 | ips 570.7
[Train] Epoch 212 | loss 1.7378 | top1 82.00% | top5 93.83% | lr 0.000039 | ips 387.3
[Val] Epoch 212 | loss 1.8493 | top1 77.64% | top5 93.73% | lr 0.000039 | ips 708.7
[Train] Epoch 213 | loss 1.7316 | top1 82.20% | top5 93.89% | lr 0.000028 | ips 387.2
[Val] Epoch 213 | loss 1.8472 | top1 77.62% | top5 93.72% | lr 0.000028 | ips 740.1
[Train] Epoch 214 | loss 1.7264 | top1 82.33% | top5 93.96% | lr 0.000020 | ips 384.5
[Val] Epoch 214 | loss 1.8447 | top1 77.73% | top5 93.78% | lr 0.000020 | ips 1076.9
[Train] Epoch 215 | loss 1.7210 | top1 82.49% | top5 94.01% | lr 0.000013 | ips 384.6
[Val] Epoch 215 | loss 1.8437 | top1 77.64% | top5 93.77% | lr 0.000013 | ips 560.5
[Train] Epoch 216 | loss 1.7186 | top1 82.54% | top5 94.04% | lr 0.000007 | ips 385.3
[Val] Epoch 216 | loss 1.8427 | top1 77.73% | top5 93.82% | lr 0.000007 | ips 713.2
[Train] Epoch 217 | loss 1.7160 | top1 82.60% | top5 94.05% | lr 0.000003 | ips 386.0
[Val] Epoch 217 | loss 1.8437 | top1 77.69% | top5 93.77% | lr 0.000003 | ips 797.7
[Train] Epoch 218 | loss 1.7106 | top1 82.76% | top5 94.15% | lr 0.000001 | ips 387.9
[Val] Epoch 218 | loss 1.8424 | top1 77.71% | top5 93.78% | lr 0.000001 | ips 683.4
[Train] Epoch 219 | loss 1.7146 | top1 82.65% | top5 94.06% | lr 0.000000 | ips 390.1
[Val] Epoch 219 | loss 1.8440 | top1 77.81% | top5 93.79% | lr 0.000000 | ips 703.2
```
```
[Val] Epoch 220 | loss 1.8440 | top1 77.81% | top5 93.79% | lr 0.000000 | ips 779.9
[Train] Epoch 220 | loss 1.7136 | top1 82.70% | top5 94.09% | lr 0.000000 | ips 381.9
[Val] Epoch 220 | loss 1.8415 | top1 77.71% | top5 93.77% | lr 0.000000 | ips 949.6
```
[Val] Epoch 220 | loss 1.8440 | top1 77.81% | top5 93.79% | lr 0.000000 | ips 790.9
[Train] Epoch 220 | loss 1.7132 | top1 82.69% | top5 94.10% | lr 0.000000 | ips 384.9
[Val] Epoch 220 | loss 1.8424 | top1 77.78% | top5 93.79% | lr 0.000000 | ips 733.4
[Train] Epoch 221 | loss 1.7122 | top1 82.73% | top5 94.10% | lr 0.000000 | ips 390.8
[Val] Epoch 221 | loss 1.8418 | top1 77.68% | top5 93.80% | lr 0.000000 | ips 700.3
[Train] Epoch 222 | loss 1.7143 | top1 82.69% | top5 94.07% | lr 0.000000 | ips 391.1
[Val] Epoch 222 | loss 1.8430 | top1 77.68% | top5 93.77% | lr 0.000000 | ips 1127.5
[Train] Epoch 223 | loss 1.7116 | top1 82.75% | top5 94.11% | lr 0.000000 | ips 387.7
[Val] Epoch 223 | loss 1.8398 | top1 77.73% | top5 93.84% | lr 0.000000 | ips 1131.6
[Train] Epoch 224 | loss 1.7151 | top1 82.67% | top5 94.07% | lr 0.000000 | ips 391.3
[Val] Epoch 224 | loss 1.8416 | top1 77.72% | top5 93.75% | lr 0.000000 | ips 1128.4
[Train] Epoch 225 | loss 1.7159 | top1 82.60% | top5 94.05% | lr 0.000000 | ips 391.4
[Val] Epoch 225 | loss 1.8414 | top1 77.75% | top5 93.77% | lr 0.000000 | ips 1128.4
[Train] Epoch 226 | loss 1.7144 | top1 82.68% | top5 94.07% | lr 0.000000 | ips 387.8
[Val] Epoch 226 | loss 1.8432 | top1 77.70% | top5 93.78% | lr 0.000000 | ips 1129.0
[Train] Epoch 227 | loss 1.7126 | top1 82.72% | top5 94.12% | lr 0.000000 | ips 391.3
[Val] Epoch 227 | loss 1.8433 | top1 77.71% | top5 93.79% | lr 0.000000 | ips 1118.8
[Train] Epoch 228 | loss 1.7128 | top1 82.67% | top5 94.12% | lr 0.000000 | ips 391.4
[Val] Epoch 228 | loss 1.8419 | top1 77.82% | top5 93.82% | lr 0.000000 | ips 679.0
[Train] Epoch 229 | loss 1.7140 | top1 82.70% | top5 94.05% | lr 0.000000 | ips 393.7
[Val] Epoch 229 | loss 1.8425 | top1 77.67% | top5 93.79% | lr 0.000000 | ips 572.2
[Train] Epoch 230 | loss 1.7130 | top1 82.72% | top5 94.08% | lr 0.000000 | ips 390.5
[Val] Epoch 230 | loss 1.8437 | top1 77.77% | top5 93.76% | lr 0.000000 | ips 810.8
[Train] Epoch 231 | loss 1.7151 | top1 82.68% | top5 94.04% | lr 0.000000 | ips 389.5
[Val] Epoch 231 | loss 1.8433 | top1 77.74% | top5 93.75% | lr 0.000000 | ips 708.7
[Train] Epoch 232 | loss 1.7124 | top1 82.70% | top5 94.10% | lr 0.000000 | ips 390.8
[Val] Epoch 232 | loss 1.8417 | top1 77.64% | top5 93.74% | lr 0.000000 | ips 523.5
[Train] Epoch 233 | loss 1.7137 | top1 82.69% | top5 94.08% | lr 0.000000 | ips 392.3
[Val] Epoch 233 | loss 1.8429 | top1 77.72% | top5 93.81% | lr 0.000000 | ips 1140.3
[Train] Epoch 234 | loss 1.7160 | top1 82.63% | top5 94.05% | lr 0.000000 | ips 389.2
[Val] Epoch 234 | loss 1.8430 | top1 77.71% | top5 93.75% | lr 0.000000 | ips 1088.6
```

```
</details>

<details>
<summary><strong>6.A.2 Model summary</strong></summary>

```text
============================================================================================================================================
Layer (type:depth-idx)                   Input Shape               Output Shape              Param #                   Kernel Shape
============================================================================================================================================
ResNet                                   [64, 3, 224, 224]         [64, 1000]                --                        --
├─Conv2d: 1-1                            [64, 3, 224, 224]         [64, 64, 112, 112]        9,408                     [7, 7]
├─BatchNorm2d: 1-2                       [64, 64, 112, 112]        [64, 64, 112, 112]        128                       --
├─ReLU: 1-3                              [64, 64, 112, 112]        [64, 64, 112, 112]        --                        --
├─MaxPool2d: 1-4                         [64, 64, 112, 112]        [64, 64, 56, 56]          --                        3
├─Sequential: 1-5                        [64, 64, 56, 56]          [64, 256, 56, 56]         --                        --
│    └─Bottleneck: 2-1                   [64, 64, 56, 56]          [64, 256, 56, 56]         --                        --
│    │    └─Conv2d: 3-1                  [64, 64, 56, 56]          [64, 64, 56, 56]          4,096                     [1, 1]
│    │    └─BatchNorm2d: 3-2             [64, 64, 56, 56]          [64, 64, 56, 56]          128                       --
│    │    └─ReLU: 3-3                    [64, 64, 56, 56]          [64, 64, 56, 56]          --                        --
│    │    └─Conv2d: 3-4                  [64, 64, 56, 56]          [64, 64, 56, 56]          36,864                    [3, 3]
│    │    └─BatchNorm2d: 3-5             [64, 64, 56, 56]          [64, 64, 56, 56]          128                       --
│    │    └─ReLU: 3-6                    [64, 64, 56, 56]          [64, 64, 56, 56]          --                        --
│    │    └─Conv2d: 3-7                  [64, 64, 56, 56]          [64, 256, 56, 56]         16,384                    [1, 1]
│    │    └─BatchNorm2d: 3-8             [64, 256, 56, 56]         [64, 256, 56, 56]         512                       --
│    │    └─Sequential: 3-9              [64, 64, 56, 56]          [64, 256, 56, 56]         16,896                    --
│    │    └─ReLU: 3-10                   [64, 256, 56, 56]         [64, 256, 56, 56]         --                        --
│    └─Bottleneck: 2-2                   [64, 256, 56, 56]         [64, 256, 56, 56]         --                        --
│    │    └─Conv2d: 3-11                 [64, 256, 56, 56]         [64, 64, 56, 56]          16,384                    [1, 1]
│    │    └─BatchNorm2d: 3-12            [64, 64, 56, 56]          [64, 64, 56, 56]          128                       --
│    │    └─ReLU: 3-13                   [64, 64, 56, 56]          [64, 64, 56, 56]          --                        --
│    │    └─Conv2d: 3-14                 [64, 64, 56, 56]          [64, 64, 56, 56]          36,864                    [3, 3]
│    │    └─BatchNorm2d: 3-15            [64, 64, 56, 56]          [64, 64, 56, 56]          128                       --
│    │    └─ReLU: 3-16                   [64, 64, 56, 56]          [64, 64, 56, 56]          --                        --
│    │    └─Conv2d: 3-17                 [64, 64, 56, 56]          [64, 256, 56, 56]         16,384                    [1, 1]
│    │    └─BatchNorm2d: 3-18            [64, 256, 56, 56]         [64, 256, 56, 56]         512                       --
│    │    └─ReLU: 3-19                   [64, 256, 56, 56]         [64, 256, 56, 56]         --                        --
│    └─Bottleneck: 2-3                   [64, 256, 56, 56]         [64, 256, 56, 56]         --                        --
│    │    └─Conv2d: 3-20                 [64, 256, 56, 56]         [64, 64, 56, 56]          16,384                    [1, 1]
│    │    └─BatchNorm2d: 3-21            [64, 64, 56, 56]          [64, 64, 56, 56]          128                       --
│    │    └─ReLU: 3-22                   [64, 64, 56, 56]          [64, 64, 56, 56]          --                        --
│    │    └─Conv2d: 3-23                 [64, 64, 56, 56]          [64, 64, 56, 56]          36,864                    [3, 3]
│    │    └─BatchNorm2d: 3-24            [64, 64, 56, 56]          [64, 64, 56, 56]          128                       --
│    │    └─ReLU: 3-25                   [64, 64, 56, 56]          [64, 64, 56, 56]          --                        --
│    │    └─Conv2d: 3-26                 [64, 64, 56, 56]          [64, 256, 56, 56]         16,384                    [1, 1]
│    │    └─BatchNorm2d: 3-27            [64, 256, 56, 56]         [64, 256, 56, 56]         512                       --
│    │    └─ReLU: 3-28                   [64, 256, 56, 56]         [64, 256, 56, 56]         --                        --
├─Sequential: 1-6                        [64, 256, 56, 56]         [64, 512, 28, 28]         --                        --
│    └─Bottleneck: 2-4                   [64, 256, 56, 56]         [64, 512, 28, 28]         --                        --
│    │    └─Conv2d: 3-29                 [64, 256, 56, 56]         [64, 128, 56, 56]         32,768                    [1, 1]
│    │    └─BatchNorm2d: 3-30            [64, 128, 56, 56]         [64, 128, 56, 56]         256                       --
│    │    └─ReLU: 3-31                   [64, 128, 56, 56]         [64, 128, 56, 56]         --                        --
│    │    └─Conv2d: 3-32                 [64, 128, 56, 56]         [64, 128, 28, 28]         147,456                   [3, 3]
│    │    └─BatchNorm2d: 3-33            [64, 128, 28, 28]         [64, 128, 28, 28]         256                       --
│    │    └─ReLU: 3-34                   [64, 128, 28, 28]         [64, 128, 28, 28]         --                        --
│    │    └─Conv2d: 3-35                 [64, 128, 28, 28]         [64, 512, 28, 28]         65,536                    [1, 1]
│    │    └─BatchNorm2d: 3-36            [64, 512, 28, 28]         [64, 512, 28, 28]         1,024                     --
│    │    └─Sequential: 3-37             [64, 256, 56, 56]         [64, 512, 28, 28]         132,096                   --
│    │    └─ReLU: 3-38                   [64, 512, 28, 28]         [64, 512, 28, 28]         --                        --
│    └─Bottleneck: 2-5                   [64, 512, 28, 28]         [64, 512, 28, 28]         --                        --
│    │    └─Conv2d: 3-39                 [64, 512, 28, 28]         [64, 128, 28, 28]         65,536                    [1, 1]
│    │    └─BatchNorm2d: 3-40            [64, 128, 28, 28]         [64, 128, 28, 28]         256                       --
│    │    └─ReLU: 3-41                   [64, 128, 28, 28]         [64, 128, 28, 28]         --                        --
│    │    └─Conv2d: 3-42                 [64, 128, 28, 28]         [64, 128, 28, 28]         147,456                   [3, 3]
│    │    └─BatchNorm2d: 3-43            [64, 128, 28, 28]         [64, 128, 28, 28]         256                       --
│    │    └─ReLU: 3-44                   [64, 128, 28, 28]         [64, 128, 28, 28]         --                        --
│    │    └─Conv2d: 3-45                 [64, 128, 28, 28]         [64, 512, 28, 28]         65,536                    [1, 1]
│    │    └─BatchNorm2d: 3-46            [64, 512, 28, 28]         [64, 512, 28, 28]         1,024                     --
│    │    └─ReLU: 3-47                   [64, 512, 28, 28]         [64, 512, 28, 28]         --                        --
│    └─Bottleneck: 2-6                   [64, 512, 28, 28]         [64, 512, 28, 28]         --                        --
│    │    └─Conv2d: 3-48                 [64, 512, 28, 28]         [64, 128, 28, 28]         65,536                    [1, 1]
│    │    └─BatchNorm2d: 3-49            [64, 128, 28, 28]         [64, 128, 28, 28]         256                       --
│    │    └─ReLU: 3-50                   [64, 128, 28, 28]         [64, 128, 28, 28]         --                        --
│    │    └─Conv2d: 3-51                 [64, 128, 28, 28]         [64, 128, 28, 28]         147,456                   [3, 3]
│    │    └─BatchNorm2d: 3-52            [64, 128, 28, 28]         [64, 128, 28, 28]         256                       --
│    │    └─ReLU: 3-53                   [64, 128, 28, 28]         [64, 128, 28, 28]         --                        --
│    │    └─Conv2d: 3-54                 [64, 128, 28, 28]         [64, 512, 28, 28]         65,536                    [1, 1]
│    │    └─BatchNorm2d: 3-55            [64, 512, 28, 28]         [64, 512, 28, 28]         1,024                     --
│    │    └─ReLU: 3-56                   [64, 512, 28, 28]         [64, 512, 28, 28]         --                        --
│    └─Bottleneck: 2-7                   [64, 512, 28, 28]         [64, 512, 28, 28]         --                        --
│    │    └─Conv2d: 3-57                 [64, 512, 28, 28]         [64, 128, 28, 28]         65,536                    [1, 1]
│    │    └─BatchNorm2d: 3-58            [64, 128, 28, 28]         [64, 128, 28, 28]         256                       --
│    │    └─ReLU: 3-59                   [64, 128, 28, 28]         [64, 128, 28, 28]         --                        --
│    │    └─Conv2d: 3-60                 [64, 128, 28, 28]         [64, 128, 28, 28]         147,456                   [3, 3]
│    │    └─BatchNorm2d: 3-61            [64, 128, 28, 28]         [64, 128, 28, 28]         256                       --
│    │    └─ReLU: 3-62                   [64, 128, 28, 28]         [64, 128, 28, 28]         --                        --
│    │    └─Conv2d: 3-63                 [64, 128, 28, 28]         [64, 512, 28, 28]         65,536                    [1, 1]
│    │    └─BatchNorm2d: 3-64            [64, 512, 28, 28]         [64, 512, 28, 28]         1,024                     --
│    │    └─ReLU: 3-65                   [64, 512, 28, 28]         [64, 512, 28, 28]         --                        --
├─Sequential: 1-7                        [64, 512, 28, 28]         [64, 1024, 14, 14]        --                        --
│    └─Bottleneck: 2-8                   [64, 512, 28, 28]         [64, 1024, 14, 14]        --                        --
│    │    └─Conv2d: 3-66                 [64, 512, 28, 28]         [64, 256, 28, 28]         131,072                   [1, 1]
│    │    └─BatchNorm2d: 3-67            [64, 256, 28, 28]         [64, 256, 28, 28]         512                       --
│    │    └─ReLU: 3-68                   [64, 256, 28, 28]         [64, 256, 28, 28]         --                        --
│    │    └─Conv2d: 3-69                 [64, 256, 28, 28]         [64, 256, 14, 14]         589,824                   [3, 3]
│    │    └─BatchNorm2d: 3-70            [64, 256, 14, 14]         [64, 256, 14, 14]         512                       --
│    │    └─ReLU: 3-71                   [64, 256, 14, 14]         [64, 256, 14, 14]         --                        --
│    │    └─Conv2d: 3-72                 [64, 256, 14, 14]         [64, 1024, 14, 14]        262,144                   [1, 1]
│    │    └─BatchNorm2d: 3-73            [64, 1024, 14, 14]        [64, 1024, 14, 14]        2,048                     --
│    │    └─Sequential: 3-74             [64, 512, 28, 28]         [64, 1024, 14, 14]        526,336                   --
│    │    └─ReLU: 3-75                   [64, 1024, 14, 14]        [64, 1024, 14, 14]        --                        --
│    └─Bottleneck: 2-9                   [64, 1024, 14, 14]        [64, 1024, 14, 14]        --                        --
│    │    └─Conv2d: 3-76                 [64, 1024, 14, 14]        [64, 256, 14, 14]         262,144                   [1, 1]
│    │    └─BatchNorm2d: 3-77            [64, 256, 14, 14]         [64, 256, 14, 14]         512                       --
│    │    └─ReLU: 3-78                   [64, 256, 14, 14]         [64, 256, 14, 14]         --                        --
│    │    └─Conv2d: 3-79                 [64, 256, 14, 14]         [64, 256, 14, 14]         589,824                   [3, 3]
│    │    └─BatchNorm2d: 3-80            [64, 256, 14, 14]         [64, 256, 14, 14]         512                       --
│    │    └─ReLU: 3-81                   [64, 256, 14, 14]         [64, 256, 14, 14]         --                        --
│    │    └─Conv2d: 3-82                 [64, 256, 14, 14]         [64, 1024, 14, 14]        262,144                   [1, 1]
│    │    └─BatchNorm2d: 3-83            [64, 1024, 14, 14]        [64, 1024, 14, 14]        2,048                     --
│    │    └─ReLU: 3-84                   [64, 1024, 14, 14]        [64, 1024, 14, 14]        --                        --
│    └─Bottleneck: 2-10                  [64, 1024, 14, 14]        [64, 1024, 14, 14]        --                        --
│    │    └─Conv2d: 3-85                 [64, 1024, 14, 14]        [64, 256, 14, 14]         262,144                   [1, 1]
│    │    └─BatchNorm2d: 3-86            [64, 256, 14, 14]         [64, 256, 14, 14]         512                       --
│    │    └─ReLU: 3-87                   [64, 256, 14, 14]         [64, 256, 14, 14]         --                        --
│    │    └─Conv2d: 3-88                 [64, 256, 14, 14]         [64, 256, 14, 14]         589,824                   [3, 3]
│    │    └─BatchNorm2d: 3-89            [64, 256, 14, 14]         [64, 256, 14, 14]         512                       --
│    │    └─ReLU: 3-90                   [64, 256, 14, 14]         [64, 256, 14, 14]         --                        --
│    │    └─Conv2d: 3-91                 [64, 256, 14, 14]         [64, 1024, 14, 14]        262,144                   [1, 1]
│    │    └─BatchNorm2d: 3-92            [64, 1024, 14, 14]        [64, 1024, 14, 14]        2,048                     --
│    │    └─ReLU: 3-93                   [64, 1024, 14, 14]        [64, 1024, 14, 14]        --                        --
│    └─Bottleneck: 2-11                  [64, 1024, 14, 14]        [64, 1024, 14, 14]        --                        --
│    │    └─Conv2d: 3-94                 [64, 1024, 14, 14]        [64, 256, 14, 14]         262,144                   [1, 1]
│    │    └─BatchNorm2d: 3-95            [64, 256, 14, 14]         [64, 256, 14, 14]         512                       --
│    │    └─ReLU: 3-96                   [64, 256, 14, 14]         [64, 256, 14, 14]         --                        --
│    │    └─Conv2d: 3-97                 [64, 256, 14, 14]         [64, 256, 14, 14]         589,824                   [3, 3]
│    │    └─BatchNorm2d: 3-98            [64, 256, 14, 14]         [64, 256, 14, 14]         512                       --
│    │    └─ReLU: 3-99                   [64, 256, 14, 14]         [64, 256, 14, 14]         --                        --
│    │    └─Conv2d: 3-100                [64, 256, 14, 14]         [64, 1024, 14, 14]        262,144                   [1, 1]
│    │    └─BatchNorm2d: 3-101           [64, 1024, 14, 14]        [64, 1024, 14, 14]        2,048                     --
│    │    └─ReLU: 3-102                  [64, 1024, 14, 14]        [64, 1024, 14, 14]        --                        --
│    └─Bottleneck: 2-12                  [64, 1024, 14, 14]        [64, 1024, 14, 14]        --                        --
│    │    └─Conv2d: 3-103                [64, 1024, 14, 14]        [64, 256, 14, 14]         262,144                   [1, 1]
│    │    └─BatchNorm2d: 3-104           [64, 256, 14, 14]         [64, 256, 14, 14]         512                       --
│    │    └─ReLU: 3-105                  [64, 256, 14, 14]         [64, 256, 14, 14]         --                        --
│    │    └─Conv2d: 3-106                [64, 256, 14, 14]         [64, 256, 14, 14]         589,824                   [3, 3]
│    │    └─BatchNorm2d: 3-107           [64, 256, 14, 14]         [64, 256, 14, 14]         512                       --
│    │    └─ReLU: 3-108                  [64, 256, 14, 14]         [64, 256, 14, 14]         --                        --
│    │    └─Conv2d: 3-109                [64, 256, 14, 14]         [64, 1024, 14, 14]        262,144                   [1, 1]
│    │    └─BatchNorm2d: 3-110           [64, 1024, 14, 14]        [64, 1024, 14, 14]        2,048                     --
│    │    └─ReLU: 3-111                  [64, 1024, 14, 14]        [64, 1024, 14, 14]        --                        --
│    └─Bottleneck: 2-13                  [64, 1024, 14, 14]        [64, 1024, 14, 14]        --                        --
│    │    └─Conv2d: 3-112                [64, 1024, 14, 14]        [64, 256, 14, 14]         262,144                   [1, 1]
│    │    └─BatchNorm2d: 3-113           [64, 256, 14, 14]         [64, 256, 14, 14]         512                       --
│    │    └─ReLU: 3-114                  [64, 256, 14, 14]         [64, 256, 14, 14]         --                        --
│    │    └─Conv2d: 3-115                [64, 256, 14, 14]         [64, 256, 14, 14]         589,824                   [3, 3]
│    │    └─BatchNorm2d: 3-116           [64, 256, 14, 14]         [64, 256, 14, 14]         512                       --
│    │    └─ReLU: 3-117                  [64, 256, 14, 14]         [64, 256, 14, 14]         --                        --
│    │    └─Conv2d: 3-118                [64, 256, 14, 14]         [64, 1024, 14, 14]        262,144                   [1, 1]
│    │    └─BatchNorm2d: 3-119           [64, 1024, 14, 14]        [64, 1024, 14, 14]        2,048                     --
│    │    └─ReLU: 3-120                  [64, 1024, 14, 14]        [64, 1024, 14, 14]        --                        --
├─Sequential: 1-8                        [64, 1024, 14, 14]        [64, 2048, 7, 7]          --                        --
│    └─Bottleneck: 2-14                  [64, 1024, 14, 14]        [64, 2048, 7, 7]          --                        --
│    │    └─Conv2d: 3-121                [64, 1024, 14, 14]        [64, 512, 14, 14]         524,288                   [1, 1]
│    │    └─BatchNorm2d: 3-122           [64, 512, 14, 14]         [64, 512, 14, 14]         1,024                     --
│    │    └─ReLU: 3-123                  [64, 512, 14, 14]         [64, 512, 14, 14]         --                        --
│    │    └─Conv2d: 3-124                [64, 512, 14, 14]         [64, 512, 7, 7]           2,359,296                 [3, 3]
│    │    └─BatchNorm2d: 3-125           [64, 512, 7, 7]           [64, 512, 7, 7]           1,024                     --
│    │    └─ReLU: 3-126                  [64, 512, 7, 7]           [64, 512, 7, 7]           --                        --
│    │    └─Conv2d: 3-127                [64, 512, 7, 7]           [64, 2048, 7, 7]          1,048,576                 [1, 1]
│    │    └─BatchNorm2d: 3-128           [64, 2048, 7, 7]          [64, 2048, 7, 7]          4,096                     --
│    │    └─Sequential: 3-129            [64, 1024, 14, 14]        [64, 2048, 7, 7]          2,101,248                 --
│    │    └─ReLU: 3-130                  [64, 2048, 7, 7]          [64, 2048, 7, 7]          --                        --
│    └─Bottleneck: 2-15                  [64, 2048, 7, 7]          [64, 2048, 7, 7]          --                        --
│    │    └─Conv2d: 3-131                [64, 2048, 7, 7]          [64, 512, 7, 7]           1,048,576                 [1, 1]
│    │    └─BatchNorm2d: 3-132           [64, 512, 7, 7]           [64, 512, 7, 7]           1,024                     --
│    │    └─ReLU: 3-133                  [64, 512, 7, 7]           [64, 512, 7, 7]           --                        --
│    │    └─Conv2d: 3-134                [64, 512, 7, 7]           [64, 512, 7, 7]           2,359,296                 [3, 3]
│    │    └─BatchNorm2d: 3-135           [64, 512, 7, 7]           [64, 512, 7, 7]           1,024                     --
│    │    └─ReLU: 3-136                  [64, 512, 7, 7]           [64, 512, 7, 7]           --                        --
│    │    └─Conv2d: 3-137                [64, 512, 7, 7]           [64, 2048, 7, 7]          1,048,576                 [1, 1]
│    │    └─BatchNorm2d: 3-138           [64, 2048, 7, 7]          [64, 2048, 7, 7]          4,096                     --
│    │    └─ReLU: 3-139                  [64, 2048, 7, 7]          [64, 2048, 7, 7]          --                        --
│    └─Bottleneck: 2-16                  [64, 2048, 7, 7]          [64, 2048, 7, 7]          --                        --
│    │    └─Conv2d: 3-140                [64, 2048, 7, 7]          [64, 512, 7, 7]           1,048,576                 [1, 1]
│    │    └─BatchNorm2d: 3-141           [64, 512, 7, 7]           [64, 512, 7, 7]           1,024                     --
│    │    └─ReLU: 3-142                  [64, 512, 7, 7]           [64, 512, 7, 7]           --                        --
│    │    └─Conv2d: 3-143                [64, 512, 7, 7]           [64, 512, 7, 7]           2,359,296                 [3, 3]
│    │    └─BatchNorm2d: 3-144           [64, 512, 7, 7]           [64, 512, 7, 7]           1,024                     --
│    │    └─ReLU: 3-145                  [64, 512, 7, 7]           [64, 512, 7, 7]           --                        --
│    │    └─Conv2d: 3-146                [64, 512, 7, 7]           [64, 2048, 7, 7]          1,048,576                 [1, 1]
│    │    └─BatchNorm2d: 3-147           [64, 2048, 7, 7]          [64, 2048, 7, 7]          4,096                     --
│    │    └─ReLU: 3-148                  [64, 2048, 7, 7]          [64, 2048, 7, 7]          --                        --
├─AdaptiveAvgPool2d: 1-9                 [64, 2048, 7, 7]          [64, 2048, 1, 1]          --                        --
├─Linear: 1-10                           [64, 2048]                [64, 1000]                2,049,000                 --
============================================================================================================================================
Total params: 25,557,032
Trainable params: 25,557,032
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 261.71
============================================================================================================================================
Input size (MB): 38.54
Forward/backward pass size (MB): 11381.23
Params size (MB): 102.23
Estimated Total Size (MB): 11521.99
============================================================================================================================================
```
</details>


---

### ☁️ B) AWS Training (g5.xlarge A10G)

**Infrastructure**

| Component | Specification |
|------------|----------------|
| **Instance Type** | `g5.xlarge` (4 vCPU / 16 GiB RAM) |
| **GPU** | NVIDIA A10G Tensor Core (24 GB) |
| **Region** | ap-south-1 (Mumbai) |
| **Storage** | 500 GB EBS Volume mounted → `/mnt/imagenet1k` |
| **Approval** | Quota raised for 8 vCPUs of “All G and VT Spot Requests” |

**Training Profile**

| Parameter | Value |
|------------|--------|
| **Epochs Completed** | 195 |
| **Batch Size** | 256 |
| **Max LR** | 0.125 |
| **Training Time** | ~30 min / epoch  (≈ 90 hours total) |
| **Optimizations Used** | DALI pipeline • OneCycle LR • AMP • Efficient Data Loading • Distributed Sampling • tmux management • Memory/Error handling |

**Auto-filled Metrics** (from `out/imagenet1kfull_g5x_1gpu_dali_nvme_lr0p125_bs256_e150_work6/train_log.csv`)

| Metric | Value |
|--------|--------|
| Training Top-1 | 81.01% |
| Training Top-5 | 93.54% |
| Validation Top-1 | 77.66% |
| Validation Top-5 | 93.84% |


**Runtime Snapshots & Logs**

**CLI snapshot**
![AWS Training CLI](images/AWS_Training_Image1.png)

**GPU Usage Training**
![GPU Usage](images/AWS_EC2_gpu_usage_training.png)

**EC2 CPU Usage Training**
![EC2 Memory Monitor](images/AWS_EC2_CPU_memory_usage_training.png)

**Epoch progress**
![Epoch 1–12](images/AWS_Training_Image2.png)
![Epoch 40-48](images/AWS_Training_Image3.png)
![Epoch 120-132](images/AWS_Training_Image4.png)
![Epoch 185-195](images/Local_Training_Image5.png)

**wanbb metrics**

A consolidated W&B report combining all AWS Spot-instance runs (0–195 epochs):
[View full W&B Report — *ImageNet1k Full Combined ResNet-50 AWS Training*](https://api.wandb.ai/links/sagar1doshi-bits-pilani/i0uhx5xj)
![Trainining and Evaluation Progress](images/AWS_Training_Image7_wanb_report.png)


<details>
<summary><strong>6.B.1 Training logs (markdown)</strong></summary>

```markdown
# Training Log
| epoch | phase | loss | top1 | top5 | lr | imgs/s |
|---:|---|---:|---:|---:|---:|---:|
| -1 | val | 41.1403 | 0.10 | 0.50 | 0.00500 | |
| 0 | train | 5.9682 | 4.74 | 13.47 | 0.00515 | 810 |
| 0 | val | 5.4989 | 8.61 | 22.27 | 0.00515 | |
| 1 | train | 4.9048 | 15.66 | 34.77 | 0.00558 | 811 |
| 1 | val | 4.5541 | 20.75 | 43.19 | 0.00558 | |
| 2 | train | 4.3336 | 24.77 | 47.84 | 0.00631 | 810 |
| 2 | val | 4.0242 | 29.68 | 55.14 | 0.00631 | |
| 3 | train | 3.9653 | 31.53 | 55.95 | 0.00732 | 810 |
| 3 | val | 3.7891 | 34.51 | 60.04 | 0.00732 | |
| 4 | train | 3.7164 | 36.37 | 61.30 | 0.00862 | 810 |
| 4 | val | 3.4566 | 40.80 | 66.68 | 0.00862 | |
| 5 | train | 3.5379 | 40.03 | 64.95 | 0.01019 | 809 |
| 5 | val | 3.3149 | 43.41 | 69.65 | 0.01019 | |
| 6 | train | 3.4009 | 42.87 | 67.74 | 0.01202 | 810 |
| 6 | val | 3.1670 | 46.80 | 72.73 | 0.01202 | |
| 7 | train | 3.2984 | 44.98 | 69.74 | 0.01412 | 811 |
| 7 | val | 3.0751 | 48.62 | 74.30 | 0.01412 | |
| 8 | train | 3.2179 | 46.71 | 71.28 | 0.01646 | 810 |
| 8 | val | 3.0401 | 49.10 | 75.13 | 0.01646 | |
| 9 | train | 3.1560 | 47.98 | 72.49 | 0.01904 | 811 |
| 9 | val | 2.9504 | 51.58 | 76.88 | 0.01904 | |
| 10 | train | 3.1068 | 49.05 | 73.39 | 0.02184 | 810 |
| 10 | val | 2.9202 | 51.89 | 77.14 | 0.02184 | |
| 11 | train | 3.0689 | 49.84 | 74.08 | 0.02485 | 810 |
| 11 | val | 2.9098 | 52.09 | 77.51 | 0.02485 | |
| 12 | train | 3.0394 | 50.44 | 74.65 | 0.02806 | 810 |
| 12 | val | 2.8526 | 53.55 | 78.58 | 0.02806 | |
| 13 | train | 3.0173 | 50.88 | 75.04 | 0.03145 | 810 |
| 13 | val | 2.8124 | 54.32 | 79.28 | 0.03145 | |
| 14 | train | 2.9946 | 51.36 | 75.39 | 0.03500 | 810 |
| 14 | val | 2.8930 | 52.24 | 77.81 | 0.03500 | |
| 15 | train | 2.9813 | 51.60 | 75.69 | 0.03870 | 810 |
| 15 | val | 2.7875 | 54.83 | 79.78 | 0.03870 | |
| 16 | train | 2.9700 | 51.85 | 75.83 | 0.04252 | 810 |
| 16 | val | 2.7541 | 55.54 | 80.37 | 0.04252 | |
| 17 | train | 2.9540 | 52.20 | 76.13 | 0.04646 | 811 |
| 17 | val | 2.8202 | 54.08 | 79.20 | 0.04646 | |
# Training Log
| epoch | phase | loss | top1 | top5 | lr | imgs/s |
|---:|---|---:|---:|---:|---:|---:|
| -1 | val | 2.8202 | 54.08 | 79.20 | 0.04646 | |
| 18 | train | 2.9435 | 52.39 | 76.32 | 0.05049 | 810 |
| 18 | val | 2.8377 | 53.49 | 78.80 | 0.05049 | |
| 19 | train | 2.9330 | 52.61 | 76.51 | 0.05458 | 811 |
| 19 | val | 2.8829 | 52.60 | 77.96 | 0.05458 | |
| 20 | train | 2.9252 | 52.77 | 76.61 | 0.05873 | 811 |
| 20 | val | 2.7714 | 55.02 | 80.17 | 0.05873 | |
| 21 | train | 2.9203 | 52.89 | 76.72 | 0.06291 | 811 |
| 21 | val | 2.7695 | 55.19 | 80.03 | 0.06291 | |
| 22 | train | 2.9152 | 53.07 | 76.82 | 0.06709 | 811 |
| 22 | val | 2.7361 | 55.89 | 80.52 | 0.06709 | |
| 23 | train | 2.9096 | 53.16 | 76.90 | 0.07127 | 811 |
| 23 | val | 2.7907 | 54.97 | 79.59 | 0.07127 | |
| 24 | train | 2.9095 | 53.15 | 76.92 | 0.07542 | 811 |
| 24 | val | 2.7316 | 56.20 | 80.70 | 0.07542 | |
| 25 | train | 2.9049 | 53.19 | 77.04 | 0.07952 | 811 |
| 25 | val | 2.9373 | 51.54 | 76.87 | 0.07952 | |
| 26 | train | 2.9000 | 53.35 | 77.11 | 0.08354 | 811 |
| 26 | val | 2.8201 | 53.96 | 79.19 | 0.08354 | |
| 27 | train | 2.8974 | 53.38 | 77.12 | 0.08748 | 811 |
| 27 | val | 2.7450 | 55.66 | 80.52 | 0.08748 | |
| 28 | train | 2.8951 | 53.41 | 77.13 | 0.09130 | 811 |
| 28 | val | 2.7648 | 55.40 | 80.26 | 0.09130 | |
| 29 | train | 2.8926 | 53.44 | 77.22 | 0.09500 | 811 |
| 29 | val | 2.8171 | 53.91 | 79.13 | 0.09500 | |
| 30 | train | 2.8909 | 53.48 | 77.22 | 0.09855 | 811 |
| 30 | val | 2.7663 | 55.38 | 80.12 | 0.09855 | |
| 31 | train | 2.8881 | 53.60 | 77.31 | 0.10194 | 811 |
| 31 | val | 2.7848 | 54.67 | 79.85 | 0.10194 | |
| 32 | train | 2.8876 | 53.55 | 77.31 | 0.10515 | 811 |
| 32 | val | 2.7497 | 55.24 | 80.35 | 0.10515 | |
| 33 | train | 2.8850 | 53.64 | 77.32 | 0.10816 | 811 |
| 33 | val | 2.7372 | 55.74 | 80.63 | 0.10816 | |
| 34 | train | 2.8808 | 53.74 | 77.41 | 0.11096 | 811 |
| 34 | val | 2.7789 | 54.55 | 79.86 | 0.11096 | |
| 35 | train | 2.8838 | 53.63 | 77.41 | 0.11354 | 811 |
| 35 | val | 2.6884 | 56.88 | 81.64 | 0.11354 | |
| 36 | train | 2.8775 | 53.82 | 77.49 | 0.11588 | 811 |
| 36 | val | 2.8365 | 53.52 | 78.93 | 0.11588 | |
| 37 | train | 2.8773 | 53.82 | 77.46 | 0.11798 | 811 |
| 37 | val | 2.7827 | 54.92 | 79.71 | 0.11798 | |
| 38 | train | 2.8754 | 53.85 | 77.53 | 0.11981 | 811 |
| 38 | val | 2.6928 | 56.85 | 81.48 | 0.11981 | |
| 39 | train | 2.8717 | 53.88 | 77.59 | 0.12138 | 811 |
| 39 | val | 2.7346 | 55.73 | 80.79 | 0.12138 | |
# Training Log
| epoch | phase | loss | top1 | top5 | lr | imgs/s |
|---:|---|---:|---:|---:|---:|---:|
# Training Log
| epoch | phase | loss | top1 | top5 | lr | imgs/s |
|---:|---|---:|---:|---:|---:|---:|
# Training Log
| epoch | phase | loss | top1 | top5 | lr | imgs/s |
|---:|---|---:|---:|---:|---:|---:|
| -1 | val | 2.7346 | 55.73 | 80.79 | 0.12138 | |
| 40 | train | 2.8679 | 54.00 | 77.59 | 0.12268 | 810 |
| 40 | val | 2.8079 | 54.30 | 79.20 | 0.12268 | |
| 41 | train | 2.8684 | 53.97 | 77.66 | 0.12369 | 811 |
| 41 | val | 2.7060 | 56.45 | 80.99 | 0.12369 | |
| 42 | train | 2.8660 | 54.09 | 77.65 | 0.12442 | 811 |
| 42 | val | 2.7870 | 54.68 | 79.80 | 0.12442 | |
| 43 | train | 2.8641 | 54.06 | 77.69 | 0.12485 | 811 |
| 43 | val | 2.8235 | 53.73 | 78.96 | 0.12485 | |
| 44 | train | 2.8603 | 54.15 | 77.77 | 0.12500 | 811 |
| 44 | val | 2.7186 | 56.17 | 81.12 | 0.12500 | |
| 45 | train | 2.8588 | 54.25 | 77.78 | 0.12497 | 811 |
| 45 | val | 2.9786 | 50.79 | 76.05 | 0.12497 | |
| 46 | train | 2.8550 | 54.30 | 77.85 | 0.12489 | 811 |
| 46 | val | 2.7939 | 54.48 | 79.43 | 0.12489 | |
| 47 | train | 2.8534 | 54.30 | 77.89 | 0.12475 | 811 |
| 47 | val | 2.7104 | 56.23 | 81.37 | 0.12475 | |
| 48 | train | 2.8479 | 54.44 | 78.02 | 0.12455 | 811 |
| 48 | val | 2.7660 | 55.43 | 79.86 | 0.12455 | |
| 49 | train | 2.8483 | 54.40 | 77.96 | 0.12430 | 811 |
| 49 | val | 2.7197 | 56.35 | 80.88 | 0.12430 | |
| 50 | train | 2.8462 | 54.51 | 78.00 | 0.12400 | 811 |
| 50 | val | 2.7382 | 55.79 | 80.68 | 0.12400 | |
| 51 | train | 2.8445 | 54.59 | 78.01 | 0.12363 | 811 |
| 51 | val | 2.6693 | 57.38 | 81.80 | 0.12363 | |
| 52 | train | 2.8396 | 54.64 | 78.11 | 0.12322 | 811 |
| 52 | val | 2.7040 | 56.68 | 81.11 | 0.12322 | |
| 53 | train | 2.8381 | 54.70 | 78.15 | 0.12275 | 811 |
| 53 | val | 2.7093 | 56.72 | 81.00 | 0.12275 | |
| 54 | train | 2.8326 | 54.79 | 78.26 | 0.12222 | 811 |
| 54 | val | 2.6895 | 57.00 | 81.32 | 0.12222 | |
| 55 | train | 2.8332 | 54.76 | 78.22 | 0.12165 | 811 |
| 55 | val | 2.7141 | 56.22 | 81.00 | 0.12165 | |
| 56 | train | 2.8310 | 54.80 | 78.26 | 0.12101 | 811 |
| 56 | val | 2.6739 | 57.49 | 81.65 | 0.12101 | |
| 57 | train | 2.8297 | 54.86 | 78.25 | 0.12033 | 811 |
| 57 | val | 2.6798 | 57.34 | 81.73 | 0.12033 | |
| 58 | train | 2.8271 | 54.89 | 78.34 | 0.11960 | 811 |
| 58 | val | 2.6520 | 57.44 | 82.14 | 0.11960 | |
| 59 | train | 2.8265 | 54.97 | 78.33 | 0.11881 | 811 |
| 59 | val | 2.7510 | 55.66 | 80.33 | 0.11881 | |
| 60 | train | 2.8224 | 55.01 | 78.40 | 0.11797 | 811 |
| 60 | val | 2.6755 | 57.36 | 81.75 | 0.11797 | |
| 61 | train | 2.8211 | 55.02 | 78.43 | 0.11709 | 811 |
| 61 | val | 2.6306 | 58.21 | 82.56 | 0.11709 | |
| 62 | train | 2.8177 | 55.06 | 78.47 | 0.11615 | 811 |
| 62 | val | 2.6791 | 57.23 | 81.59 | 0.11615 | |
| 63 | train | 2.8151 | 55.21 | 78.53 | 0.11517 | 811 |
| 63 | val | 2.6549 | 57.67 | 81.96 | 0.11517 | |
| 64 | train | 2.8129 | 55.21 | 78.57 | 0.11414 | 811 |
| 64 | val | 2.7341 | 55.86 | 80.57 | 0.11414 | |
| 65 | train | 2.8108 | 55.26 | 78.63 | 0.11306 | 811 |
| 65 | val | 2.6601 | 57.52 | 81.89 | 0.11306 | |
| 66 | train | 2.8088 | 55.28 | 78.61 | 0.11194 | 811 |
| 66 | val | 2.6278 | 58.24 | 82.47 | 0.11194 | |
| 67 | train | 2.8056 | 55.36 | 78.72 | 0.11078 | 811 |
| 67 | val | 2.6992 | 56.76 | 81.29 | 0.11078 | |
| 68 | train | 2.8028 | 55.49 | 78.72 | 0.10957 | 811 |
| 68 | val | 2.6473 | 57.99 | 82.05 | 0.10957 | |
| 69 | train | 2.8000 | 55.50 | 78.74 | 0.10832 | 811 |
| 69 | val | 2.6418 | 58.18 | 82.08 | 0.10832 | |
| 70 | train | 2.7986 | 55.56 | 78.82 | 0.10702 | 811 |
| 70 | val | 2.6372 | 58.05 | 82.43 | 0.10702 | |
| 71 | train | 2.7957 | 55.61 | 78.90 | 0.10569 | 811 |
| 71 | val | 2.6544 | 57.75 | 81.99 | 0.10569 | |
| 72 | train | 2.7949 | 55.64 | 78.84 | 0.10432 | 811 |
| 72 | val | 2.6884 | 56.75 | 81.44 | 0.10432 | |
| 73 | train | 2.7900 | 55.75 | 78.95 | 0.10291 | 811 |
| 73 | val | 2.5916 | 59.14 | 83.27 | 0.10291 | |
| 74 | train | 2.7868 | 55.83 | 79.02 | 0.10147 | 811 |
| 74 | val | 2.6557 | 57.48 | 82.12 | 0.10147 | |
| 75 | train | 2.7847 | 55.86 | 79.04 | 0.09999 | 811 |
| 75 | val | 2.6483 | 57.73 | 82.17 | 0.09999 | |
| 76 | train | 2.7831 | 55.89 | 79.03 | 0.09848 | 811 |
| 76 | val | 2.6373 | 58.28 | 82.22 | 0.09848 | |
| 77 | train | 2.7772 | 56.05 | 79.16 | 0.09693 | 811 |
| 77 | val | 2.6153 | 58.50 | 82.68 | 0.09693 | |
| 78 | train | 2.7764 | 56.09 | 79.18 | 0.09536 | 811 |
| 78 | val | 2.8513 | 53.15 | 78.50 | 0.09536 | |
| 79 | train | 2.7730 | 56.14 | 79.22 | 0.09375 | 811 |
| 79 | val | 2.6141 | 58.84 | 82.57 | 0.09375 | |
| 80 | train | 2.7717 | 56.18 | 79.23 | 0.09212 | 811 |
| 80 | val | 2.6121 | 58.76 | 82.84 | 0.09212 | |
| 81 | train | 2.7650 | 56.39 | 79.34 | 0.09046 | 811 |
| 81 | val | 2.6214 | 58.38 | 82.70 | 0.09046 | |
| 82 | train | 2.7615 | 56.38 | 79.42 | 0.08877 | 811 |
| 82 | val | 2.5851 | 59.44 | 83.27 | 0.08877 | |
| 83 | train | 2.7582 | 56.48 | 79.52 | 0.08706 | 811 |
| 83 | val | 2.6125 | 58.59 | 82.60 | 0.08706 | |
| 84 | train | 2.7557 | 56.54 | 79.48 | 0.08533 | 809 |
| 84 | val | 2.5861 | 59.41 | 83.06 | 0.08533 | |
| 85 | train | 2.7505 | 56.66 | 79.57 | 0.08358 | 808 |
| 85 | val | 2.6366 | 58.40 | 82.05 | 0.08358 | |
# Training Log
| epoch | phase | loss | top1 | top5 | lr | imgs/s |
|---:|---|---:|---:|---:|---:|---:|
| -1 | val | 2.6366 | 58.40 | 82.05 | 0.08358 | |
| 86 | train | 2.7472 | 56.76 | 79.63 | 0.08181 | 810 |
| 86 | val | 2.6344 | 57.90 | 82.30 | 0.08181 | |
| 87 | train | 2.7422 | 56.88 | 79.72 | 0.08003 | 810 |
| 87 | val | 2.6252 | 58.46 | 82.44 | 0.08003 | |
| 88 | train | 2.7374 | 57.00 | 79.80 | 0.07822 | 811 |
| 88 | val | 2.5797 | 59.44 | 83.41 | 0.07822 | |
| 89 | train | 2.7344 | 57.06 | 79.86 | 0.07641 | 811 |
| 89 | val | 2.5844 | 59.52 | 83.38 | 0.07641 | |
| 90 | train | 2.7296 | 57.13 | 79.95 | 0.07458 | 811 |
| 90 | val | 2.5414 | 60.53 | 83.87 | 0.07458 | |
| 91 | train | 2.7223 | 57.34 | 80.03 | 0.07274 | 811 |
| 91 | val | 2.5235 | 60.82 | 84.31 | 0.07274 | |
| 92 | train | 2.7190 | 57.37 | 80.15 | 0.07089 | 811 |
| 92 | val | 2.5845 | 59.32 | 83.09 | 0.07089 | |
| 93 | train | 2.7140 | 57.51 | 80.19 | 0.06903 | 811 |
| 93 | val | 2.5229 | 60.83 | 84.30 | 0.06903 | |
| 94 | train | 2.7078 | 57.66 | 80.28 | 0.06717 | 811 |
| 94 | val | 2.5340 | 60.65 | 83.96 | 0.06717 | |
| 95 | train | 2.7047 | 57.72 | 80.36 | 0.06530 | 811 |
| 95 | val | 2.5498 | 60.17 | 83.61 | 0.06530 | |
| 96 | train | 2.6994 | 57.86 | 80.44 | 0.06343 | 811 |
| 96 | val | 2.4827 | 62.08 | 84.86 | 0.06343 | |
| 97 | train | 2.6925 | 57.98 | 80.53 | 0.06156 | 811 |
| 97 | val | 2.5160 | 61.01 | 84.25 | 0.06156 | |
| 98 | train | 2.6857 | 58.18 | 80.66 | 0.05970 | 811 |
| 98 | val | 2.5395 | 60.45 | 84.12 | 0.05970 | |
| 99 | train | 2.6818 | 58.22 | 80.71 | 0.05783 | 811 |
| 99 | val | 2.5775 | 59.82 | 83.13 | 0.05783 | |
| 100 | train | 2.6734 | 58.51 | 80.86 | 0.05597 | 811 |
| 100 | val | 2.5249 | 60.77 | 84.27 | 0.05597 | |
| 101 | train | 2.6683 | 58.57 | 80.94 | 0.05411 | 811 |
| 101 | val | 2.4416 | 62.87 | 85.62 | 0.05411 | |
| 102 | train | 2.6574 | 58.83 | 81.13 | 0.05226 | 811 |
| 102 | val | 2.5051 | 61.23 | 84.56 | 0.05226 | |
| 103 | train | 2.6535 | 58.93 | 81.17 | 0.05042 | 811 |
| 103 | val | 2.4721 | 62.26 | 85.01 | 0.05042 | |
| 104 | train | 2.6429 | 59.14 | 81.36 | 0.04859 | 811 |
| 104 | val | 2.5892 | 59.46 | 83.06 | 0.04859 | |
| 105 | train | 2.6375 | 59.30 | 81.44 | 0.04678 | 811 |
| 105 | val | 2.4493 | 62.47 | 85.48 | 0.04678 | |
| 106 | train | 2.6289 | 59.50 | 81.61 | 0.04497 | 811 |
| 106 | val | 2.4971 | 61.53 | 84.42 | 0.04497 | |
| 107 | train | 2.6200 | 59.70 | 81.77 | 0.04319 | 811 |
| 107 | val | 2.4180 | 63.33 | 86.04 | 0.04319 | |
| 108 | train | 2.6111 | 59.93 | 81.87 | 0.04142 | 811 |
| 108 | val | 2.5006 | 61.37 | 84.51 | 0.04142 | |
| 109 | train | 2.6009 | 60.13 | 82.05 | 0.03967 | 811 |
| 109 | val | 2.4283 | 63.16 | 85.90 | 0.03967 | |
| 110 | train | 2.5908 | 60.38 | 82.20 | 0.03794 | 811 |
| 110 | val | 2.4383 | 63.08 | 85.42 | 0.03794 | |
| 111 | train | 2.5820 | 60.57 | 82.32 | 0.03623 | 811 |
| 111 | val | 2.3968 | 63.92 | 86.17 | 0.03623 | |
| 112 | train | 2.5701 | 60.89 | 82.55 | 0.03454 | 811 |
| 112 | val | 2.4192 | 63.26 | 85.80 | 0.03454 | |
| 113 | train | 2.5599 | 61.10 | 82.67 | 0.03288 | 811 |
| 113 | val | 2.3810 | 64.11 | 86.66 | 0.03288 | |
| 114 | train | 2.5489 | 61.36 | 82.86 | 0.03125 | 811 |
| 114 | val | 2.4403 | 62.94 | 85.36 | 0.03125 | |
| 115 | train | 2.5380 | 61.61 | 83.02 | 0.02964 | 811 |
| 115 | val | 2.3566 | 64.94 | 86.87 | 0.02964 | |
| 116 | train | 2.5254 | 61.92 | 83.22 | 0.02807 | 811 |
| 116 | val | 2.3312 | 65.28 | 87.30 | 0.02807 | |
| 117 | train | 2.5104 | 62.28 | 83.50 | 0.02652 | 811 |
| 117 | val | 2.3709 | 64.49 | 86.65 | 0.02652 | |
| 118 | train | 2.4986 | 62.56 | 83.65 | 0.02501 | 811 |
| 118 | val | 2.3242 | 65.59 | 87.40 | 0.02501 | |
| 119 | train | 2.4835 | 62.96 | 83.91 | 0.02353 | 811 |
| 119 | val | 2.3009 | 66.38 | 87.65 | 0.02353 | |
# Training Log
| epoch | phase | loss | top1 | top5 | lr | imgs/s |
|---:|---|---:|---:|---:|---:|---:|
| -1 | val | 2.3009 | 66.38 | 87.65 | 0.02353 | |
# Training Log
| epoch | phase | loss | top1 | top5 | lr | imgs/s |
|---:|---|---:|---:|---:|---:|---:|
| -1 | val | 2.3009 | 66.38 | 87.65 | 0.00500 | |
| 120 | train | 2.5830 | 60.53 | 82.31 | 0.03381 | 809 |
| 120 | val | 2.4087 | 63.42 | 86.03 | 0.03381 | |
| 121 | train | 2.5616 | 61.07 | 82.67 | 0.03227 | 811 |
| 121 | val | 2.4308 | 62.92 | 85.49 | 0.03227 | |
| 122 | train | 2.5417 | 61.53 | 83.00 | 0.03075 | 811 |
| 122 | val | 2.3409 | 65.31 | 87.00 | 0.03075 | |
| 123 | train | 2.5284 | 61.86 | 83.20 | 0.02925 | 811 |
| 123 | val | 2.3405 | 65.15 | 87.05 | 0.02925 | |
| 124 | train | 2.5162 | 62.10 | 83.40 | 0.02778 | 811 |
| 124 | val | 2.3236 | 65.56 | 87.36 | 0.02778 | |
| 125 | train | 2.5024 | 62.48 | 83.56 | 0.02633 | 811 |
| 125 | val | 2.3302 | 65.36 | 87.23 | 0.02633 | |
| 126 | train | 2.4895 | 62.79 | 83.81 | 0.02492 | 811 |
| 126 | val | 2.3206 | 65.50 | 87.18 | 0.02492 | |
| 127 | train | 2.4769 | 63.03 | 83.99 | 0.02353 | 811 |
| 127 | val | 2.2749 | 66.81 | 88.24 | 0.02353 | |
| 128 | train | 2.4625 | 63.39 | 84.21 | 0.02218 | 811 |
| 128 | val | 2.2523 | 67.23 | 88.55 | 0.02218 | |
| 129 | train | 2.4472 | 63.79 | 84.45 | 0.02085 | 811 |
| 129 | val | 2.2872 | 66.41 | 87.80 | 0.02085 | |
| 130 | train | 2.4340 | 64.12 | 84.64 | 0.01956 | 811 |
| 130 | val | 2.2678 | 67.05 | 88.26 | 0.01956 | |
| 131 | train | 2.4197 | 64.44 | 84.88 | 0.01831 | 811 |
| 131 | val | 2.2712 | 67.08 | 88.05 | 0.01831 | |
| 132 | train | 2.4020 | 64.87 | 85.13 | 0.01708 | 811 |
| 132 | val | 2.2340 | 67.97 | 88.68 | 0.01708 | |
| 133 | train | 2.3843 | 65.31 | 85.41 | 0.01590 | 811 |
| 133 | val | 2.2064 | 68.27 | 89.04 | 0.01590 | |
| 134 | train | 2.3676 | 65.68 | 85.65 | 0.01475 | 811 |
| 134 | val | 2.2156 | 68.28 | 88.88 | 0.01475 | |
| 135 | train | 2.3492 | 66.15 | 85.93 | 0.01364 | 811 |
| 135 | val | 2.1873 | 68.89 | 89.40 | 0.01364 | |
| 136 | train | 2.3302 | 66.65 | 86.20 | 0.01256 | 811 |
| 136 | val | 2.1972 | 68.78 | 89.21 | 0.01256 | |
| 137 | train | 2.3112 | 67.04 | 86.50 | 0.01153 | 811 |
| 137 | val | 2.1619 | 69.73 | 89.76 | 0.01153 | |
| 138 | train | 2.2907 | 67.55 | 86.80 | 0.01053 | 811 |
| 138 | val | 2.1297 | 70.39 | 90.00 | 0.01053 | |
| 139 | train | 2.2705 | 68.07 | 87.05 | 0.00958 | 811 |
| 139 | val | 2.1356 | 70.38 | 90.07 | 0.00958 | |
| 140 | train | 2.2459 | 68.66 | 87.43 | 0.00867 | 811 |
| 140 | val | 2.1157 | 70.66 | 90.29 | 0.00867 | |
| 141 | train | 2.2224 | 69.22 | 87.78 | 0.00780 | 811 |
| 141 | val | 2.0775 | 71.49 | 90.85 | 0.00780 | |
| 142 | train | 2.1968 | 69.87 | 88.13 | 0.00697 | 811 |
| 142 | val | 2.0625 | 72.04 | 91.09 | 0.00697 | |
| 143 | train | 2.1739 | 70.45 | 88.45 | 0.00619 | 811 |
| 143 | val | 2.0499 | 72.51 | 91.28 | 0.00619 | |
| 144 | train | 2.1464 | 71.11 | 88.86 | 0.00545 | 811 |
| 144 | val | 2.0256 | 73.01 | 91.55 | 0.00545 | |
| 145 | train | 2.1195 | 71.86 | 89.21 | 0.00476 | 811 |
| 145 | val | 2.0212 | 73.03 | 91.65 | 0.00476 | |
| 146 | train | 2.0908 | 72.53 | 89.63 | 0.00411 | 811 |
| 146 | val | 2.0003 | 73.76 | 91.75 | 0.00411 | |
| 147 | train | 2.0617 | 73.27 | 89.98 | 0.00351 | 811 |
| 147 | val | 1.9725 | 74.25 | 92.15 | 0.00351 | |
| 148 | train | 2.0325 | 74.04 | 90.40 | 0.00295 | 811 |
| 148 | val | 1.9619 | 74.67 | 92.32 | 0.00295 | |
| 149 | train | 2.0055 | 74.73 | 90.73 | 0.00244 | 811 |
| 149 | val | 1.9369 | 75.38 | 92.66 | 0.00244 | |
| 150 | train | 1.9746 | 75.53 | 91.12 | 0.00198 | 810 |
| 150 | val | 1.9184 | 75.66 | 92.84 | 0.00198 | |
| 151 | train | 1.9458 | 76.28 | 91.51 | 0.00157 | 811 |
| 151 | val | 1.9104 | 75.94 | 92.92 | 0.00157 | |
| 152 | train | 1.9179 | 77.02 | 91.83 | 0.00120 | 811 |
| 152 | val | 1.8939 | 76.36 | 93.12 | 0.00120 | |
| 153 | train | 1.8925 | 77.69 | 92.14 | 0.00088 | 811 |
| 153 | val | 1.8791 | 76.56 | 93.40 | 0.00088 | |
| 154 | train | 1.8698 | 78.32 | 92.41 | 0.00061 | 811 |
| 154 | val | 1.8668 | 76.99 | 93.53 | 0.00061 | |
| 155 | train | 1.8496 | 78.86 | 92.64 | 0.00039 | 811 |
| 155 | val | 1.8578 | 77.33 | 93.59 | 0.00039 | |
| 156 | train | 1.8332 | 79.34 | 92.84 | 0.00022 | 811 |
| 156 | val | 1.8531 | 77.40 | 93.64 | 0.00022 | |
| 157 | train | 1.8224 | 79.61 | 92.97 | 0.00010 | 809 |
| 157 | val | 1.8488 | 77.56 | 93.73 | 0.00010 | |
| 158 | train | 1.8154 | 79.78 | 93.06 | 0.00003 | 806 |
| 158 | val | 1.8468 | 77.62 | 93.75 | 0.00003 | |
| 159 | train | 1.8115 | 79.91 | 93.08 | 0.00000 | 806 |
| 159 | val | 1.8463 | 77.59 | 93.75 | 0.00000 | |
# Training Log
| epoch | phase | loss | top1 | top5 | lr | imgs/s |
|---:|---|---:|---:|---:|---:|---:|
| -1 | val | 1.8463 | 77.59 | 93.75 | 0.00000 | |
| 160 | train | 2.0736 | 72.87 | 89.91 | 0.00399 | 810 |
| 160 | val | 2.0209 | 73.19 | 91.57 | 0.00399 | |
| 161 | train | 2.0650 | 73.16 | 89.99 | 0.00344 | 810 |
| 161 | val | 1.9975 | 73.59 | 91.90 | 0.00344 | |
| 162 | train | 2.0335 | 73.96 | 90.42 | 0.00294 | 810 |
| 162 | val | 1.9691 | 74.21 | 92.36 | 0.00294 | |
| 163 | train | 2.0018 | 74.81 | 90.79 | 0.00247 | 810 |
| 163 | val | 1.9722 | 74.38 | 92.25 | 0.00247 | |
| 164 | train | 1.9691 | 75.68 | 91.22 | 0.00204 | 810 |
| 164 | val | 1.9329 | 75.23 | 92.67 | 0.00204 | |
| 165 | train | 1.9387 | 76.46 | 91.56 | 0.00166 | 810 |
| 165 | val | 1.9159 | 75.66 | 92.92 | 0.00166 | |
| 166 | train | 1.9079 | 77.28 | 91.97 | 0.00131 | 810 |
| 166 | val | 1.9055 | 75.99 | 92.90 | 0.00131 | |
| 167 | train | 1.8791 | 78.09 | 92.32 | 0.00100 | 810 |
| 167 | val | 1.8846 | 76.62 | 93.35 | 0.00100 | |
| 168 | train | 1.8540 | 78.77 | 92.60 | 0.00074 | 810 |
| 168 | val | 1.8756 | 76.79 | 93.39 | 0.00074 | |
| 169 | train | 1.8304 | 79.41 | 92.90 | 0.00051 | 810 |
| 169 | val | 1.8682 | 76.95 | 93.49 | 0.00051 | |
| 170 | train | 1.8099 | 79.97 | 93.09 | 0.00033 | 810 |
| 170 | val | 1.8570 | 77.18 | 93.54 | 0.00033 | |
| 171 | train | 1.7930 | 80.41 | 93.33 | 0.00019 | 811 |
| 171 | val | 1.8507 | 77.33 | 93.72 | 0.00019 | |
| 172 | train | 1.7815 | 80.72 | 93.46 | 0.00008 | 811 |
| 172 | val | 1.8472 | 77.51 | 93.80 | 0.00008 | |
| 173 | train | 1.7751 | 80.94 | 93.53 | 0.00002 | 808 |
| 173 | val | 1.8455 | 77.61 | 93.79 | 0.00002 | |
| 174 | train | 1.7713 | 81.03 | 93.56 | 0.00000 | 810 |
| 174 | val | 1.8445 | 77.66 | 93.84 | 0.00000 | |
# Training Log
| epoch | phase | loss | top1 | top5 | lr | imgs/s |
|---:|---|---:|---:|---:|---:|---:|
# Training Log
| epoch | phase | loss | top1 | top5 | lr | imgs/s |
|---:|---|---:|---:|---:|---:|---:|
| -1 | val | 1.8445 | 77.66 | 93.84 | 0.00000 | |
# Training Log
| epoch | phase | loss | top1 | top5 | lr | imgs/s |
|---:|---|---:|---:|---:|---:|---:|
# Training Log
| epoch | phase | loss | top1 | top5 | lr | imgs/s |
|---:|---|---:|---:|---:|---:|---:|
| -1 | val | 1.8445 | 77.66 | 93.84 | 0.00000 | |
| 175 | train | 1.7712 | 81.02 | 93.55 | 0.00000 | 810 |
| 175 | val | 1.8449 | 77.60 | 93.81 | 0.00000 | |
# Training Log
| epoch | phase | loss | top1 | top5 | lr | imgs/s |
|---:|---|---:|---:|---:|---:|---:|
| -1 | val | 1.8449 | 77.60 | 93.81 | 0.00000 | |
| 176 | train | 1.7711 | 81.07 | 93.57 | 0.00000 | 810 |
| 176 | val | 1.8454 | 77.59 | 93.83 | 0.00000 | |
| 177 | train | 1.7710 | 81.03 | 93.56 | 0.00000 | 810 |
| 177 | val | 1.8450 | 77.56 | 93.75 | 0.00000 | |
| 178 | train | 1.7710 | 81.05 | 93.56 | 0.00000 | 811 |
| 178 | val | 1.8452 | 77.59 | 93.79 | 0.00000 | |
| 179 | train | 1.7709 | 81.06 | 93.56 | 0.00000 | 811 |
| 179 | val | 1.8445 | 77.61 | 93.79 | 0.00000 | |
| 180 | train | 1.7697 | 81.10 | 93.57 | 0.00000 | 811 |
| 180 | val | 1.8445 | 77.62 | 93.80 | 0.00000 | |
| 181 | train | 1.7720 | 81.04 | 93.53 | 0.00000 | 811 |
| 181 | val | 1.8445 | 77.57 | 93.83 | 0.00000 | |
| 182 | train | 1.7705 | 81.09 | 93.57 | 0.00000 | 811 |
| 182 | val | 1.8447 | 77.57 | 93.80 | 0.00000 | |
| 183 | train | 1.7714 | 81.03 | 93.54 | 0.00000 | 811 |
| 183 | val | 1.8449 | 77.62 | 93.80 | 0.00000 | |
| 184 | train | 1.7702 | 81.08 | 93.59 | 0.00000 | 811 |
| 184 | val | 1.8451 | 77.55 | 93.82 | 0.00000 | |
# Training Log
| epoch | phase | loss | top1 | top5 | lr | imgs/s |
|---:|---|---:|---:|---:|---:|---:|
| -1 | val | 1.4043 | 77.55 | 93.82 | 0.00000 | |
| 185 | train | 1.3344 | 81.03 | 93.54 | 0.00000 | 810 |
| 185 | val | 1.4037 | 77.58 | 93.81 | 0.00000 | |
| 186 | train | 1.3314 | 81.01 | 93.53 | 0.00000 | 811 |
| 186 | val | 1.4005 | 77.61 | 93.81 | 0.00000 | |
| 187 | train | 1.3281 | 81.11 | 93.56 | 0.00000 | 810 |
| 187 | val | 1.3999 | 77.63 | 93.82 | 0.00000 | |
| 188 | train | 1.3297 | 80.99 | 93.51 | 0.00000 | 809 |
| 188 | val | 1.3982 | 77.62 | 93.80 | 0.00000 | |
| 189 | train | 1.3279 | 81.02 | 93.53 | 0.00000 | 808 |
| 189 | val | 1.3984 | 77.62 | 93.83 | 0.00000 | |
| 190 | train | 1.3265 | 81.00 | 93.52 | 0.00000 | 810 |
| 190 | val | 1.3981 | 77.62 | 93.80 | 0.00000 | |
| 191 | train | 1.3271 | 81.00 | 93.50 | 0.00000 | 810 |
| 191 | val | 1.3977 | 77.60 | 93.83 | 0.00000 | |
| 192 | train | 1.3260 | 81.01 | 93.51 | 0.00000 | 810 |
| 192 | val | 1.3974 | 77.60 | 93.83 | 0.00000 | |
| 193 | train | 1.3251 | 81.00 | 93.52 | 0.00000 | 810 |
| 193 | val | 1.3964 | 77.64 | 93.82 | 0.00000 | |
| 194 | train | 1.3245 | 81.01 | 93.54 | 0.00000 | 810 |
| 194 | val | 1.3966 | 77.61 | 93.85 | 0.00000 | |

```
</details>

<details>
<summary><strong>6.B.2 Model summary</strong></summary>

```text
=================================================================================================================================================
Layer (type:depth-idx)                        Input Shape               Output Shape              Param #                   Kernel Shape
=================================================================================================================================================
ResNet50                                      [256, 3, 224, 224]        [256, 1000]               --                        --
├─ResNet: 1-1                                 [256, 3, 224, 224]        [256, 1000]               --                        --
│    └─Conv2d: 2-1                            [256, 3, 224, 224]        [256, 64, 112, 112]       9,408                     [7, 7]
│    └─BatchNorm2d: 2-2                       [256, 64, 112, 112]       [256, 64, 112, 112]       128                       --
│    └─ReLU: 2-3                              [256, 64, 112, 112]       [256, 64, 112, 112]       --                        --
│    └─MaxPool2d: 2-4                         [256, 64, 112, 112]       [256, 64, 56, 56]         --                        3
│    └─Sequential: 2-5                        [256, 64, 56, 56]         [256, 256, 56, 56]        --                        --
│    │    └─Bottleneck: 3-1                   [256, 64, 56, 56]         [256, 256, 56, 56]        --                        --
│    │    │    └─Conv2d: 4-1                  [256, 64, 56, 56]         [256, 64, 56, 56]         4,096                     [1, 1]
│    │    │    └─BatchNorm2d: 4-2             [256, 64, 56, 56]         [256, 64, 56, 56]         128                       --
│    │    │    └─ReLU: 4-3                    [256, 64, 56, 56]         [256, 64, 56, 56]         --                        --
│    │    │    └─Conv2d: 4-4                  [256, 64, 56, 56]         [256, 64, 56, 56]         36,864                    [3, 3]
│    │    │    └─BatchNorm2d: 4-5             [256, 64, 56, 56]         [256, 64, 56, 56]         128                       --
│    │    │    └─ReLU: 4-6                    [256, 64, 56, 56]         [256, 64, 56, 56]         --                        --
│    │    │    └─Conv2d: 4-7                  [256, 64, 56, 56]         [256, 256, 56, 56]        16,384                    [1, 1]
│    │    │    └─BatchNorm2d: 4-8             [256, 256, 56, 56]        [256, 256, 56, 56]        512                       --
│    │    │    └─Sequential: 4-9              [256, 64, 56, 56]         [256, 256, 56, 56]        --                        --
│    │    │    │    └─Conv2d: 5-1             [256, 64, 56, 56]         [256, 256, 56, 56]        16,384                    [1, 1]
│    │    │    │    └─BatchNorm2d: 5-2        [256, 256, 56, 56]        [256, 256, 56, 56]        512                       --
│    │    │    └─ReLU: 4-10                   [256, 256, 56, 56]        [256, 256, 56, 56]        --                        --
│    │    └─Bottleneck: 3-2                   [256, 256, 56, 56]        [256, 256, 56, 56]        --                        --
│    │    │    └─Conv2d: 4-11                 [256, 256, 56, 56]        [256, 64, 56, 56]         16,384                    [1, 1]
│    │    │    └─BatchNorm2d: 4-12            [256, 64, 56, 56]         [256, 64, 56, 56]         128                       --
│    │    │    └─ReLU: 4-13                   [256, 64, 56, 56]         [256, 64, 56, 56]         --                        --
│    │    │    └─Conv2d: 4-14                 [256, 64, 56, 56]         [256, 64, 56, 56]         36,864                    [3, 3]
│    │    │    └─BatchNorm2d: 4-15            [256, 64, 56, 56]         [256, 64, 56, 56]         128                       --
│    │    │    └─ReLU: 4-16                   [256, 64, 56, 56]         [256, 64, 56, 56]         --                        --
│    │    │    └─Conv2d: 4-17                 [256, 64, 56, 56]         [256, 256, 56, 56]        16,384                    [1, 1]
│    │    │    └─BatchNorm2d: 4-18            [256, 256, 56, 56]        [256, 256, 56, 56]        512                       --
│    │    │    └─ReLU: 4-19                   [256, 256, 56, 56]        [256, 256, 56, 56]        --                        --
│    │    └─Bottleneck: 3-3                   [256, 256, 56, 56]        [256, 256, 56, 56]        --                        --
│    │    │    └─Conv2d: 4-20                 [256, 256, 56, 56]        [256, 64, 56, 56]         16,384                    [1, 1]
│    │    │    └─BatchNorm2d: 4-21            [256, 64, 56, 56]         [256, 64, 56, 56]         128                       --
│    │    │    └─ReLU: 4-22                   [256, 64, 56, 56]         [256, 64, 56, 56]         --                        --
│    │    │    └─Conv2d: 4-23                 [256, 64, 56, 56]         [256, 64, 56, 56]         36,864                    [3, 3]
│    │    │    └─BatchNorm2d: 4-24            [256, 64, 56, 56]         [256, 64, 56, 56]         128                       --
│    │    │    └─ReLU: 4-25                   [256, 64, 56, 56]         [256, 64, 56, 56]         --                        --
│    │    │    └─Conv2d: 4-26                 [256, 64, 56, 56]         [256, 256, 56, 56]        16,384                    [1, 1]
│    │    │    └─BatchNorm2d: 4-27            [256, 256, 56, 56]        [256, 256, 56, 56]        512                       --
│    │    │    └─ReLU: 4-28                   [256, 256, 56, 56]        [256, 256, 56, 56]        --                        --
│    └─Sequential: 2-6                        [256, 256, 56, 56]        [256, 512, 28, 28]        --                        --
│    │    └─Bottleneck: 3-4                   [256, 256, 56, 56]        [256, 512, 28, 28]        --                        --
│    │    │    └─Conv2d: 4-29                 [256, 256, 56, 56]        [256, 128, 56, 56]        32,768                    [1, 1]
│    │    │    └─BatchNorm2d: 4-30            [256, 128, 56, 56]        [256, 128, 56, 56]        256                       --
│    │    │    └─ReLU: 4-31                   [256, 128, 56, 56]        [256, 128, 56, 56]        --                        --
│    │    │    └─Conv2d: 4-32                 [256, 128, 56, 56]        [256, 128, 28, 28]        147,456                   [3, 3]
│    │    │    └─BatchNorm2d: 4-33            [256, 128, 28, 28]        [256, 128, 28, 28]        256                       --
│    │    │    └─ReLU: 4-34                   [256, 128, 28, 28]        [256, 128, 28, 28]        --                        --
│    │    │    └─Conv2d: 4-35                 [256, 128, 28, 28]        [256, 512, 28, 28]        65,536                    [1, 1]
│    │    │    └─BatchNorm2d: 4-36            [256, 512, 28, 28]        [256, 512, 28, 28]        1,024                     --
│    │    │    └─Sequential: 4-37             [256, 256, 56, 56]        [256, 512, 28, 28]        --                        --
│    │    │    │    └─Conv2d: 5-3             [256, 256, 56, 56]        [256, 512, 28, 28]        131,072                   [1, 1]
│    │    │    │    └─BatchNorm2d: 5-4        [256, 512, 28, 28]        [256, 512, 28, 28]        1,024                     --
│    │    │    └─ReLU: 4-38                   [256, 512, 28, 28]        [256, 512, 28, 28]        --                        --
│    │    └─Bottleneck: 3-5                   [256, 512, 28, 28]        [256, 512, 28, 28]        --                        --
│    │    │    └─Conv2d: 4-39                 [256, 512, 28, 28]        [256, 128, 28, 28]        65,536                    [1, 1]
│    │    │    └─BatchNorm2d: 4-40            [256, 128, 28, 28]        [256, 128, 28, 28]        256                       --
│    │    │    └─ReLU: 4-41                   [256, 128, 28, 28]        [256, 128, 28, 28]        --                        --
│    │    │    └─Conv2d: 4-42                 [256, 128, 28, 28]        [256, 128, 28, 28]        147,456                   [3, 3]
│    │    │    └─BatchNorm2d: 4-43            [256, 128, 28, 28]        [256, 128, 28, 28]        256                       --
│    │    │    └─ReLU: 4-44                   [256, 128, 28, 28]        [256, 128, 28, 28]        --                        --
│    │    │    └─Conv2d: 4-45                 [256, 128, 28, 28]        [256, 512, 28, 28]        65,536                    [1, 1]
│    │    │    └─BatchNorm2d: 4-46            [256, 512, 28, 28]        [256, 512, 28, 28]        1,024                     --
│    │    │    └─ReLU: 4-47                   [256, 512, 28, 28]        [256, 512, 28, 28]        --                        --
│    │    └─Bottleneck: 3-6                   [256, 512, 28, 28]        [256, 512, 28, 28]        --                        --
│    │    │    └─Conv2d: 4-48                 [256, 512, 28, 28]        [256, 128, 28, 28]        65,536                    [1, 1]
│    │    │    └─BatchNorm2d: 4-49            [256, 128, 28, 28]        [256, 128, 28, 28]        256                       --
│    │    │    └─ReLU: 4-50                   [256, 128, 28, 28]        [256, 128, 28, 28]        --                        --
│    │    │    └─Conv2d: 4-51                 [256, 128, 28, 28]        [256, 128, 28, 28]        147,456                   [3, 3]
│    │    │    └─BatchNorm2d: 4-52            [256, 128, 28, 28]        [256, 128, 28, 28]        256                       --
│    │    │    └─ReLU: 4-53                   [256, 128, 28, 28]        [256, 128, 28, 28]        --                        --
│    │    │    └─Conv2d: 4-54                 [256, 128, 28, 28]        [256, 512, 28, 28]        65,536                    [1, 1]
│    │    │    └─BatchNorm2d: 4-55            [256, 512, 28, 28]        [256, 512, 28, 28]        1,024                     --
│    │    │    └─ReLU: 4-56                   [256, 512, 28, 28]        [256, 512, 28, 28]        --                        --
│    │    └─Bottleneck: 3-7                   [256, 512, 28, 28]        [256, 512, 28, 28]        --                        --
│    │    │    └─Conv2d: 4-57                 [256, 512, 28, 28]        [256, 128, 28, 28]        65,536                    [1, 1]
│    │    │    └─BatchNorm2d: 4-58            [256, 128, 28, 28]        [256, 128, 28, 28]        256                       --
│    │    │    └─ReLU: 4-59                   [256, 128, 28, 28]        [256, 128, 28, 28]        --                        --
│    │    │    └─Conv2d: 4-60                 [256, 128, 28, 28]        [256, 128, 28, 28]        147,456                   [3, 3]
│    │    │    └─BatchNorm2d: 4-61            [256, 128, 28, 28]        [256, 128, 28, 28]        256                       --
│    │    │    └─ReLU: 4-62                   [256, 128, 28, 28]        [256, 128, 28, 28]        --                        --
│    │    │    └─Conv2d: 4-63                 [256, 128, 28, 28]        [256, 512, 28, 28]        65,536                    [1, 1]
│    │    │    └─BatchNorm2d: 4-64            [256, 512, 28, 28]        [256, 512, 28, 28]        1,024                     --
│    │    │    └─ReLU: 4-65                   [256, 512, 28, 28]        [256, 512, 28, 28]        --                        --
│    └─Sequential: 2-7                        [256, 512, 28, 28]        [256, 1024, 14, 14]       --                        --
│    │    └─Bottleneck: 3-8                   [256, 512, 28, 28]        [256, 1024, 14, 14]       --                        --
│    │    │    └─Conv2d: 4-66                 [256, 512, 28, 28]        [256, 256, 28, 28]        131,072                   [1, 1]
│    │    │    └─BatchNorm2d: 4-67            [256, 256, 28, 28]        [256, 256, 28, 28]        512                       --
│    │    │    └─ReLU: 4-68                   [256, 256, 28, 28]        [256, 256, 28, 28]        --                        --
│    │    │    └─Conv2d: 4-69                 [256, 256, 28, 28]        [256, 256, 14, 14]        589,824                   [3, 3]
│    │    │    └─BatchNorm2d: 4-70            [256, 256, 14, 14]        [256, 256, 14, 14]        512                       --
│    │    │    └─ReLU: 4-71                   [256, 256, 14, 14]        [256, 256, 14, 14]        --                        --
│    │    │    └─Conv2d: 4-72                 [256, 256, 14, 14]        [256, 1024, 14, 14]       262,144                   [1, 1]
│    │    │    └─BatchNorm2d: 4-73            [256, 1024, 14, 14]       [256, 1024, 14, 14]       2,048                     --
│    │    │    └─Sequential: 4-74             [256, 512, 28, 28]        [256, 1024, 14, 14]       --                        --
│    │    │    │    └─Conv2d: 5-5             [256, 512, 28, 28]        [256, 1024, 14, 14]       524,288                   [1, 1]
│    │    │    │    └─BatchNorm2d: 5-6        [256, 1024, 14, 14]       [256, 1024, 14, 14]       2,048                     --
│    │    │    └─ReLU: 4-75                   [256, 1024, 14, 14]       [256, 1024, 14, 14]       --                        --
│    │    └─Bottleneck: 3-9                   [256, 1024, 14, 14]       [256, 1024, 14, 14]       --                        --
│    │    │    └─Conv2d: 4-76                 [256, 1024, 14, 14]       [256, 256, 14, 14]        262,144                   [1, 1]
│    │    │    └─BatchNorm2d: 4-77            [256, 256, 14, 14]        [256, 256, 14, 14]        512                       --
│    │    │    └─ReLU: 4-78                   [256, 256, 14, 14]        [256, 256, 14, 14]        --                        --
│    │    │    └─Conv2d: 4-79                 [256, 256, 14, 14]        [256, 256, 14, 14]        589,824                   [3, 3]
│    │    │    └─BatchNorm2d: 4-80            [256, 256, 14, 14]        [256, 256, 14, 14]        512                       --
│    │    │    └─ReLU: 4-81                   [256, 256, 14, 14]        [256, 256, 14, 14]        --                        --
│    │    │    └─Conv2d: 4-82                 [256, 256, 14, 14]        [256, 1024, 14, 14]       262,144                   [1, 1]
│    │    │    └─BatchNorm2d: 4-83            [256, 1024, 14, 14]       [256, 1024, 14, 14]       2,048                     --
│    │    │    └─ReLU: 4-84                   [256, 1024, 14, 14]       [256, 1024, 14, 14]       --                        --
│    │    └─Bottleneck: 3-10                  [256, 1024, 14, 14]       [256, 1024, 14, 14]       --                        --
│    │    │    └─Conv2d: 4-85                 [256, 1024, 14, 14]       [256, 256, 14, 14]        262,144                   [1, 1]
│    │    │    └─BatchNorm2d: 4-86            [256, 256, 14, 14]        [256, 256, 14, 14]        512                       --
│    │    │    └─ReLU: 4-87                   [256, 256, 14, 14]        [256, 256, 14, 14]        --                        --
│    │    │    └─Conv2d: 4-88                 [256, 256, 14, 14]        [256, 256, 14, 14]        589,824                   [3, 3]
│    │    │    └─BatchNorm2d: 4-89            [256, 256, 14, 14]        [256, 256, 14, 14]        512                       --
│    │    │    └─ReLU: 4-90                   [256, 256, 14, 14]        [256, 256, 14, 14]        --                        --
│    │    │    └─Conv2d: 4-91                 [256, 256, 14, 14]        [256, 1024, 14, 14]       262,144                   [1, 1]
│    │    │    └─BatchNorm2d: 4-92            [256, 1024, 14, 14]       [256, 1024, 14, 14]       2,048                     --
│    │    │    └─ReLU: 4-93                   [256, 1024, 14, 14]       [256, 1024, 14, 14]       --                        --
│    │    └─Bottleneck: 3-11                  [256, 1024, 14, 14]       [256, 1024, 14, 14]       --                        --
│    │    │    └─Conv2d: 4-94                 [256, 1024, 14, 14]       [256, 256, 14, 14]        262,144                   [1, 1]
│    │    │    └─BatchNorm2d: 4-95            [256, 256, 14, 14]        [256, 256, 14, 14]        512                       --
│    │    │    └─ReLU: 4-96                   [256, 256, 14, 14]        [256, 256, 14, 14]        --                        --
│    │    │    └─Conv2d: 4-97                 [256, 256, 14, 14]        [256, 256, 14, 14]        589,824                   [3, 3]
│    │    │    └─BatchNorm2d: 4-98            [256, 256, 14, 14]        [256, 256, 14, 14]        512                       --
│    │    │    └─ReLU: 4-99                   [256, 256, 14, 14]        [256, 256, 14, 14]        --                        --
│    │    │    └─Conv2d: 4-100                [256, 256, 14, 14]        [256, 1024, 14, 14]       262,144                   [1, 1]
│    │    │    └─BatchNorm2d: 4-101           [256, 1024, 14, 14]       [256, 1024, 14, 14]       2,048                     --
│    │    │    └─ReLU: 4-102                  [256, 1024, 14, 14]       [256, 1024, 14, 14]       --                        --
│    │    └─Bottleneck: 3-12                  [256, 1024, 14, 14]       [256, 1024, 14, 14]       --                        --
│    │    │    └─Conv2d: 4-103                [256, 1024, 14, 14]       [256, 256, 14, 14]        262,144                   [1, 1]
│    │    │    └─BatchNorm2d: 4-104           [256, 256, 14, 14]        [256, 256, 14, 14]        512                       --
│    │    │    └─ReLU: 4-105                  [256, 256, 14, 14]        [256, 256, 14, 14]        --                        --
│    │    │    └─Conv2d: 4-106                [256, 256, 14, 14]        [256, 256, 14, 14]        589,824                   [3, 3]
│    │    │    └─BatchNorm2d: 4-107           [256, 256, 14, 14]        [256, 256, 14, 14]        512                       --
│    │    │    └─ReLU: 4-108                  [256, 256, 14, 14]        [256, 256, 14, 14]        --                        --
│    │    │    └─Conv2d: 4-109                [256, 256, 14, 14]        [256, 1024, 14, 14]       262,144                   [1, 1]
│    │    │    └─BatchNorm2d: 4-110           [256, 1024, 14, 14]       [256, 1024, 14, 14]       2,048                     --
│    │    │    └─ReLU: 4-111                  [256, 1024, 14, 14]       [256, 1024, 14, 14]       --                        --
│    │    └─Bottleneck: 3-13                  [256, 1024, 14, 14]       [256, 1024, 14, 14]       --                        --
│    │    │    └─Conv2d: 4-112                [256, 1024, 14, 14]       [256, 256, 14, 14]        262,144                   [1, 1]
│    │    │    └─BatchNorm2d: 4-113           [256, 256, 14, 14]        [256, 256, 14, 14]        512                       --
│    │    │    └─ReLU: 4-114                  [256, 256, 14, 14]        [256, 256, 14, 14]        --                        --
│    │    │    └─Conv2d: 4-115                [256, 256, 14, 14]        [256, 256, 14, 14]        589,824                   [3, 3]
│    │    │    └─BatchNorm2d: 4-116           [256, 256, 14, 14]        [256, 256, 14, 14]        512                       --
│    │    │    └─ReLU: 4-117                  [256, 256, 14, 14]        [256, 256, 14, 14]        --                        --
│    │    │    └─Conv2d: 4-118                [256, 256, 14, 14]        [256, 1024, 14, 14]       262,144                   [1, 1]
│    │    │    └─BatchNorm2d: 4-119           [256, 1024, 14, 14]       [256, 1024, 14, 14]       2,048                     --
│    │    │    └─ReLU: 4-120                  [256, 1024, 14, 14]       [256, 1024, 14, 14]       --                        --
│    └─Sequential: 2-8                        [256, 1024, 14, 14]       [256, 2048, 7, 7]         --                        --
│    │    └─Bottleneck: 3-14                  [256, 1024, 14, 14]       [256, 2048, 7, 7]         --                        --
│    │    │    └─Conv2d: 4-121                [256, 1024, 14, 14]       [256, 512, 14, 14]        524,288                   [1, 1]
│    │    │    └─BatchNorm2d: 4-122           [256, 512, 14, 14]        [256, 512, 14, 14]        1,024                     --
│    │    │    └─ReLU: 4-123                  [256, 512, 14, 14]        [256, 512, 14, 14]        --                        --
│    │    │    └─Conv2d: 4-124                [256, 512, 14, 14]        [256, 512, 7, 7]          2,359,296                 [3, 3]
│    │    │    └─BatchNorm2d: 4-125           [256, 512, 7, 7]          [256, 512, 7, 7]          1,024                     --
│    │    │    └─ReLU: 4-126                  [256, 512, 7, 7]          [256, 512, 7, 7]          --                        --
│    │    │    └─Conv2d: 4-127                [256, 512, 7, 7]          [256, 2048, 7, 7]         1,048,576                 [1, 1]
│    │    │    └─BatchNorm2d: 4-128           [256, 2048, 7, 7]         [256, 2048, 7, 7]         4,096                     --
│    │    │    └─Sequential: 4-129            [256, 1024, 14, 14]       [256, 2048, 7, 7]         --                        --
│    │    │    │    └─Conv2d: 5-7             [256, 1024, 14, 14]       [256, 2048, 7, 7]         2,097,152                 [1, 1]
│    │    │    │    └─BatchNorm2d: 5-8        [256, 2048, 7, 7]         [256, 2048, 7, 7]         4,096                     --
│    │    │    └─ReLU: 4-130                  [256, 2048, 7, 7]         [256, 2048, 7, 7]         --                        --
│    │    └─Bottleneck: 3-15                  [256, 2048, 7, 7]         [256, 2048, 7, 7]         --                        --
│    │    │    └─Conv2d: 4-131                [256, 2048, 7, 7]         [256, 512, 7, 7]          1,048,576                 [1, 1]
│    │    │    └─BatchNorm2d: 4-132           [256, 512, 7, 7]          [256, 512, 7, 7]          1,024                     --
│    │    │    └─ReLU: 4-133                  [256, 512, 7, 7]          [256, 512, 7, 7]          --                        --
│    │    │    └─Conv2d: 4-134                [256, 512, 7, 7]          [256, 512, 7, 7]          2,359,296                 [3, 3]
│    │    │    └─BatchNorm2d: 4-135           [256, 512, 7, 7]          [256, 512, 7, 7]          1,024                     --
│    │    │    └─ReLU: 4-136                  [256, 512, 7, 7]          [256, 512, 7, 7]          --                        --
│    │    │    └─Conv2d: 4-137                [256, 512, 7, 7]          [256, 2048, 7, 7]         1,048,576                 [1, 1]
│    │    │    └─BatchNorm2d: 4-138           [256, 2048, 7, 7]         [256, 2048, 7, 7]         4,096                     --
│    │    │    └─ReLU: 4-139                  [256, 2048, 7, 7]         [256, 2048, 7, 7]         --                        --
│    │    └─Bottleneck: 3-16                  [256, 2048, 7, 7]         [256, 2048, 7, 7]         --                        --
│    │    │    └─Conv2d: 4-140                [256, 2048, 7, 7]         [256, 512, 7, 7]          1,048,576                 [1, 1]
│    │    │    └─BatchNorm2d: 4-141           [256, 512, 7, 7]          [256, 512, 7, 7]          1,024                     --
│    │    │    └─ReLU: 4-142                  [256, 512, 7, 7]          [256, 512, 7, 7]          --                        --
│    │    │    └─Conv2d: 4-143                [256, 512, 7, 7]          [256, 512, 7, 7]          2,359,296                 [3, 3]
│    │    │    └─BatchNorm2d: 4-144           [256, 512, 7, 7]          [256, 512, 7, 7]          1,024                     --
│    │    │    └─ReLU: 4-145                  [256, 512, 7, 7]          [256, 512, 7, 7]          --                        --
│    │    │    └─Conv2d: 4-146                [256, 512, 7, 7]          [256, 2048, 7, 7]         1,048,576                 [1, 1]
│    │    │    └─BatchNorm2d: 4-147           [256, 2048, 7, 7]         [256, 2048, 7, 7]         4,096                     --
│    │    │    └─ReLU: 4-148                  [256, 2048, 7, 7]         [256, 2048, 7, 7]         --                        --
│    └─AdaptiveAvgPool2d: 2-9                 [256, 2048, 7, 7]         [256, 2048, 1, 1]         --                        --
│    └─Linear: 2-10                           [256, 2048]               [256, 1000]               2,049,000                 --
=================================================================================================================================================
Total params: 25,557,032
Trainable params: 25,557,032
Non-trainable params: 0
Total mult-adds (Units.TERABYTES): 1.05
=================================================================================================================================================
Input size (MB): 154.14
Forward/backward pass size (MB): 45524.93
Params size (MB): 102.23
Estimated Total Size (MB): 45781.30
=================================================================================================================================================
```
</details>


**Cost Breakdown**

| Task | Est. Cost (USD) |
|------|-----------------|
| Dataset Download + Unzip | ≈ $6 |
| Training (195 epochs) | ≈ $79 |
| **Total** | **≈ $85** |

**Notes**
- Dataset mounted at `/mnt/imagenet1k`.  
- ~2× faster per epoch vs local (run on A10G + DALI).  
- W&B Dashboard: add link → `https://wandb.ai/<user>/imagenet1k_runs`


---

**Summary Comparison**

| Feature | Local (4060 Ti) | AWS (A10G) |
|----------|-----------------|-------------|
| Batch Size | 64 | 256 |
| Total Epoch | 235 | 195 |
| Precision | AMP | AMP + DALI |
| Epoch Time | ~60 min | ~30 min |
| Total Training Hours | ~240 | ~90 |
| Best Val Top-1 | 77.82% | 77.66% |
| Best Val Top-5 | 93.82% | 93.84% |
| Storage Path | `data/imagenet` | `/mnt/imagenet1k` |

---