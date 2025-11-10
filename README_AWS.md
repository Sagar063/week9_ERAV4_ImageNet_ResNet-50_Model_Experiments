# 📘 AWS ImageNet Dataset & Training Pipeline Setup

This document details the full process of **downloading, organizing, snapshotting, and training** on the **ImageNet‑1K** dataset using **AWS EC2 (g5 family, NVIDIA A10G)** with **ResNet‑50 / OneCycle / AMP**. It reflects the **actual steps** we executed in this project and includes safe **cost‑control** procedures.

---

## 🧠 Appendix A — ImageNet Dataset Setup on AWS (Manual Steps Followed)

This section documents how the **ImageNet‑1K (ILSVRC‑2017)** dataset and **Imagenette (mini‑ImageNet)** were downloaded, organized, and preserved as **EBS snapshots**.

### ⚙️ EC2 Configuration (Download/Prep Phase)

| Parameter | Value |
|---|---|
| **Region** | Asia Pacific (Mumbai) – `ap-south-1` |
| **Instance Type** | `t3.2xlarge` (8 vCPU / 32 GiB RAM) |
| **Root Volume** | 30 GiB (gp3) |
| **Attached Data Volume** | 400 GiB (gp3) mounted at `/mnt/data` |
| **AMI** | *Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.8 (Ubuntu 24.04)* |
| **Key Pair** | `imagenet_resnet_gpu.pem` |
| **Security Group** | SSH (22) open for setup (later restrict to your IP) |
| **Purpose** | Download + organize ImageNet; create reusable snapshot(s) |

### 🪜 Procedure (Actual Commands Used)

#### 1) Attach & mount the 400 GB volume
> We attached the new EBS volume in the console as **/dev/sdf**; on the instance it appeared as **/dev/nvme1n1** (Nitro).

```bash
sudo mkfs -t ext4 /dev/nvme1n1
sudo mkdir -p /mnt/data
sudo mount /dev/nvme1n1 /mnt/data
sudo chown -R ubuntu:ubuntu /mnt/data
df -h
# Expected:
# /dev/nvme1n1  393G   28K  373G   1%  /mnt/data
```

#### 2) Install packages
```bash
sudo apt update -y
sudo apt install -y aria2 unzip wget
```

#### 3) Download ImageNet via Academic Torrents
```bash
cd /mnt/data
mkdir -p imagenet_full && cd imagenet_full

MAGNET_LINK='magnet:?xt=urn:btih:943977d8c96892d24237638335e481f3ccd54cfb&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce'

# Background download
aria2c --dir=/mnt/data/imagenet_full --enable-rpc=false \
  --max-concurrent-downloads=1 --continue=true --seed-time=0 "$MAGNET_LINK" &

# Monitor
ps aux | grep aria2c
ls -lh /mnt/data/imagenet_full
```

#### 4) Extract & organize
```bash
cd /mnt/data/imagenet_full
tar -xzf ILSVRC2017_CLS-LOC.tar.gz

mkdir -p imagenet
mv ILSVRC/Data/CLS-LOC/train imagenet/
mv ILSVRC/Data/CLS-LOC/val   imagenet/
mv ILSVRC/Data/CLS-LOC/test  imagenet/
```

#### 🔧 Fix the validation folder structure
ImageNet validation images ship in a flat directory. We organized them into **1000 class sub‑folders** using Soumith’s script:

```bash
cd /mnt/data/imagenet_full/imagenet/val
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
bash valprep.sh
rm valprep.sh
```

**Resulting structure:**
```
imagenet/
  ├── train/ n01440764/ ... n15075141/
  ├── val/   n01440764/ ... n15075141/
  └── test/  (optional)
```

#### 5) Create snapshot (Full ImageNet)
Console: **EC2 → Volumes → (400 GB volume) → Actions → Create snapshot**  
**Description:** `ImageNet-Full-Backup-Oct2025`  
Snapshot **ID:** `snap-05ad5ff67e9c9d9b2` ✅

---

### 🧩 Imagenette (Mini‑ImageNet subset)
We also prepared a small subset for quick pipeline tests.

```bash
cd /mnt/data
mkdir -p imagenet_mini && cd imagenet_mini
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xvf imagenette2-320.tgz
```
Structure:
```
imagenette2-320/
  ├── train/
  ├── val/
  └── noisy_imagenette.csv
```
Stored on a 10 GB EBS volume and snapshotted as: **`snap-08f4facd61be06644`** ✅

---

## ⚡ Appendix B — GPU Instance & Training Setup

### ⚙️ GPU Instance Configuration (Training Phase)

| Parameter | Value |
|---|---|
| **Instance Types used** | `g5.xlarge`, `g5.4xlarge`, `g5.12xlarge` |
| **GPU** | NVIDIA **A10G** (24 GB) |
| **AMI** | *Deep Learning OSS Nvidia Driver AMI PyTorch 2.8 (Ubuntu 24.04)* |
| **Root Volume** | 45–60 GiB (gp3) |
| **Pricing** | On‑Demand or Spot (Spot: persistent off, interruption = stop) |
| **Key Pair** | `imagenet_resnet_gpu.pem` |

### 1) Recreate a dataset volume from snapshot
**Full ImageNet (400 GB):**
```bash
aws ec2 create-volume \
  --snapshot-id snap-05ad5ff67e9c9d9b2 \
  --availability-zone ap-south-1a \
  --volume-type gp3 --size 400 \
  --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=ImageNet-Full-Data}]'
```
*(For Imagenette mini use `snap-08f4facd61be06644` and size 10 GB.)*

### 2) Attach to running GPU instance
Console: **EC2 → Volumes → (select volume) → Attach**  
- **Instance:** your `g5.*` EC2  
- **Device name:** `/dev/sdf` (shows as `/dev/nvme1n1` on the instance)

### 3) Mount inside the instance
```bash
lsblk
sudo mkdir -p /mnt/data
sudo mount /dev/nvme1n1 /mnt/data
sudo chown -R ubuntu:ubuntu /mnt/data
df -h /mnt/data
# Expect to see ~400GB mounted at /mnt/data
```

**Sanity check:**
```bash
ls /mnt/data/imagenet_full/imagenet/train | wc -l  # 1000
ls /mnt/data/imagenet_full/imagenet/val   | wc -l  # 1000
```

### 4) Environment setup
```bash
nvidia-smi

# virtual env
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

### 5) Clone your repo
```bash
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/Sagar063/week9_ERAV4_ImageNet_ResNet-50_Model_Experiments.git
cd week9_ERAV4_ImageNet_ResNet-50_Model_Experiments
```

### 6) VS Code Remote‑SSH
Local `~/.ssh/config`:
```sshconfig
Host aws-g5
  HostName <EC2_Public_IP>
  User ubuntu
  IdentityFile ~/.ssh/imagenet_resnet_gpu.pem
  ServerAliveInterval 60
  ServerAliveCountMax 5
```
VS Code → *Remote‑SSH: Connect to Host…* → `aws-g5`

### 7) Launch training

**Single GPU (g5.xlarge):**
```bash
tmux new -s imagenet1k_full -n train

# Inside tmux
source /opt/dlami/nvme/envs/imagenet1k_venv/bin/activate
cd ~/week9_ERAV4_ImageNet_ResNet-50_Model_Experiments

bash scripts/launch_single_gpu.sh /mnt/imagenet1k 150 256 6   --max-lr 0.125   --stats-file data_stats/imagenet_1k_aws_stats.json   --show-progress   --amp --channels-last   --out-dir imagenet1kfull_g5x_1gpu_dali_nvme_lr0p125_bs256_e150_work6   --wandb --wandb-project imagenet1k_runs   --wandb-tags imagenet1k_full,dali,1gpu,nvme,lr0p125,bs256,e150,work6
```

**Multi‑GPU (e.g., g5.12xlarge with 4 GPUs):**
```bash
tmux new -s imagenet1k_full -n train

# inside tmux
source /opt/dlami/nvme/envs/imagenet1k_venv/bin/activate
cd ~/week9_ERAV4_ImageNet_ResNet-50_Model_Experiments

# NGPUS  DATA_PATH    EPOCHS  PER_GPU_BATCH  WORKERS   --extra-args...
bash scripts/launch_multi_gpu.sh 4 /mnt/imagenet1k 150 256 12 \
  --max-lr 0.25 \
  --stats-file data_stats/imagenet_1k_aws_stats.json \
  --amp --channels-last \
  --out-dir imagenet1kfull_g5_4gpu_dali_nvme_lr0p25_bs256x4_e150_work12 \
  --wandb --wandb-project imagenet1k_runs \
  --wandb-tags imagenet1k_full,dali,4gpu,nvme,lr0p25,bs256x4,e150,work12
```

🔁 For Resume training  refer to Appendix C

Notes

launch_multi_gpu.sh takes: NGPUS DATA EPOCHS BATCH WORKERS ...EXTRA_ARGS.

BATCH is per-GPU in your multi script (it passes --batch-size straight through to your train script). So with 256 per GPU and 4 GPUs, global batch = 1024.

Scaled LR rule-of-thumb: if single-GPU 256 used --max-lr 0.125, 4× GPUs → LR ≈ 0.5. I suggested a conservative 0.25. Tweak after a short burn-in/val check.

Workers: g5.12xlarge has 48 vCPUs; 12 workers is a good start (scale 10–16 if input pipeline is the bottleneck).

> Ensure your script is **DDP‑ready** (uses `DistributedSampler`, logs on rank‑0 only, and AMP via `autocast`/`GradScaler`).

---
## 🧾 Appendix C — Resuming trainings 

The training logic now supports **resuming, extending, or freezing LR schedules** cleanly using environment variables — no code edits needed.

### 1️⃣ Fresh Training Run
**Single GPU (g5.xlarge):**
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

### 2️⃣ Resume Training (Normal Resume)
When continuing from a saved checkpoint (`--resume checkpoints/.../last_epochXXX.pth`),  
just run normally — no env vars required:
```bash
bash scripts/launch_single_gpu.sh /mnt/imagenet1k 150 256 6 \
  --resume checkpoints/imagenet1kfull_g5x_1gpu_dali_nvme_lr0p125_bs256_e150_work6/last_epoch150.pth \
  --max-lr 0.125 --amp --channels-last --show-progress
```

The scheduler resumes **exactly** where it left off, restoring all states (optimizer, scaler, scheduler).

---

### 3️⃣ Resume and Extend Total Epochs
If you trained 0‑120 and now want to continue to 150 epochs:
```bash
export RESET_SCHED=1      # rebuild scheduler for the new end‑epoch
unset FREEZE_LR           # let OneCycleLR handle learning rate
unset FREEZE_LR_VALUE
```
Then launch as usual with `--epochs 150`.  
The scheduler realigns itself from the current `start_epoch` → new total.

---

### 4️⃣ Resume and Freeze Learning Rate
If validation accuracy is stable and you want constant LR continuation:
```bash
export FREEZE_LR=1
unset RESET_SCHED
```
Optionally, specify a custom LR:
```bash
export FREEZE_LR_VALUE=0.00005
```
This freezes the LR for the rest of the run — useful for convergence stabilization.

---

### 5️⃣ Combined: Resume + Extend + Freeze
```bash
export RESET_SCHED=1
export FREEZE_LR=1
unset FREEZE_LR_VALUE
```
This rebuilds the scheduler for new epochs but immediately replaces it with a constant‑LR version.

---

### ✅ Example Scenarios

| Case | Environment | Result |
|------|--------------|--------|
| Train fresh run with fixed LR | `FREEZE_LR=1`, no resume | Constant LR entire run |
| Resume normally | (no env vars) | Scheduler continues smoothly |
| Resume + extend epochs | `RESET_SCHED=1` | Scheduler realigned, OneCycle continues |
| Resume + freeze LR | `FREEZE_LR=1` | LR frozen to last used value |
| Resume + extend + freeze | `RESET_SCHED=1 + FREEZE_LR=1` | Scheduler rebuilt, then frozen immediately |

---

### 💡 Why this approach is better

| Concept | Controlled by | Purpose |
|----------|---------------|----------|
| Resume logic | `--resume`, `RESET_SCHED` | Decides where to start, which weights, how to align scheduler |
| Freeze logic | `FREEZE_LR`, `FREEZE_LR_VALUE` | Controls whether LR stays constant or dynamic |

---

### 🧾 Summary of Improvements

| Issue | Old Behavior | New Behavior |
|--------|---------------|---------------|
| Resume with new total epochs | LR curve restarted or spiked | Smoothly realigned & clamped |
| Freeze LR | Sometimes froze to 0 | Properly freezes to valid last LR or custom value |
| No resume | Scheduler still looked for ckpt | Clean start path (`ckpt=None`) |
| Scheduler handling | Custom patchwork | Unified inside PyTorch `_LRScheduler` |
| Readability | Mixed resume/freeze logic | Clearly separated, labeled sections |

### 🧾Notes
If your last checkpoint was produced by a single-GPU run, you can still resume on multi-GPU (and vice-versa) as long as your training script saves the usual state_dict and optimizer/scheduler state—DDP will wrap the model on launch. Just make sure:

You pass --resume <.pth> that contains model, optimizer, scaler, and epoch.

The data path and class count match what the checkpoint was trained on.
---

## 🧾 Appendix D — Shutdown & Cost‑Control Guide

| Task | Action | Why |
|---|---|---|
| Pause compute | **Stop instance** | Stops GPU billing; keeps root EBS intact |
| End session | **Terminate instance** | Deletes root volume (unless “Delete on termination” disabled) |
| Preserve datasets | **Keep snapshots** | Cheap, S3‑backed; recreate volumes anytime |
| Avoid idle storage | **Delete detached volumes** | Prevents EBS $/GB‑month charges |
| Monitor cost | **Billing → Credits / Bills / Cost Explorer** | Credits are applied before card is charged |
| Alerts | **Budget alert at $20** | Early warning if something runs unexpectedly |

### Final checklist
- [x] Instances stopped when not training  
- [x] Only snapshots retained (`ImageNet‑Full`, `Imagenette`)  
- [x] Unused volumes deleted  
- [x] Budget alert configured  
- [x] Reproducible restore path from snapshot(s)

---

### ✅ Summary

- **Datasets**: Full ImageNet‑1K + Imagenette prepared and organized; validation folder fixed via `valprep.sh`.  
- **Persistence**: Both datasets preserved as snapshots (`snap-05ad5ff67e9c9d9b2`, `snap-08f4facd61be06644`).  
- **Training**: g5 family (A10G) instances with AMP, OneCycle, DDP‑ready.  
- **Cost safety**: Stop/terminate policy, delete detached volumes, keep snapshots only.

---
