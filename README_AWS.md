# 📘 AWS ImageNet Dataset & Training Pipeline Setup (Final)

This document details the **end‑to‑end AWS workflow** we actually used for **ImageNet‑1K** with **ResNet‑50 / OneCycle / AMP**: dataset download and snapshotting, GPU training setup (single & multi‑GPU), robust resume/freeze logic, cost control, and a clean way to **store heavy artifacts on the NVMe data volume** while keeping the repo light via symlinks.

---

## 🧠 Appendix A — ImageNet Dataset Setup on AWS

This section documents how the **ImageNet‑1K (ILSVRC‑2017)** dataset and **Imagenette (mini‑ImageNet)** were downloaded, organized, and preserved as EBS snapshots.

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

### 🪜 Steps performed

#### 1) Attach & mount the 400 GB volume
(Attached in console as `/dev/sdf`, shows on the instance as `/dev/nvme1n1`)
```bash
sudo mkfs -t ext4 /dev/nvme1n1
sudo mkdir -p /mnt/data
sudo mount /dev/nvme1n1 /mnt/data
sudo chown -R ubuntu:ubuntu /mnt/data
df -h
# Expect something like:
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
```bash
cd /mnt/data/imagenet_full/imagenet/val
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
bash valprep.sh
rm valprep.sh
```
**Resulting structure:**
```
/mnt/data/imagenet_full/imagenet/
  ├── train/ n01440764/ ... n15075141/
  ├── val/   n01440764/ ... n15075141/
  └── test/  (optional)
```

#### 5) Create snapshot(s)
- Full ImageNet snapshot: `snap-05ad5ff67e9c9d9b2`  
- Imagenette snapshot: `snap-08f4facd61be06644`

> We will **mount the dataset as `/mnt/imagenet1k`** in training instances for a stable path.

---

## ⚡ Appendix B — GPU Instance & Training Setup

### ⚙️ GPU Instance (Training Phase)

| Parameter | Value |
|---|---|
| **Instance Types used** | `g5.xlarge`, `g5.4xlarge`, `g5.12xlarge` |
| **GPU** | NVIDIA **A10G** (24 GB) |
| **AMI** | *Deep Learning OSS Nvidia Driver AMI PyTorch 2.8 (Ubuntu 24.04)* |
| **Root Volume** | 45–60 GiB (gp3) |
| **Pricing** | On‑Demand or Spot (if you need “Stop on interruption”, set Spot Request **Persistent**) |
| **Key Pair** | `imagenet_resnet_gpu.pem` |

### 1) Create a volume from snapshot & attach to the GPU instance
**Full ImageNet (400 GB):**
```bash
aws ec2 create-volume \
  --snapshot-id snap-05ad5ff67e9c9d9b2 \
  --availability-zone ap-south-1a \
  --volume-type gp3 --size 400 \
  --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=ImageNet-Full-Data}]'
```
Attach in **EC2 → Volumes → Attach** (device `/dev/sdf`, appears as `/dev/nvme1n1`).

### 2) Mount dataset at `/mnt/imagenet1k`
```bash
# Check available volumes
lsblk
# Mount dataset (after attaching the EBS volume)
sudo mkdir -p /mnt/imagenet1k
sudo mount /dev/nvme1n1 /mnt/imagenet1k
sudo chown -R ubuntu:ubuntu /mnt/imagenet1k
df -h /mnt/imagenet1k
# Expect ~400GB mounted
# Verify dataset contents
ls /mnt/imagenet1k
# Expected: train/  val/  test/

```

### 3) Clone repo → create venv → install requirements
```bash
# Repo location (as requested)
cd ~
git clone https://github.com/<YOUR_USERNAME>/week9_ERAV4_ImageNet_ResNet-50_Model_Experiments.git
cd ~/week9_ERAV4_ImageNet_ResNet-50_Model_Experiments

# Venv on fast local NVMe
sudo mkdir -p /opt/dlami/nvme/envs
sudo chown -R ubuntu:ubuntu /opt/dlami/nvme
python3 -m venv /opt/dlami/nvme/envs/imagenet1k_venv
source /opt/dlami/nvme/envs/imagenet1k_venv/bin/activate

# Install deps
pip install -r requirements_aws.txt
# If you keep a requirements file in the repo:
# pip install -r requirements_aws.txt
# Or install explicitly (CUDA 12.1 wheels):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# pip install timm albumentations opencv-python tqdm wandb
```

### 4) Keep large outputs on the dataset volume via symlinks

We’ll **move heavy folders** from the repo onto `/mnt/imagenet1k` and replace them with symlinks so:
- training writes to fast NVMe volume,
- the Git repo stays light and clean.

**Create target dirs on the data volume:**
```bash
mkdir -p /mnt/imagenet1k/{checkpoints,out,reports,wandb,runs}
```

**Move existing contents (if any) from repo → data volume, then symlink:**
```bash
cd ~/week9_ERAV4_ImageNet_ResNet-50_Model_Experiments

# If these folders exist in the repo, move them
[ -d checkpoints ] && mv checkpoints/* /mnt/imagenet1k/checkpoints/ 2>/dev/null || true
[ -d out ]         && mv out/*         /mnt/imagenet1k/out/         2>/dev/null || true
[ -d reports ]     && mv reports/*     /mnt/imagenet1k/reports/     2>/dev/null || true
[ -d wandb ]       && mv wandb/*       /mnt/imagenet1k/wandb/       2>/dev/null || true
[ -d runs ]        && mv runs/*        /mnt/imagenet1k/runs/        2>/dev/null || true

# Remove the (now-empty) repo dirs and replace them with symlinks
rm -rf checkpoints out reports wandb runs
#Create Symbolic Links (Symlinks)
ln -s /mnt/imagenet1k/checkpoints checkpoints
ln -s /mnt/imagenet1k/out         out
ln -s /mnt/imagenet1k/reports     reports
ln -s /mnt/imagenet1k/wandb       wandb
ln -s /mnt/imagenet1k/runs        runs
#Now all training outputs are written to the high-speed mounted volume but remain visible from your repo path.
# Verify
ls -l | egrep 'checkpoints|out|reports|wandb|runs'
```

**When you want to `git push`** (and include the latest artifacts in the repo temporarily):
```bash
cd ~/week9_ERAV4_ImageNet_ResNet-50_Model_Experiments

# 1) remove symlinks
rm checkpoints out reports wandb runs

# 2) recreate real folders and sync back
mkdir -p checkpoints out reports wandb runs
rsync -a --ignore-existing /mnt/imagenet1k/checkpoints/ checkpoints/
rsync -a --ignore-existing /mnt/imagenet1k/out/         out/
rsync -a --ignore-existing /mnt/imagenet1k/reports/     reports/
rsync -a --ignore-existing /mnt/imagenet1k/wandb/       wandb/
rsync -a --ignore-existing /mnt/imagenet1k/runs/        runs/

# 3) commit & push (adjust what you want tracked; usually you KEEP these dirs gitignored)
# git add -A && git commit -m "Sync artifacts from NVMe volume" && git push

# 4) restore the symlink setup again
rm -rf checkpoints out reports wandb runs
ln -s /mnt/imagenet1k/checkpoints checkpoints
ln -s /mnt/imagenet1k/out         out
ln -s /mnt/imagenet1k/reports     reports
ln -s /mnt/imagenet1k/wandb       wandb
ln -s /mnt/imagenet1k/runs        runs

# sanity check:
ls -l | egrep 'checkpoints|out|reports|wandb|runs'
```

> Tip: it’s typical to **.gitignore** these folders and avoid pushing heavy artifacts. The above block is only if you **intentionally** want to sync/push a subset.

### 5) Launch training (tmux) - Fresh trainings. For resuming training from other chcekpoint see next section

**Create session & attach:**
```bash
tmux new -s imagenet1k_full -n train
# reattach later:
# tmux attach -t imagenet1k_full
# list sessions:
# tmux ls
```

**Inside tmux:**
```bash
source /opt/dlami/nvme/envs/imagenet1k_venv/bin/activate
cd ~/week9_ERAV4_ImageNet_ResNet-50_Model_Experiments
```

**Single GPU (g5.xlarge):**
```bash
bash scripts/launch_single_gpu.sh /mnt/imagenet1k 150 256 6 \
  --max-lr 0.125 \
  --stats-file data_stats/imagenet_1k_aws_stats.json \
  --show-progress --amp --channels-last \
  --out-dir imagenet1kfull_g5x_1gpu_dali_nvme_lr0p125_bs256_e150_work6 \
  --wandb --wandb-project imagenet1k_runs \
  --wandb-tags imagenet1k_full,dali,1gpu,nvme,lr0p125,bs256,e150,work6
```

**Multi‑GPU (e.g., g5.12xlarge with 4 GPUs):**
```bash
bash scripts/launch_multi_gpu.sh 4 /mnt/imagenet1k 150 256 12 \
  --max-lr 0.25 \
  --stats-file data_stats/imagenet_1k_aws_stats.json \
  --amp --channels-last \
  --out-dir imagenet1kfull_g5_4gpu_dali_nvme_lr0p25_bs256x4_e150_work12 \
  --wandb --wandb-project imagenet1k_runs \
  --wandb-tags imagenet1k_full,dali,4gpu,nvme,lr0p25,bs256x4,e150,work12
```

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
### 🧩 Environment Variable Summary

| Variable                  | Description                               | Typical Use                         |
| ------------------------- | ----------------------------------------- | ----------------------------------- |
| `RESET_SCHED=1`           | Rebuild scheduler when extending epochs   | e.g., resuming from epoch 120 → 150 |
| `FREEZE_LR=1`             | Freeze LR to constant value for stability | e.g., fine-tuning or last 10 epochs |
| `FREEZE_LR_VALUE=<float>` | Explicit constant LR value                | Optional override when freezing     |

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

### 🧾 Notes
If your last checkpoint was produced by a single-GPU run, you can still resume on multi-GPU (and vice-versa) as long as your training script saves the usual state_dict and optimizer/scheduler state—DDP will wrap the model on launch. Just make sure:

- You pass `--resume <.pth>` that contains model, optimizer, scaler, and epoch.  
- The data path and class count match what the checkpoint was trained on.

✅ Tip:
Always verify the current LR and schedule in Weights & Biases or in out/train_log.csv to ensure your resume or freeze behaves as expected.
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
| Spot specifics | For **Stop on interruption**, set **Request type = Persistent** | Otherwise Spot interruption defaults to **Terminate** |

### Tmux quick ref
```bash
tmux ls                        # list sessions
tmux attach -t imagenet1k_full # reattach
tmux new -s imagenet1k_full    # new session
tmux kill-session -t imagenet1k_full  # kill it
```

### Final checklist
- [x] Instances stopped when not training  
- [x] Only snapshots retained (`ImageNet‑Full`, `Imagenette`)  
- [x] Unused volumes deleted  
- [x] Budget alert configured  
- [x] Reproducible restore path from snapshot(s)

---

### ✅ Summary

- **Datasets**: Full ImageNet‑1K + Imagenette prepared and organized; validation folder fixed via `valprep.sh`.  
- **Persistence**: Both datasets preserved as snapshots.  
- **Training**: g5 family (A10G) instances with AMP, OneCycle, DDP‑ready.  
- **Artifacts management**: Heavy outputs live on `/mnt/imagenet1k` via symlinks.  
- **Cost safety**: Stop/terminate policy, delete detached volumes, keep snapshots only.
