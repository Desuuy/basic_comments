# Vast.ai Setup Guide for MultiFrame-LPR Training

This document provides step-by-step commands and hardware recommendations for running training on Vast.ai.

---

## 1. Google Drive Links (Your Assets)

| Asset | Google Drive Link | File ID |
|-------|-------------------|---------|
| Training data (train.zip) | [train.zip](https://drive.google.com/file/d/1xrk7Cfig3PEvXRYIHYhdPVli8f1RD74t/view?usp=sharing) | `1xrk7Cfig3PEvXRYIHYhdPVli8f1RD74t` |
| val_tracks.json | [val_tracks.json](https://drive.google.com/file/d/16Rq3SELxcbM9WMvULs93lBpt-jJDyUaK/view?usp=sharing) | `16Rq3SELxcbM9WMvULs93lBpt-jJDyUaK` |
| SVTRv2 pretrained (best.pth) | [best.pth](https://drive.google.com/file/d/1wMqMWpf2K646NJ91UUT9E1oPlv7ZwqJN/view?usp=sharing) | `1wMqMWpf2K646NJ91UUT9E1oPlv7ZwqJN` |
| SR checkpoint (opt) | [I80000_E41_opt_best_psnr.pth](https://drive.google.com/file/d/12IJKQ0-K_FcMlYqfcbw9vJs-rcz4Nohq/view?usp=sharing) | `12IJKQ0-K_FcMlYqfcbw9vJs-rcz4Nohq` |

Note: For SR you need the **GEN** checkpoint (`*_gen_best_psnr.pth`). The opt file is for optimizer state. If you have `I80000_E41_gen_best_psnr.pth`, use that for `SR_CHECKPOINT_PATH`.

---

## 2. Vast.ai Hardware Recommendations

### Without SR (recommended for cost/speed)

| GPU | VRAM | Recommended batch size | Est. cost (spot) |
|-----|------|-------------------------|------------------|
| RTX 3060 | 12 GB | 32–48 | Low |
| RTX 3070 | 8 GB | 24–32 | Low |
| RTX 3080 | 10 GB | 32–48 | Medium |
| RTX 3090 | 24 GB | 64–96 | Medium |
| A100 40GB | 40 GB | 128+ | High |

### With SR (very slow; use only for experiments)

- GPU: RTX 3090 / A100 with 24+ GB VRAM.
- Batch size: 4–8.
- Expect long training time due to diffusion (1000 steps per SR call).

### Optimization tips for Vast.ai

1. **Use spot instances** – cheaper than on-demand.
2. **Choose GPU with enough VRAM** – avoid OOM; 8 GB minimum for batch 32 without SR.
3. **Storage** – 50–100 GB for data + checkpoints.
4. **Set `USE_SR=False`** – SR is expensive; train OCR first, add SR later if needed.
5. **`num_workers`** – 4–8 for faster data loading; 0 if you see multiprocessing errors.
6. **`--aug-level light`** – faster if you need quick iterations.

---

## 3. Commands: Before Training

### 3.1. Clone the repository and enter the project

```bash
git clone https://github.com/Desuuy/ICPR_Challenge.git MultiFrame-LPR
cd MultiFrame-LPR
```

### 3.2. Create virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Linux; on Windows: .venv\Scripts\activate

# Using uv (if available)
uv sync

# Or using pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install albumentations opencv-python matplotlib numpy pandas tqdm tqdm
```

### 3.3. Download data from Google Drive

Install `gdown` if needed:

```bash
pip install gdown
```

Download and extract:

```bash
# Training data
gdown "" -O train.zip
unzip train.zip -d .
# Assuming train.zip extracts to data/train or similar; adjust if needed
# If it extracts to current dir: mv train data/  (or whatever structure you need)

# val_tracks.json
mkdir -p data
gdown "" -O data/val_tracks.json

# SVTRv2 pretrained (best.pth)
mkdir -p weights
gdown "" -O weights/best.pth

# SR checkpoint (optional; only if USE_SR=True)
# You need I80000_E41_gen_best_psnr.pth for SR, not the opt file.
# If you have gen file:
# gdown "<gen_file_id>" -O weights/I80000_E41_gen_best_psnr.pth
```

Verify structure:

```bash
ls -la data/
ls -la data/train/   # or wherever train.zip extracted
ls -la weights/
```

### 3.4. Set DATA_ROOT and paths

If your data is not under `data/train`, set `--data-root` when training, e.g.:

```bash
# Example if train extracted to ./train
# --data-root ./train
```

---

## 4. Commands: Training

### 4.1. Basic training (no SR, fast)

```bash
python train.py \
  --model mf_svtrv2 \
  --experiment-name mfsvtrv2_vastai \
  --data-root data/train \
  --batch-size 32 \
  --epochs 30 \
  --lr 3.25e-4 \
  --output-dir results
```

### 4.2. With more GPU memory (batch 64)

```bash
python train.py \
  --model mf_svtrv2 \
  --experiment-name mfsvtrv2_bs64 \
  --data-root data/train \
  --batch-size 64 \
  --epochs 30 \
  --output-dir results
```

### 4.3. Light augmentation (faster)

```bash
python train.py \
  --model mf_svtrv2 \
  --aug-level light \
  --batch-size 32 \
  --epochs 30 \
  --output-dir results
```

### 4.4. Submission mode (full train + test inference)

```bash
python train.py \
  --submission-mode \
  --model mf_svtrv2 \
  --data-root data/train \
  --batch-size 32 \
  --epochs 30 \
  --output-dir results
```

### 4.5. With SR (slow; only for experiments)

```bash
python train.py \
  --model mf_svtrv2 \
  --use-sr \
  --sr-checkpoint-path weights/I80000_E41_gen_best_psnr.pth \
  --batch-size 4 \
  --epochs 5 \
  --output-dir results
```

---

## 5. One-Line Setup Script (copy-paste)

Run this block after connecting to your Vast.ai instance:

```bash
# Setup
cd ~
git clone https://github.com/Desuuy/ICPR_Challenge.git MultiFrame-LPR
cd MultiFrame-LPR

python -m venv .venv
source .venv/bin/activate
pip install uv 2>/dev/null || true
uv sync 2>/dev/null || pip install torch torchvision torchaudio albumentations opencv-python matplotlib numpy pandas tqdm

pip install gdown

# Download
mkdir -p data weights
gdown "" -O train.zip
unzip -o train.zip
gdown "" -O data/val_tracks.json
gdown "" -O weights/best.pth

# Adjust paths: if train.zip extracted to 'train/', set DATA_ROOT
# export DATA_ROOT=./train   # or full path

# Train
python train.py --model mf_svtrv2 --data-root ./train --batch-size 32 --epochs 30 --output-dir results
```

Adjust `--data-root` and `--batch-size` based on your extraction layout and GPU.

---

## 6. After Training

Outputs are in `results/`:

- `results/mf_svtrv2_vastai_best.pth` – best checkpoint
- `results/submission_*.txt` – predictions
- `results/wrong_predictions_*.txt` – error analysis

Download them from the instance (Vast.ai file browser or `scp`).

