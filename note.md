# Ghi chú lệnh - MultiFrame-LPR

```bash
git clone https://github.com/Desuuy/ICPR_Challenge.git MultiFrame-LPR
cd MultiFrame-LPR

uv sync

pip install gdown

# Train Data
mkdir Data/
cd Data/

# csv 
gdown 11MkyjJip8coA4HfrvhYaDeLR2fOmoxdY

# Text cho OCR
gdown 1wkY1VrkgYI1DvHKEEBvHbZ_pBmAOthxv

gdown 1xrk7Cfig3PEvXRYIHYhdPVli8f1RD74t 
unzip /workspace/MultiFrame-LPR/Data/train.zip

# Tải test 
gdown 1Wix1U-m1UOBMo2u72VA65j76VaL9g9bb 
unzip /workspace/MultiFrame-LPR/Data/test.zip

# tải blind_test
gdown 1NKPQFzYu5uV1CfQbafrWrpyUghFT3j_b
unzip blind_test

# Val_track.json
gdown 16Rq3SELxcbM9WMvULs93lBpt-jJDyUaK 

# weights

mkdir weights/
cd /workspace/MultiFrame-LPR/weights

# best ocr
gdown 1DHTvwn84lUqr0xGsmw7IIIQtyOHJ7Tdm

# best v2
gdown 1ADI9mCCMhCQyFcFOxQ7h1Sx55MI8NYAU

# v3
gdown 1BAcWme2c-bA-6N52cZHgIgewzqGysfEg



# SR weight
gdown 1Bw-uZqqsdnqL5j7QCx1Zxa5U5wc_h9hQ 

# Active Venv
cd /workspace/MultiFrame-LPR
source .venv/bin/activate

# ===== Clone OpenOCR rút gọn =====
cd /workspace
git clone https://github.com/Desuuy/MultiFrame_With_OpenOCR.git OpenOCR
cd OpenOCR

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# tải LMDB cho OpenOCR
cd /workspace
gdown 13VvTP5yhlUgJVABAKf0lpZiBEujIjmUf
unzip lmdb_for_openocr.zip

```
## 1.2 Thực hiện train

```bash
# Ếnmble 
cd C:\Users\anhhu\MultiFrame-LPR
.\scripts\run_ensemble.ps1


cd C:\Users\anhhu\MultiFrame-LPR

python scripts/ensemble_three_models.py weights/ocr_best.pth weights/v2_best.pth weights/v3_best.pth weights/best.pth --mode val --output results/ensemble_val.csv --strategy vote 


python scripts/ensemble_three_models.py weights/ocr_best.pth weights/v2_best.pth weights/v3_best.pth weights/best.pth --mode test --data-root Data/public_test --output results/ensemble_test.csv --save-txt --strategy vote




# TTạonhanx 
python make_lp_label_list.py

python tools/infer_rec.py -c configs/rec/lp_svtrv2_gtc.yml


python train_optimized.py --no-checkpoint
```



## 2. Kích hoạt virtual environment (venv)

```bash
# Linux - dùng dấu gạch ngang (không dùng .ps1)
source /workspace/MultiFrame-LPR/.venv/bin/activate

# Hoặc nếu đã ở trong thư mục project
cd /workspace/MultiFrame-LPR
source .venv/bin/activate
```

---

## 3. Chạy training

```bash
cd /workspace/MultiFrame-LPR
python train.py --no-stn --epochs 30

python train.py --batch-size 64 --use-sr

python train.py --batch-size 64

# Train từ đầu có pretrained từ SVTRv2

python train_optimized.py --no-checkpoint --batch-size 64 --epochs 35

# Train từ đầu không pretrained không checkpoint
python train_optimized.py --no-pretrained --no-checkpoint --epochs 50

# 


# Theo LP-Diff.json (hiện tại là 100)
python train.py --use-sr

# Chạy nhanh với 10 steps
python train.py --use-sr --sr-n-timestep 10

# Chất lượng cao với 1000 steps
python train.py --use-sr --sr-n-timestep 1000
```

---








Nguyên nhân: loss bị clamp ở 20.0
Trong trainer.py (Focal CTC):
trainer.py
Lines 111-117
                    loss_per_sample_safe = torch.clamp(
                        loss_per_sample, min=-20.0, max=20.0)
                    loss_per_sample_safe = torch.nan_to_num(
                        loss_per_sample_safe, nan=20.0, posinf=20.0, neginf=-20.0)
                    clamped = loss_per_sample_safe
                    weight = (1 - torch.exp(-clamped)) ** 2
                    loss = (loss_per_sample_safe * weight).mean()
                    loss_per_sample_safe = torch.clamp(                        loss_per_sample, min=-20.0, max=20.0)                    loss_per_sample_safe = torch.nan_to_num(                        loss_per_sample_safe, nan=20.0, posinf=20.0, neginf=-20.0)                    clamped = loss_per_sample_safe                    weight = (1 - torch.exp(-clamped)) ** 2                    loss = (loss_per_sample_safe * weight).mean()
Khi CTC loss trả về:
inf hoặc nan → được thay bằng 20.0
giá trị > 20 → bị clamp max=20.0
→ Nếu nhiều sample có loss rất lớn/inf, trung bình sẽ gần 20.0.
Vì sao CTC sinh inf/loss lớn?
CTC trả về inf khi xác suất của alignment đúng ≈ 0. Thường do:
Nguyên nhân	Gợi ý
Model dự đoán quá tệ	Kiểm tra pretrained weights, init
Chỉ train 1 epoch	Tăng số epoch (vd: 30–50)
Data/label không khớp	Kiểm tra CHAR2IDX, encoding
Learning rate	Thử giảm LR
Input sai format	Kiểm tra shape, normalize
Cách xử lý
Giảm ceiling khi clamp (vd: 10 thay vì 20) để loss phản ánh tốt hơn, nhưng vẫn có thể gây gradient không ổn định.
Tắt Focal CTC tạm thời:
# configs/config.py
USE_FOCAL_CTC: bool = False
# configs/config.pyUSE_FOCAL_CTC: bool = False
Kiểm tra pretrained: PRETRAINED_PATH có đúng checkpoint tương thích với model không.
Tăng epochs: Chạy nhiều epoch hơn (vd: 30+) để model có thời gian học.
Nếu cần, có thể bổ sung logging để đếm số sample có loss = inf/nan hoặc > 20 trước khi clamp.
vậy nó là bình thường đúng không

Không hẳn bình thường, nó phụ thuộc thời điểm:
Có thể coi là bình thường
Epoch 1, những batch đầu: Model mới khởi tạo / load pretrained, dự đoán rất tệ → CTC trả inf → bị clamp về 20.0.
Dấu hiệu bất thường
Sau vài epoch (5–10) mà loss vẫn ~20.0 → Có vấn đề.
Val Acc 0% như output trước đó → Model gần như không dự đoán được gì.
Nên làm gì
Chạy thêm epochs (ví dụ 10–20) với USE_FOCAL_CTC=False hoặc giữ nguyên.
Theo dõi loss:
Nếu loss giảm dần (20 → 15 → 10 → …) → ổn.
Nếu vẫn ~20 suốt → kiểm tra pretrained, data, label.
Tóm lại: Loss ~20 ở đầu training có thể bình thường, nhưng nếu kéo dài qua nhiều epoch thì không.
vấn đề sẽ nằm ở đâu 

Đã rà soát, các điểm có thể gây vấn đề:
1. Thiếu pretrained weights (khả năng cao nhất)
weights/
  - config.yml   ← chỉ có file này
  - best.pth    ← KHÔNG TỒN TẠI
weights/  - config.yml   ← chỉ có file này  - best.pth    ← KHÔNG TỒN TẠI
PRETRAINED_PATH = weights/best.pth nhưng file không có → model train từ random init → dự đoán rất tệ → CTC → inf → bị clamp 20.
Cách xử lý: Tải checkpoint best.pth (UniRec/SVTRv2) đúng với kiến trúc và đặt vào weights/.
2. Kiểm tra dữ liệu
Mẫu annotations.json trông chuẩn:
Label: "ASC7C56" (khớp CHARS)
5 lr + 5 hr frames
Nên kiểm tra thêm:
Có track nào có label rỗng hoặc toàn ký tự lạ không?
Label có ký tự không có trong CHARS không? (sẽ bị bỏ qua → target = [0] → gần như chỉ blank)
3. Focal CTC clamp
Đoạn clamp max=20 khiến loss không xuống dưới 20 khi model dự đoán quá tệ.
Có thể tạm thời đổi thành:
# trainer.py - giảm max clamp để thấy loss thật
loss_per_sample_safe = torch.clamp(loss_per_sample, min=-20.0, max=15.0)  # hoặc 10
# trainer.py - giảm max clamp để thấy loss thậtloss_per_sample_safe = torch.clamp(loss_per_sample, min=-20.0, max=15.0)  # hoặc 10
4. Một số điểm phụ
Chỗ cần kiểm tra	Cách kiểm tra
Pretrained có đúng kiến trúc không	So sánh config.yml với model hiện tại
Số lớp (num_classes)	Bằng len(CHARS) + 1
Định dạng ảnh	lr-*.png/jpg có đủ 5 file mỗi track
Thứ tự nên làm
Đặt best.pth vào weights/ hoặc tắt pretrained và train từ đầu với nhiều epoch.
Chạy thêm epochs (20–30) để xem loss có giảm không.
Tạm tắt Focal CTC (USE_FOCAL_CTC=False) để xem loss thật (không bị kẹp ở 20).
Kiểm tra labels – in vài sample để xem có label lạ hoặc rỗng không.











(multiframe-lpr) PS C:\Users\anhhu\MultiFrame-LPR> python train.py --batch-size 16 
🔒 Deterministic mode ENABLED (Reproducibility optimized). Benchmark mode DISABLED.
🔄 Đã xóa cache CUDA. Bộ nhớ GPU:
   Total:     4095.69 MiB (4.00 GiB)
   Allocated: 0.00 MiB
   Reserved:  0.00 MiB
   Free:      4095.69 MiB
🚀 Configuration:
   EXPERIMENT: mf_svtrv2
   MODEL: mf_svtrv2
   USE_STN: True
   DATA_ROOT: C:\Users\anhhu\MultiFrame-LPR\Data\train
   EPOCHS: 1
   BATCH_SIZE: 16
   LEARNING_RATE: 0.000325
   DEVICE: cuda
   USE_SR: False
   USE_FOCAL_CTC: False  ->  LOSS: CTC
   SUBMISSION_MODE: False

ℹ️  USE_SR=False -> Pipeline chạy KHÔNG có Super-Resolution

[TRAIN] Scanning: C:\Users\anhhu\MultiFrame-LPR\Data\train
📂 Loading split from 'C:\Users\anhhu\MultiFrame-LPR\Data\val_tracks.json'...
[TRAIN] Loaded 19001 tracks.
Indexing train: 100%|██████████████████████████████████████████████| 19001/19001 [01:30<00:00, 210.84it/s]
-> Total: 38002 samples.
[VAL] Scanning: C:\Users\anhhu\MultiFrame-LPR\Data\train
📂 Loading split from 'C:\Users\anhhu\MultiFrame-LPR\Data\val_tracks.json'...
[VAL] Loaded 999 tracks.
Indexing val: 100%|████████████████████████████████████████████████████| 999/999 [00:01<00:00, 996.10it/s]
-> Total: 999 samples.
C:\Users\anhhu\MultiFrame-LPR\.venv\Lib\site-packages\torch\nn\modules\module.py:1357: UserWarning: expandable_segments not supported on this platform (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\c10/cuda/CUDAAllocatorConfig.h:35.)
  return t.to(

✅ Model đã được khởi tạo với:
   - STN: ✅
   - Backbone: SVTRv2LNConvTwo33 ✅
   - Fusion: AttentionFusion ✅
   - Head: RCTCDecoder ✅

🔄 Loading Pretrained Weights: C:\Users\anhhu\MultiFrame-LPR\weights\best.pth
✅ Đã nạp thành công 236 layers từ checkpoint.

============================================================
📋 MODEL ARCHITECTURE & PARAMETERS
============================================================
   Type: MultiFrameSVTRv2
   STN: ✅ ENABLED
   Backbone: SVTRv2LNConvTwo33
   Decoder: RCTCDecoder (CTC)
   Fusion: AttentionFusion (5 frames)
   STN params: 283,974 (1.08 MB)
   Backbone params: 17,682,368 (67.45 MB)
   Fusion params: 18,529 (0.07 MB)
   Head params: 2,083,621 (7.95 MB)

   Pretrained Weights: ✅ LOADED
   Path: C:\Users\anhhu\MultiFrame-LPR\weights\best.pth

   📊 Total params: 20,068,492 (76.56 MB)
   📊 Trainable: 20,068,492
   📊 Non-trainable: 0
============================================================

📊 Bộ nhớ cần để chạy model (ước tính):
   Tham số:        20,068,492 (~76.56 MiB)
   Trainable:      20,068,492
   Optimizer (AdamW): ~153.11 MiB
   Batch size:     16 (input ~3.75 MiB)
   Activations (ước tính): ~15.00 MiB
   Tổng ước tính: ~244.67 MiB (~0.24 GiB)
   ✓ GPU free ~4006 MiB, đủ cho ước tính ~245 MiB.
🔒 Deterministic mode ENABLED (Reproducibility optimized). Benchmark mode DISABLED.
🚀 TRAINING START | Device: cuda | Epochs: 1 | Loss: CTC
Ep 1/1: 100%|████████████████████████████| 2376/2376 [1:36:01<00:00,  1.75s/it, loss=nan(skip), lr=1.3e-5][W208 01:44:55.000000000 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator ())
[W208 01:44:55.000000000 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator ())
[W208 01:44:55.000000000 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator ())
[W208 01:44:55.000000000 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator ())
[W208 01:44:55.000000000 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator ())
[W208 01:44:55.000000000 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator ())
[W208 01:44:55.000000000 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator ())
[W208 01:44:55.000000000 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator ())
[W208 01:44:55.000000000 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator ())
[W208 01:44:55.000000000 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator ())
Ep 1/1: 100%|████████████████████████████| 2376/2376 [1:36:11<00:00,  2.43s/it, loss=nan(skip), lr=1.3e-5]
[W208 01:48:11.000000000 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator ())
[W208 01:48:11.000000000 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator ())
[W208 01:48:11.000000000 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator ())
[W208 01:48:11.000000000 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator ())
[W208 01:48:11.000000000 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator ())
[W208 01:48:11.000000000 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator ())
[W208 01:48:12.000000000 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator ())
[W208 01:48:12.000000000 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator ())
[W208 01:48:12.000000000 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator ())
[W208 01:48:12.000000000 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator ())
Epoch 1/1: Train Loss: 0.0000 | Val Loss: nan | Val Acc: 0.00% | Val CER: 1.0000 | LR: 1.30e-05
📝 Saved 999 lines to results\submission_mf_svtrv2.txt
📋 Saved 999 wrong predictions to results\wrong_predictions_mf_svtrv2.txt
📁 Copied 4995 wrong images to results\wrong_images_mf_svtrv2
  💾 Saved final model: results\mf_svtrv2_final.pth

✅ Training complete! Best Val Acc: 0.00%
[W208 01:48:24.000000000 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator ())
(multiframe-lpr) PS C:\Users\anhhu\MultiFrame-LPR> 