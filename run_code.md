# Lệnh chạy pipeline MultiFrame-LPR + OpenOCR

Tất cả lệnh dưới đây chạy được từ workspace. CWD: `/workspace`.

---

## 1. Pipeline VALIDATION (999 tracks)

### Bước 1.1: Tạo tracks_val.csv từ openocr_infer_list.txt

```bash
cd /workspace/MultiFrame-LPR

python scripts/build_tracks_from_infer_list.py \
  --infer-list Data/openocr_infer_list.txt \
  --output Data/tracks_val.csv
```

### Bước 1.2: Tạo submission từ kết quả OpenOCR đã infer sẵn

```bash
cd /workspace/MultiFrame-LPR

python scripts/make_submission.py \
  --tracks-csv Data/tracks_val.csv \
  --openocr-results Data/openocr_results.txt \
  --output Data/submission.txt \
  --verbose
```

**Output:** `Data/submission.txt` (999 dòng)

---

## 2. Pipeline PARSEQ (999 tracks, Scenario-B từ val_tracks_parseq.json)

### Bước 2.0: Tạo infer list và chạy full pipeline

```bash
cd /workspace/MultiFrame-LPR

# 2.1 Infer list
python3 scripts/prepare_openocr_input.py \
  --tracks-csv Data/tracks.csv \
  --output-list Data/openocr_infer_list_parseq.txt \
  --images-root /workspace/MultiFrame-LPR \
  --filter-tracks Data/val_tracks_parseq.json

# 2.2 OpenOCR inference (chạy trong OpenOCR với venv)
cd /workspace/OpenOCR && source .venv/bin/activate
python3 tools/infer_rec.py -c configs/rec/lp_svtrv2_gtc.yml \
  -o Global.infer_img=/workspace/MultiFrame-LPR/Data/openocr_infer_list_parseq.txt \
     Global.save_res_path=/workspace/MultiFrame-LPR/Data/openocr_results_parseq.txt

# 2.3 Tạo tracks và submission
cd /workspace/MultiFrame-LPR
python3 scripts/build_tracks_from_infer_list.py \
  --infer-list Data/openocr_infer_list_parseq.txt \
  --output Data/tracks_parseq_val.csv

python3 scripts/make_submission.py \
  --tracks-csv Data/tracks_parseq_val.csv \
  --openocr-results Data/openocr_results_parseq.txt \
  --output Data/submission_parseq.txt \
  --verbose
```

**Output:** `Data/submission_parseq.txt` (999 dòng)

---

## 3. Pipeline BLIND TEST (3000 tracks)

### Bước 3.1: Tạo tracks_blind.csv từ dataset blind_test

```bash
cd /workspace/MultiFrame-LPR

python scripts/build_tracks_csv_from_dataset.py \
  --root-dir Data/blind_test \
  --output Data/tracks_blind.csv \
  --rel-to /workspace/MultiFrame-LPR
```

### Bước 3.2: Tạo openocr_infer_list_blind.txt

```bash
cd /workspace/MultiFrame-LPR

python scripts/prepare_openocr_input.py \
  --tracks-csv Data/tracks_blind.csv \
  --output-list Data/openocr_infer_list_blind.txt \
  --images-root /workspace/MultiFrame-LPR
```

### Bước 3.3: Chạy OpenOCR inference trên blind_test

```bash
cd /workspace/OpenOCR

python tools/infer_rec.py -c configs/rec/lp_svtrv2_gtc.yml \
  -o Global.infer_img=/workspace/MultiFrame-LPR/Data/openocr_infer_list_blind.txt \
     Global.save_res_path=/workspace/MultiFrame-LPR/Data/openocr_results_blind.txt
```

### Bước 3.4: Tạo submission_blind.txt

**Cách A – Dùng tracks_blind.csv + images-root:**

```bash
cd /workspace/MultiFrame-LPR

python scripts/make_submission.py \
  --tracks-csv Data/tracks_blind.csv \
  --openocr-results Data/openocr_results_blind.txt \
  --output Data/submission_blind.txt \
  --images-root /workspace/MultiFrame-LPR \
  --verbose
```

**Cách B – Dùng tracks_blind_val.csv (path khớp infer list):**

```bash
cd /workspace/MultiFrame-LPR

python scripts/build_tracks_from_infer_list.py \
  --infer-list Data/openocr_infer_list_blind.txt \
  --output Data/tracks_blind_val.csv

python scripts/make_submission.py \
  --tracks-csv Data/tracks_blind_val.csv \
  --openocr-results Data/openocr_results_blind.txt \
  --output Data/submission_blind.txt \
  --verbose
```

**Output:** `Data/submission_blind.txt` (3000 dòng)

---

## 4. Pipeline TRAIN (build tracks từ toàn bộ train – 20000 tracks)

### Bước 4.1: Tạo tracks.csv từ train

```bash
cd /workspace/MultiFrame-LPR

python scripts/build_tracks_csv_from_dataset.py \
  --root-dir Data/train \
  --output Data/tracks.csv \
  --rel-to /workspace/MultiFrame-LPR
```

### Bước 4.2: Tạo openocr_infer_list từ tracks.csv (chỉ val nếu cần)

```bash
cd /workspace/MultiFrame-LPR

# Tất cả train
python scripts/prepare_openocr_input.py \
  --tracks-csv Data/tracks.csv \
  --output-list Data/openocr_infer_list.txt \
  --images-root /workspace/MultiFrame-LPR

# Chỉ validation
python scripts/prepare_openocr_input.py \
  --tracks-csv Data/tracks.csv \
  --output-list Data/openocr_infer_list.txt \
  --images-root /workspace/MultiFrame-LPR \
  --filter-tracks Data/val_tracks.json
```

---

## 5. Config OpenOCR (lp_svtrv2_gtc.yml)

Các tham số override thường dùng khi infer:

```yaml
Global:
  infer_img: /workspace/MultiFrame-LPR/Data/openocr_infer_list.txt   # hoặc _blind.txt
  save_res_path: /workspace/MultiFrame-LPR/Data/openocr_results.txt # hoặc _blind.txt
  pretrained_model: /workspace/MultiFrame-LPR/weights/v3_best.pth
```

Hoặc override khi chạy:

```bash
python tools/infer_rec.py -c configs/rec/lp_svtrv2_gtc.yml \
  -o Global.infer_img=/path/to/list.txt Global.save_res_path=/path/to/output.txt
```

---

## 6. Tóm tắt file

| File | Mô tả |
|------|-------|
| `Data/tracks.csv` | Tracks từ train (20000) |
| `Data/tracks_val.csv` | Tracks validation, path khớp infer list |
| `Data/tracks_parseq_val.csv` | Tracks parseq (Scenario-B), path khớp infer list |
| `Data/tracks_blind.csv` | Tracks từ blind_test |
| `Data/tracks_blind_val.csv` | Tracks blind, path khớp infer list |
| `Data/openocr_infer_list.txt` | List ảnh infer (val) |
| `Data/openocr_infer_list_parseq.txt` | List ảnh infer (parseq) |
| `Data/openocr_infer_list_blind.txt` | List ảnh infer (blind) |
| `Data/openocr_results.txt` | Kết quả OpenOCR (val) |
| `Data/openocr_results_parseq.txt` | Kết quả OpenOCR (parseq) |
| `Data/openocr_results_blind.txt` | Kết quả OpenOCR (blind) |
| `Data/submission.txt` | File nộp (val) |
| `Data/submission_parseq.txt` | File nộp (parseq) |
| `Data/submission_blind.txt` | File nộp (blind) |
