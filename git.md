<p align="center">
  <img src="09af1d89-6b93-469b-9740-241f750e067a_Git+and+GitHub.jpg" alt="Git và GitHub" width="50%">
</p>

# 🚀 MLOps & DevOps Standard Operating Procedures (SOP)

- Tài liệu này hệ thống hóa các quy trình làm việc từ quản lý mã nguồn (Git), đóng gói ứng dụng (Docker) đến vận hành mô hình (MLflow) và kiểm thử (Testing).

- Được ghi chú trong quá trình học tập với Anh Khiêm siêu giỏi, được giải thích rõ ràng và thực hiện để dễ hiểu.

- Các lệnh Git nên được thực hiện nhiều lần và trong thời gian dài để nhớ rõ và chính xác hơn.

- Cảm ơn đã đọc.

---

## 1. Quản lý Môi trường & Hệ thống (System Setup)

Trước khi bắt đầu bất kỳ thao tác nào, hãy xác định vị trí và kích hoạt môi trường làm việc cô lập.

* **Kiểm tra thư mục hiện hành:** `pwd`
* **Tạo thư mục mới:** `mkdir <tên_thư_mục>` (Dùng `touch <tên_file>` nếu muốn tạo file).
* **Kích hoạt Virtual Environment (venv):**
    ```bash
    source .venv/bin/activate
    ```

---

## 2. Quy trình Git Workflow (Chuyên sâu)

### 2.1. Thao tác Cơ bản & Nhánh

* **Xem nhánh hiện tại:** `git branch`
* **Xem tất cả nhánh (bao gồm remote):** `git branch -a`
* **Tạo và chuyển sang nhánh mới:** `git checkout -b <tên_nhánh>`
* **Chuyển nhánh hiện có:** `git checkout <tên_nhánh>`

### 2.2. Đồng bộ hóa với Remote (Origin & Upstream)
* **Kiểm tra danh sách remote:** `git remote -v`
* **Cập nhật dữ liệu mới nhất:**
    * `git fetch origin`: Cập nhật từ bản fork của bạn.
    * `git fetch --all`: Cập nhật từ tất cả các nguồn.
* **Xử lý Divergent Branch (Khi nhánh local và remote khác biệt):**
    * **Cách 1 (Merge - An toàn):** `git pull --no-rebase upstream main`
    * **Cách 2 (Rebase - Sạch lịch sử):** `git pull --rebase upstream main`

### 2.3. Quy trình Đẩy Code từ Dev lên Main
1. **Tại nhánh dev:** `git commit -m "feat: update api and prediction logic"`
2. **Chuyển về main:** `git checkout main`
3. **Gộp code:** `git merge dev`
4. **Đẩy lên GitHub:** * Lần đầu: `git push -u origin main`
    * Các lần sau: `git push`

---

## 3. Containerization với Docker

### 3.1. Build & Tagging
* **Lệnh build cơ bản:** `docker build -t <tên_image>:<tag> .`
* **Build cho môi trường Production (AMD64):**
    ```bash
    docker build --platform linux/amd64 -t huyynguyenn/mlops:v1 .
    ```
* **Gắn tag để push lên Docker Hub:**
    ```bash
    docker tag mlops:v1 yourusername/mlops:v1
    ```

### 3.2. Quản lý Image & Container
* **Xem danh sách image:** `docker images`
* **Đăng nhập Docker Hub:** `docker login`
* **Đẩy image:** `docker push yourusername/mlops:v1`
* **Triển khai với Docker Compose:**
    ```bash
    docker-compose up -d --build
    ```
    * `-d`: Chạy ngầm (Detached mode).
    * `--build`: Ép buộc build lại image trước khi chạy.

---

## 4. MLOps Stack & Quality Control

### 4.1. Model Tracking (MLflow)
* **Khởi chạy MLflow Server:**
    ```bash
    mlflow server --host 127.0.0.1 --port 8080
    ```

### 4.2. Testing & Coverage
Đảm bảo code không có lỗi logic trước khi đóng gói.
* **Chạy kiểm thử & báo cáo độ bao phủ:**
    ```bash
    pytest --cov --cov-report=html
    ```
* **Xem báo cáo trực quan trên trình duyệt:**
    ```bash
    cd htmlcov/  # Vào thư mục báo cáo
    python -m http.server 8000
    ```
    *Sau đó truy cập: http://localhost:8000*

---

## 5. Phím tắt & Thủ thuật (Tips)

### 5.1. VS Code Shortcuts
* `Ctrl + Shift + E`: Mở nhanh thanh Side Bar (Explorer).
* `Ctrl + ` ` (Backtick): Bật/tắt Terminal tích hợp.

### 5.2. Giải thích các Flag Git phổ biến
* `-m`: Message (Tin nhắn commit).
* `-b`: Branch (Tạo nhánh mới).
* `-a`: All (Tự động stage các file đã tracked).
* `-u`: Update (Thiết lập upstream tracking).
* `-v`: Verbose (Hiển thị chi tiết thông tin).

---

**💡 Ghi chú DevOps:** Luôn kiểm tra `git status` trước khi thực hiện commit để tránh đẩy nhầm các file rác hoặc dữ liệu nhạy cảm vào kho lưu trữ.


