# 📘 Ultralytics YOLO Model Overview (Object Detection)

> **Mục tiêu báo cáo**
> Tài liệu này cung cấp cái nhìn tổng quan, có hệ thống về các phiên bản YOLO hiện đại (YOLOv8 → YOLO11) và một số mô hình liên quan (RT-DETR), tập trung vào **số tham số (Parameters)**, **độ phức tạp tính toán (GFLOPs)** và **định hướng sử dụng** trong bài toán **Object Detection**.
> Nội dung được trình bày nhằm giúp người đọc dễ theo dõi, so sánh và lựa chọn mô hình phù hợp cho nghiên cứu, học tập và triển khai thực tế.

---

## 1. Tổng quan về YOLO trong bài toán Object Detection

YOLO (*You Only Look Once*) là dòng mô hình **one-stage detector**, thực hiện đồng thời việc định vị (localization) và phân loại (classification) đối tượng chỉ trong **một lần suy luận**. Điều này giúp YOLO đạt tốc độ cao và rất phù hợp cho các ứng dụng **real-time**.

Các phiên bản YOLO hiện đại tập trung vào:

* Giảm số tham số nhưng vẫn giữ độ chính xác
* Tối ưu GFLOPs để triển khai trên nhiều phần cứng (Edge → Server)
* Hướng tới **End-to-End Detection** (giảm hoặc loại bỏ NMS)

---

## 2. Thông số kỹ thuật chính

* **Layers**: số lớp trong mạng, phản ánh độ sâu kiến trúc
* **Parameters**: số tham số học được (ảnh hưởng đến dung lượng model)
* **GFLOPs**: độ phức tạp tính toán (ảnh hưởng trực tiếp đến tốc độ suy luận)

---

## 3. YOLOv8 (Ultralytics – 2023)

🔗 Tài liệu chính thức: [https://docs.ultralytics.com/models/yolov8/](https://docs.ultralytics.com/models/yolov8/)

YOLOv8 là phiên bản **anchor-free**, đơn giản hóa pipeline huấn luyện và suy luận. Đây là phiên bản được sử dụng rộng rãi nhất trong thực tế.

| Model   | Layers | Parameters | GFLOPs |
| ------- | ------ | ---------- | ------ |
| YOLOv8n | 129    | 3,157,200  | 8.9    |
| YOLOv8s | 129    | 11,166,560 | 28.8   |
| YOLOv8m | 169    | 25,902,640 | 79.3   |
| YOLOv8l | 209    | 43,691,520 | 165.7  |
| YOLOv8x | 209    | 68,229,648 | 258.5  |

**Nhận xét**:

* Dễ huấn luyện, code ổn định
* Cộng đồng lớn, tài liệu đầy đủ
* Phù hợp làm baseline cho hầu hết các bài toán Object Detection

---

## 4. YOLOv9 (2024 – GELAN Backbone)

🔗 Paper: [https://arxiv.org/abs/2402.13616](https://arxiv.org/abs/2402.13616)
🔗 Repository: [https://github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)

YOLOv9 giới thiệu **GELAN Backbone** và cơ chế **re-parameterization**, cải thiện hiệu quả học biểu diễn mà không tăng chi phí suy luận.

| Model   | Layers | Parameters | GFLOPs |
| ------- | ------ | ---------- | ------ |
| YOLOv9t | 544    | 2,128,720  | 8.5    |
| YOLOv9s | 544    | 7,318,368  | 27.6   |
| YOLOv9m | 348    | 20,216,160 | 77.9   |
| YOLOv9c | 358    | 25,590,912 | 104.0  |

**Nhận xét**:

* Số layer lớn → kiến trúc sâu
* Hiệu quả tham số tốt
* Phù hợp cho nghiên cứu và benchmark

---

## 5. YOLOv10 (Real-Time End-to-End – 2024)

🔗 Paper: [https://arxiv.org/abs/2405.14458](https://arxiv.org/abs/2405.14458)
🔗 Repository: [https://github.com/THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)

YOLOv10 tập trung vào **End-to-End Object Detection**, loại bỏ NMS để giảm độ trễ suy luận.

| Model    | Layers | Parameters | GFLOPs |
| -------- | ------ | ---------- | ------ |
| YOLOv10n | 223    | 2,775,520  | 8.7    |
| YOLOv10s | 234    | 8,128,272  | 25.1   |
| YOLOv10m | 288    | 16,576,768 | 64.5   |
| YOLOv10l | 364    | 25,888,688 | 127.9  |
| YOLOv10x | 400    | 31,808,960 | 171.8  |

**Nhận xét**:

* Ít tham số hơn YOLOv8/9 cùng phân khúc
* Độ trễ thấp, phù hợp hệ thống real-time
* Hướng tới triển khai sản phẩm

---

## 6. YOLO11 (Ultralytics – Thế hệ mới)

🔗 Tài liệu: [https://docs.ultralytics.com/models/yolo11/](https://docs.ultralytics.com/models/yolo11/)

YOLO11 là thế hệ kế nhiệm YOLOv8, tối ưu mạnh về **tỷ lệ Accuracy / Compute**.

| Model   | Layers | Parameters | GFLOPs |
| ------- | ------ | ---------- | ------ |
| YOLO11n | 181    | 2,624,080  | 6.6    |
| YOLO11s | 181    | 9,458,752  | 21.7   |
| YOLO11m | 231    | 20,114,688 | 68.5   |
| YOLO11l | 357    | 25,372,160 | 87.6   |
| YOLO11x | 357    | 56,966,176 | 196.0  |

**Nhận xét**:

* GFLOPs thấp hơn đáng kể so với YOLOv8 cùng kích thước
* Phù hợp cho cả Edge và Server
* Nên ưu tiên cho các dự án mới

---

## 7. RT-DETR (Transformer-based Detector)

🔗 Paper: [https://arxiv.org/abs/2304.08069](https://arxiv.org/abs/2304.08069)
🔗 Repository: [https://github.com/IDEA-Research/RT-DETR](https://github.com/IDEA-Research/RT-DETR)

RT-DETR là mô hình **Transformer-based**, không cần NMS, đạt độ chính xác cao.

| Model     | Layers | Parameters | GFLOPs |
| --------- | ------ | ---------- | ------ |
| RT-DETR-l | 449    | 32,970,476 | 108.3  |
| RT-DETR-x | 567    | 67,467,852 | 232.7  |

**Nhận xét**:

* Độ chính xác cao
* Chi phí tính toán lớn
* Phù hợp server, không phù hợp edge

---

## 8. So sánh & Định hướng lựa chọn mô hình

| Nhu cầu sử dụng           | Mô hình gợi ý     |
| ------------------------- | ----------------- |
| Edge / Mobile             | YOLOv8n, YOLO11n  |
| Realtime (GPU yếu)        | YOLOv8s, YOLO11s  |
| Cân bằng Speed / Accuracy | YOLOv8m, YOLO11m  |
| Độ chính xác cao          | YOLO11l, YOLOv10l |
| Nghiên cứu / Benchmark    | YOLOv9, RT-DETR   |
| End-to-End, latency thấp  | YOLOv10           |

---

## 9. Kết luận

Sự phát triển của YOLO cho thấy xu hướng rõ ràng:

* Tối ưu **hiệu quả tham số**
* Giảm **độ trễ suy luận**
* Hướng tới **End-to-End Object Detection**

Trong bối cảnh hiện tại, **YOLO11** là lựa chọn cân bằng và hiện đại nhất cho đa số bài toán Object Detection, trong khi **RT-DETR** phù hợp cho các hệ thống yêu cầu độ chính xác cao trên hạ tầng mạnh.

---

> 📌 *Tài liệu này được thiết kế để sử dụng trực tiếp cho GitHub (Markdown), phù hợp làm README hoặc tài liệu kỹ thuật cho dự án AI / Computer Vision.*
