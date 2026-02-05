# 📘 Ultralytics YOLO Model Overview (Object Detection)

> **Report Objective**  
> This document provides a structured and comprehensive overview of modern YOLO versions (YOLOv8 → YOLO11) and related models (RT-DETR), with a focus on **model parameters**, **computational complexity (GFLOPs)**, and **practical usage scenarios** in **Object Detection** tasks.  
> The goal is to help readers clearly understand, compare, and select suitable models for research, learning, and real-world deployment.

---

## 1. Overview of YOLO in Object Detection

YOLO (*You Only Look Once*) is a family of **one-stage object detectors** that perform object localization and classification in a **single forward pass**. This design enables high inference speed and makes YOLO particularly suitable for **real-time applications**.

Modern YOLO versions primarily focus on:

- Reducing the number of parameters while maintaining high accuracy  
- Optimizing GFLOPs for deployment across diverse hardware (Edge → Server)  
- Moving toward **End-to-End Detection** (reducing or eliminating NMS)

---

## 2. Key Technical Metrics

- **Layers**: Number of network layers, reflecting architectural depth  
- **Parameters**: Number of learnable parameters (affects model size and capacity)  
- **GFLOPs**: Computational complexity (directly impacts inference speed)

---

## 3. YOLOv8 (Ultralytics – 2023)

🔗 Official documentation: https://docs.ultralytics.com/models/yolov8/

YOLOv8 adopts an **anchor-free** design, simplifying both training and inference pipelines. It is currently one of the most widely used YOLO versions in practical applications.

| Model   | Layers | Parameters | GFLOPs |
| ------- | ------ | ---------- | ------ |
| YOLOv8n | 129    | 3,157,200  | 8.9    |
| YOLOv8s | 129    | 11,166,560 | 28.8   |
| YOLOv8m | 169    | 25,902,640 | 79.3   |
| YOLOv8l | 209    | 43,691,520 | 165.7  |
| YOLOv8x | 209    | 68,229,648 | 258.5  |

**Remarks**:
- Easy to train and deploy  
- Stable codebase with extensive documentation  
- Well-suited as a baseline for most Object Detection tasks  

---

## 4. YOLOv9 (2024 – GELAN Backbone)

🔗 Paper: https://arxiv.org/abs/2402.13616  
🔗 Repository: https://github.com/WongKinYiu/yolov9  

YOLOv9 introduces the **GELAN backbone** and **re-parameterization strategies**, improving representation efficiency without increasing inference cost.

| Model   | Layers | Parameters | GFLOPs |
| ------- | ------ | ---------- | ------ |
| YOLOv9t | 544    | 2,128,720  | 8.5    |
| YOLOv9s | 544    | 7,318,368  | 27.6   |
| YOLOv9m | 348    | 20,216,160 | 77.9   |
| YOLOv9c | 358    | 25,590,912 | 104.0  |

**Remarks**:
- Deep architectures with a large number of layers  
- High parameter efficiency  
- Well-suited for research and benchmarking  

---

## 5. YOLOv10 (Real-Time End-to-End – 2024)

🔗 Paper: https://arxiv.org/abs/2405.14458  
🔗 Repository: https://github.com/THU-MIG/yolov10  

YOLOv10 focuses on **End-to-End Object Detection**, removing NMS to further reduce inference latency.

| Model    | Layers | Parameters | GFLOPs |
| -------- | ------ | ---------- | ------ |
| YOLOv10n | 223    | 2,775,520  | 8.7    |
| YOLOv10s | 234    | 8,128,272  | 25.1   |
| YOLOv10m | 288    | 16,576,768 | 64.5   |
| YOLOv10l | 364    | 25,888,688 | 127.9  |
| YOLOv10x | 400    | 31,808,960 | 171.8  |

**Remarks**:
- Fewer parameters compared to YOLOv8/YOLOv9 at similar scales  
- Lower latency, suitable for real-time systems  
- Strong focus on production deployment  

---

## 6. YOLO11 (Ultralytics – Next Generation)

🔗 Documentation: https://docs.ultralytics.com/models/yolo11/

YOLO11 is the successor to YOLOv8, significantly optimizing the **accuracy-to-compute ratio**.

| Model   | Layers | Parameters | GFLOPs |
| ------- | ------ | ---------- | ------ |
| YOLO11n | 181    | 2,624,080  | 6.6    |
| YOLO11s | 181    | 9,458,752  | 21.7   |
| YOLO11m | 231    | 20,114,688 | 68.5   |
| YOLO11l | 357    | 25,372,160 | 87.6   |
| YOLO11x | 357    | 56,966,176 | 196.0  |

**Remarks**:
- Significantly lower GFLOPs compared to YOLOv8 at similar model sizes  
- Suitable for both edge devices and server environments  
- Recommended choice for new projects  

---

## 7. RT-DETR (Transformer-based Detector)

🔗 Paper: https://arxiv.org/abs/2304.08069  
🔗 Repository: https://github.com/IDEA-Research/RT-DETR  

RT-DETR is a **Transformer-based object detector** that eliminates the need for NMS and achieves high detection accuracy.

| Model     | Layers | Parameters | GFLOPs |
| --------- | ------ | ---------- | ------ |
| RT-DETR-l | 449    | 32,970,476 | 108.3  |
| RT-DETR-x | 567    | 67,467,852 | 232.7  |

**Remarks**:
- High accuracy  
- High computational cost  
- Best suited for server-side deployment  

---

## 8. Model Selection Guidelines

| Use Case                  | Recommended Models |
| ------------------------- | ------------------ |
| Edge / Mobile             | YOLOv8n, YOLO11n   |
| Real-time (Low-end GPU)   | YOLOv8s, YOLO11s   |
| Balanced Speed / Accuracy | YOLOv8m, YOLO11m   |
| High Accuracy             | YOLO11l, YOLOv10l  |
| Research / Benchmarking   | YOLOv9, RT-DETR    |
| End-to-End, Low Latency   | YOLOv10            |

---

## 9. Conclusion

The evolution of YOLO models highlights clear trends toward:

- Improved **parameter efficiency**  
- Reduced **inference latency**  
- Fully **End-to-End Object Detection** pipelines  

At present, **YOLO11** offers the most balanced and modern solution for the majority of Object Detection tasks, while **RT-DETR** is better suited for scenarios that prioritize maximum accuracy on high-performance infrastructure.

---
