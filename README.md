# 校园自行车检测系统

基于 YOLOv8 + BoT-SORT 的实时校园自行车检测与统计系统，针对校园场景下密集停放和动态行驶两种场景分别优化。

---

## 功能概览

| 功能 | 说明 |
|------|------|
| 静态停放检测 | 检测并统计停放自行车，SAHI 切片推理解决密集场景漏检 |
| 动态行驶计数 | 虚拟计数线统计经过的自行车数量，支持双向计数 |
| 跨帧追踪去重 | BoT-SORT 追踪器分配唯一 track ID，同一辆车不重复计数 |
| 遮挡恢复 | 三层遮挡保护，短暂遮挡后恢复原 track ID |
| 多输入源 | 摄像头实时流 / 视频文件 |
| 多推理后端 | `.pt` / `.onnx` / `.tflite`，支持 CPU / CUDA / NPU |

---

## 技术栈

### 1. 目标检测 — YOLOv8

YOLOv8 是 Ultralytics 推出的单阶段目标检测模型，采用 anchor-free 设计，在速度和精度上均优于前代。本项目使用 `yolov8n`（nano 版本），参数量约 3.2M，适合边缘设备部署。

检测目标限定为 COCO 数据集中的 class 1（bicycle），通过 `classes=[1]` 参数过滤其他类别，减少无关检测干扰。

```
输入帧 → YOLOv8n → [bbox, confidence, class_id] → 过滤 class=1 → Detection 列表
```

推理时传入 `imgsz=640`，在精度和速度之间取得平衡。置信度阈值默认 0.25，NMS IoU 阈值 0.3，密集场景下适当降低阈值以提升召回率。

---

### 2. 密集目标检测 — SAHI（Slicing Aided Hyper Inference）

校园停车场景中自行车经常密集排列，标准全图推理在小目标和密集目标上漏检严重。SAHI 通过将图像切片后分别推理再合并，显著提升密集场景的检测召回率。

**本项目的 SAHI 实现流程：**

```
原始帧（如 1080×1920）
    ↓
切片生成：640×640，重叠 20%，生成 N 个子图
    ↓
+ 全图缩放图（resize 到 640×640）
    ↓
所有切片批量送入 YOLOv8（一次 forward pass）
    ↓
局部坐标 → 全图归一化坐标还原
    ↓
NMS 去重（IoU 阈值 0.45）
    ↓
最终检测结果
```

关键优化：所有切片**批量推理**而非逐片串行，GPU 并行处理多张子图，推理时间接近单次推理，避免了朴素切片方案 FPS 极低的问题。

---

### 3. 目标追踪 — BoT-SORT

BoT-SORT（Bootstrapped Object Tracking with SORT）是在 ByteTrack 基础上增加了全局运动补偿（GMC）和 ReID 特征的追踪算法，由 Ultralytics 内置支持。

**追踪流程：**

```
当前帧检测框
    ↓
卡尔曼滤波预测各轨迹位置
    ↓
高置信度检测框与活跃轨迹 IoU 匹配（匈牙利算法）
    ↓
未匹配轨迹与低置信度检测框二次匹配
    ↓
仍未匹配的检测框创建新轨迹
    ↓
超过 track_buffer 帧未匹配的轨迹删除
```

**卡尔曼滤波**用于预测目标在下一帧的位置，即使目标短暂消失也能维持轨迹状态，是追踪稳定性的核心。

本项目自定义了 `botsort_custom.yaml`，关键参数调整：

| 参数 | 默认值 | 本项目 | 原因 |
|------|--------|--------|------|
| `track_buffer` | 30 | 90 | 遮挡后保留更长时间再放弃 |
| `new_track_thresh` | 0.25 | 0.4 | 减少误创建新轨迹 |
| `match_thresh` | 0.8 | 0.9 | 更容易重匹配回原轨迹 |
| `gmc_method` | sparseOptFlow | sparseOptFlow | 稀疏光流做全局运动补偿 |

---

### 4. 遮挡恢复 — 三层缓冲重匹配

单靠追踪器的 `track_buffer` 不够，目标被遮挡时间较长后追踪器会彻底丢失轨迹，重新出现时分配新 ID，导致重复计数。本项目在应用层额外实现了两层保护：

**第二层：应用层 lost_pool**

轨迹从 Ultralytics 追踪器彻底丢失后，不立即删除，而是移入 `_lost_pool` 再保留 60 帧：

```python
# 轨迹消失 → 移入 lost_pool
_lost_pool[tid] = (track, lost_frame_count=0)

# 每帧递增计数，超过 60 帧才真正删除
if lost_frame_count > 60:
    del _lost_pool[tid]
```

**第三层：中心点距离重匹配**

新出现的 track 若中心点与 `_lost_pool` 中某条轨迹末尾中心点的欧氏距离小于阈值（0.08，归一化坐标），认为是同一辆车，复用旧 track ID：

```python
dist = sqrt((cx - lx)² + (cy - ly)²)
if dist < 0.08:
    canonical_id = lost_id  # 复用旧 ID
```

三层合计保护时长约 5 秒（90帧 + 60帧，按 25FPS 计算），覆盖绝大多数短暂遮挡场景。

---

### 5. 虚拟计数线 — 线段相交检测

动态模式下，通过判断目标轨迹是否穿越预设的虚拟计数线来统计车流量。

**穿越检测算法（叉积跨立实验）：**

给定轨迹上相邻两帧的中心点 `prev_pos → curr_pos`，以及计数线端点 `line.start → line.end`，判断两线段是否相交：

```
d1 = cross(line.start, line.end, prev_pos)
d2 = cross(line.start, line.end, curr_pos)
d3 = cross(prev_pos, curr_pos, line.start)
d4 = cross(prev_pos, curr_pos, line.end)

相交条件：d1 和 d2 异号 且 d3 和 d4 异号
```

**穿越方向判断：** 用运动向量与计数线法向量的点积符号判断正向/反向穿越。

**去重机制：** 同一 `(track_id, line_id)` 上次穿越后，轨迹至少再增加 2 个点才能再次触发，防止同一次穿越在相邻帧重复计数。

---

### 6. 推理引擎 — 多格式支持

`YOLOInferenceEngine` 统一封装了三种模型格式的推理：

| 格式 | 后端 | 适用场景 |
|------|------|----------|
| `.pt` | PyTorch / Ultralytics | 开发调试，GPU 加速 |
| `.onnx` | ONNX Runtime | 跨平台部署，CPU/GPU 均可 |
| `.tflite` | TFLite / TFLite Runtime | 移动端、嵌入式设备 |

模型加载时自动检测 CUDA 是否可用，不可用时自动降级到 CPU，避免启动报错。

---

### 7. 轻量化策略

针对边缘设备算力限制：

- **模型选型**：YOLOv8n（nano），3.2M 参数，是 YOLOv8 系列中最轻量的版本
- **输入分辨率**：推理时使用 `imgsz=640`，静态检测可降至 320 进一步提速
- **半精度推理**：CUDA 设备上 Ultralytics 自动使用 FP16，显存占用减半，速度提升约 2x
- **批量推理**：SAHI 切片批量处理，充分利用 GPU 并行能力
- **跳帧机制**：主循环连续读两帧取最新帧，防止摄像头缓冲积压导致画面延迟

---

## 环境要求

- Python 3.9+
- PyTorch（CUDA 版本推荐）

```bash
# RTX 50 系列（Blackwell，CUDA 12.8）需要 nightly 版本
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# RTX 30/40 系列（CUDA 12.1）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

验证 CUDA 是否可用：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

安装其他依赖：

```bash
pip install -r requirements.txt
```

---

## 快速开始

```bash
# 摄像头实时检测
python -m campus_bike_detection.main --source 0 --device cuda

# 视频文件，静态停放检测
python -m campus_bike_detection.main --source video.mp4 --mode static --device cuda

# 视频文件，动态行驶计数
python -m campus_bike_detection.main --source video.mp4 --mode dynamic --device cuda

# 两种模式同时运行
python -m campus_bike_detection.main --source video.mp4 --mode both --device cuda
```

按 `q` 退出。

---

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source` | `0` | 摄像头编号或视频文件路径 |
| `--model` | `campus_bike_detection/yolov8n.pt` | 模型文件路径（支持 .pt/.onnx/.tflite） |
| `--mode` | `both` | `static` / `dynamic` / `both` |
| `--device` | `cuda` | `cpu` / `cuda` / `npu` |
| `--conf` | `0.4` | 检测置信度阈值 |
| `--fps` | `30` | 目标处理帧率 |

---

## 项目结构

```
campus_bike_detection/
├── main.py               # 入口脚本，参数解析
├── system.py             # 系统主循环、可视化渲染
├── models.py             # 数据模型（Detection, Track, CountLine 等）
├── frame_capture.py      # 帧采集，统一封装摄像头和视频文件
├── inference_engine.py   # YOLO 推理引擎，支持 .pt/.onnx/.tflite
├── dense_detector.py     # SAHI 密集检测，批量切片推理 + NMS
├── object_tracker.py     # BoT-SORT 追踪器 + 应用层遮挡缓冲重匹配
├── flow_counter.py       # 虚拟计数线，叉积相交检测 + 方向判断
├── device_adapter.py     # 设备适配（CPU/CUDA/NPU）
├── botsort_custom.yaml   # 自定义追踪器配置
└── yolov8n.pt            # 预训练模型文件
```

---

## 可视化说明

画面左上角信息面板：

- `FPS`：当前推理帧率
- `Now`：当前帧检测到的唯一自行车数（活跃 track 数量）
- `Cross`：累计穿越计数线次数

检测框上显示置信度分数，红色横线为虚拟计数线。

---

## 会话报告

程序退出后输出：

```
===== 会话报告 =====
总帧数      : 381
峰值停放数  : 12
平均停放数  : 8.3
累计检测数  : 25  (唯一自行车，track ID 去重)
各线路流量  : {'main': 18}
运行时长    : 42.1s
```

| 字段 | 说明 |
|------|------|
| 峰值停放数 | 单帧最多同时检测到的自行车数 |
| 平均停放数 | 所有帧的平均检测数 |
| 累计检测数 | 整个视频中出现过的唯一自行车总数（track ID 去重） |
| 各线路流量 | 每条计数线的穿越总次数 |

---

## 依赖

```
ultralytics>=8.0     # YOLOv8 检测与追踪
onnxruntime>=1.16    # ONNX 模型推理
opencv-python>=4.8   # 视频读取、图像处理、可视化
numpy>=1.24          # 数值计算
ensemble-boxes>=1.0.4 # Weighted Box Fusion（备用）
scipy>=1.10          # 科学计算
```
