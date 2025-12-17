---
description: "SwinUNet-VOG 术语表、概念定义"
alwaysApply: false
---

# 术语表

## 模型相关

### SwinUNet
基于 **Swin Transformer** 和 **U-Net** 的混合架构。结合了 Transformer 的全局注意力机制和 U-Net 的跳跃连接结构，用于从眼部图像回归视线方向。

### Swin Transformer
**Shifted Window Transformer** 的缩写。通过窗口注意力机制降低计算复杂度，同时通过窗口移动实现跨窗口信息交互。

### Patch Embedding
将输入图像分割成固定大小的 patch（如 4×4），每个 patch 展平后通过线性层映射为 token 向量。

### Window Attention
在局部窗口内计算自注意力，复杂度从 O(n²) 降低到 O(n·w²)，其中 w 为窗口大小。

---

## 视线估计

### Gaze Vector / 注视向量
表示视线方向的 3D 单位向量 (x, y, z)，满足 x² + y² + z² = 1。

### Pitch / 俯仰角
垂直方向的视线角度。向上为正，向下为负。
```python
Pitch = arcsin(-y)  # 范围: [-90°, 90°]
```

### Yaw / 偏航角
水平方向的视线角度。向右为正，向左为负。
```python
Yaw = arctan2(-x, -z)  # 范围: [-180°, 180°]
```

### Angular Error / 角度误差
预测视线方向与真实视线方向之间的夹角（度）。通过向量点积计算。

---

## 眨眼处理

### EAR (Eye Aspect Ratio)
**眼睛纵横比**，用于检测眨眼。计算公式：
```
EAR = (上下眼睑距离之和) / (2 × 左右眼角距离)
```
典型阈值：EAR < 0.20 认为是眨眼。

### Blink Window / 眨眼窗口
眨眼事件前后的时间范围，该范围内的数据被标记为不可靠。
- 默认值：±300ms
- 目的：覆盖眼睛完全闭合和微闭（半眨眼）状态

### Two-Phase Processing / 两阶段处理
1. **阶段1**：快速扫描视频，记录所有眨眼时刻
2. **阶段2**：重新处理，对每个眨眼时刻应用对称窗口

优势：确保眨眼前的帧也能被正确标记。

### Linear Interpolation / 线性插值
用相邻有效数据点填充眨眼期间的无效数据，保持曲线连续性。

---

## 信号处理

### Median Filter / 中值滤波
用于去除信号中的尖峰噪声（脉冲噪声）。取滑动窗口内的中值作为输出。
- 默认 kernel_size: 5

### Low-pass Filter / 低通滤波
去除高频噪声，平滑信号曲线。使用 Butterworth 滤波器。
- 默认截止频率: 8.0 Hz
- 滤波阶数: 4

### Butterworth Filter
具有最平坦通带响应的滤波器设计。在截止频率处衰减 -3dB。

---

## 数据集

### MPIIGaze
由 MPI 信息学研究所发布的视线估计数据集。
- **规模**: 15 人，~45,000 张眼部图像
- **图像尺寸**: 36×60×3（归一化后）
- **标注**: 3D 注视向量
- **采集环境**: 日常办公场景

### Geometric Normalization / 几何归一化
将眼部图像旋转到标准视角的预处理步骤：
1. 检测人脸关键点
2. 计算头部姿态（旋转矩阵）
3. 将眼部图像旋转到正视状态
4. 裁剪并缩放到 36×60

### Cross-Subject Evaluation / 跨被试评估
**留一法 (Leave-One-Out)**：用 14 人训练，1 人测试。评估模型的泛化能力。

### Person-Specific Evaluation / 个人化评估
单人数据分割为训练集和测试集。评估模型对特定用户的适应能力。

---

## 部署相关

### ONNX
**Open Neural Network Exchange**，开放神经网络交换格式。用于跨平台模型部署。

### ONNX Runtime Web
在浏览器中运行 ONNX 模型的 JavaScript 库，使用 WebAssembly (WASM) 加速。

### MediaPipe
Google 开发的多媒体处理框架。本项目使用其 Face Mesh 模块检测 468 个人脸关键点。

### WASM (WebAssembly)
一种低级字节码格式，可在浏览器中以接近原生速度运行。

---

## GUI 相关

### CustomTkinter
基于 Tkinter 的现代化 GUI 库，提供更美观的界面组件。

### CTkToplevel
CustomTkinter 的顶级窗口类，用于创建独立的子窗口。

### Animation Mode / 动画模式
实时显示处理过程的模式。开启时较慢但可观察处理进度，关闭时仅显示进度条。

---

## 数据格式

### Plist (Property List)
苹果开发的 XML 格式文件，用于存储结构化数据。本项目用于导出视线数据，兼容 pVestibular 分析平台。

### 导出字段
```python
{
    'TimeList': [...],           # 时间戳 (秒)
    'LeftEyeXDegList': [...],    # Yaw 角度 (度)
    'LeftEyeYDegList': [...],    # Pitch 角度 (度)
    'GazePitchDegList': [...],   # Pitch 角度 (度)
    'GazeYawDegList': [...],     # Yaw 角度 (度)
    'GazeXList': [...],          # 3D 向量 X
    'GazeYList': [...],          # 3D 向量 Y
    'GazeZList': [...],          # 3D 向量 Z
}
```

---

## 性能指标

### FPS (Frames Per Second)
每秒处理的帧数。
- CPU 模式: ~56 FPS
- GPU 模式: 200-300 FPS

### Inference Time / 推理时间
单次模型前向传播的耗时。
- CPU: ~10 ms
- GPU: ~3 ms

### Memory Usage / 内存使用
处理过程中的内存占用。
- 基础: ~400 MB
- 长视频: 使用临时文件缓存，增长有限

