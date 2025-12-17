---
description: "SwinUNet-VOG 项目背景、目标、当前状态"
alwaysApply: false
---

# 项目上下文

## 项目概述

**SwinUNet-VOG** 是一个基于 Swin Transformer 的视线估计系统，用于从眼部图像估计视线方向（Pitch/Yaw 角度）。

### 核心目标
- 在 MPIIGaze 数据集上实现高精度视线估计（~5.7° 平均角度误差）
- 提供实时 GUI 可视化工具，支持视频处理和眨眼滤波
- 支持 Web 浏览器端推理（ONNX Runtime + MediaPipe）

### 应用场景
- 眼动追踪研究
- 医学视频分析（前庭功能检查）
- 人机交互
- 驾驶员注意力监测

---

## 技术栈

### 核心框架
- **Python 3.8+**
- **PyTorch 2.x** - 深度学习框架
- **ONNX** - 模型导出与跨平台部署

### 关键依赖
- **MediaPipe** - 人脸/眼部关键点检测
- **OpenCV** - 图像处理
- **CustomTkinter** - GUI 界面
- **Matplotlib** - 数据可视化
- **SciPy** - 信号滤波

### Web 部署
- **ONNX Runtime Web** - 浏览器端模型推理
- **MediaPipe JS** - 浏览器端人脸检测
- **Chart.js** - 实时图表

---

## 当前状态

### ✅ 已完成功能

1. **模型训练**
   - SwinUNet 架构实现（~7.6M 参数）
   - 跨被试评估（leave-one-out）
   - 个人化评估
   - 训练曲线可视化

2. **GUI 可视化器** (`gui_visualizer.py`)
   - 视频拖拽处理
   - MediaPipe 眼部提取
   - 实时角度曲线
   - 眨眼检测与时间窗口扩展（±300ms）
   - 信号滤波（中值 + 低通）
   - 3D 注视向量可视化
   - Plist 数据导出（原始/滤波）
   - 内存优化（临时文件缓存）
   - 动画模式可选

3. **Web 部署**
   - ONNX 模型导出
   - 浏览器端推理 demo
   - 本地依赖下载脚本

### 🔄 进行中
- 无

### 📋 计划中
- GPU 加速优化
- 批量视频处理
- 更多导出格式支持

---

## 项目里程碑

| 日期 | 事件 |
|------|------|
| 2024-Q4 | 项目初始化，SwinUNet 模型实现 |
| 2024-Q4 | MPIIGaze 数据集训练完成 |
| 2024-Q4 | GUI 可视化工具开发 |
| 2024-12 | 眨眼检测两阶段处理优化 |
| 2024-12 | 内存优化（临时文件缓存） |
| 2024-12 | Plist 导出功能完善 |
| 2024-12 | 项目结构精简 |

---

## 关键文件

### 核心代码
- `model.py` - SwinUNet 模型定义
- `train.py` - 训练流程
- `test.py` - 评估脚本
- `gui_visualizer.py` - GUI 可视化工具
- `preprocessing.py` - 数据预处理

### 配置与资源
- `config.json` - 训练配置
- `checkpoints/` - 模型权重
- `models/` - 导出的 ONNX 模型
- `MPIIGaze/` - 数据集

### Web 部署
- `js/demo.html` - 演示页面
- `js/server.py` - HTTP 服务器
- `js/swinunet-gaze-api.js` - JavaScript API

