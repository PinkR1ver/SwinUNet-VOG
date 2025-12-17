---
description: "SwinUNet-VOG 开发规则、环境管理、代码标准"
alwaysApply: true
---

# 开发规则

## 🌐 语言规则

- **始终使用中文回复用户**
- 代码注释可使用英文或中文
- 文档优先使用中文

---

## 🐍 Python 环境

### 环境配置
```bash
# 使用项目本地 conda 环境
.conda\python.exe

# 安装依赖
pip install -r requirements.txt
```

### 关键依赖版本
- Python: 3.8+
- PyTorch: 2.x
- MediaPipe: 0.10.21
- protobuf: 4.25.x（与 MediaPipe 兼容）
- CustomTkinter: 5.2.2

### 依赖冲突处理
⚠️ **注意**: TensorFlow 与 MediaPipe 存在 protobuf 版本冲突
```bash
# 如果出现 protobuf 导入错误，卸载 TensorFlow
pip uninstall tensorflow tensorflow-intel keras tensorboard -y
pip install "protobuf>=4.25.3,<5"
```

---

## 📁 代码标准

### 文件命名
- Python 文件: `snake_case.py`
- 类名: `PascalCase`
- 函数/变量: `snake_case`
- 常量: `UPPER_SNAKE_CASE`

### 代码风格
- 使用 4 空格缩进
- 类型提示推荐但不强制
- 文档字符串使用三引号

### 导入顺序
```python
# 1. 标准库
import os
import sys

# 2. 第三方库
import torch
import numpy as np
import cv2

# 3. 本地模块
from model import SwinUNetGaze
from preprocessing import EyeImagePreprocessor
```

---

## 📝 文档维护规则

### 必须更新的情况

1. **功能变更时**
   - 更新 `README.md` 对应章节
   - 更新 `@project-context` 的当前状态

2. **架构变更时**
   - 更新 `@architecture` 的模块设计
   - 更新数据流图

3. **新增术语时**
   - 添加到 `@glossary`

4. **性能优化时**
   - 更新 `PERFORMANCE.md`

### 禁止创建的文件
- 临时测试脚本（用完即删）
- 重复的文档文件
- 未使用的配置文件

---

## 🧪 测试规则

### 临时测试脚本
- 创建后必须删除
- 命名格式: `test_*.py` 或 `verify_*.py`
- 不要提交到版本控制

### 调试输出
- 使用 `print()` 进行临时调试
- 正式代码使用 `logging` 模块
- 提交前删除调试输出

---

## 🚀 运行命令

### GUI 可视化器
```bash
.conda\python.exe gui_visualizer.py
```

### Web 服务器
```bash
cd js
python server.py
# 访问 http://localhost:8000/demo.html
```

### 模型训练
```bash
python train.py --eval_mode cross_subject
```

### ONNX 导出
```bash
python export_to_onnx.py --checkpoint checkpoints/checkpoint_best.pth --output models/swinunet_web.onnx
```

---

## ⚠️ 常见问题处理

### protobuf 版本冲突
```bash
pip uninstall tensorflow tensorflow-intel keras tensorboard -y
pip install "protobuf>=4.25.3,<5"
```

### MediaPipe 初始化失败
- 确保 webcam 未被占用
- 检查 GPU 驱动版本

### GUI 窗口不显示
- 检查 CustomTkinter 版本
- 尝试重启 Python 进程

---

## 📊 性能考虑

### CPU 模式
- 处理速度: ~56 FPS
- 内存需求: 4-8 GB

### GPU 模式（推荐）
- 处理速度: 200-300 FPS
- 显存需求: 4 GB VRAM

### 内存优化
- 长视频处理使用临时文件缓存
- 每 1000 帧自动写入磁盘
- 处理完成后自动清理

