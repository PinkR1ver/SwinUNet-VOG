# 图像预处理：处理不同采集标准

## 问题背景

在实际应用中，眼图数据可能来自不同的采集设备，具有不同的特性：

1. **不同相机分辨率**：640×480、1280×720、1920×1080等
2. **不同光照条件**：室内/室外、自然光/人工光、明暗差异
3. **不同色彩空间**：RGB、BGR、灰度图
4. **不同图像质量**：清晰度、对比度、噪声水平
5. **不同几何变换**：旋转、缩放、裁剪位置

如果直接将不同采集标准的图像输入模型，会导致：
- **分布不匹配**：测试数据与训练数据分布差异大
- **性能下降**：模型在新的数据上表现不佳
- **泛化能力差**：无法适应实际应用场景

## 解决方案

本项目提供了两种预处理模式来处理这个问题：

### 1. Simple模式（默认）

**适用场景**：
- MPIIGaze等已经标准化过的数据集
- 所有数据使用相同采集设备
- 数据已经过预处理

**功能**：
- 尺寸归一化到目标尺寸（36×60）
- 基本的数值归一化（[0, 255] → [0, 1]）
- 灰度图转RGB

**配置**：
```json
{
    "preprocessing": {
        "mode": "simple"
    }
}
```

### 2. Full模式（完整预处理）

**适用场景**：
- 不同采集标准的混合数据
- 实际应用中的新数据
- 需要增强模型鲁棒性

**功能**：
- ✅ **尺寸归一化**：自动调整到目标尺寸
- ✅ **光照归一化**：去除不均匀光照的影响（LAB颜色空间）
- ✅ **对比度归一化**：使用CLAHE标准化对比度
- ✅ **颜色归一化**：处理不同相机色彩特性
- ✅ **Gamma校正**：调整图像亮度分布
- ✅ **自适应直方图均衡化**（可选）：增强图像细节

**配置**：
```json
{
    "preprocessing": {
        "mode": "full",
        "normalize_illumination": true,
        "normalize_contrast": true,
        "normalize_color": true,
        "gamma_correction": true,
        "adaptive_hist_eq": false
    }
}
```

## 使用方法

### 方式1：通过配置文件

在 `config.json` 中设置预处理模式：

```json
{
    "preprocessing": {
        "mode": "full",
        "normalize_illumination": true,
        "normalize_contrast": true,
        "normalize_color": true,
        "gamma_correction": true,
        "adaptive_hist_eq": false
    }
}
```

然后正常训练：
```bash
python train.py
```

### 方式2：在代码中使用

```python
from preprocessing import EyeImagePreprocessor, SimplePreprocessor
from data import MPIIGazeDataset

# 创建预处理器
preprocessor = EyeImagePreprocessor(
    target_size=(36, 60),
    normalize_illumination=True,
    normalize_contrast=True,
    normalize_color=True,
    gamma_correction=True,
    adaptive_hist_eq=False
)

# 使用预处理器
dataset = MPIIGazeDataset(
    data_dir="MPIIGaze/Data/Normalized",
    participants=['p00', 'p01'],
    preprocessor=preprocessor
)
```

### 方式3：处理单个图像

```python
from preprocessing import EyeImagePreprocessor
import numpy as np
from PIL import Image

# 创建预处理器
preprocessor = EyeImagePreprocessor(
    target_size=(36, 60),
    normalize_illumination=True,
    normalize_contrast=True
)

# 加载图像（可以是任何尺寸、格式）
image = Image.open("eye_image.jpg")
image_np = np.array(image)

# 预处理
processed = preprocessor(image_np)
# 返回: torch.Tensor, shape (3, 36, 60), range [0, 1]

# 可以直接输入模型
model.eval()
with torch.no_grad():
    gaze = model(processed.unsqueeze(0))  # 添加batch维度
```

## 预处理步骤详解

### 1. 尺寸归一化
- **方法**：使用`cv2.resize`，插值方法为`INTER_AREA`（适合缩小）
- **目标**：将所有图像统一到36×60像素
- **注意**：保持宽高比可能导致变形，但对于眼图通常可接受

### 2. 光照归一化
- **方法**：在LAB颜色空间的L通道上进行高斯模糊和加权混合
- **原理**：通过估计和去除不均匀光照，标准化亮度分布
- **效果**：减少光照条件差异的影响

### 3. 对比度归一化
- **方法**：CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **参数**：clipLimit=2.0, tileGridSize=(8, 8)
- **效果**：增强局部对比度，同时限制过度增强

### 4. 颜色归一化
- **方法**：对每个通道进行均值-方差归一化，然后重新映射到[0, 255]
- **效果**：标准化不同相机的色彩特性差异

### 5. Gamma校正
- **方法**：使用查找表(LUT)进行gamma=1.2的校正
- **效果**：调整图像亮度分布，使其更符合人眼感知

### 6. 自适应直方图均衡化（可选）
- **方法**：在YCbCr颜色空间的Y通道上应用CLAHE
- **注意**：可能对某些图像过度增强，默认关闭

## 性能考虑

### 计算开销
- **Simple模式**：几乎无额外开销
- **Full模式**：每张图像增加约5-10ms处理时间（取决于图像尺寸）

### 内存占用
- 预处理在数据加载时进行，不增加模型内存占用
- 建议使用多进程数据加载（`num_workers > 0`）以并行化预处理

## 最佳实践

### 1. 训练阶段
- **MPIIGaze数据集**：使用`simple`模式（数据已标准化）
- **混合数据集**：使用`full`模式
- **增强鲁棒性**：即使使用MPIIGaze，也可以尝试`full`模式训练

### 2. 测试/推理阶段
- **新数据**：如果新数据采集标准不同，使用`full`模式
- **相同标准**：如果测试数据与训练数据相同标准，使用`simple`模式

### 3. 调试建议
- 可视化预处理前后的图像，确认预处理效果
- 对比不同预处理配置的模型性能
- 根据具体数据特性调整预处理参数

## 示例：处理不同采集标准的图像

```python
import numpy as np
from preprocessing import EyeImagePreprocessor

# 创建预处理器
preprocessor = EyeImagePreprocessor(
    target_size=(36, 60),
    normalize_illumination=True,
    normalize_contrast=True,
    normalize_color=True,
    gamma_correction=True
)

# 模拟不同采集标准的图像
different_images = {
    'high_res': np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8),
    'low_res': np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
    'grayscale': np.random.randint(0, 255, (480, 640), dtype=np.uint8),
    'bright': np.random.randint(200, 255, (480, 640, 3), dtype=np.uint8),
    'dark': np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8),
}

# 统一预处理
processed_images = {}
for name, img in different_images.items():
    processed = preprocessor(img)
    processed_images[name] = processed
    print(f"{name}: {img.shape} -> {processed.shape}")

# 所有图像现在都是相同的格式: (3, 36, 60), range [0, 1]
# 可以直接输入模型进行推理
```

## 常见问题

### Q1: 预处理会改变gaze向量的标签吗？
**A**: 不会。预处理只改变图像，不改变gaze向量。但是，如果图像发生了翻转等几何变换，需要相应调整gaze向量（这在数据增强中处理）。

### Q2: 预处理会影响模型精度吗？
**A**: 
- 对于已经标准化的数据（如MPIIGaze），`simple`模式通常足够
- 对于不同采集标准的数据，`full`模式通常会**提高**精度
- 过度的预处理（如adaptive_hist_eq）可能对某些图像过度增强，反而降低精度

### Q3: 如何处理旋转的图像？
**A**: 当前的预处理不处理旋转。如果数据中有旋转，需要：
1. 使用关键点进行几何归一化（需要landmark信息）
2. 在数据增强中添加随机旋转
3. 使用更大的模型或增强训练数据多样性

### Q4: 可以自定义预处理流程吗？
**A**: 可以。继承`EyeImagePreprocessor`类，重写`__call__`方法，或直接修改`preprocessing.py`。

## 总结

不同采集标准的图片是实际应用中必须面对的问题。本项目提供了灵活的预处理方案：

- **Simple模式**：快速、高效，适用于标准化数据
- **Full模式**：全面、鲁棒，适用于混合数据和实际应用

通过合理配置预处理参数，可以显著提高模型对不同采集标准的适应能力，增强模型的实用性和泛化能力。
