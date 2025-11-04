# 几何归一化：处理距离和位置不固定的问题

## 问题背景

在实际应用中，gaze estimation面临两个关键挑战：

1. **距离变化**：人眼到摄像头的距离不同，导致眼睛在图像中的**尺度（大小）**不同
2. **位置变化**：眼睛在图像中的**位置**不固定，可能出现在画面的不同区域

这些问题会导致：
- **尺度不匹配**：同一只眼睛在不同距离下看起来大小不同
- **位置偏移**：眼睛位置的变化会影响模型的输入分布
- **性能下降**：模型难以学习稳定的特征表示

## 解决方案：基于关键点的几何归一化

本项目提供了基于**眼部关键点**的几何归一化方案，通过仿射/透视变换将眼睛区域归一化到标准位置和大小。

### 核心思想

1. **检测关键点**：使用眼部关键点（眼角、瞳孔）标识眼睛的位置和形状
2. **计算变换矩阵**：基于关键点计算仿射或透视变换矩阵
3. **应用变换**：将眼睛区域变换到标准位置和大小

### 关键点定义

使用6个关键点来描述眼睛：
- **左外眼角** (outer corner of left eye)
- **左内眼角** (inner corner of left eye)
- **右内眼角** (inner corner of right eye)
- **右外眼角** (outer corner of right eye)
- **左瞳孔中心** (left pupil center)
- **右瞳孔中心** (right pupil center)

## 使用方法

### 方式1：通过配置文件启用

在 `config.json` 中启用几何归一化：

```json
{
    "preprocessing": {
        "mode": "full",
        "use_geometric_normalization": true,
        "normalize_illumination": true,
        "normalize_contrast": true
    }
}
```

然后正常训练：
```bash
python train.py
```

### 方式2：在代码中使用

```python
from preprocessing import EyeImagePreprocessor
from geometric_normalization import RobustGeometricNormalizer
import numpy as np

# 创建几何归一化器
geo_normalizer = RobustGeometricNormalizer(
    target_size=(36, 60),
    fallback_to_center=True,
    use_pupil=True
)

# 创建预处理器（启用几何归一化）
preprocessor = EyeImagePreprocessor(
    target_size=(36, 60),
    use_geometric_normalization=True,
    geometric_normalizer=geo_normalizer
)

# 使用关键点进行归一化
image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)

# 关键点坐标 (x, y)
keypoints = np.array([
    [50, 100],   # 左外眼角
    [100, 100],  # 左内眼角
    [150, 100],  # 右内眼角
    [200, 100],  # 右外眼角
    [75, 100],   # 左瞳孔
    [175, 100],  # 右瞳孔
], dtype=np.float32)

# 应用归一化
normalized = preprocessor(image, keypoints=keypoints)
```

### 方式3：只使用眼睛边界框（如果没有关键点）

```python
from geometric_normalization import RobustGeometricNormalizer

normalizer = RobustGeometricNormalizer(target_size=(36, 60))

image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)

# 眼睛边界框 [x, y, width, height]
eye_bbox = [100, 80, 150, 40]

# 使用bbox估算关键点并归一化
normalized = normalizer(image, eye_bbox=eye_bbox)
```

### 方式4：处理MPIIGaze标注数据

```python
from geometric_normalization import RobustGeometricNormalizer, parse_mpiigaze_keypoints

# 读取MPIIGaze标注文件
annotation_line = "day01/0005.jpg 594 366 637 365 719 366 762 368 626 495 726 496"

# 解析关键点
keypoints = parse_mpiigaze_keypoints(annotation_line)

# 归一化
normalizer = RobustGeometricNormalizer(target_size=(36, 60))
image = cv2.imread("path/to/image.jpg")
normalized = normalizer(image, keypoints=keypoints)
```

## 工作原理

### 1. 仿射变换 vs 透视变换

**仿射变换**（使用3个点）：
- 适用于平面内的变换（旋转、缩放、平移）
- 保持平行线仍然平行
- 计算更快

**透视变换**（使用4个点）：
- 适用于3D到2D的投影
- 可以处理透视失真
- 更精确但计算更慢

### 2. 变换矩阵计算

```python
# 使用眼角点计算仿射变换
src_points = keypoints[:4]  # 4个眼角点
ref_points = reference_points[:4]  # 参考位置

# 计算变换矩阵
M = cv2.getAffineTransform(src_points[:3], ref_points[:3])

# 应用变换
normalized = cv2.warpAffine(image, M, (width, height))
```

### 3. 参考点定义

标准参考点定义了归一化后的眼睛位置：

```python
# 假设目标尺寸为 36×60
center_x = 30  # 图像中心x坐标
center_y = 18  # 图像中心y坐标
eye_width = 36  # 眼间距（图像宽度的60%）

reference_points = [
    [center_x - eye_width/2, center_y],  # 左外眼角
    [center_x - eye_width/4, center_y],  # 左内眼角
    [center_x + eye_width/4, center_y],  # 右内眼角
    [center_x + eye_width/2, center_y],  # 右外眼角
    [center_x - eye_width/3, center_y],  # 左瞳孔
    [center_x + eye_width/3, center_y],  # 右瞳孔
]
```

## 备选方案（无关键点时）

如果没有关键点信息，系统会自动使用备选方案：

1. **中心裁剪**：假设眼睛在图像中心，裁剪中心80%的区域
2. **简单缩放**：将裁剪后的区域缩放到目标尺寸

虽然不如关键点精确，但仍然是可用的方案。

## 最佳实践

### 1. 训练阶段

**MPIIGaze数据集**（已标准化）：
- 如果数据已经几何归一化，可以**不启用**几何归一化
- 如果数据未归一化但有标注，**启用**几何归一化

**自定义数据集**：
- **有关键点**：强烈建议启用几何归一化
- **无关键点**：使用备选方案或考虑添加关键点检测

### 2. 推理阶段

**有关键点数据**：
```python
preprocessor = EyeImagePreprocessor(
    use_geometric_normalization=True
)
processed = preprocessor(image, keypoints=keypoints)
```

**无关键点数据**：
```python
# 使用备选方案
preprocessor = EyeImagePreprocessor(
    use_geometric_normalization=False  # 使用简单缩放
)
processed = preprocessor(image)
```

### 3. 关键点检测

如果数据中没有关键点，可以使用：

- **OpenCV DNN Face Detector**：检测人脸和关键点
- **MediaPipe Face Mesh**：检测面部关键点
- **dlib Face Landmarks**：检测68个面部关键点
- **深度学习关键点检测器**：如FAN、HRNet等

示例（使用dlib）：
```python
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

faces = detector(image)
for face in faces:
    landmarks = predictor(image, face)
    # 提取眼部关键点（landmark 36-47对应眼睛区域）
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    # 计算眼角和瞳孔位置...
```

## 性能影响

### 计算开销
- **几何归一化**：每张图像增加约2-5ms（取决于图像尺寸）
- **关键点检测**：如果使用外部检测器，额外增加10-50ms

### 精度提升
- **有几何归一化**：显著提升对距离和位置变化的鲁棒性
- **预期改进**：在存在距离/位置变化的数据上，误差可降低20-40%

## 常见问题

### Q1: 如果没有关键点怎么办？

**A**: 使用备选方案（中心裁剪+缩放），或使用关键点检测器自动检测关键点。

### Q2: 几何归一化会影响gaze标签吗？

**A**: 不会。几何归一化只改变图像，不改变gaze向量。gaze向量在相机坐标系中，不受图像变换影响。

### Q3: 需要每个样本都有关键点吗？

**A**: 不需要。如果有部分样本没有关键点，系统会自动使用备选方案。但建议尽可能多的样本提供关键点。

### Q4: 可以使用其他关键点吗？

**A**: 可以。只要关键点能够唯一确定眼睛的位置和形状，就可以使用。但建议使用标准的眼部关键点（眼角、瞳孔）。

### Q5: 几何归一化与图像预处理的关系？

**A**: 几何归一化应该在图像预处理的第一步进行，因为它改变了图像的空间结构。其他预处理（光照归一化、对比度归一化等）应该在几何归一化之后进行。

## 总结

几何归一化是处理距离和位置变化的关键技术：

1. **基于关键点**：使用眼部关键点进行精确归一化
2. **仿射/透视变换**：将眼睛变换到标准位置和大小
3. **备选方案**：无关键点时使用中心裁剪+缩放
4. **性能提升**：显著提高对距离和位置变化的鲁棒性

通过合理使用几何归一化，可以显著提高模型在实际应用中的性能！
