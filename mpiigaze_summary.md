# MPIIGaze Dataset Summary

## 数据集概览

MPIIGaze是一个用于视线估计（gaze estimation）的公开数据集。数据集包含从笔记本电脑摄像头收集的自然图像，用户在日常生活中使用计算机时的真实场景。

- **许可证**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
- **用途**: 基于外观的视线估计（Appearance-based gaze estimation）
- **应用**: 眼动追踪、人机交互、注意力分析

## 数据组织结构

### 1. 参与者信息
- **参与者数量**: 15人（p00-p14）
- **记录天数**: 共521天
- **每日记录**: 不同参与者的记录天数不同（从7天到69天不等）

### 2. 数据目录结构

```
MPIIGaze/
├── Data/
│   ├── Normalized/           # 标准化处理后的数据
│   │   └── p00-p14/         # 按参与者组织
│   │       └── day01.mat, day02.mat, ...  # 每日的MAT文件
│   └── Original/             # 原始数据
│       └── p00-p14/
│           ├── Calibration/ # 校准数据
│           ├── day01/       # 每日原始图像和标注
│           └── ...
├── Annotation Subset/        # 手动标注的子集
│   └── p00.txt, p01.txt, ... (10,654张图像)
└── Evaluation Subset/        # 评估子集
    ├── sample list for eye image/  # 眼图评估集（45,000个样本）
    └── annotation for face image/  # 面部评估集（45,000个样本）
```

## 数据格式详解

### 1. Normalized 数据 (.mat文件)

**位置**: `MPIIGaze/Data/Normalized/<participant>/<day>.mat`

每个MAT文件包含预处理过的眼图数据和对应的3D gaze向量。

**结构**:
```python
{
    'data': {
        'left':  [gaze, image, pose],   # 左眼数据
        'right': [gaze, image, pose]    # 右眼数据
    },
    'filenames': [...]  # 对应的文件名列表
}
```

**字段说明**:
- **gaze** (3D vector): 3D gaze方向向量，shape = (995, 3)
- **image** (RGB图像): 标准化后的眼图，shape = (995, 36, 60)，dtype = uint8
- **pose**: 头部姿态信息
- **filenames**: 995个文件名，对应原始图像

**示例**: p00/day01.mat包含995个样本

### 2. Original 数据

**位置**: `MPIIGaze/Data/Original/<participant>/<day>/`

包含原始的人脸图像和详细标注。

**文件**:
- `*.jpg`: 原始人脸图像
- `annotation.txt`: 每帧的详细标注
  - 12个面部关键点（24个坐标）
  - 屏幕上的注视目标（x, y）
  - 相机参数和额外的gaze信息
- `Calibration/*`: 校准数据
  - `Camera.mat`: 相机标定参数
  - `monitorPose.mat`: 显示器位置/朝向
  - `screenSize.mat`: 屏幕尺寸

### 3. Annotation Subset

**位置**: `MPIIGaze/Annotation Subset/p*.txt`

手动标注的子集，包含详细的面部landmarks。

- **总数**: 10,654张图像
- **标注内容**:
  - 12个面部关键点
  - 人脸边界框
  - 左右眼边界框
  - 瞳孔位置

**格式**:
```
<day>/<filename> <landmark_x1> <landmark_y1> ... <pupil_x_left> <pupil_y_left> <pupil_x_right> <pupil_y_right>
```

### 4. Evaluation Subset

#### 4.1 眼图评估集

**位置**: `MPIIGaze/Evaluation Subset/sample list for eye image/p*.txt`

- **每个参与者**: 3,000个样本（1,500个左眼 + 1,500个右眼）
- **总共**: 45,000个样本
- **说明**: 右眼图像已翻转，以匹配左眼的方向

**格式**:
```
<day>/<filename> <left|right>
```

示例:
```
day08/0069.jpg left
day28/0646.jpg left
day04/0900.jpg right
```

#### 4.2 面部评估集

**位置**: `MPIIGaze/Evaluation Subset/annotation for face image/p*.txt`

- **数量**: 每个参与者3,000个样本
- **对应**: 与眼图评估集相同的图像（对应的面部区域）
- **格式**: 图像文件名 + 6个面部关键点

## 数据集统计

| 项目 | 数量 |
|------|------|
| 参与者 | 15 |
| 总记录天数 | 521 |
| 眼图评估样本 | 45,000 |
| 面部评估样本 | 45,000 |
| 手动标注图像 | 10,654 |
| 眼图尺寸 | 36×60 pixels |
| Gaze向量维度 | 3D |

## 使用建议

### 训练视线估计模型

1. **使用Normalized数据**:
   - 预处理后的眼图
   - 标准化的3D gaze向量
   - 适用于深度学习模型训练

2. **使用Evaluation Subset**:
   - 标准的训练/测试划分
   - 包含左右眼数据
   - 适合进行论文复现和对比

3. **使用Annotation Subset**:
   - 精确的landmark标注
   - 适合基于关键点的approaches

### 数据预处理注意事项

1. **右眼翻转**: Evaluation Subset中的右眼已翻转
2. **标准化**: Normalized数据已经过预处理
3. **gaze表示**: 使用3D相机坐标系中的方向向量
4. **参与者划分**: 某些研究使用person-independent evaluation

### 评估协议

根据原论文，建议的评估方式：
- **Person-specific**: 在同一个人的数据上训练和测试
- **Person-independent**: 在未见过的参与者上测试

## 相关论文

- [1] Appearance-Based Gaze Estimation in the Wild (2015)
- [2] It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation

## 数据引用

```
@inproceedings{zhang2015appearance,
  title={Appearance-based gaze estimation in the wild},
  author={Zhang, Xucong and Sugano, Yusuke and Fritz, Mario and Bulling, Andreas},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2015}
}
```

## 总结

MPIIGaze是一个高质量、大规模的自然场景gaze estimation数据集。适合用于：
- 基于外观的视线估计研究
- 深度学习模型训练
- 人机交互应用开发
- 眼动追踪算法评估

数据集提供了三种不同格式的数据（Normalized, Original, Annotations），可以满足不同的研究需求。

