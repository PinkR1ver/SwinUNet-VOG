# 训练和测试输出说明

本文档详细说明训练和测试过程中生成的所有输出文件和可视化结果。

## 📁 输出目录结构

```
checkpoints/          # 训练输出目录（由config.json中的save_dir指定）
├── checkpoint_best.pth      # 最佳模型（验证集上表现最好）
├── checkpoint_latest.pth    # 最新模型（最后一个epoch）
├── training_curves.png      # 训练曲线综合图（loss + angle error）
├── loss_curves.png          # Loss曲线详细图
├── angle_error_curves.png   # 角度误差曲线详细图
└── training_summary.json    # 训练摘要（JSON格式）

results/              # 测试输出目录（由test.py的--save_dir指定）
├── results.json                    # 测试结果统计（JSON格式）
├── error_distribution.png          # 误差分布直方图
├── cumulative_distribution.png     # 累积误差分布图
├── predictions_vs_targets.png      # 预测vs真实值散点图
├── prediction_samples.png          # 预测样本可视化（16个样本）
└── best_worst_predictions.png      # 最好和最差预测对比图
```

---

## 🏋️ 训练输出 (`train.py`)

### 1. 模型检查点文件

#### `checkpoint_best.pth`
- **描述**: 验证集上表现最好的模型权重
- **内容**:
  - `model_state_dict`: 模型参数
  - `optimizer_state_dict`: 优化器状态
  - `scheduler_state_dict`: 学习率调度器状态
  - `epoch`: 训练轮数
  - `best_val_angle`: 最佳验证集角度误差
  - `train_losses`: 所有epoch的训练loss历史
  - `val_losses`: 所有epoch的验证loss历史
  - `train_angles`: 所有epoch的训练角度误差历史
  - `val_angles`: 所有epoch的验证角度误差历史

#### `checkpoint_latest.pth`
- **描述**: 最后一个epoch的模型权重
- **内容**: 与`checkpoint_best.pth`相同，但保存的是最新状态

**使用方法**:
```python
checkpoint = torch.load('checkpoints/checkpoint_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### 2. 训练曲线图

#### `training_curves.png`
- **描述**: 综合训练曲线（包含loss和角度误差）
- **内容**:
  - 左图: 训练和验证Loss曲线
  - 右图: 训练和验证角度误差曲线
- **用途**: 快速查看整体训练趋势

#### `loss_curves.png`
- **描述**: Loss曲线详细图
- **内容**:
  - 训练Loss (蓝色)
  - 验证Loss (红色)
  - 带有数据点标记
- **用途**: 分析模型是否过拟合、学习率是否合适

#### `angle_error_curves.png`
- **描述**: 角度误差曲线详细图
- **内容**:
  - 训练集角度误差 (蓝色)
  - 验证集角度误差 (红色)
  - 带有数据点标记
- **用途**: 监控模型在gaze estimation任务上的性能

### 3. 训练摘要

#### `training_summary.json`
- **描述**: 训练过程摘要（JSON格式）
- **内容**:
  ```json
  {
    "total_epochs": 50,
    "best_val_angle": 8.5,
    "final_train_loss": 0.001234,
    "final_val_loss": 0.001456,
    "final_train_angle": 7.2,
    "final_val_angle": 8.7
  }
  ```
- **用途**: 快速查看训练结果，便于对比不同实验

### 4. 控制台输出

训练过程中的控制台输出包括：
- 每个batch的进度和loss
- 每个epoch的总结（训练loss、验证loss、角度误差、耗时）
- 最佳模型保存提示
- 训练完成后的统计信息

---

## 🧪 测试输出 (`test.py`)

### 1. 测试结果统计

#### `results.json`
- **描述**: 测试集上的详细统计信息
- **内容**:
  ```json
  {
    "mean_error": 8.5,
    "median_error": 6.2,
    "std_error": 5.3,
    "percentile_95": 18.7,
    "percentile_99": 25.3,
    "avg_loss": 0.001456
  }
  ```
- **用途**: 量化模型性能，便于对比和报告

### 2. 误差分析图

#### `error_distribution.png`
- **描述**: 角度误差分布直方图
- **内容**:
  - 误差分布直方图
  - 均值线（红色虚线）
  - 中位数线（绿色虚线）
- **用途**: 了解误差分布情况，识别异常值

#### `cumulative_distribution.png`
- **描述**: 累积误差分布图
- **内容**:
  - X轴: 角度误差（度）
  - Y轴: 累积百分比（%）
  - 曲线显示有多少比例的样本误差小于某个值
- **用途**: 回答"有多少比例的预测误差小于X度？"

#### `predictions_vs_targets.png`
- **描述**: 预测值与真实值的散点图
- **内容**:
  - 3个子图，分别对应gaze向量的3个分量（x, y, z）
  - 散点图显示预测vs真实值
  - 对角线（红色虚线）表示完美预测
- **用途**: 检查预测的系统性偏差

### 3. 可视化样本

#### `prediction_samples.png`
- **描述**: 随机选择的预测样本可视化
- **内容**:
  - 16个测试样本的眼图
  - 每个样本显示: 预测gaze向量、真实gaze向量、角度误差
- **用途**: 直观了解模型在具体样本上的表现

#### `best_worst_predictions.png` (通过 `visualize_results.py` 生成)
- **描述**: 最好和最差预测的对比
- **内容**:
  - 上半部分: 误差最小的8个样本
  - 下半部分: 误差最大的8个样本
- **用途**: 分析模型在哪些样本上表现好/差，找出规律

---

## 📊 可视化工具 (`visualize_results.py`)

### 使用方法

```bash
python visualize_results.py \
    --checkpoint checkpoints/checkpoint_best.pth \
    --data_dir MPIIGaze/Data/Normalized \
    --save_dir results \
    --num_samples 16
```

### 输出文件

- `prediction_samples.png`: 随机样本可视化
- `best_worst_predictions.png`: 最好/最差预测对比

---

## 📈 如何解读输出

### 1. 训练曲线解读

**Loss曲线**:
- ✅ **正常**: Loss持续下降，训练和验证loss接近
- ⚠️ **过拟合**: 训练loss继续下降，但验证loss不再下降或上升
- ⚠️ **欠拟合**: 训练和验证loss都很高且下降缓慢
- ⚠️ **学习率过大**: Loss震荡严重，不稳定

**角度误差曲线**:
- 好的训练: 角度误差持续减小，最终稳定
- 理想结果: 训练集误差 < 验证集误差 < 10度

### 2. 测试结果解读

**Mean Error** (平均误差):
- < 5°: 优秀
- 5-10°: 良好
- 10-15°: 中等
- > 15°: 需要改进

**95th Percentile** (95%分位数):
- 表示95%的样本误差小于此值
- 用于评估模型的稳定性

### 3. 误差分布解读

- **窄且偏左**: 大部分预测准确（好）
- **宽或偏右**: 预测不稳定或有系统性偏差（需改进）

---

## 🔄 完整工作流程示例

### 1. 训练模型
```bash
python train.py --epochs 50 --batch_size 64
```

**输出到**: `checkpoints/`
- `checkpoint_best.pth`
- `checkpoint_latest.pth`
- `training_curves.png`
- `loss_curves.png`
- `angle_error_curves.png`
- `training_summary.json`

### 2. 测试模型
```bash
python test.py --checkpoint checkpoints/checkpoint_best.pth
```

**输出到**: `results/`
- `results.json`
- `error_distribution.png`
- `cumulative_distribution.png`
- `predictions_vs_targets.png`
- `prediction_samples.png`

### 3. 可视化结果
```bash
python visualize_results.py --checkpoint checkpoints/checkpoint_best.pth
```

**输出到**: `results/`
- `best_worst_predictions.png` (更新)

---

## 📝 总结

### 训练阶段输出
✅ 模型检查点（best + latest）  
✅ Loss曲线图  
✅ 角度误差曲线图  
✅ 训练摘要JSON  
✅ 控制台日志  

### 测试阶段输出
✅ 误差统计JSON  
✅ 误差分布直方图  
✅ 累积分布图  
✅ 预测vs真实值散点图  
✅ 样本可视化  

### 可视化工具输出
✅ 最好/最差预测对比  
✅ 随机样本展示  

所有这些输出帮助您全面了解模型的训练过程和性能表现！
