# SwinUNet Gaze API 参考文档

视线估计 JavaScript API，用于在浏览器中实时预测视线方向。

## 快速开始

```javascript
const api = new SwinUNetGazeAPI();
await api.initialize('models/tfjs_model/model.json');

const video = document.getElementById('myVideo');
const result = await api.processFrame(video, 'left');

console.log(`Pitch: ${result.pitch}°, Yaw: ${result.yaw}°`);
```

---

## 类：SwinUNetGazeAPI

### 构造函数

```javascript
const api = new SwinUNetGazeAPI();
```

创建 API 实例。无需参数。

---

### 方法

#### initialize(modelPath)

初始化模型和 MediaPipe Face Mesh。

**参数：**
- `modelPath` (string): TensorFlow.js 模型路径，指向 `model.json` 文件

**返回：**
- `Promise<void>`

**示例：**
```javascript
await api.initialize('models/tfjs_model/model.json');
```

**说明：**
此方法会加载两个组件：
1. TensorFlow.js 模型（SwinUNet）
2. MediaPipe Face Mesh（用于检测人脸关键点）

---

#### processFrame(source, eye)

处理单帧图像或视频，返回视线估计结果。

**参数：**
- `source` (HTMLVideoElement | HTMLImageElement | HTMLCanvasElement): 输入源
- `eye` (string): 选择眼睛，`'left'` 或 `'right'`，默认 `'left'`

**返回：**
- `Promise<Object | null>`: 视线数据对象，如果检测失败返回 `null`

**返回对象结构：**
```javascript
{
    vector: [x, y, z],  // 3D 单位向量
    pitch: number,      // 俯仰角（度）
    yaw: number         // 偏航角（度）
}
```

**示例：**
```javascript
const video = document.getElementById('video');
const gaze = await api.processFrame(video, 'left');

if (gaze) {
    console.log('3D Vector:', gaze.vector);
    console.log('Pitch:', gaze.pitch, '°');
    console.log('Yaw:', gaze.yaw, '°');
}
```

---

#### extractEyeROI(source, eye)

从输入源中提取眼部 ROI（感兴趣区域）。

**参数：**
- `source` (HTMLVideoElement | HTMLImageElement | HTMLCanvasElement): 输入源
- `eye` (string): `'left'` 或 `'right'`

**返回：**
- `Promise<tf.Tensor | null>`: 眼部图像张量 (1, 3, 36, 60)，失败返回 `null`

**示例：**
```javascript
const eyeTensor = await api.extractEyeROI(video, 'left');

if (eyeTensor) {
    // 使用张量进行推理
    const gaze = await api.predictGaze(eyeTensor);
    eyeTensor.dispose(); // 释放内存
}
```

**注意：**
返回的张量需要手动释放内存（调用 `.dispose()`）。

---

#### predictGaze(eyeROI)

对眼部 ROI 张量进行视线预测。

**参数：**
- `eyeROI` (tf.Tensor): 眼部图像张量，形状为 (1, 3, 36, 60)

**返回：**
- `Promise<Object>`: 视线数据对象

**示例：**
```javascript
const eyeTensor = await api.extractEyeROI(video, 'left');
const gaze = await api.predictGaze(eyeTensor);
eyeTensor.dispose();
```

---

#### vectorToAngles(vector)

将 3D 视线向量转换为 Pitch 和 Yaw 角度。

**参数：**
- `vector` (Array): 3D 向量 `[x, y, z]`

**返回：**
- `Object`: `{ pitch: number, yaw: number }`，单位为度

**示例：**
```javascript
const vector = [0.1, -0.2, -0.97];
const angles = api.vectorToAngles(vector);
console.log(angles); // { pitch: 11.5, yaw: -5.9 }
```

**转换公式：**
```
pitch = arcsin(-y) * 180 / π
yaw = arctan2(-x, -z) * 180 / π
```

---

#### getHistory()

获取视线历史记录。

**返回：**
- `Object`: `{ gaze: Array, time: Array }`

**示例：**
```javascript
const history = api.getHistory();
console.log('Gaze history:', history.gaze);
console.log('Time history:', history.time);
```

**说明：**
- `gaze`: 视线数据数组，每个元素为 `{ vector, pitch, yaw }`
- `time`: 时间戳数组（毫秒）
- 历史记录最多保存 300 帧

---

#### clearHistory()

清空历史记录。

**返回：**
- `void`

**示例：**
```javascript
api.clearHistory();
```

---

#### dispose()

释放所有资源（模型和 MediaPipe）。

**返回：**
- `void`

**示例：**
```javascript
api.dispose();
```

**说明：**
在页面卸载或不再使用 API 时调用，避免内存泄漏。

---

## 属性

### isInitialized

**类型：** `boolean`

**说明：** 指示 API 是否已初始化。

**示例：**
```javascript
if (api.isInitialized) {
    console.log('API ready');
}
```

---

### inputSize

**类型：** `Object`

**说明：** 模型输入尺寸 `{ width: 60, height: 36 }`。

---

## MediaPipe 集成说明

API 内部使用 MediaPipe Face Mesh 进行人脸关键点检测。

### 检测流程

1. **人脸检测**：MediaPipe 检测人脸并返回 468 个关键点
2. **眼部定位**：根据预定义的关键点索引提取眼角和眼睑位置
3. **仿射变换**：将眼部区域变换到标准坐标系
4. **归一化**：调整为模型输入尺寸 (36×60)

### 关键点索引

```javascript
// 左眼
LEFT_EYE_OUTER = 263   // 外眼角
LEFT_EYE_INNER = 362   // 内眼角
LEFT_EYE_UPPER = 386   // 上眼睑中心

// 右眼
RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133
RIGHT_EYE_UPPER = 159
```

### 坐标变换

MediaPipe 返回归一化坐标 (0-1)，需要转换为像素坐标：

```javascript
const pixelX = landmark.x * imageWidth;
const pixelY = landmark.y * imageHeight;
```

### 仿射变换矩阵

使用 3 个源点和 3 个目标点计算仿射变换：

**源点**（从 MediaPipe 获取）：
- 外眼角
- 内眼角
- 上眼睑中心

**目标点**（标准化坐标）：
```javascript
const padding = 0.15;
const dstPoints = [
    [width * padding, height * 0.55],           // 外眼角
    [width * (1 - padding), height * 0.55],     // 内眼角
    [width * 0.5, height * 0.25]                // 上眼睑
];
```

这确保了眼部区域在标准位置和大小，与训练数据一致。

---

## 完整示例

### 基础用法

```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.11.0"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4"></script>
    <script src="js/swinunet-gaze-api.js"></script>
</head>
<body>
    <video id="video" src="video.mp4" controls></video>
    <div id="result"></div>
    
    <script>
        const api = new SwinUNetGazeAPI();
        const video = document.getElementById('video');
        const result = document.getElementById('result');
        
        async function main() {
            await api.initialize('models/tfjs_model/model.json');
            
            video.addEventListener('play', processLoop);
        }
        
        async function processLoop() {
            if (video.paused || video.ended) return;
            
            const gaze = await api.processFrame(video, 'left');
            
            if (gaze) {
                result.textContent = 
                    `Pitch: ${gaze.pitch.toFixed(1)}°, ` +
                    `Yaw: ${gaze.yaw.toFixed(1)}°`;
            }
            
            requestAnimationFrame(processLoop);
        }
        
        main();
    </script>
</body>
</html>
```

### 高级用法：手动控制流程

```javascript
const api = new SwinUNetGazeAPI();
await api.initialize('models/tfjs_model/model.json');

// 1. 提取 ROI
const eyeTensor = await api.extractEyeROI(video, 'left');

if (eyeTensor) {
    // 2. 预测视线
    const gaze = await api.predictGaze(eyeTensor);
    
    // 3. 转换角度
    const angles = api.vectorToAngles(gaze.vector);
    
    // 4. 释放资源
    eyeTensor.dispose();
    
    console.log('Gaze:', gaze);
    console.log('Angles:', angles);
}
```

### 批量处理

```javascript
const api = new SwinUNetGazeAPI();
await api.initialize('models/tfjs_model/model.json');

const frames = [frame1, frame2, frame3];
const results = [];

for (const frame of frames) {
    const gaze = await api.processFrame(frame, 'left');
    if (gaze) {
        results.push(gaze);
    }
}

console.log('Processed', results.length, 'frames');
```

---

## 性能优化

### 1. 降低处理频率

```javascript
let frameCount = 0;

async function processLoop() {
    frameCount++;
    
    // 每 3 帧处理一次
    if (frameCount % 3 === 0) {
        const gaze = await api.processFrame(video, 'left');
        // 更新 UI
    }
    
    requestAnimationFrame(processLoop);
}
```

### 2. 使用 WebGL 后端

```javascript
// 在初始化前设置
await tf.setBackend('webgl');
await tf.ready();

const api = new SwinUNetGazeAPI();
await api.initialize('models/tfjs_model/model.json');
```

### 3. 及时释放张量

```javascript
// 使用 tf.tidy 自动管理内存
const gaze = await tf.tidy(() => {
    return api.processFrame(video, 'left');
});
```

---

## 错误处理

```javascript
try {
    const api = new SwinUNetGazeAPI();
    await api.initialize('models/tfjs_model/model.json');
    
    const gaze = await api.processFrame(video, 'left');
    
    if (!gaze) {
        console.warn('No face detected in frame');
    }
    
} catch (error) {
    if (error.message.includes('model')) {
        console.error('Model loading failed:', error);
    } else if (error.message.includes('MediaPipe')) {
        console.error('Face detection failed:', error);
    } else {
        console.error('Unknown error:', error);
    }
}
```

---

## 浏览器兼容性

| 浏览器 | 最低版本 | 支持状态 |
|--------|---------|---------|
| Chrome | 90+ | ✅ 完全支持 |
| Firefox | 88+ | ✅ 完全支持 |
| Safari | 14+ | ⚠️ 需启用 WebGL |
| Edge | 90+ | ✅ 完全支持 |

---

## 常见问题

### Q: 为什么有时返回 null？

A: 当 MediaPipe 无法检测到人脸或眼部关键点时返回 null。确保：
- 人脸清晰可见
- 光照充足
- 眼睛未被遮挡

### Q: 如何提高检测准确率？

A: 
- 使用高质量视频（至少 480p）
- 确保人脸占画面 30% 以上
- 避免过度曝光或阴影

### Q: 内存占用过高怎么办？

A: 
- 定期调用 `clearHistory()`
- 手动释放不用的张量（`.dispose()`）
- 使用 `tf.tidy()` 包裹推理代码

---

## 技术细节

### 模型架构

- **输入**: (1, 3, 36, 60) - NCHW 格式
- **输出**: (1, 3) - 3D 单位向量
- **参数量**: 7.6M
- **模型大小**: ~15 MB (压缩后)

### 角度范围

基于 MPIIGaze 数据集统计：
- **Pitch**: -25° 到 45°（常见范围 -20° 到 5°）
- **Yaw**: -45° 到 45°（常见范围 -20° 到 20°）

### 坐标系

MPIIGaze 标准坐标系：
- **X 轴**: 向右为正
- **Y 轴**: 向上为正
- **Z 轴**: 向前为正（远离相机）
- **原点**: 眼球中心

---

## 许可证

本 API 基于 SwinUNet-VOG 项目，仅供教育和研究使用。

---

## 更新日志

### v1.0.0 (2024)
- 初始版本发布
- 支持 TensorFlow.js 推理
- 集成 MediaPipe Face Mesh
- 支持视频和图像输入

