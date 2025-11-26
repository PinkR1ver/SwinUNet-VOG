# SDK 集成说明

本文档说明了如何在 `capture.py` 中集成 CapLib SDK 以提升摄像头捕获效果。

## 背景

原始的 `capture.py` 使用 OpenCV 进行摄像头捕获，但在某些情况下效果不理想。CapLib SDK 提供了更底层的摄像头控制功能，可以显著提升捕获质量和稳定性。

## 集成方案

### 1. SDK 封装 (`sdk_wrapper.py`)

创建了 `UnifiedCameraCapture` 类，自动选择最佳的摄像头捕获实现：

- **Windows 平台**：优先使用 CapLib SDK
- **其他平台**：使用 OpenCV 作为备选方案

### 2. 自动回退机制

当 SDK 不可用时，系统会自动回退到 OpenCV，确保程序在任何环境下都能正常运行。

## 使用方法

### Windows 平台（推荐）

1. **获取 CapLib.dll**：
   - 从 SDK 供应商处获取 `CapLib.dll` 文件
   - 将 DLL 文件放置在系统 PATH 中，或程序运行目录中

2. **安装依赖**：
   ```bash
   pip install opencv-python mediapipe pillow
   ```

3. **运行程序**：
   ```bash
   python capture.py
   ```

程序会自动检测并使用 SDK，提升捕获效果。

### 其他平台

在 Linux/macOS 上，程序会自动使用 OpenCV 备选方案，无需额外配置。

## 主要改进

### SDK 优势

1. **更精确的格式控制**：
   - 支持 YUY2、MJPEG 等原生格式
   - 精确的帧率和分辨率控制

2. **更好的硬件兼容性**：
   - 直接调用摄像头驱动
   - 支持更多摄像头属性设置

3. **更稳定的捕获**：
   - 减少格式协商失败
   - 更好的 USB 带宽管理

### 集成特性

1. **透明集成**：
   - 无需修改现有代码逻辑
   - 自动选择最佳实现

2. **错误处理**：
   - SDK 不可用时自动回退
   - 详细的错误日志

3. **跨平台兼容**：
   - Windows: SDK 优先
   - 其他: OpenCV 备选

## 测试

运行测试脚本验证集成：

```bash
python test_sdk_integration.py
```

测试结果会显示：
- SDK 封装导入状态
- capture.py 导入状态
- CaptureApp 基础功能状态

## 文件结构

```
.
├── capture.py              # 主程序（已修改）
├── sdk_wrapper.py          # SDK 封装模块
├── test_sdk_integration.py # 集成测试脚本
└── sdk/                    # SDK 文件夹
    ├── include/
    │   └── cap_card.h      # SDK 头文件
    ├── MFC-demo/
    │   └── MFCApplication2/
    │       └── CapLib.lib  # 静态库（示例）
    └── manual.pdf          # SDK 手册
```

## 注意事项

1. **DLL 依赖**：
   - CapLib.dll 需要从 SDK 供应商处获取
   - 确保 DLL 与系统架构匹配（32位/64位）

2. **权限要求**：
   - Windows: 需要摄像头访问权限
   - 可能需要管理员权限访问某些摄像头属性

3. **兼容性**：
   - SDK 只在 Windows 上工作
   - OpenCV 备选方案确保跨平台兼容

## 故障排除

### SDK 无法加载
```
Warning: Failed to load CapLib SDK: Could not find module 'CapLib.dll'
```

**解决方案**：
1. 确认 CapLib.dll 文件存在
2. 将 DLL 放置在系统 PATH 中
3. 检查 DLL 架构（32位/64位）

### 摄像头无法打开
```
SDK 打开设备失败，回退到 OpenCV
```

**解决方案**：
1. 检查摄像头连接
2. 确认摄像头未被其他程序占用
3. 尝试不同的摄像头索引

### 性能问题
如果使用 SDK 后性能不佳，可以强制使用 OpenCV：

```python
# 在 sdk_wrapper.py 中修改
self.current_impl = self.opencv_capture  # 强制使用 OpenCV
```

## 技术细节

### SDK API 映射

| SDK 函数 | Python 方法 | 说明 |
|---------|------------|------|
| `InitLib()` | `initialize()` | 初始化库 |
| `EnumCameras()` | `enum_devices()` | 枚举设备 |
| `OpenCamera()` | `open_device()` | 打开设备 |
| `SetFormat()` | `set_format()` | 设置格式 |
| `RunCamera()` | `start_capture()` | 开始捕获 |

### 回调机制

SDK 使用回调函数处理帧数据：

```c
int CAP_GRABBER(double sampleTime, uint8_t* buf, long bufSize, void* ptrClass)
```

Python 封装将缓冲区转换为 NumPy 数组并放入队列。

## 贡献

如需改进集成方案，请修改 `sdk_wrapper.py` 中的相应类，并运行测试确保兼容性。
