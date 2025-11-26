# Simple Recorder - VS Code控制台版本

这是一个专为Visual Studio Code设计的1920x1080 120FPS录制程序控制台版本，使用CapLib SDK。

## 🚀 快速开始

### 1. 环境要求
- Windows 10/11
- Visual Studio 2022 (或 MSVC编译器)
- CMake 3.10+
- 支持1920x1080的摄像头

### 2. VS Code配置
确保安装以下扩展：
- C/C++ (ms-vscode.cpptools)
- CMake Tools (ms-vscode.cmake-tools)

### 3. 编译步骤

#### 方法一：使用VS Code任务
1. 打开项目文件夹：`File > Open Folder > sdk/console_recorder`
2. 按 `Ctrl+Shift+P` 运行 `CMake: Configure`
3. 按 `Ctrl+Shift+P` 运行 `CMake: Build`

#### 方法二：命令行
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Debug
```

### 4. 运行程序
```bash
./build/bin/SimpleRecorder.exe
```

## 🎮 使用方法

程序启动后会显示命令提示符 `>`，支持以下命令：

```
p      - Start preview (开始预览)
s      - Stop preview (停止预览)
r      - Start recording (开始录制)
t      - Stop recording (停止录制)
d      - Show devices (显示设备)
f      - Show formats (显示格式)
h      - Show help (显示帮助)
q      - Quit (退出)
```

### 典型使用流程：
1. 启动程序
2. 输入 `d` 查看可用设备
3. 输入 `f` 查看可用格式
4. 输入 `p` 开始预览
5. 输入 `r` 开始录制
6. 输入 `t` 停止录制
7. 输入 `s` 停止预览
8. 输入 `q` 退出

## 📁 输出文件

录制文件保存在程序运行目录，命名格式：`recording_YYYYMMDD_HHMMSS.raw`

## 🔧 调试

### 在VS Code中调试：
1. 按 `F5` 或点击调试面板的运行按钮
2. 选择 "Debug SimpleRecorder" 配置
3. 程序会在断点处停止

### 添加断点：
- 点击行号左侧添加断点
- 或按 `F9` 在当前行添加断点

## 🛠️ 项目结构

```
sdk/console_recorder/
├── CMakeLists.txt              # CMake配置文件
├── SimpleRecorder.cpp          # 主程序
├── cap_card.h                  # CapLib API定义
├── .vscode/                    # VS Code配置
│   ├── tasks.json             # 构建任务
│   ├── launch.json            # 调试配置
│   └── settings.json          # 编辑器设置
└── README.md                  # 本文件
```

## ⚡ 优势

- **VS Code友好**：完全兼容VS Code开发环境
- **控制台界面**：无GUI依赖，调试友好
- **智能格式选择**：自动选择最佳的1920x1080格式
- **实时反馈**：详细的状态和错误信息
- **跨平台构建**：使用CMake，支持多种编译器

## 🔧 故障排除

### 编译错误
- 确保安装了Visual Studio C++构建工具
- 检查CMake版本 >= 3.10
- 确保CapLib.lib在库搜索路径中

### 运行错误
- 确保CapLib.dll在PATH中或与exe同目录
- 检查摄像头是否被其他程序占用
- 确认摄像头支持1920x1080分辨率

### 录制问题
- 确保有足够的磁盘空间
- 检查摄像头权限设置
- 尝试不同的USB端口

## 📝 技术说明

- 使用C++17标准
- 多线程安全的文件写入
- 信号处理支持优雅退出
- 实时帧率统计和显示
