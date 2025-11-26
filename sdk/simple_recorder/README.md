# Simple Recorder - 简化版1920x1080 120FPS录制程序

这是一个基于CapLib SDK的简化摄像头录制程序，专门针对1920x1080分辨率120FPS的高帧率录制需求。

## 功能特性

- **专用分辨率**：固定支持1920x1080分辨率
- **高帧率录制**：优先选择120FPS，如果不支持则选择该分辨率下的最高FPS
- **简化界面**：去除复杂的摄像头控制选项
- **实时预览**：支持录制前的视频预览
- **原始数据录制**：直接保存摄像头原始数据流

## 使用方法

1. **启动程序**：运行SimpleRecorder.exe
2. **开始预览**：点击"Start Preview"查看视频
3. **开始录制**：点击"Start Record"开始录制到文件
4. **停止录制**：点击"Stop Record"结束录制
5. **停止预览**：点击"Stop Preview"关闭预览

## 输出文件

录制文件保存在当前目录，命名格式：`recording_YYYYMMDD_HHMMSS.raw`

## 技术实现

- 使用CapLib SDK进行摄像头控制
- 自动检测并选择最佳的1920x1080格式
- 通过回调函数实时接收帧数据
- 多线程安全的文件写入

## 依赖项

- CapLib.dll：摄像头SDK库
- Windows 10/11
- 支持1920x1080分辨率的摄像头

## 编译说明

使用Visual Studio 2022打开SimpleRecorder.vcxproj进行编译。

确保：
1. CapLib.lib在链接库路径中
2. cap_card.h在包含路径中
3. 项目配置为Unicode字符集
