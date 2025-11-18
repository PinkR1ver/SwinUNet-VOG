#!/bin/bash

# SwinUNet-VOG 数据采集工具启动脚本

echo "🎬 SwinUNet-VOG 眼睛数据采集工具"
echo "=================================="
echo ""

# 检查 Python 版本
python3 --version || { echo "❌ 未找到 Python 3"; exit 1; }

# 检查依赖
echo "📦 检查依赖..."
python3 -c "import cv2; import mediapipe; import tkinter" 2>/dev/null || {
    echo "⚠️  缺少依赖，开始安装..."
    pip install -r requirements.txt
}

echo ""
echo "✅ 准备就绪！"
echo ""
echo "使用说明："
echo "1️⃣  选择摄像头"
echo "2️⃣  配置分辨率和帧率"
echo "3️⃣  点击 '👁️ 开始预览' 查看预览"
echo "4️⃣  点击 '🔴 开始录制' 开始采集"
echo "5️⃣  按 'q' 或 'Esc' 停止录制"
echo ""

# 运行应用
python3 capture.py



