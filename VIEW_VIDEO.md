# 📹 如何查看录制的视频文件

你的 `capture.avi` 文件使用了 YUY2 等无损编码，不是所有播放器都支持。以下有几种方法查看。

---

## 方案 1: 用专用工具查看 (推荐) ⭐

我为你创建了一个专用的视频查看工具：

### 快速查看
```bash
python3 view_video.py capture.avi
```

**功能**：
- ✅ 播放 AVI/MP4/MOV 视频
- ✅ 逐帧查看（按 → ←）
- ✅ 暂停/播放（按 Space）
- ✅ 查看文件信息

**按键说明**：
| 按键 | 功能 |
|------|------|
| Space | 暂停/播放 |
| → | 下一帧 |
| ← | 上一帧 |
| q | 退出 |

### 查看文件信息
```bash
python3 view_video.py capture.avi --info
```

输出示例：
```
============================================================
📹 视频信息
============================================================
文件：capture.avi
分辨率：1280×720
帧率：30.00 FPS
总帧数：900
时长：30.00 秒
============================================================
```

### 转换为 MP4（可在任何系统上播放）
```bash
python3 view_video.py capture.avi --export output.mp4
```

这会自动转换为 H.264 MP4 格式，macOS 和 Windows 都能打开。

---

## 方案 2: 用 VLC 播放器

VLC 支持几乎所有视频格式，包括无损编码。

### 安装 VLC
```bash
# 用 Homebrew 安装
brew install vlc

# 或从 App Store 下载
# https://apps.apple.com/us/app/vlc-media-player/id681907227
```

### 播放视频
```bash
# 命令行
open -a VLC capture.avi

# 或拖入 VLC 窗口
# 或右键 → 打开方式 → VLC
```

**优点**：
- ✅ 支持所有编码格式
- ✅ 免费开源
- ✅ 播放流畅

---

## 方案 3: 用 FFmpeg 转换

FFmpeg 是专业的视频处理工具。

### 安装 FFmpeg
```bash
brew install ffmpeg
```

### 查看文件信息
```bash
ffprobe capture.avi
```

### 转换为通用 MP4
```bash
# 快速转换（只重新容器化，不重新编码）
ffmpeg -i capture.avi -c:v copy -c:a copy output.mp4

# 转换为 H.264（兼容性最好）
ffmpeg -i capture.avi -c:v libx264 -preset medium -crf 18 output.mp4

# 转换为 H.265（文件更小）
ffmpeg -i capture.avi -c:v libx265 -preset medium -crf 18 output.mp4
```

### 参数说明
| 参数 | 说明 |
|------|------|
| `-i input.avi` | 输入文件 |
| `-c:v copy` | 视频编码方式（copy = 不重新编码，最快） |
| `-c:v libx264` | H.264 编码（兼容性好） |
| `-c:v libx265` | H.265 编码（文件更小） |
| `-preset fast/medium/slow` | 编码速度（fast最快，slow最好质量） |
| `-crf 18` | 质量（0=最好，23=默认，51=最差） |

---

## 方案 4: 用 Python + OpenCV 读取

如果你需要逐帧处理数据，可以用 Python：

```python
import cv2

# 打开视频
cap = cv2.VideoCapture('capture.avi')

# 获取信息
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"分辨率: {width}×{height}")
print(f"帧率: {fps} FPS")
print(f"总帧数: {frame_count}")

# 逐帧读取
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 处理 frame
    # ...
    
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 方案 5: 用 QuickTime (仅 macOS)

macOS 自带的 QuickTime Player 可能不支持 YUY2，但可以尝试：

```bash
open -a "QuickTime Player" capture.avi
```

如果不行，就用其他方案。

---

## 🎯 推荐方案

| 用途 | 推荐方案 |
|------|---------|
| 快速查看视频 | ⭐ `python3 view_video.py` |
| 在任何系统上播放 | 转换为 MP4：`python3 view_video.py --export output.mp4` |
| 专业播放 | VLC 播放器 |
| 数据处理/分析 | Python + OpenCV |
| 质量最好的转换 | FFmpeg |

---

## 🚨 常见问题

### Q: 为什么不能直接打开？
**A:** YUY2 等无损格式是专业编码，macOS 默认播放器不支持。需要转换或用专门工具。

### Q: 转换会损失质量吗？
**A:** 取决于目标编码：
- `copy` 方式：完全无损，只改容器
- `libx264` H.264：质量损失微小（CRF 18 时几乎无损）
- `libx265` H.265：同等质量，文件更小

### Q: 哪个方案最简单？
**A:** 
1. 快速查看：`python3 view_video.py capture.avi`
2. 永久使用：转换为 MP4：`python3 view_video.py --export output.mp4`
3. 然后用任何播放器打开 output.mp4

### Q: 文件太大了怎么办？
**A:** 用更强的压缩：
```bash
# 方案 1: 用 H.265 编码（文件小 40%）
ffmpeg -i capture.avi -c:v libx265 -preset medium -crf 18 output.mp4

# 方案 2: 降低分辨率
ffmpeg -i capture.avi -vf scale=640:360 output.mp4

# 方案 3: 降低帧率
ffmpeg -i capture.avi -r 15 output.mp4

# 方案 4: 组合使用（最激进）
ffmpeg -i capture.avi -vf scale=640:360 -r 15 -c:v libx265 -crf 23 output.mp4
```

### Q: 如何确保不丢失眼睛检测信息？
**A:** 如果视频中有绿色眼眶轮廓：
- ✅ 任何转换都会保留（是画在视频里的）
- ✅ 使用 CRF 18 以上确保最高质量
- ✅ 不要过度压缩

### Q: 能批量转换多个文件吗？
**A:** 
```bash
# 批量转换所有 AVI 为 MP4
for file in *.avi; do
    python3 view_video.py "$file" --export "${file%.avi}.mp4"
done
```

---

## 📊 格式对比

| 格式 | 文件大小 | 兼容性 | 质量 |
|------|---------|--------|------|
| YUY2 (原始) | 最大 | 差 | 最好 |
| MP4 (H.264) | 中等 | 最好 | 很好 |
| MP4 (H.265) | 最小 | 好 | 很好 |
| MOV | 较大 | 中等 | 最好 |

**建议**：YUY2 用于科研分析，MP4 H.264 用于分享和播放。

---

立即开始查看你的视频吧！🎬



