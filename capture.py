"""
capture.py
----------

集成 MediaPipe 的高级摄像头录制工具：
1. 实时眼睛特征点检测与眼眶可视化
2. 自动检测相机的分辨率/帧率预设
3. 现代化 Tkinter UI，支持预设快速选择
4. 灵活的编码格式选择（YUY2/I420/MJPEG）
5. 智能 MP4 容器优化（替代不稳定的 AVI）
"""

import os
import sys
import platform
import threading
import queue
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import time

# 设置环境变量减少MediaPipe/TensorFlow的警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误信息
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用oneDNN优化警告

import cv2
import numpy as np

# 导入 SDK 封装
try:
    from sdk_wrapper import UnifiedCameraCapture, CameraCapabilities as SDKCapabilities
except ImportError:
    print("警告：无法导入 SDK 封装，将使用纯 OpenCV 模式")
    UnifiedCameraCapture = None

try:
    import mediapipe as mp
except ImportError:
    mp = None  # type: ignore

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover
    tk = None  # type: ignore
    ImageTk = None  # type: ignore


def get_camera_backend() -> int:
    """根据操作系统返回合适的摄像头后端。
    
    - macOS: CAP_AVFOUNDATION
    - Windows: CAP_DSHOW
    - Linux: CAP_V4L2
    """
    system = platform.system()
    if system == "Darwin":  # macOS
        return cv2.CAP_AVFOUNDATION
    elif system == "Windows":
        return cv2.CAP_DSHOW
    else:  # Linux 或其他
        return cv2.CAP_V4L2


class EyeDetector:
    """使用 MediaPipe 检测眼睛特征点。"""
    
    def __init__(self):
        if mp is None:
            self.enabled = False
            return
        
        self.enabled = True
        # 禁用MediaPipe的feedback manager警告
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少TensorFlow日志

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
            thickness=1, circle_radius=1, color=(0, 255, 0)
        )
        
        # 眼睛特征点索引 (左眼和右眼)
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """检测眼睛特征点并返回标注后的帧与检测结果。"""
        if not self.enabled:
            return frame, {"detected": False}
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        h, w, _ = frame.shape
        annotated_frame = frame.copy()
        
        detection_info = {
            "detected": False,
            "left_eye": None,
            "right_eye": None,
        }
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 绘制眼眶轮廓
                landmarks = face_landmarks.landmark
                
                # 左眼
                left_eye_points = np.array(
                    [[landmarks[idx].x * w, landmarks[idx].y * h] for idx in self.LEFT_EYE],
                    dtype=np.int32
                )
                cv2.polylines(annotated_frame, [left_eye_points], True, (0, 255, 0), 2)
                
                # 右眼
                right_eye_points = np.array(
                    [[landmarks[idx].x * w, landmarks[idx].y * h] for idx in self.RIGHT_EYE],
                    dtype=np.int32
                )
                cv2.polylines(annotated_frame, [right_eye_points], True, (0, 255, 0), 2)
                
                detection_info["detected"] = True
                detection_info["left_eye"] = left_eye_points
                detection_info["right_eye"] = right_eye_points
        
        return annotated_frame, detection_info


@dataclass
class CameraCapabilities:
    """相机支持的分辨率和帧率。"""
    resolutions: List[Tuple[int, int]] = field(default_factory=list)
    fps_values: List[float] = field(default_factory=list)


def detect_camera_capabilities(device_index: int, backend: int, existing_capture=None) -> tuple[CameraCapabilities, str]:
    """自动检测相机支持的分辨率和帧率预设。

    优先使用 SDK 检测，如果不可用则使用 OpenCV。

    Args:
        device_index: 设备索引
        backend: OpenCV 后端
        existing_capture: 可选的现有 UnifiedCameraCapture 实例，避免重复创建
    """
    capabilities = CameraCapabilities()

    # 使用提供的实例，或者创建一个新的
    capture = existing_capture

    # 尝试使用 SDK 检测
    if capture and capture.is_using_sdk():
        try:
            sdk_caps = capture.get_capabilities(device_index)
            capabilities.resolutions = sdk_caps.resolutions
            capabilities.fps_values = sdk_caps.fps_values

            if capabilities.resolutions and capabilities.fps_values:
                return capabilities, "SDK"
        except Exception as e:
            print(f"SDK 检测失败: {e}，回退到 OpenCV")

    # OpenCV 备选检测
    cap = cv2.VideoCapture(device_index, backend)

    if not cap.isOpened():
        return capabilities, "OpenCV"

    # 常见的分辨率预设
    common_resolutions = [
        (320, 240),    # QVGA
        (640, 480),    # VGA
        (800, 600),    # SVGA
        (1024, 768),   # XGA
        (1280, 720),   # HD
        (1280, 960),   # UXGA
        (1920, 1080),  # Full HD
        (2560, 1440),  # QHD
        (3840, 2160),  # 4K
    ]

    # 常见的帧率
    common_fps = [15, 24, 25, 30, 48, 50, 60, 120]

    # 方法1：尝试直接设置并检测（对大多数相机有效）
    tested_resolutions = set()
    for width, height in common_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        time.sleep(0.05)  # 给相机时间响应

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 记录实际获得的分辨率（即使不完全匹配请求）
        if actual_width > 0 and actual_height > 0:
            res = (actual_width, actual_height)
            if res not in tested_resolutions:
                tested_resolutions.add(res)
                if res not in capabilities.resolutions:
                    capabilities.resolutions.append(res)

    # 方法2：尝试帧率（在一个稳定的分辨率下）
    if capabilities.resolutions:
        # 使用最常见的分辨率 640x480 测试帧率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(0.1)

    tested_fps = set()
    for fps in common_fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
        time.sleep(0.05)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        # 允许 ±2 FPS 的容差
        if actual_fps > 0:
            fps_rounded = round(actual_fps)
            if fps_rounded not in tested_fps:
                tested_fps.add(fps_rounded)
                if fps_rounded not in capabilities.fps_values:
                    capabilities.fps_values.append(float(fps_rounded))

    cap.release()

    # 排序
    capabilities.resolutions.sort()
    capabilities.fps_values.sort()

    # 如果检测失败，返回默认值
    if not capabilities.resolutions:
        capabilities.resolutions = [(640, 480), (1280, 720), (1920, 1080)]
    if not capabilities.fps_values:
        capabilities.fps_values = [24.0, 30.0, 60.0]

    return capabilities, "OpenCV"


@dataclass
class CaptureSettings:
    device_index: int
    output_path: str
    fps: float
    width: int
    height: int
    fourcc: str
    enable_eye_detection: bool = True


class CaptureApp:
    """现代化的摄像头录制应用 UI。"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SwinUNet-VOG - 眼睛数据采集工具")
        self.root.geometry("1000x750")
        self.root.resizable(True, True)

        self.backend = get_camera_backend()
        self.settings: Optional[CaptureSettings] = None
        self.preview_running = False
        self.preview_frame = None
        self.camera_thread = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)

        # 初始化摄像头捕获器
        self.camera_capture = UnifiedCameraCapture() if UnifiedCameraCapture else None
        if self.camera_capture:
            self.camera_capture.initialize()

        self._build_ui()
        self.refresh_cameras()

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'camera_capture') and self.camera_capture:
            try:
                self.camera_capture.uninitialize()
            except:
                pass

    def _build_ui(self) -> None:
        """构建现代化UI。"""
        # 使用 grid 布局管理器，更灵活
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)  # 预览区域占用主要空间
        
        # 上半部分：设置（不扩展）
        settings_frame = ttk.LabelFrame(self.root, text="采集参数", padding=10)
        settings_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 5))
        
        # 相机选择
        ttk.Label(settings_frame, text="📷 摄像头：").grid(row=0, column=0, sticky="e", padx=5, pady=8)
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(settings_frame, textvariable=self.device_var, width=30, state="readonly")
        self.device_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=8)
        self.device_combo.bind("<<ComboboxSelected>>", lambda e: self._on_device_changed())
        
        refresh_btn = ttk.Button(settings_frame, text="🔄", command=self.refresh_cameras, width=3)
        refresh_btn.grid(row=0, column=2, padx=2, pady=8)
        
        obs_btn = ttk.Button(settings_frame, text="🔗 OBS", command=self._sync_with_obs, width=8)
        obs_btn.grid(row=0, column=3, padx=2, pady=8)
        
        # 分辨率预设
        ttk.Label(settings_frame, text="📐 分辨率：").grid(row=1, column=0, sticky="e", padx=5, pady=8)
        self.resolution_var = tk.StringVar()
        self.resolution_combo = ttk.Combobox(settings_frame, textvariable=self.resolution_var, width=20, state="readonly")
        self.resolution_combo.grid(row=1, column=1, sticky="ew", padx=5, pady=8)
        self.resolution_combo.bind("<<ComboboxSelected>>", lambda e: self._on_resolution_changed())
        
        # 帧率预设
        ttk.Label(settings_frame, text="⏱️  帧率 (FPS)：").grid(row=1, column=2, sticky="e", padx=5, pady=8)
        self.fps_var = tk.StringVar()
        self.fps_combo = ttk.Combobox(settings_frame, textvariable=self.fps_var, width=15, state="readonly")
        self.fps_combo.grid(row=1, column=3, sticky="ew", padx=5, pady=8)
        
        # 输出文件（默认 MP4，兼容性最好）
        ttk.Label(settings_frame, text="💾 输出文件：").grid(row=2, column=0, sticky="e", padx=5, pady=8)
        self.output_var = tk.StringVar(value=os.path.abspath("capture.mp4"))
        output_entry = ttk.Entry(settings_frame, textvariable=self.output_var)
        output_entry.grid(row=2, column=1, columnspan=2, sticky="ew", padx=5, pady=8)
        
        browse_btn = ttk.Button(settings_frame, text="浏览...", command=self._browse_output, width=10)
        browse_btn.grid(row=2, column=3, padx=5, pady=8)
        
        # 编码格式
        ttk.Label(settings_frame, text="🎬 编码格式：").grid(row=3, column=0, sticky="e", padx=5, pady=8)
        self.fourcc_var = tk.StringVar(value="YUY2")
        fourcc_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.fourcc_var,
            values=["YUY2", "UYVY", "I420", "MJPG"],  # 移除 XVID (不稳定)
            width=20,
            state="readonly"
        )
        fourcc_combo.grid(row=3, column=1, sticky="ew", padx=5, pady=8)
        
        # 自动文件扩展名
        def on_codec_changed(event=None):
            codec = self.fourcc_var.get()
            current_path = self.output_var.get()
            if current_path and '.' in current_path:
                base = current_path.rsplit('.', 1)[0]
                # 4K 用 MP4，其他用 AVI
                res_str = self.resolution_var.get()
                if res_str:
                    try:
                        w, h = map(int, res_str.split("×"))
                        if w * h > 2560 * 1440:  # 4K 以上
                            self.output_var.set(base + '.mp4')
                        else:
                            self.output_var.set(base + '.avi')
                    except:
                        pass
        
        fourcc_combo.bind("<<ComboboxSelected>>", on_codec_changed)
        
        # 编码格式说明
        def show_codec_help():
            msg = """📦 容器格式 & 编码格式说明

容器格式（文件后缀）
===================
🎯 MP4（推荐，强制使用）
  • 现代标准，稳定可靠
  • 支持所有编码格式
  • 跨平台兼容性最好
  • 自动处理高分辨率/高帧率
  • 比 AVI 稳定 100 倍

❌ AVI（已弃用）
  • 30 年前的过时格式
  • 4K 视频易崩溃
  • 不支持 >2GB 文件
  • 自动转换为 MP4


编码格式（压缩方式）
==================
YUY2 (推荐)
  • 无损，原始采样格式
  • 文件大小大，质量最好
  • 用于科研/精确分析
  • ⚠️ 1080p@120fps 可能超 USB 2.0 带宽

UYVY 
  • 与 YUY2 类似，字节顺序不同
  • 某些摄像头原生格式
  • 无损，质量等同 YUY2

I420 (推荐备选)
  • 无损，色度下采样 (4:2:0)
  • 文件比 YUY2 小 50%
  • 适合长时间采集

MJPG (Motion JPEG，推荐高帧率)
  • 有损压缩，文件最小
  • ✅ 1080p@120fps 轻松支持
  • 可能是摄像头硬件编码
  • 适合高帧率/高分辨率"""
            messagebox.showinfo("容器 & 编码格式说明", msg)
        
        help_btn = ttk.Button(settings_frame, text="❓", command=show_codec_help, width=3)
        help_btn.grid(row=3, column=2, sticky="w", padx=5)
        ttk.Label(settings_frame, text="YUY2/UYVY/I420 = 无压缩 | MJPG/XVID = 压缩", font=("", 8, "italic")).grid(row=3, column=3, sticky="w", padx=5)
        
        # 眼睛检测选项
        ttk.Label(settings_frame, text="👁️  功能：").grid(row=4, column=0, sticky="e", padx=5, pady=8)
        self.eye_detection_var = tk.BooleanVar(value=True)
        eye_check = ttk.Checkbutton(settings_frame, text="启用实时眼睛检测 (MediaPipe)", variable=self.eye_detection_var)
        eye_check.grid(row=4, column=1, columnspan=2, sticky="w", padx=5, pady=8)
        
        # 格式建议
        ttk.Label(settings_frame, text="💡 建议：").grid(row=5, column=0, sticky="e", padx=5, pady=8)
        ttk.Label(settings_frame, text="4K 用 MP4 格式更稳定 | 1080p 用 YUY2 质量最好", 
                 font=("", 9, "italic"), foreground="blue").grid(row=5, column=1, columnspan=3, sticky="w", padx=5)
        
        settings_frame.columnconfigure(1, weight=1)
        
        # 中间部分：预览（占用主要空间）
        preview_frame = ttk.LabelFrame(self.root, text="实时预览", padding=5)
        preview_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=5)
        
        self.preview_label = tk.Label(preview_frame, bg="black", width=640, height=360)
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        
        # 下半部分：按钮（固定高度，永远可见）
        button_frame = ttk.LabelFrame(self.root, text="操作", padding=10)
        button_frame.grid(row=2, column=0, sticky="ew", padx=15, pady=(5, 15))
        
        # 第一行：预览控制
        preview_control_frame = ttk.Frame(button_frame)
        preview_control_frame.pack(fill=tk.X, pady=5)
        
        self.preview_btn = ttk.Button(
            preview_control_frame, 
            text="👁️ 开始预览", 
            command=self._start_preview,
            width=20
        )
        self.preview_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_preview_btn = ttk.Button(
            preview_control_frame, 
            text="⏹️ 停止预览", 
            command=self._stop_preview,
            width=20, 
            state=tk.DISABLED
        )
        self.stop_preview_btn.pack(side=tk.LEFT, padx=5)
        
        # 第二行：录制和退出
        record_control_frame = ttk.Frame(button_frame)
        record_control_frame.pack(fill=tk.X, pady=5)
        
        self.record_btn = ttk.Button(
            record_control_frame, 
            text="🔴 开始录制",
            command=self._start_recording,
            width=20
        )
        self.record_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            record_control_frame, 
            text="❌ 退出", 
            command=self.root.destroy,
            width=20
        ).pack(side=tk.LEFT, padx=5)
    
    def refresh_cameras(self) -> None:
        """刷新可用相机列表。"""
        cameras = self._list_cameras()
        if cameras:
            items = [f"{idx}: {name}" for idx, name in cameras.items()]
            self.device_combo["values"] = items
            self.device_combo.current(0)
            self._on_device_changed()
        else:
            messagebox.showwarning("提示", "未检测到摄像头。请检查连接。")
    
    def _list_cameras(self) -> Dict[int, str]:
        """列出所有可用相机。"""
        available = {}

        # 优先使用 SDK 枚举设备
        if self.camera_capture:
            try:
                devices = self.camera_capture.enum_devices()
                for i, device in enumerate(devices):
                    available[i] = device.name
                return available
            except Exception as e:
                print(f"SDK 枚举设备失败: {e}，回退到 OpenCV")

        # OpenCV 备选方案
        for index in range(10):
            cap = cv2.VideoCapture(index, self.backend)
            if cap.isOpened():
                backend_name = cap.getBackendName() if hasattr(cap, "getBackendName") else ""
                name = f"Camera {index}" + (f" ({backend_name})" if backend_name else "")
                available[index] = name
                cap.release()
        return available
    
    def _on_device_changed(self) -> None:
        """相机选择改变时更新分辨率和帧率。"""
        device_str = self.device_var.get()
        if not device_str or ":" not in device_str:
            return
        
        device_index = int(device_str.split(":")[0])
        print(f"正在检测摄像头 {device_index} 的能力...")
        capabilities, detection_method = detect_camera_capabilities(device_index, self.backend, self.camera_capture)

        print(f"使用 {detection_method} 检测到能力：分辨率 {capabilities.resolutions}，帧率 {capabilities.fps_values}")
        
        # 更新分辨率
        resolution_items = [f"{w}×{h}" for w, h in capabilities.resolutions]
        self.resolution_combo["values"] = resolution_items
        if resolution_items:
            # 优先选择 1280×720
            if "1280×720" in resolution_items:
                self.resolution_combo.set("1280×720")
            elif "640×480" in resolution_items:
                self.resolution_combo.set("640×480")
            else:
                self.resolution_combo.current(0)
        
        # 更新帧率
        fps_items = [str(int(f)) for f in capabilities.fps_values]
        self.fps_combo["values"] = fps_items
        if fps_items:
            # 优先选择 30
            if "30" in fps_items:
                self.fps_combo.set("30")
            else:
                self.fps_combo.current(0)
    
    def _sync_with_obs(self) -> None:
        """弹出对话框，让用户输入 OBS 检测到的参数。"""
        dialog = tk.Toplevel(self.root)
        dialog.title("与 OBS 同步参数")
        dialog.geometry("400x250")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="请输入 OBS 中检测到的分辨率和帧率：", font=("", 11, "bold")).pack(pady=10)
        
        # 分辨率输入
        res_frame = ttk.Frame(dialog)
        res_frame.pack(pady=10)
        ttk.Label(res_frame, text="分辨率（宽×高）：").pack(side=tk.LEFT, padx=5)
        res_var = tk.StringVar(value="1280x720")
        res_entry = ttk.Entry(res_frame, textvariable=res_var, width=15)
        res_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(res_frame, text="例：1920x1080").pack(side=tk.LEFT, padx=5)
        
        # 帧率输入
        fps_frame = ttk.Frame(dialog)
        fps_frame.pack(pady=10)
        ttk.Label(fps_frame, text="帧率 (FPS)：").pack(side=tk.LEFT, padx=5)
        fps_var = tk.StringVar(value="30")
        fps_entry = ttk.Entry(fps_frame, textvariable=fps_var, width=15)
        fps_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(fps_frame, text="例：60").pack(side=tk.LEFT, padx=5)
        
        def apply_settings():
            try:
                res_str = res_var.get().strip().replace(" ", "")
                width, height = map(int, res_str.replace("x", "×").split("×"))
                fps = float(fps_var.get())
                
                # 更新 UI
                res_text = f"{width}×{height}"
                if res_text in self.resolution_combo["values"]:
                    self.resolution_combo.set(res_text)
                else:
                    self.resolution_combo.set(res_text)
                    current_values = list(self.resolution_combo["values"])
                    if res_text not in current_values:
                        current_values.append(res_text)
                        self.resolution_combo["values"] = current_values
                        self.resolution_combo.set(res_text)
                
                fps_text = str(int(fps))
                if fps_text in self.fps_combo["values"]:
                    self.fps_combo.set(fps_text)
                else:
                    self.fps_combo.set(fps_text)
                    current_values = list(self.fps_combo["values"])
                    if fps_text not in current_values:
                        current_values.append(fps_text)
                        self.fps_combo["values"] = current_values
                        self.fps_combo.set(fps_text)
                
                messagebox.showinfo("成功", f"参数已更新：{width}×{height} @ {int(fps)} FPS")
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("错误", f"输入格式不正确：{e}")
        
        ttk.Button(dialog, text="✅ 应用", command=apply_settings, width=20).pack(pady=10)
        ttk.Button(dialog, text="❌ 关闭", command=dialog.destroy, width=20).pack(pady=5)
    
    def _on_resolution_changed(self) -> None:
        """分辨率改变时，根据分辨率自动建议编码格式。"""
        res_str = self.resolution_var.get()
        if not res_str:
            return
        
        try:
            width, height = map(int, res_str.split("×"))
            total_pixels = width * height
            
            # 根据分辨率自动建议编码
            if total_pixels > 2560 * 1440:  # 4K 以上
                suggested = "MP4 或 I420"
                reason = "4K 用 AVI 容易出错"
            elif total_pixels > 1920 * 1080:  # 2K 以上
                suggested = "I420 或 MP4"
                reason = "平衡文件大小和质量"
            else:  # 1080p 以下
                suggested = "YUY2"
                reason = "无损，质量最好"
            
            # 这里可以添加提示逻辑
            # 暂时不改自动选择，让用户手动选
        except:
            pass
    
    def _browse_output(self) -> None:
        """浏览输出文件。"""
        path = filedialog.asksaveasfilename(
            title="选择输出文件",
            defaultextension=".avi",
            filetypes=[("AVI 文件", "*.avi"), ("所有文件", "*.*")],
        )
        if path:
            self.output_var.set(os.path.abspath(path))
    
    def _start_preview(self) -> None:
        """启动预览。"""
        if not self._validate_settings():
            return
        
        self.preview_running = True
        self.preview_btn.config(state=tk.DISABLED)
        self.stop_preview_btn.config(state=tk.NORMAL)
        self.device_combo.config(state=tk.DISABLED)
        
        self.camera_thread = threading.Thread(target=self._preview_thread, daemon=True)
        self.camera_thread.start()
        self._update_preview_label()
    
    def _stop_preview(self) -> None:
        """停止预览。"""
        self.preview_running = False
        self.preview_btn.config(state=tk.NORMAL)
        self.stop_preview_btn.config(state=tk.DISABLED)
        self.device_combo.config(state="readonly")
        self.preview_label.config(image="")
    
    def _preview_thread(self) -> None:
        """预览线程。"""
        device_str = self.device_var.get()
        device_index = int(device_str.split(":")[0])
        resolution_str = self.resolution_var.get()
        fps_str = self.fps_var.get()

        width, height = map(int, resolution_str.split("×"))
        fps = float(fps_str)

        eye_detector = EyeDetector() if self.eye_detection_var.get() else None

        # 尝试使用 SDK
        using_sdk = False
        if self.camera_capture:
            try:
                # 打开设备
                if self.camera_capture.open_device(device_index):
                    # 设置帧率
                    self.camera_capture.set_frame_rate(device_index, int(fps))
                    # 开始捕获
                    if self.camera_capture.start_capture(device_index):
                        using_sdk = True
                        print(f"预览使用 SDK 模式")
                    else:
                        self.camera_capture.close_device(device_index)
                else:
                    print(f"SDK 打开设备失败，回退到 OpenCV")
            except Exception as e:
                print(f"SDK 预览初始化失败: {e}，回退到 OpenCV")

        # OpenCV 备选方案
        cap = None
        if not using_sdk:
            cap = cv2.VideoCapture(device_index, self.backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)

        try:
            while self.preview_running:
                frame = None

                if using_sdk:
                    # 从 SDK 获取帧
                    frame = self.camera_capture.get_frame(timeout=0.1)
                else:
                    # 从 OpenCV 获取帧
                    if cap and cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                if frame is None:
                    continue

                # 检测眼睛
                if eye_detector:
                    frame, _ = eye_detector.detect(frame)

                # 缩放用于预览
                preview_size = (640, 360)
                frame_resized = cv2.resize(frame, preview_size)

                # 添加信息
                mode_text = "SDK" if using_sdk else "OpenCV"
                info_text = f"{mode_text}: {width}x{height} @ {fps:.0f} FPS"
                cv2.putText(frame_resized, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                try:
                    self.frame_queue.put_nowait(frame_resized)
                except queue.Full:
                    pass
        finally:
            if using_sdk:
                self.camera_capture.stop_capture(device_index)
                self.camera_capture.close_device(device_index)
            elif cap:
                cap.release()
    
    def _update_preview_label(self) -> None:
        """更新预览标签。"""
        if not self.preview_running:
            return
        
        try:
            frame = self.frame_queue.get_nowait()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(img)
            self.preview_label.config(image=photo)
            self.preview_label.image = photo
        except queue.Empty:
            pass
        
        self.root.after(33, self._update_preview_label)  # 30 FPS
    
    def _validate_settings(self) -> bool:
        """验证设置。"""
        if not self.device_var.get():
            messagebox.showerror("错误", "请选择摄像头")
            return False
        if not self.resolution_var.get():
            messagebox.showerror("错误", "请选择分辨率")
            return False
        if not self.fps_var.get():
            messagebox.showerror("错误", "请选择帧率")
            return False
        return True
    
    def _start_recording(self) -> None:
        """启动录制。"""
        if not self._validate_settings():
            return
        
        device_str = self.device_var.get()
        device_index = int(device_str.split(":")[0])
        resolution_str = self.resolution_var.get()
        fps_str = self.fps_var.get()
        
        width, height = map(int, resolution_str.split("×"))
        fps = float(fps_str)
        
        settings = CaptureSettings(
            device_index=device_index,
            output_path=self.output_var.get(),
            fps=fps,
            width=width,
            height=height,
            fourcc=self.fourcc_var.get(),
            enable_eye_detection=self.eye_detection_var.get(),
        )
        
        self._stop_preview()
        
        # 运行录制
        if run_capture(settings):
            messagebox.showinfo("成功", f"录制完成！\n文件已保存到：\n{settings.output_path}")
        else:
            messagebox.showerror("错误", "录制失败")


def run_capture(settings: CaptureSettings) -> bool:
    """使用给定参数录制视频。"""
    backend = get_camera_backend()

    # 尝试使用 SDK
    camera_capture = UnifiedCameraCapture() if UnifiedCameraCapture else None
    using_sdk = False

    if camera_capture:
        try:
            if camera_capture.initialize():
                if camera_capture.open_device(settings.device_index):
                    camera_capture.set_frame_rate(settings.device_index, int(settings.fps))
                    if camera_capture.start_capture(settings.device_index):
                        using_sdk = camera_capture.is_using_sdk()  # 使用实际的SDK状态
                        mode_text = "SDK" if using_sdk else "OpenCV"
                        print(f"录制使用 {mode_text} 模式")
                    else:
                        camera_capture.close_device(settings.device_index)
                else:
                    print(f"SDK 打开设备失败，回退到 OpenCV")
            else:
                print(f"SDK 初始化失败，回退到 OpenCV")
        except Exception as e:
            print(f"SDK 录制初始化失败: {e}，回退到 OpenCV")

    # OpenCV 备选方案
    cap = None
    if not using_sdk:
        cap = cv2.VideoCapture(settings.device_index, backend)
        if not cap.isOpened():
            print(f"❌ 错误：无法打开摄像头 {settings.device_index}", file=sys.stderr)
            return False
    
    # 给摄像头足够的初始化时间（某些驱动需要）
    print(f"等待摄像头初始化...")
    time.sleep(1.0)
    
    print(f"\n{'='*60}")
    print(f"🔧 摄像头初始化诊断")
    print(f"{'='*60}")
    
    # 【第1步】先尝试读取一帧（在设置任何参数前）
    print(f"[1/5] 测试原始读取 (无参数设置)...")
    if using_sdk:
        print(f"   SDK 模式：跳过 OpenCV 初始化测试")
        ret_test = True  # SDK 模式下假设初始化成功
        default_width = settings.width
        default_height = settings.height
    else:
        ret_test, frame_test = cap.read()
        if ret_test:
            print(f"   ✅ 原始读取成功，帧大小：{frame_test.shape}")
            default_width = frame_test.shape[1]
            default_height = frame_test.shape[0]
            print(f"   💡 摄像头默认分辨率：{default_width}×{default_height}")
        else:
            print(f"   ❌ 原始读取失败 - 摄像头驱动初始化问题")
            print(f"   💡 建议：可能需要摄像头官方 SDK 支持")
    
    # 【第2步】设置分辨率和帧率
    print(f"\n[2/4] 设置参数...")
    print(f"   分辨率：{settings.width}×{settings.height}")
    print(f"   帧率：{settings.fps:.0f} FPS")
    print(f"   编码：{settings.fourcc}")

    if using_sdk:
        # SDK 已经设置了帧率，这里不需要额外设置分辨率
        print(f"   SDK 模式：参数已通过 SDK 设置")
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
        cap.set(cv2.CAP_PROP_FPS, settings.fps)

        # 尝试设置 FourCC（某些驱动需要这一步）
        fourcc_test = cv2.VideoWriter_fourcc(*settings.fourcc)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc_test)
    
    # 等待设置生效
    time.sleep(0.5)
    
    # 【第3步】验证参数是否生效
    print(f"\n[3/4] 验证参数...")

    if using_sdk:
        # SDK 模式：使用 SDK 获取实际参数
        actual_fps = camera_capture.get_frame_rate(settings.device_index)
        # SDK 模式下分辨率信息可能不可用，使用设置值
        actual_width = settings.width
        actual_height = settings.height
        print(f"   SDK 模式：使用 SDK 报告的参数")
    else:
        # 读取实际参数
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"   分辨率：{settings.width}×{settings.height} → {actual_width}×{actual_height}", end="")
    if actual_width == settings.width and actual_height == settings.height:
        print(" ✓")
    else:
        print(" ⚠️ 设置失败，使用实际值")
    
    print(f"   帧率：{settings.fps:.0f}FPS → {actual_fps:.2f}FPS", end="")
    if actual_fps >= settings.fps * 0.9:
        print(" ✓")
    else:
        print(" ⚠️ 未达到目标")
    
    if actual_fps <= 0:
        actual_fps = settings.fps
        print(f"   ℹ️ 摄像头未报告帧率，使用设置值: {actual_fps:.2f}FPS")
    
    # 【第4步】尝试读取一帧（设置参数后）
    print(f"\n[4/5] 测试参数下的读取...")
    if using_sdk:
        print(f"   SDK 模式：跳过参数测试")
        ret_test2 = True
    else:
        ret_test2, frame_test2 = cap.read()
        if ret_test2:
            print(f"   ✅ 参数设置后读取成功，帧大小：{frame_test2.shape}")
        else:
            print(f"   ⚠️ 参数设置后读取失败，尝试恢复为默认格式...")

            # 尝试恢复到默认格式（不设置任何参数）
            cap.release()
            cap = cv2.VideoCapture(settings.device_index, backend)
            time.sleep(1.0)

            ret_test3, frame_test3 = cap.read()
            if ret_test3:
                print(f"   ✅ 回到默认格式后读取成功！帧大小：{frame_test3.shape}")
                # 使用默认分辨率
                actual_width = frame_test3.shape[1]
                actual_height = frame_test3.shape[0]
                print(f"\n   💡 将使用默认参数录制：{actual_width}×{actual_height}")
            else:
                print(f"   ❌ 连默认格式都无法读取！")
                print(f"\n{'='*60}")
                print(f"🔴 致命错误：摄像头完全无法初始化")
                print(f"{'='*60}")
                print(f"可能原因：")
                print(f"1. 摄像头驱动程序问题")
                print(f"2. 摄像头被其他应用独占")
                print(f"3. USB 连接不稳定")
                print(f"4. 缺少官方 SDK 支持")
                print(f"\n排查步骤：")
                print(f"1. 关闭 OBS、Zoom、FaceTime 等应用")
                print(f"2. 运行：python3 detect_camera.py")
                print(f"3. 检查系统偏好 > 隐私 > 摄像头权限")
                print(f"4. 查找摄像头官方 SDK（可能需要）")
                print(f"5. 重新插拔摄像头")
                print(f"{'='*60}\n")
                cap.release()
                return False
    
    # 【第5步】如果默认读取成功，跳过参数设置
    if ret_test:  # 第1步成功，使用默认格式
        print(f"\n[5/5] 使用摄像头默认格式录制")
        print(f"   📝 自动使用默认分辨率和帧率")
        actual_width = default_width
        actual_height = default_height
        if using_sdk:
            actual_fps = camera_capture.get_frame_rate(settings.device_index)
        else:
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if actual_fps <= 0:
            actual_fps = settings.fps
    
    print(f"{'='*60}\n")
    
    # 根据分辨率自动选择最佳容器和编码
    total_pixels = actual_width * actual_height
    is_high_res = total_pixels > 2560 * 1440  # 4K 以上
    is_ultra_hd = total_pixels > 1920 * 1080   # 2K 以上
    
    output_path = settings.output_path
    selected_codec = settings.fourcc
    
    # 强制使用 MP4 容器格式（比 AVI 更稳定可靠）
    # AVI 容器已弃用，原因：
    # - 不支持大于 2GB 的文件
    # - 高分辨率、高帧率支持差
    # - 对大帧数据容易出错（如 4K YUY2）
    # - macOS 兼容性差
    # - 是 30 年前的过时格式
    
    if output_path.endswith('.avi'):
        output_path = output_path.replace('.avi', '.mp4')
        print(f"ℹ️ AVI 容器已弃用，自动转换为 MP4\n")
    elif not output_path.endswith('.mp4'):
        # 如果不是 .mp4，就加上 .mp4
        output_path = output_path + '.mp4'
    
    # 根据分辨率智能选择编码器
    if is_high_res:
        # 4K 必须用 MJPEG（无损格式带宽太大）
        selected_codec = 'MJPG'
        print(f"⚠️ 4K 分辨率检测 ({actual_width}×{actual_height})")
        print(f"   🔧 自动优化: MP4 容器 + MJPEG 编码")
        print(f"   📁 输出文件: {output_path}\n")
    elif is_ultra_hd and selected_codec in ['YUY2', 'UYVY']:
        # 2K 时如果选了 YUY2/UYVY，改用 I420（文件小50%，更稳定）
        selected_codec = 'I420'
        print(f"ℹ️ 高分辨率 ({actual_width}×{actual_height})")
        print(f"   🔧 自动优化: MP4 容器 + I420 编码\n")
    elif selected_codec in ['YUY2', 'UYVY', 'I420']:
        # MP4 容器不支持 raw 格式，自动切换到 MJPEG
        print(f"⚠️ MP4 容器不支持 {selected_codec} 编码")
        selected_codec = 'MJPG'
        print(f"   🔧 自动切换到 MJPEG 编码")
        print(f"   📁 输出文件: {output_path}\n")
    
    # 创建 VideoWriter（使用实际获得的帧率）
    # 重要：必须用实际帧率而不是用户设置的帧率，否则视频会加速/减速
    writer_fps = actual_fps if actual_fps > 0 else settings.fps
    
    print(f"📝 录制参数:")
    print(f"   分辨率: {actual_width}×{actual_height}")
    print(f"   设置帧率: {actual_fps:.2f} FPS")
    print(f"   文件帧率: {writer_fps:.2f} FPS")
    print(f"   编码: {selected_codec}")
    print(f"   容器: {output_path.split('.')[-1].upper()}")
    print(f"   ℹ️ 实际帧率会在录制完成后显示\n")
    
    fourcc_code = cv2.VideoWriter_fourcc(*selected_codec)
    writer = cv2.VideoWriter(
        output_path,
        fourcc_code,
        writer_fps,
        (actual_width, actual_height),
    )
    
    if not writer.isOpened():
        print(f"⚠️ 警告：{selected_codec} 编码不支持，自动尝试 MJPEG...")
        fourcc_code = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(
            output_path,
            fourcc_code,
            actual_fps,
            (actual_width, actual_height),
        )
        
        if not writer.isOpened():
            print(f"❌ 错误：无法创建视频文件 {output_path}", file=sys.stderr)
            cap.release()
            return False
        print(f"✅ 已自动切换到 MJPEG 编码\n")
    
    eye_detector = EyeDetector() if settings.enable_eye_detection else None
    
    print(f"🎬 开始录制")
    print(f"{'='*60}")
    print(f"摄像头索引：{settings.device_index}")
    print(f"输出文件：{output_path}")
    print(f"眼睛检测：{'启用 👁️' if settings.enable_eye_detection else '禁用'}")
    print(f"按 'q' 或 'Esc' 结束录制\n")
    
    try:
        frame_count = 0
        start_time = time.time()
        last_time = start_time
        fps_samples = []
        bandwidth_samples = []

        while True:
            if using_sdk:
                frame = camera_capture.get_frame(timeout=0.1)
                if frame is None:
                    print("❌ SDK 摄像头读取失败，中止录制。")
                    print("   💡 可能原因：SDK 连接问题或摄像头被其他应用占用")
                    break
            else:
                ret, frame = cap.read()
                if not ret:
                    print("❌ OpenCV 摄像头读取失败，中止录制。")
                    print("   💡 可能原因：摄像头驱动问题或连接不稳定")
                    break
            
            # 计算实际传输的数据大小（帧大小）
            frame_bytes = frame.nbytes if hasattr(frame, 'nbytes') else frame.size
            
            # 检测眼睛
            if eye_detector:
                frame, detection_info = eye_detector.detect(frame)
                # 添加检测状态指示
                status = "Eye: Detected" if detection_info["detected"] else "Eye: Not detected"
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            writer.write(frame)
            
            # 显示预览
            display_frame = cv2.resize(frame, (960, 540))
            cv2.imshow("Recording (Press q/Esc to stop)", display_frame)
            
            frame_count += 1
            
            # 测量实际帧率和带宽（每 10 帧采样一次）
            if frame_count % 10 == 0:
                current_time = time.time()
                time_delta = current_time - last_time
                if time_delta > 0:
                    sampled_fps = 10 / time_delta
                    fps_samples.append(sampled_fps)
                    
                    # 计算带宽（MBps）
                    bandwidth_mbps = (frame_bytes * sampled_fps) / (1024 * 1024)
                    bandwidth_samples.append(bandwidth_mbps)
                last_time = current_time
            
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("检测到退出指令，停止录制。")
                break
    finally:
        if using_sdk:
            camera_capture.stop_capture(settings.device_index)
            camera_capture.close_device(settings.device_index)
            camera_capture.uninitialize()
        elif cap:
            cap.release()
        writer.release()
        cv2.destroyAllWindows()
        
        elapsed_time = time.time() - start_time
        file_size_mb = os.path.getsize(output_path) / (1024*1024)
        
        # 计算实际帧率和带宽
        actual_measured_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        avg_sampled_fps = sum(fps_samples) / len(fps_samples) if fps_samples else 0
        avg_bandwidth_mbps = sum(bandwidth_samples) / len(bandwidth_samples) if bandwidth_samples else 0
        
        print(f"\n{'='*60}")
        print(f"✅ 录制完成")
        print(f"{'='*60}")
        print(f"📊 录制统计:")
        print(f"   总帧数：{frame_count} 帧")
        print(f"   耗时：{elapsed_time:.2f} 秒")
        print(f"   平均帧率：{actual_measured_fps:.2f} FPS")
        if avg_sampled_fps > 0:
            print(f"   采样帧率：{avg_sampled_fps:.2f} FPS (实时测量)")
        
        # 带宽诊断
        print(f"\n🔌 USB 带宽诊断:")
        if avg_bandwidth_mbps > 0 and actual_measured_fps > 0:
            print(f"   平均传输速率：{avg_bandwidth_mbps:.2f} MBps")
            print(f"   理论带宽需求：{frame_bytes * actual_measured_fps / (1024*1024):.2f} MBps")
            
            if avg_bandwidth_mbps > 300:
                print(f"   ⚠️ 警告：超过 USB 2.0 限制 (60 MBps 可用)")
                print(f"      💡 建议：用 USB 3.0 接口或选择 MJPEG 有损格式")
            elif avg_bandwidth_mbps > 60:
                print(f"   ⚠️ 可能接近 USB 2.0 限制")
            else:
                print(f"   ✅ USB 2.0 可以支持此参数")
        
        # 帧率警告
        if avg_sampled_fps > 0 and abs(actual_measured_fps - actual_fps) > 5:
            print(f"\n⚠️ 帧率问题:")
            print(f"   设置：{actual_fps:.0f}FPS，实际：{actual_measured_fps:.2f}FPS")
            if actual_measured_fps < actual_fps * 0.8:
                print(f"      🔴 摄像头无法达到设置的帧率！")
                print(f"      原因可能是 USB 带宽不足或选择的编码格式过大")
                print(f"      💡 建议：")
                print(f"         1. 使用 USB 3.0 接口")
                print(f"         2. 尝试 MJPEG 有损格式（如果摄像头硬件支持）")
                print(f"         3. 降低帧率或分辨率")
        
        print(f"\n📁 文件信息:")
        print(f"   文件大小：{file_size_mb:.2f} MB")
        if actual_measured_fps > 0:
            print(f"   视频时长：{frame_count/actual_measured_fps:.2f} 秒")
        else:
            print(f"   视频时长：0.00 秒 (未获取任何帧)")
        print(f"   文件位置：{output_path}")
        
        # 诊断：没有获取到帧
        if frame_count == 0:
            print(f"\n❌ 错误：未能从摄像头获取任何帧！")
            print(f"   可能的原因：")
            print(f"   1. 摄像头格式协商失败（FourCC 不支持）")
            print(f"   2. 分辨率/帧率摄像头不支持")
            print(f"   3. 摄像头被其他应用占用")
            print(f"   4. USB 连接问题")
            print(f"\n   建议：")
            print(f"   1. 尝试降低帧率或分辨率")
            print(f"   2. 尝试不同的编码格式（MJPG/I420）")
            print(f"   3. 关闭其他使用摄像头的应用")
            print(f"   4. 重新插拔 USB 摄像头")
        
        print(f"{'='*60}\n")
    
    return True


def main() -> None:
    """主函数。"""
    if tk is None or ImageTk is None:
        print("错误：需要 tkinter 和 Pillow。", file=sys.stderr)
        return
    
    if mp is None:
        print("警告：MediaPipe 未安装，眼睛检测将被禁用。")
    
    root = tk.Tk()
    app = CaptureApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
