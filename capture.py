"""
capture.py
----------

通过图形界面选择摄像头与拍摄参数，录制原始（无损压缩）视频到当前目录。

主要特性：
1. 直接在 UI 中选择摄像头索引（0~N），无需预先检测设备是否连接。
2. 在界面里自定义输出文件、分辨率、帧率以及 FourCC 编码。
3. 默认使用 YUY2 无压缩编码保存 AVI，尽可能保留原始数据。
4. 录制过程中会显示实时预览，按 `q` 或 `Esc` 结束。
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional

import cv2

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # pragma: no cover - tkinter 在大多数桌面环境默认存在
    tk = None  # type: ignore


@dataclass
class CaptureSettings:
    device_index: int
    output_path: str
    fps: float
    width: int
    height: int
    fourcc: str


def list_available_cameras(max_index: int = 10) -> Dict[int, str]:
    """枚举摄像头，返回 {索引: 名称} 映射。"""
    available: Dict[int, str] = {}
    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            name = f"Camera {index}"
            backend_name = cap.getBackendName() if hasattr(cap, "getBackendName") else ""
            if backend_name:
                name += f" ({backend_name})"
            available[index] = name
            cap.release()
    return available


def show_ui(max_index: int = 10) -> Optional[CaptureSettings]:
    """弹出 Tkinter UI，返回用户选择的录制参数。"""
    if tk is None:
        message = "未检测到 Tkinter，无法显示图形界面。"
        print(message, file=sys.stderr)
        return None

    root = tk.Tk()
    root.title("摄像头录制 - SwinUNet-VOG")
    root.resizable(False, False)

    # 保存用户选择
    result: dict[str, str] = {}

    # 默认值
    default_device = 0
    default_output = os.path.abspath("capture.avi")
    default_fps = "30"
    default_width = "1280"
    default_height = "720"
    default_fourcc = "YUY2"

    # 设备选择
    tk.Label(root, text="摄像头索引：").grid(row=0, column=0, sticky="e", padx=6, pady=6)
    device_var = tk.StringVar(value=str(default_device))
    device_combo = ttk.Combobox(root, textvariable=device_var, width=32)
    device_combo.grid(row=0, column=1, sticky="w", padx=6, pady=6)
    hint_label = tk.Label(root, text="（若列表为空，可手动输入索引或点击刷新）")
    hint_label.grid(row=0, column=2, sticky="w", padx=6, pady=6)

    def refresh_devices() -> None:
        cameras = list_available_cameras(max_index=max_index)
        if cameras:
            device_combo["values"] = [f"{idx}: {name}" for idx, name in cameras.items()]
            device_var.set(f"{next(iter(cameras.keys()))}: {cameras[next(iter(cameras.keys()))]}")
            hint_label.config(text="（可从列表选择或手动修改索引）")
        else:
            device_combo["values"] = [str(i) for i in range(max_index)]
            device_var.set(str(default_device))
            hint_label.config(text="（未检测到设备，可手动输入索引后开始录制）")
            if tk is not None:
                messagebox.showinfo("提示", "未检测到摄像头，您仍可手动输入索引尝试录制。")

    refresh_devices_button = ttk.Button(root, text="刷新列表", command=refresh_devices)
    refresh_devices_button.grid(row=0, column=3, padx=6, pady=6)
    refresh_devices()

    # 输出文件
    tk.Label(root, text="输出文件：").grid(row=1, column=0, sticky="e", padx=6, pady=6)
    output_var = tk.StringVar(value=default_output)
    output_entry = tk.Entry(root, textvariable=output_var, width=34)
    output_entry.grid(row=1, column=1, sticky="w", padx=6, pady=6)

    def browse_output() -> None:
        path = filedialog.asksaveasfilename(
            title="选择输出文件",
            defaultextension=".avi",
            filetypes=[("AVI 文件", "*.avi"), ("所有文件", "*.*")],
            initialfile=os.path.basename(default_output),
        )
        if path:
            output_var.set(os.path.abspath(path))

    browse_button = ttk.Button(root, text="浏览...", command=browse_output)
    browse_button.grid(row=1, column=2, padx=6, pady=6)

    # FPS
    tk.Label(root, text="帧率 (FPS)：").grid(row=2, column=0, sticky="e", padx=6, pady=6)
    fps_var = tk.StringVar(value=default_fps)
    tk.Entry(root, textvariable=fps_var, width=10).grid(row=2, column=1, sticky="w", padx=6, pady=6)

    # 分辨率
    tk.Label(root, text="分辨率：").grid(row=3, column=0, sticky="e", padx=6, pady=6)
    width_var = tk.StringVar(value=default_width)
    height_var = tk.StringVar(value=default_height)
    res_frame = tk.Frame(root)
    res_frame.grid(row=3, column=1, sticky="w", padx=6, pady=6)
    tk.Entry(res_frame, textvariable=width_var, width=8).pack(side="left")
    tk.Label(res_frame, text=" × ").pack(side="left")
    tk.Entry(res_frame, textvariable=height_var, width=8).pack(side="left")

    # FourCC
    tk.Label(root, text="FourCC 编码：").grid(row=4, column=0, sticky="e", padx=6, pady=6)
    fourcc_var = tk.StringVar(value=default_fourcc)
    tk.Entry(root, textvariable=fourcc_var, width=10).grid(row=4, column=1, sticky="w", padx=6, pady=6)
    tk.Label(root, text="（建议使用无压缩，如 YUY2、UYVY、I420）").grid(
        row=4, column=2, sticky="w", padx=6, pady=6
    )

    def on_start() -> None:
        try:
            device_choice = device_var.get().strip()
            if ":" in device_choice:
                device_choice = device_choice.split(":", 1)[0].strip()
            device_index = int(device_choice)
            fps = float(fps_var.get())
            width = int(width_var.get())
            height = int(height_var.get())
            fourcc = fourcc_var.get().strip().upper()
            output_path = output_var.get()
        except ValueError:
            messagebox.showerror("输入错误", "请确保帧率、分辨率为正确数字。")
            return

        if len(fourcc) != 4:
            messagebox.showerror("输入错误", "FourCC 编码必须是 4 个字符。")
            return
        if not output_path:
            messagebox.showerror("输入错误", "请指定输出文件路径。")
            return

        result["device_index"] = str(device_index)
        result["fps"] = str(fps)
        result["width"] = str(width)
        result["height"] = str(height)
        result["fourcc"] = fourcc
        result["output"] = os.path.abspath(output_path)
        root.destroy()

    def on_cancel() -> None:
        root.destroy()

    button_frame = tk.Frame(root)
    button_frame.grid(row=5, column=0, columnspan=3, pady=10)
    ttk.Button(button_frame, text="开始录制", command=on_start).pack(side="left", padx=8)
    ttk.Button(button_frame, text="取消", command=on_cancel).pack(side="left", padx=8)

    root.mainloop()

    if not result:
        return None

    return CaptureSettings(
        device_index=int(result["device_index"]),
        output_path=result["output"],
        fps=float(result["fps"]),
        width=int(result["width"]),
        height=int(result["height"]),
        fourcc=result["fourcc"],
    )


def run_capture(settings: CaptureSettings) -> bool:
    """使用给定参数录制视频。"""
    print(f"使用摄像头索引 {settings.device_index}")
    cap = cv2.VideoCapture(settings.device_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        message = f"无法打开摄像头设备 {settings.device_index}"
        if tk is not None:
            messagebox.showerror("摄像头错误", message)
        else:
            print(message, file=sys.stderr)
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    cap.set(cv2.CAP_PROP_FPS, settings.fps)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or settings.fps

    fourcc_code = cv2.VideoWriter_fourcc(*settings.fourcc)
    writer = cv2.VideoWriter(
        settings.output_path,
        fourcc_code,
        actual_fps,
        (actual_width, actual_height),
    )
    if not writer.isOpened():
        message = f"无法创建视频文件 {settings.output_path}"
        if tk is not None:
            messagebox.showerror("写入错误", message)
        else:
            print(message, file=sys.stderr)
        cap.release()
        return False

    print("开始录制，按 `q` 或 `Esc` 结束。")
    print(f"输出文件：{settings.output_path}")
    print(f"分辨率：{actual_width}×{actual_height} @ {actual_fps:.2f} FPS")
    print(f"FourCC：{settings.fourcc}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("摄像头读取失败，中止录制。")
                break

            writer.write(frame)
            cv2.imshow("Recording Preview (press q/Esc to stop)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("检测到退出指令，停止录制。")
                break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print("录制完成。")
    return True


def main() -> None:
    settings = show_ui()
    if settings is None:
        print("用户取消或未能获取录制参数。")
        return
    success = run_capture(settings)
    if not success:
        print("录制未开始或中途失败。")


if __name__ == "__main__":
    main()

