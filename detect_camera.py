"""
detect_camera.py
----------------

检测摄像头的 USB 协议、支持的分辨率/帧率等详细信息。
仅在 macOS 上有效。
"""

import subprocess
import json
import sys
import platform


def get_usb_device_info():
    """获取 macOS USB 设备信息。"""
    
    if platform.system() != "Darwin":
        print("❌ 本工具仅支持 macOS")
        return []
    
    print("📱 检测 USB 摄像头...\n")
    
    # 使用 system_profiler 获取 USB 设备信息
    try:
        result = subprocess.run(
            ["system_profiler", "SPUSBDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            print(f"❌ 系统命令失败")
            return []
        
        data = json.loads(result.stdout)
        cameras = []
        
        # 遍历 USB 设备
        for item in data.get("SPUSBDataType", []):
            cameras.extend(find_cameras_in_device(item, []))
        
        return cameras
    except Exception as e:
        print(f"❌ 错误: {e}")
        return []


def find_cameras_in_device(device, path):
    """递归搜索 USB 设备中的摄像头。"""
    cameras = []
    current_path = path + [device.get("_name", "Unknown")]
    
    # 检查是否为摄像头
    if "camera" in device.get("_name", "").lower() or \
       "video" in device.get("_name", "").lower() or \
       "capture" in device.get("_name", "").lower():
        cameras.append({
            "name": device.get("_name", "Unknown"),
            "path": " → ".join(current_path),
            "product_id": device.get("product_id", "N/A"),
            "vendor_id": device.get("vendor_id", "N/A"),
            "speed": device.get("_speed", "Unknown"),
            "manufacturer": device.get("manufacturer", "Unknown"),
            "serial_number": device.get("serial_number", "N/A"),
            "raw": device
        })
    
    # 递归检查子设备
    for item in device.get("_items", []):
        cameras.extend(find_cameras_in_device(item, current_path))
    
    return cameras


def get_cv_camera_info():
    """使用 OpenCV 获取摄像头信息（需要已安装 cv2）。"""
    try:
        import cv2
        cameras = []
        
        print("🎥 OpenCV 检测的摄像头:\n")
        
        backend = cv2.CAP_AVFOUNDATION  # macOS
        for idx in range(5):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                name = f"Camera {idx}"
                backend_name = cap.getBackendName() if hasattr(cap, "getBackendName") else "AVFoundation"
                
                # 获取支持的分辨率和帧率
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                cap.set(cv2.CAP_PROP_FPS, 120)
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"📹 摄像头 {idx}: {name}")
                print(f"   后端: {backend_name}")
                print(f"   分辨率: {actual_w}×{actual_h}")
                print(f"   帧率设置: 120 FPS → 实际: {actual_fps:.2f} FPS")
                
                # 尝试高帧率
                for test_fps in [240, 180, 150, 120, 90, 60]:
                    cap.set(cv2.CAP_PROP_FPS, test_fps)
                    got_fps = cap.get(cv2.CAP_PROP_FPS)
                    if got_fps >= test_fps * 0.9:
                        print(f"   ✅ 支持 {test_fps} FPS → {got_fps:.2f}")
                        break
                
                cameras.append({
                    "index": idx,
                    "name": name,
                    "resolution": f"{actual_w}×{actual_h}",
                    "fps": actual_fps
                })
                cap.release()
                print()
        
        return cameras
    except ImportError:
        print("⚠️ OpenCV 未安装，跳过 OpenCV 检测\n")
        return []


def get_ioreg_camera_info():
    """使用 ioreg 获取详细的摄像头 USB 信息。"""
    print("🔧 USB 协议详情 (ioreg):\n")
    
    try:
        result = subprocess.run(
            ["ioreg", "-p", "IOUSB", "-l"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines):
                if 'camera' in line.lower() or 'video' in line.lower():
                    # 打印该设备及其周围上下文
                    start = max(0, i - 3)
                    end = min(len(lines), i + 10)
                    print("发现摄像头相关信息:")
                    for j in range(start, end):
                        print(lines[j])
                    print()
    except Exception as e:
        print(f"⚠️ ioreg 查询失败: {e}\n")


def get_lsusb_info():
    """使用 lsusb 风格的命令获取 USB 信息 (macOS 上可能不可用)。"""
    print("📋 尝试 USB 设备列表:\n")
    
    try:
        result = subprocess.run(
            ["system_profiler", "SPUSBDataType"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            capture = False
            for line in lines:
                if 'camera' in line.lower() or 'video' in line.lower() or 'capture' in line.lower():
                    capture = True
                if capture:
                    print(line)
                    if line.strip() == "" and capture:
                        capture = False
    except Exception as e:
        print(f"⚠️ 查询失败: {e}\n")


def main():
    print(f"\n{'='*70}")
    print(f"🎥 摄像头 USB 协议检测工具")
    print(f"{'='*70}\n")
    
    # 1. OpenCV 检测
    cv_cameras = get_cv_camera_info()
    
    # 2. 系统 USB 信息
    get_usb_device_info()
    
    # 3. 详细的 ioreg 信息
    get_ioreg_camera_info()
    
    # 4. system_profiler 完整信息
    get_lsusb_info()
    
    print(f"\n{'='*70}")
    print(f"💡 建议:")
    print(f"{'='*70}")
    print("""
如果看到 'USB 3.0' 或 'High-Speed' 且 'Super-Speed':
  ✅ 你的摄像头支持高速 USB
  
如果只看到 'USB 2.0' 或 'High-Speed':
  ⚠️ 可能是 USB 2.0 带宽限制
  - 1080p@120FPS 可能超出 USB 2.0 带宽 (480 Mbps)
  - 建议用 USB 3.0 接口（USB-A 蓝色或 USB-C）
  
如果看不到摄像头:
  ❌ 摄像头可能未被系统识别
  - 重新插拔摄像头
  - 检查驱动是否安装
    """)


def test_camera_opencv_formats():
    """测试 OpenCV 能否打开摄像头及支持的格式。"""
    print(f"\n{'='*70}")
    print(f"🔧 OpenCV 摄像头兼容性测试")
    print(f"{'='*70}\n")
    
    try:
        import cv2
    except ImportError:
        print("❌ OpenCV 未安装，跳过测试")
        return
    
    # 尝试打开摄像头
    print("[1/3] 尝试打开摄像头索引 0...")
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    
    if not cap.isOpened():
        print("   ❌ 无法打开摄像头")
        print("   可能原因：")
        print("   - 摄像头被其他应用占用（检查 OBS、Zoom 等）")
        print("   - 权限问题（检查系统偏好设置 > 隐私）")
        print("   - 摄像头驱动问题")
        return
    
    print("   ✅ 摄像头已打开")
    
    # 读取默认格式
    print("\n[2/3] 检测默认格式...")
    ret, frame = cap.read()
    if ret:
        print(f"   ✅ 能读取帧：{frame.shape}")
        print(f"   数据类型：{frame.dtype}")
    else:
        print(f"   ⚠️ 无法读取默认格式")
    
    # 尝试设置分辨率和帧率
    print("\n[3/3] 尝试常见参数组合...")
    
    test_configs = [
        (640, 480, 30),
        (1280, 720, 30),
        (1920, 1080, 30),
        (1280, 720, 60),
        (1920, 1080, 60),
        (1920, 1080, 120),
    ]
    
    for width, height, fps in test_configs:
        cap_test = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        cap_test.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap_test.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap_test.set(cv2.CAP_PROP_FPS, fps)
        
        time.sleep(0.2)
        
        actual_w = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap_test.get(cv2.CAP_PROP_FPS)
        
        ret, _ = cap_test.read()
        status = "✅" if ret else "❌"
        
        print(f"   {status} {width}×{height}@{fps}FPS → {actual_w}×{actual_h}@{actual_fps:.0f}FPS")
        
        cap_test.release()
    
    cap.release()
    print()


if __name__ == "__main__":
    import time
    main()
    test_camera_opencv_formats()

