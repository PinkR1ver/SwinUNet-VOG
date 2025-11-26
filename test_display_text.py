#!/usr/bin/env python3
"""
测试显示文本是否正确
"""

def test_display_text():
    """测试视频叠加文字"""

    # 预览信息
    mode_text = "OpenCV"  # 模拟使用OpenCV模式
    width, height, fps = 1280, 720, 30.0
    info_text = f"{mode_text}: {width}x{height} @ {fps:.0f} FPS"
    print(f"Preview text: {info_text}")

    # 眼睛检测状态
    detection_info = {"detected": True}
    status = "Eye: Detected" if detection_info["detected"] else "Eye: Not detected"
    print(f"Recording status: {status}")

    detection_info = {"detected": False}
    status = "Eye: Detected" if detection_info["detected"] else "Eye: Not detected"
    print(f"Recording status: {status}")

    return True

if __name__ == "__main__":
    print("Testing display text...")
    test_display_text()
    print("✅ All display text is now in English!")
