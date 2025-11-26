#!/usr/bin/env python3
"""
测试 capture.py 中的修复
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from capture import CaptureSettings, run_capture

def test_sdk_mode():
    """测试 SDK 模式下的录制（模拟）"""

    # 创建一个模拟的设置
    settings = CaptureSettings(
        device_index=0,
        output_path="test_output.mp4",
        fps=30.0,
        width=1280,
        height=720,
        fourcc="MJPG",
        enable_eye_detection=False
    )

    print("Testing SDK mode fix...")
    print("Note: This is just a syntax test, actual recording requires camera hardware")

    # Here we just check if the function can start normally without crashing
    # Since there's no actual camera, we expect it to fail, but not with AttributeError
    try:
        result = run_capture(settings)
        print(f"Function executed, result: {result}")
        print("✅ Fix successful: no AttributeError")
    except AttributeError as e:
        if "'NoneType' object has no attribute 'isOpened'" in str(e):
            print("❌ Fix failed: still have the original error")
            return False
        else:
            print(f"Other AttributeError: {e}")
            return False
    except Exception as e:
        print(f"Other exception (expected, since no camera): {type(e).__name__}: {e}")
        print("✅ Fix successful: no original AttributeError")

    return True

if __name__ == "__main__":
    test_sdk_mode()
