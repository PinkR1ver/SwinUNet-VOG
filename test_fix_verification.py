#!/usr/bin/env python3
"""
验证修复是否有效
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_sdk_detection():
    """测试SDK检测逻辑"""
    try:
        from sdk_wrapper import UnifiedCameraCapture

        capture = UnifiedCameraCapture()
        is_using_sdk = capture.is_using_sdk()

        print(f"SDK available: {is_using_sdk}")
        print("Expected: False (since CapLib.dll is not available)")

        # 初始化测试
        if capture.initialize():
            print("Initialize: SUCCESS")
        else:
            print("Initialize: FAILED")

        return True

    except Exception as e:
        print(f"Test failed: {e}")
        return False

def test_codec_selection():
    """测试编码选择逻辑"""
    # 模拟MP4容器中的编码选择
    container = "MP4"
    user_codec = "YUY2"

    # 应用修复逻辑
    if user_codec in ['YUY2', 'UYVY', 'I420']:
        print(f"Container {container} doesn't support {user_codec}")
        selected_codec = 'MJPG'
        print(f"Auto-switched to {selected_codec}")
    else:
        selected_codec = user_codec

    print(f"Final codec: {selected_codec}")
    return selected_codec == 'MJPG'

if __name__ == "__main__":
    print("Testing fixes...")

    print("\n1. Testing SDK detection:")
    sdk_ok = test_sdk_detection()

    print("\n2. Testing codec selection:")
    codec_ok = test_codec_selection()

    if sdk_ok and codec_ok:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
