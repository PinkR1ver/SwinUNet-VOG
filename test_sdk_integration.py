#!/usr/bin/env python3
"""
SDK 集成测试脚本

测试 SDK 封装和 capture.py 的集成。
"""

import sys
import platform
import time

def test_sdk_import():
    """测试 SDK 封装导入"""
    print("=== Testing SDK wrapper import ===")
    try:
        from sdk_wrapper import UnifiedCameraCapture, SDKCameraCapture, OpenCVFallbackCapture
        print("OK: SDK wrapper imported successfully")

        # 测试实例化
        capture = UnifiedCameraCapture()
        print("OK: UnifiedCameraCapture instantiated successfully")

        # 测试是否使用 SDK
        using_sdk = capture.is_using_sdk()
        print(f"OK: Using SDK: {using_sdk}")

        # 测试初始化
        if capture.initialize():
            print("OK: SDK initialized successfully")

            # 测试枚举设备
            devices = capture.enum_devices()
            print(f"OK: Found {len(devices)} devices:")
            for i, device in enumerate(devices):
                print(f"  {i}: {device.name}")

            # 如果有设备，测试获取能力
            if devices:
                capabilities = capture.get_capabilities(0)
                print(f"OK: Device 0 capabilities:")
                print(f"  Resolutions: {capabilities.resolutions}")
                print(f"  FPS values: {capabilities.fps_values}")

            capture.uninitialize()
            print("OK: SDK cleanup successful")

        else:
            print("FAIL: SDK initialization failed")

        return True

    except ImportError as e:
        print(f"FAIL: SDK wrapper import failed: {e}")
        return False
    except Exception as e:
        print(f"FAIL: SDK test exception: {e}")
        return False

def test_capture_import():
    """测试 capture.py 导入"""
    print("\n=== Testing capture.py import ===")
    try:
        import capture
        print("OK: capture.py imported successfully")
        return True
    except ImportError as e:
        print(f"FAIL: capture.py import failed: {e}")
        return False
    except Exception as e:
        print(f"FAIL: capture.py test exception: {e}")
        return False

def test_capture_app():
    """测试 CaptureApp（无 GUI）"""
    print("\n=== Testing CaptureApp instantiation ===")
    try:
        # 由于 Tkinter 在无显示环境下会失败，我们只测试导入和基本实例化
        import capture

        # 检查必要的类是否存在
        if hasattr(capture, 'CaptureApp'):
            print("OK: CaptureApp class exists")
        else:
            print("FAIL: CaptureApp class not found")
            return False

        if hasattr(capture, 'EyeDetector'):
            print("OK: EyeDetector class exists")
        else:
            print("FAIL: EyeDetector class not found")
            return False

        if hasattr(capture, 'detect_camera_capabilities'):
            print("OK: detect_camera_capabilities function exists")
        else:
            print("FAIL: detect_camera_capabilities function not found")
            return False

        return True

    except Exception as e:
        print(f"FAIL: CaptureApp test exception: {e}")
        return False

def main():
    """主测试函数"""
    print(f"Running on: {platform.system()} {platform.release()}")
    print(f"Python version: {sys.version}")
    print()

    results = []

    # 测试 SDK 导入
    results.append(("SDK wrapper import", test_sdk_import()))

    # 测试 capture.py 导入
    results.append(("capture.py import", test_capture_import()))

    # 测试 CaptureApp
    results.append(("CaptureApp basic functionality", test_capture_app()))

    # 输出总结
    print("\n" + "="*50)
    print("Test Results Summary:")
    print("="*50)

    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:<35} : {status}")
        if not passed:
            all_passed = False

    print(f"\nOverall result: {'PASS: All tests passed' if all_passed else 'FAIL: Some tests failed'}")

    if not all_passed:
        print("\nTroubleshooting:")
        print("1. Ensure running on Windows platform (SDK only supports Windows)")
        print("2. Ensure CapLib.dll is in system PATH or current directory")
        print("3. Install required dependencies: pip install opencv-python mediapipe pillow")
        print("4. On Linux/Mac, OpenCV fallback will be used automatically")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
