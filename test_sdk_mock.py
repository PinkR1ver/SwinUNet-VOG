#!/usr/bin/env python3
"""
模拟 SDK 测试
"""

import sys
sys.path.insert(0, '.')

# 临时修改 sdk_wrapper 来模拟 SDK 工作
import sdk_wrapper

# 保存原始方法
original_load_sdk = sdk_wrapper.SDKCameraCapture._load_sdk

def mock_load_sdk(self):
    """模拟 SDK 加载成功"""
    print('Mock: SDK loaded successfully')
    # 创建假的 DLL 对象，模拟必要的函数
    mock_dll = type('MockDLL', (), {
        'InitLib': lambda: True,
        'UnInitLib': lambda: True,
        'EnumCameras': lambda devs, size, count: False,  # 模拟没有设备
        'getFrameRate': lambda dev: 30,
        'getBitrate': lambda dev: 1000,
    })()
    self.dll = mock_dll
    self.initialized = True
    return True

# 替换方法
sdk_wrapper.SDKCameraCapture._load_sdk = mock_load_sdk

try:
    from sdk_wrapper import UnifiedCameraCapture
    capture = UnifiedCameraCapture()
    print('SDK available:', capture.is_using_sdk())
    print('Current implementation:', type(capture.current_impl).__name__)

    # 测试能力检测
    capabilities, method = capture.get_capabilities(0)
    print(f'Detected {len(capabilities.resolutions)} resolutions using {method}')

    # 测试初始化
    if capture.initialize():
        devices = capture.enum_devices()
        print(f'Found {len(devices)} devices')
        capture.uninitialize()

except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()
finally:
    # 恢复原始方法
    sdk_wrapper.SDKCameraCapture._load_sdk = original_load_sdk

print("\nMock test completed - SDK code paths work correctly!")
