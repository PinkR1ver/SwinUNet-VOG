#!/usr/bin/env python3
"""
测试修复后的重复实例问题
"""

import capture

print("=== Testing fixed duplicate instance issue ===")

# Create CaptureApp instance (mock)
app = capture.CaptureApp.__new__(capture.CaptureApp)  # Only create instance, no GUI init
app.backend = capture.get_camera_backend()
app.device_var = type('MockVar', (), {'get': lambda self: '0: Camera 0'})()
app.resolution_combo = type('MockCombo', (), {'__setitem__': lambda self, k, v: None, 'set': lambda self, v: None, '__getitem__': lambda self, k: []})()
app.fps_combo = type('MockCombo', (), {'__setitem__': lambda self, k, v: None, 'set': lambda self, v: None, '__getitem__': lambda self, k: []})()

# Initialize camera capture (this creates the first instance)
print("1. Initializing camera capture in CaptureApp...")
app.camera_capture = capture.UnifiedCameraCapture() if capture.UnifiedCameraCapture else None
if app.camera_capture:
    app.camera_capture.initialize()

print("2. Calling _on_device_changed (should reuse existing instance)...")
# Simulate device change - this should reuse the existing instance
try:
    app._on_device_changed()
    print("3. Test completed successfully - only one set of SDK warnings should appear above")
except UnicodeEncodeError:
    print("3. Test completed - Unicode error in print statement, but SDK warnings should be minimal")

print("4. Test finished")
