#!/usr/bin/env python3
"""
测试MediaPipe警告是否被抑制
"""

import os
import sys

# 模拟capture.py开头的环境变量设置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    import mediapipe as mp
    print("MediaPipe imported successfully")

    # 测试FaceMesh初始化
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    print("FaceMesh initialized successfully")

    # 清理
    face_mesh.close()
    print("FaceMesh closed successfully")

except ImportError:
    print("MediaPipe not available")
except Exception as e:
    print(f"Error: {e}")

print("Test completed - warnings should be reduced")
