"""
SDK Camera Capture Wrapper Module

Provides Python encapsulation for CapLib SDK, supporting advanced camera control on Windows.
Automatically falls back to OpenCV on non-Windows platforms.
"""

import platform
import ctypes
import threading
import queue
import time
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import numpy as np
import cv2

# Check if running on Windows platform
IS_WINDOWS = platform.system() == "Windows"


@dataclass
class VideoFormat:
    """Video format information"""
    color_space: str  # 'MJPG', 'YUY2', 'Unknown'
    width: int
    height: int
    fps: float
    guid: bytes = field(default_factory=lambda: b'\x00' * 16)


@dataclass
class CameraDevice:
    """Camera device information"""
    vidpid: str  # VID_PID format
    name: str


@dataclass
class CameraCapabilities:
    """Camera capabilities information"""
    resolutions: List[Tuple[int, int]] = field(default_factory=list)
    fps_values: List[float] = field(default_factory=list)
    formats: List[VideoFormat] = field(default_factory=list)


class SDKCameraCapture:
    """Camera capture class based on CapLib SDK"""

    def __init__(self):
        self.dll = None
        self.initialized = False
        self.devices: List[CameraDevice] = []
        self.current_device: Optional[int] = None
        self.current_format: Optional[VideoFormat] = None
        self.is_running = False
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.capture_thread: Optional[threading.Thread] = None
        self.callback_ptr = None

        if IS_WINDOWS:
            self._load_sdk()

    def _load_sdk(self) -> bool:
        """Load CapLib SDK"""
        try:
            # Try to load DLL (assumed to be in system PATH or current directory)
            self.dll = ctypes.windll.LoadLibrary("CapLib.dll")

            # Define callback function type
            self.CAP_GRABBER = ctypes.WINFUNCTYPE(
                ctypes.c_int,  # return value
                ctypes.c_double,  # sampleTime
                ctypes.POINTER(ctypes.c_uint8),  # buf
                ctypes.c_long,  # bufSize
                ctypes.c_void_p  # ptrClass
            )

            self.initialized = True
            return True

        except Exception as e:
            print(f"Warning: Failed to load CapLib SDK: {e}")
            print("Will use OpenCV as fallback")
            self.initialized = False
            return False

    def initialize(self) -> bool:
        """Initialize SDK"""
        if not IS_WINDOWS or not self.initialized:
            return False

        try:
            if self.dll.InitLib():
                return True
            else:
                print("SDK initialization failed")
                return False
        except Exception as e:
            print(f"SDK initialization exception: {e}")
            return False

    def uninitialize(self) -> bool:
        """Uninitialize SDK"""
        if not IS_WINDOWS or not self.initialized:
            return True

        try:
            self.stop_capture()
            return bool(self.dll.UnInitLib())
        except Exception as e:
            print(f"SDK uninitialization exception: {e}")
            return False

    def enum_devices(self) -> List[CameraDevice]:
        """Enumerate camera devices"""
        if not IS_WINDOWS or not self.initialized:
            return []

        try:
            # Define device structure
            class CaptureDeviceStruct(ctypes.Structure):
                _fields_ = [
                    ("vidpid", ctypes.c_char * 9),
                    ("devName", ctypes.c_wchar * 256)
                ]

            devices_array = (CaptureDeviceStruct * 10)()
            dev_count = ctypes.c_uint32()

            if self.dll.EnumCameras(devices_array, 10, ctypes.byref(dev_count)):
                self.devices = []
                for i in range(dev_count.value):
                    dev_struct = devices_array[i]
                    device = CameraDevice(
                        vidpid=dev_struct.vidpid.decode('utf-8'),
                        name=dev_struct.devName.decode('utf-16')
                    )
                    self.devices.append(device)
                return self.devices
            else:
                print("Device enumeration failed")
                return []

        except Exception as e:
            print(f"Device enumeration exception: {e}")
            return []

    def get_capabilities(self, device_index: int) -> CameraCapabilities:
        """Get camera capabilities"""
        capabilities = CameraCapabilities()

        if not IS_WINDOWS or not self.initialized:
            # Return default capabilities
            capabilities.resolutions = [(640, 480), (1280, 720), (1920, 1080)]
            capabilities.fps_values = [24.0, 30.0, 60.0]
            return capabilities

        try:
            # Define format structure
            class VideoFormatStruct(ctypes.Structure):
                _fields_ = [
                    ("vcs", ctypes.c_int),  # VideoColorSpace_t
                    ("w", ctypes.c_uint16),
                    ("h", ctypes.c_uint16),
                    ("fps", ctypes.c_float),
                    ("vcs_guid", ctypes.c_byte * 16)
                ]

            formats_array = (VideoFormatStruct * 50)()
            format_count = ctypes.c_uint16()

            if self.dll.GetFormats(device_index, formats_array,
                                 ctypes.byref(format_count), 50):
                for i in range(format_count.value):
                    fmt_struct = formats_array[i]
                    # Map color space enum
                    color_spaces = {0: 'Unknown', 1: 'Unknow', 2: 'MJPG', 3: 'YUY2'}
                    color_space = color_spaces.get(fmt_struct.vcs, 'Unknown')

                    fmt = VideoFormat(
                        color_space=color_space,
                        width=fmt_struct.w,
                        height=fmt_struct.h,
                        fps=fmt_struct.fps,
                        guid=bytes(fmt_struct.vcs_guid)
                    )
                    capabilities.formats.append(fmt)

                    # Extract resolution and fps
                    res = (fmt_struct.w, fmt_struct.h)
                    if res not in capabilities.resolutions:
                        capabilities.resolutions.append(res)

                    if fmt_struct.fps not in capabilities.fps_values:
                        capabilities.fps_values.append(float(fmt_struct.fps))

                # Sort
                capabilities.resolutions.sort()
                capabilities.fps_values.sort()

        except Exception as e:
            print(f"Capabilities retrieval exception: {e}")

        # Return defaults if retrieval failed
        if not capabilities.resolutions:
            capabilities.resolutions = [(640, 480), (1280, 720), (1920, 1080)]
        if not capabilities.fps_values:
            capabilities.fps_values = [24.0, 30.0, 60.0]

        return capabilities

    def open_device(self, device_index: int, hwnd: int = 0) -> bool:
        """Open camera device"""
        if not IS_WINDOWS or not self.initialized:
            return False

        try:
            # Set callback function
            def frame_callback(sample_time: float, buf: ctypes.POINTER(ctypes.c_uint8),
                             buf_size: int, ptr_class: ctypes.c_void_p) -> int:
                try:
                    # Convert buffer to numpy array
                    buffer_array = np.ctypeslib.as_array(buf, shape=(buf_size,))

                    # Parse frame according to current format
                    if self.current_format:
                        if self.current_format.color_space == 'YUY2':
                            # YUY2 format: width x height x 2
                            expected_size = self.current_format.width * self.current_format.height * 2
                            if buf_size == expected_size:
                                # Convert to BGR
                                yuy2_img = buffer_array.reshape(
                                    (self.current_format.height, self.current_format.width, 2)
                                )
                                bgr_img = cv2.cvtColor(yuy2_img, cv2.COLOR_YUV2BGR_YUY2)

                                # Put into queue (non-blocking)
                                try:
                                    self.frame_queue.put_nowait(bgr_img)
                                except queue.Full:
                                    pass
                        elif self.current_format.color_space == 'MJPG':
                            # MJPEG format: decode directly
                            try:
                                bgr_img = cv2.imdecode(buffer_array, cv2.IMREAD_COLOR)
                                if bgr_img is not None:
                                    try:
                                        self.frame_queue.put_nowait(bgr_img)
                                    except queue.Full:
                                        pass
                            except:
                                pass

                    return 0
                except Exception as e:
                    print(f"Frame callback exception: {e}")
                    return -1

            # Create callback function pointer
            self.callback_ptr = self.CAP_GRABBER(frame_callback)

            # Open device
            if self.dll.OpenCamera(device_index, hwnd, self.callback_ptr, None):
                self.current_device = device_index
                return True
            else:
                print("Device open failed")
                return False

        except Exception as e:
            print(f"Device open exception: {e}")
            return False

    def close_device(self, device_index: int) -> bool:
        """Close camera device"""
        if not IS_WINDOWS or not self.initialized:
            return True

        try:
            self.stop_capture()
            return bool(self.dll.CloseCamera(device_index))
        except Exception as e:
            print(f"Device close exception: {e}")
            return False

    def set_format(self, device_index: int, format_index: int) -> bool:
        """Set video format"""
        if not IS_WINDOWS or not self.initialized:
            return False

        try:
            if self.dll.SetFormat(device_index, format_index, 0):
                # Get current format info
                capabilities = self.get_capabilities(device_index)
                if format_index < len(capabilities.formats):
                    self.current_format = capabilities.formats[format_index]
                return True
            else:
                print("Format setting failed")
                return False
        except Exception as e:
            print(f"Format setting exception: {e}")
            return False

    def set_frame_rate(self, device_index: int, fps: int) -> bool:
        """Set frame rate"""
        if not IS_WINDOWS or not self.initialized:
            return False

        try:
            return bool(self.dll.SetFrameRate(device_index, fps))
        except Exception as e:
            print(f"Frame rate setting exception: {e}")
            return False

    def start_capture(self, device_index: int) -> bool:
        """Start capture"""
        if not IS_WINDOWS or not self.initialized:
            return False

        try:
            if self.dll.RunCamera(device_index):
                self.is_running = True
                return True
            else:
                print("Capture start failed")
                return False
        except Exception as e:
            print(f"Capture start exception: {e}")
            return False

    def stop_capture(self, device_index: Optional[int] = None) -> bool:
        """Stop capture"""
        if not IS_WINDOWS or not self.initialized:
            return True

        if device_index is None:
            device_index = self.current_device

        if device_index is None:
            return True

        try:
            self.is_running = False
            return bool(self.dll.StopCamera(device_index))
        except Exception as e:
            print(f"Capture stop exception: {e}")
            return False

    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get a frame"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_available(self) -> bool:
        """Check if SDK is available"""
        return IS_WINDOWS and self.initialized

    def get_frame_rate(self, device_index: int) -> int:
        """Get current frame rate"""
        if not IS_WINDOWS or not self.initialized:
            return 0

        try:
            return self.dll.getFrameRate(device_index)
        except:
            return 0

    def get_bitrate(self, device_index: int) -> int:
        """Get current bitrate"""
        if not IS_WINDOWS or not self.initialized:
            return 0

        try:
            return self.dll.getBitrate(device_index)
        except:
            return 0


# OpenCV Fallback Implementation
class OpenCVFallbackCapture:
    """OpenCV fallback camera capture class"""

    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.capture_thread: Optional[threading.Thread] = None
        self.devices: List[CameraDevice] = []

    def initialize(self) -> bool:
        return True

    def uninitialize(self) -> bool:
        self.stop_capture()
        return True

    def enum_devices(self) -> List[CameraDevice]:
        """Enumerate devices (OpenCV method)"""
        self.devices = []
        for i in range(10):  # Check first 10 indices
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2)
            if cap.isOpened():
                name = f"Camera {i}"
                try:
                    # Try to get device name (may not be available)
                    backend_name = cap.getBackendName()
                    if backend_name:
                        name += f" ({backend_name})"
                except:
                    pass

                device = CameraDevice(
                    vidpid="",  # OpenCV doesn't provide VID/PID
                    name=name
                )
                self.devices.append(device)
                cap.release()
        return self.devices

    def get_capabilities(self, device_index: int) -> CameraCapabilities:
        """Get camera capabilities (OpenCV method)"""
        capabilities = CameraCapabilities()

        # Common resolutions
        capabilities.resolutions = [
            (320, 240), (640, 480), (800, 600),
            (1024, 768), (1280, 720), (1280, 960),
            (1920, 1080), (2560, 1440), (3840, 2160)
        ]

        # Common frame rates
        capabilities.fps_values = [15, 24, 25, 30, 48, 50, 60, 120]

        return capabilities

    def open_device(self, device_index: int, hwnd: int = 0) -> bool:
        """Open device (OpenCV method)"""
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
        self.cap = cv2.VideoCapture(device_index, backend)
        return self.cap.isOpened()

    def close_device(self, device_index: int) -> bool:
        """Close device"""
        if self.cap:
            self.cap.release()
            self.cap = None
        return True

    def set_format(self, device_index: int, format_index: int) -> bool:
        """Set format (simplified implementation)"""
        # OpenCV doesn't directly support format index, just return success
        return True

    def set_frame_rate(self, device_index: int, fps: int) -> bool:
        """Set frame rate"""
        if self.cap:
            return self.cap.set(cv2.CAP_PROP_FPS, fps)
        return False

    def start_capture(self, device_index: int) -> bool:
        """Start capture"""
        if not self.cap or not self.cap.isOpened():
            return False

        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_thread, daemon=True)
        self.capture_thread.start()
        return True

    def stop_capture(self, device_index: Optional[int] = None) -> bool:
        """Stop capture"""
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        return True

    def _capture_thread(self):
        """Capture thread"""
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass
            else:
                time.sleep(0.01)  # Avoid high CPU usage

    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get frame"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_available(self) -> bool:
        """OpenCV is always available"""
        return True

    def get_frame_rate(self, device_index: int) -> int:
        """Get frame rate"""
        if self.cap:
            return int(self.cap.get(cv2.CAP_PROP_FPS))
        return 0

    def get_bitrate(self, device_index: int) -> int:
        """OpenCV doesn't provide bitrate info"""
        return 0


# Unified Camera Capture Interface
class UnifiedCameraCapture:
    """Unified camera capture interface, automatically selects best implementation"""

    def __init__(self):
        self.sdk_capture = SDKCameraCapture()
        self.opencv_capture = OpenCVFallbackCapture()
        self.current_impl = None

        # Prefer SDK (if available)
        if self.sdk_capture.is_available():
            self.current_impl = self.sdk_capture
            print("Using CapLib SDK for camera capture")
        else:
            self.current_impl = self.opencv_capture
            print("Using OpenCV for camera capture")

    def initialize(self) -> bool:
        return self.current_impl.initialize()

    def uninitialize(self) -> bool:
        return self.current_impl.uninitialize()

    def enum_devices(self) -> List[CameraDevice]:
        return self.current_impl.enum_devices()

    def get_capabilities(self, device_index: int) -> CameraCapabilities:
        return self.current_impl.get_capabilities(device_index)

    def open_device(self, device_index: int, hwnd: int = 0) -> bool:
        return self.current_impl.open_device(device_index, hwnd)

    def close_device(self, device_index: int) -> bool:
        return self.current_impl.close_device(device_index)

    def set_format(self, device_index: int, format_index: int) -> bool:
        return self.current_impl.set_format(device_index, format_index)

    def set_frame_rate(self, device_index: int, fps: int) -> bool:
        return self.current_impl.set_frame_rate(device_index, fps)

    def start_capture(self, device_index: int) -> bool:
        return self.current_impl.start_capture(device_index)

    def stop_capture(self, device_index: Optional[int] = None) -> bool:
        return self.current_impl.stop_capture(device_index)

    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        return self.current_impl.get_frame(timeout)

    def is_using_sdk(self) -> bool:
        """Check if SDK is being used"""
        return self.current_impl == self.sdk_capture

    def get_frame_rate(self, device_index: int) -> int:
        return self.current_impl.get_frame_rate(device_index)

    def get_bitrate(self, device_index: int) -> int:
        return self.current_impl.get_bitrate(device_index)
