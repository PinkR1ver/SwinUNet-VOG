// SimpleRecorder.cpp : 控制台版本的1920x1080 120fps录制程序
// 使用CapLib SDK，支持VS Code编译

#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <mutex>
#include <chrono>
#include <atomic>
#include <csignal>
#include <filesystem>

#include "cap_card.h"

using namespace std;
using namespace CameraCAP;
using namespace chrono;
namespace fs = filesystem;

// 全局变量
CaptureDevice g_devices[16];
uint32_t g_deviceCount = 0;
vcVideoFormat g_formats[64];
uint16_t g_formatCount = 0;
int g_selectedDevice = 0;
int g_selectedFormat = -1;
atomic<bool> g_isRecording(false);
atomic<bool> g_isRunning(false);
ofstream g_outputFile;
mutex g_fileMutex;

// 信号处理
volatile sig_atomic_t g_signalStatus = 0;

void signalHandler(int signal) {
    g_signalStatus = signal;
    g_isRecording = false;
    g_isRunning = false;
}

// 回调函数用于接收帧数据
int WinAPI FrameCallback(double sampleTime, uint8_t* buf, long bufSize, void* ptrClass)
{
    if (g_isRecording && g_outputFile.is_open())
    {
        lock_guard<mutex> lock(g_fileMutex);
        g_outputFile.write(reinterpret_cast<char*>(buf), bufSize);
    }
    return 0;
}

// 显示可用设备
void showDevices() {
    cout << "Available devices:" << endl;
    for (uint32_t i = 0; i < g_deviceCount; i++) {
        wcout << i << ": " << g_devices[i].devName << endl;
    }
}

// 显示可用格式
void showFormats() {
    cout << "Available formats for device " << g_selectedDevice << ":" << endl;
    for (uint16_t i = 0; i < g_formatCount; i++) {
        string colorSpace;
        switch (g_formats[i].vcs) {
            case vcVideoColorSpace::Yuy2: colorSpace = "YUY2"; break;
            case vcVideoColorSpace::Mjpg: colorSpace = "MJPG"; break;
            default: colorSpace = "Unknown"; break;
        }
        cout << i << ": " << g_formats[i].w << "x" << g_formats[i].h
             << " @ " << g_formats[i].fps << " FPS (" << colorSpace << ")" << endl;
    }
}

// 选择设备
bool selectDevice() {
    if (g_deviceCount == 0) {
        cout << "No devices found!" << endl;
        return false;
    }

    if (g_deviceCount == 1) {
        g_selectedDevice = 0;
        wcout << "Auto-selected device: " << g_devices[0].devName << endl;
        return true;
    }

    showDevices();
    cout << "Select device (0-" << (g_deviceCount - 1) << "): ";
    int device;
    cin >> device;

    if (device < 0 || device >= (int)g_deviceCount) {
        cout << "Invalid device selection!" << endl;
        return false;
    }

    g_selectedDevice = device;
    wcout << "Selected device: " << g_devices[device].devName << endl;
    return true;
}

// 选择格式（自动选择1920x1080的最佳格式）
bool selectFormat() {
    // 优先选择1920x1080 120fps
    for (uint16_t i = 0; i < g_formatCount; i++) {
        if (g_formats[i].w == 1920 && g_formats[i].h == 1080 &&
            abs(g_formats[i].fps - 120.0f) < 1.0f) {
            g_selectedFormat = i;
            cout << "Selected format: 1920x1080 @ 120 FPS" << endl;
            return true;
        }
    }

    // 如果没有120fps，选择1920x1080的最高fps
    float maxFps = 0.0f;
    for (uint16_t i = 0; i < g_formatCount; i++) {
        if (g_formats[i].w == 1920 && g_formats[i].h == 1080 && g_formats[i].fps > maxFps) {
            maxFps = g_formats[i].fps;
            g_selectedFormat = i;
        }
    }

    if (g_selectedFormat != -1) {
        cout << "Selected format: 1920x1080 @ " << maxFps << " FPS (highest available)" << endl;
        return true;
    }

    cout << "1920x1080 resolution not supported!" << endl;
    return false;
}

// 开始预览
bool startPreview() {
    if (!OpenCamera(g_selectedDevice, NULL, FrameCallback, NULL)) {
        cout << "Failed to open camera!" << endl;
        return false;
    }

    if (!SetFormat(g_selectedDevice, g_selectedFormat, NULL)) {
        cout << "Failed to set format!" << endl;
        CloseCamera(g_selectedDevice);
        return false;
    }

    if (!RunCamera(g_selectedDevice)) {
        cout << "Failed to start camera!" << endl;
        CloseCamera(g_selectedDevice);
        return false;
    }

    g_isRunning = true;
    cout << "Preview started successfully" << endl;
    return true;
}

// 停止预览
void stopPreview() {
    if (g_isRunning) {
        StopCamera(g_selectedDevice);
        CloseCamera(g_selectedDevice);
        g_isRunning = false;
        cout << "Preview stopped" << endl;
    }
}

// 开始录制
bool startRecording() {
    if (!g_isRunning) {
        cout << "Camera not running! Start preview first." << endl;
        return false;
    }

    // 生成文件名
    auto now = system_clock::now();
    auto time = system_clock::to_time_t(now);
    tm tm_time;
    localtime_s(&tm_time, &time);

    char filename[256];
    strftime(filename, sizeof(filename), "recording_%Y%m%d_%H%M%S.raw", &tm_time);

    g_outputFile.open(filename, ios::binary);
    if (!g_outputFile.is_open()) {
        cout << "Failed to create output file!" << endl;
        return false;
    }

    g_isRecording = true;
    cout << "Recording started: " << filename << endl;
    return true;
}

// 停止录制
void stopRecording() {
    if (g_isRecording) {
        g_isRecording = false;

        if (g_outputFile.is_open()) {
            g_outputFile.close();
            cout << "Recording stopped" << endl;
        }
    }
}

// 显示帮助
void showHelp() {
    cout << "Simple Recorder - Console Version" << endl;
    cout << "Commands:" << endl;
    cout << "  p      - Start preview" << endl;
    cout << "  s      - Stop preview" << endl;
    cout << "  r      - Start recording" << endl;
    cout << "  t      - Stop recording" << endl;
    cout << "  d      - Show devices" << endl;
    cout << "  f      - Show formats" << endl;
    cout << "  h      - Show this help" << endl;
    cout << "  q      - Quit" << endl;
}

int main(int argc, char* argv[])
{
    // 设置信号处理
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    cout << "Simple Recorder - 1920x1080 High FPS Console Version" << endl;
    cout << "Press Ctrl+C to exit" << endl << endl;

    // 初始化CapLib
    if (!InitLib()) {
        cout << "Failed to initialize CapLib!" << endl;
        return 1;
    }

    // 枚举设备
    if (!EnumCameras(g_devices, ARRAYSIZE(g_devices), &g_deviceCount)) {
        cout << "No cameras found!" << endl;
        UnInitLib();
        return 1;
    }

    // 选择设备
    if (!selectDevice()) {
        UnInitLib();
        return 1;
    }

    // 打开设备获取格式信息
    if (!OpenCamera(g_selectedDevice, NULL, NULL, NULL)) {
        cout << "Failed to open camera for format enumeration!" << endl;
        UnInitLib();
        return 1;
    }

    // 获取格式
    if (!GetFormats(g_selectedDevice, g_formats, &g_formatCount, ARRAYSIZE(g_formats))) {
        cout << "Failed to get formats!" << endl;
        CloseCamera(g_selectedDevice);
        UnInitLib();
        return 1;
    }

    CloseCamera(g_selectedDevice);

    // 选择格式
    if (!selectFormat()) {
        UnInitLib();
        return 1;
    }

    showHelp();

    // 主循环
    string command;
    while (!g_signalStatus) {
        cout << "> ";
        getline(cin, command);

        if (command.empty()) continue;

        char cmd = tolower(command[0]);

        switch (cmd) {
            case 'p':
                if (!g_isRunning) {
                    startPreview();
                } else {
                    cout << "Preview already running" << endl;
                }
                break;

            case 's':
                stopPreview();
                break;

            case 'r':
                if (!g_isRecording) {
                    startRecording();
                } else {
                    cout << "Already recording" << endl;
                }
                break;

            case 't':
                stopRecording();
                break;

            case 'd':
                showDevices();
                break;

            case 'f':
                showFormats();
                break;

            case 'h':
                showHelp();
                break;

            case 'q':
                cout << "Exiting..." << endl;
                goto cleanup;
                break;

            default:
                cout << "Unknown command. Type 'h' for help." << endl;
                break;
        }

        // 检查信号
        if (g_signalStatus) break;
    }

cleanup:
    stopRecording();
    stopPreview();
    UnInitLib();

    cout << "Goodbye!" << endl;
    return 0;
}
