// SimpleRecorderDlg.cpp : 实现文件
// 简化的1920x1080 120fps录制程序

#include "pch.h"
#include "framework.h"
#include "SimpleRecorder.h"
#include "SimpleRecorderDlg.h"
#include "afxdialogex.h"
#include "cap_card.h"

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <filesystem>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

using namespace std;
using namespace CameraCAP;
using namespace chrono;
using namespace filesystem;

// 全局变量
CaptureDevice g_devices[16];
uint32_t g_deviceCount = 0;
vcVideoFormat g_formats[64];
uint16_t g_formatCount = 0;
int g_selectedDevice = 0;
int g_selectedFormat = -1;
bool g_isRecording = false;
bool g_isPreviewing = false;
HWND g_previewHandle = NULL;
ofstream g_outputFile;
mutex g_fileMutex;

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

// CSimpleRecorderDlg 对话框
CSimpleRecorderDlg::CSimpleRecorderDlg(CWnd* pParent /*=nullptr*/)
	: CDialog(IDD_SIMPLERECORDER_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CSimpleRecorderDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_PREVIEW, m_previewWindow);
}

BEGIN_MESSAGE_MAP(CSimpleRecorderDlg, CDialog)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_WM_CLOSE()
	ON_BN_CLICKED(IDC_START_PREVIEW, &CSimpleRecorderDlg::OnBnClickedStartPreview)
	ON_BN_CLICKED(IDC_STOP_PREVIEW, &CSimpleRecorderDlg::OnBnClickedStopPreview)
	ON_BN_CLICKED(IDC_START_RECORD, &CSimpleRecorderDlg::OnBnClickedStartRecord)
	ON_BN_CLICKED(IDC_STOP_RECORD, &CSimpleRecorderDlg::OnBnClickedStopRecord)
	ON_WM_TIMER()
END_MESSAGE_MAP()

// CSimpleRecorderDlg 消息处理程序

BOOL CSimpleRecorderDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// 初始化CapLib
	if (!InitLib())
	{
		MessageBox(_T("Failed to initialize CapLib!"), _T("Error"), MB_ICONERROR);
		EndDialog(IDCANCEL);
		return FALSE;
	}

	// 枚举摄像头
	if (!EnumCameras(g_devices, ARRAYSIZE(g_devices), &g_deviceCount))
	{
		MessageBox(_T("No cameras found!"), _T("Error"), MB_ICONERROR);
		EndDialog(IDCANCEL);
		return FALSE;
	}

	if (g_deviceCount == 0)
	{
		MessageBox(_T("No cameras found!"), _T("Error"), MB_ICONERROR);
		EndDialog(IDCANCEL);
		return FALSE;
	}

	// 自动选择第一个设备
	g_selectedDevice = 0;

	// 打开摄像头（用于格式枚举）
	if (!OpenCamera(g_selectedDevice, NULL, FrameCallback, NULL))
	{
		MessageBox(_T("Failed to open camera!"), _T("Error"), MB_ICONERROR);
		EndDialog(IDCANCEL);
		return FALSE;
	}

	// 获取支持的格式
	if (!GetFormats(g_selectedDevice, g_formats, &g_formatCount, ARRAYSIZE(g_formats)))
	{
		MessageBox(_T("Failed to get formats!"), _T("Error"), MB_ICONERROR);
		CloseCamera(g_selectedDevice);
		EndDialog(IDCANCEL);
		return FALSE;
	}

	// 查找1920x1080 120fps格式
	for (uint16_t i = 0; i < g_formatCount; i++)
	{
		if (g_formats[i].w == 1920 && g_formats[i].h == 1080 && abs(g_formats[i].fps - 120.0f) < 1.0f)
		{
			g_selectedFormat = i;
			break;
		}
	}

	if (g_selectedFormat == -1)
	{
		// 如果找不到120fps，尝试找最高fps的1920x1080格式
		float maxFps = 0.0f;
		for (uint16_t i = 0; i < g_formatCount; i++)
		{
			if (g_formats[i].w == 1920 && g_formats[i].h == 1080 && g_formats[i].fps > maxFps)
			{
				maxFps = g_formats[i].fps;
				g_selectedFormat = i;
			}
		}
	}

	if (g_selectedFormat == -1)
	{
		MessageBox(_T("1920x1080 resolution not supported!"), _T("Error"), MB_ICONERROR);
		CloseCamera(g_selectedDevice);
		EndDialog(IDCANCEL);
		return FALSE;
	}

	// 关闭摄像头（稍后重新打开用于预览）
	CloseCamera(g_selectedDevice);

	// 更新UI
	CString info;
	info.Format(_T("Ready to record 1920x1080 @ %.0f FPS"), g_formats[g_selectedFormat].fps);
	SetDlgItemText(IDC_STATUS, info);

	// 设置按钮状态
	GetDlgItem(IDC_START_PREVIEW)->EnableWindow(TRUE);
	GetDlgItem(IDC_STOP_PREVIEW)->EnableWindow(FALSE);
	GetDlgItem(IDC_START_RECORD)->EnableWindow(FALSE);
	GetDlgItem(IDC_STOP_RECORD)->EnableWindow(FALSE);

	// 设置定时器用于状态更新
	SetTimer(1, 1000, NULL);

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CSimpleRecorderDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		// TODO: 在此添加命令处理程序代码
	}
	else
	{
		CDialog::OnSysCommand(nID, LPARAM);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CSimpleRecorderDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialog::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CSimpleRecorderDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void CSimpleRecorderDlg::OnClose()
{
	// 清理资源
	if (g_isRecording)
	{
		OnBnClickedStopRecord();
	}
	if (g_isPreviewing)
	{
		OnBnClickedStopPreview();
	}

	UnInitLib();
	CDialog::OnClose();
}

void CSimpleRecorderDlg::OnBnClickedStartPreview()
{
	if (g_isPreviewing) return;

	// 打开摄像头用于预览
	g_previewHandle = m_previewWindow.GetSafeHwnd();
	if (!OpenCamera(g_selectedDevice, g_previewHandle, FrameCallback, NULL))
	{
		MessageBox(_T("Failed to open camera for preview!"), _T("Error"), MB_ICONERROR);
		return;
	}

	// 设置格式
	if (!SetFormat(g_selectedDevice, g_selectedFormat, g_previewHandle))
	{
		MessageBox(_T("Failed to set format!"), _T("Error"), MB_ICONERROR);
		CloseCamera(g_selectedDevice);
		return;
	}

	// 开始预览
	if (!RunCamera(g_selectedDevice))
	{
		MessageBox(_T("Failed to start preview!"), _T("Error"), MB_ICONERROR);
		CloseCamera(g_selectedDevice);
		return;
	}

	g_isPreviewing = true;

	// 更新UI
	GetDlgItem(IDC_START_PREVIEW)->EnableWindow(FALSE);
	GetDlgItem(IDC_STOP_PREVIEW)->EnableWindow(TRUE);
	GetDlgItem(IDC_START_RECORD)->EnableWindow(TRUE);
	GetDlgItem(IDC_STOP_RECORD)->EnableWindow(FALSE);

	SetDlgItemText(IDC_STATUS, _T("Previewing..."));
}

void CSimpleRecorderDlg::OnBnClickedStopPreview()
{
	if (!g_isPreviewing) return;

	// 停止预览
	StopCamera(g_selectedDevice);
	CloseCamera(g_selectedDevice);

	g_isPreviewing = false;

	// 更新UI
	GetDlgItem(IDC_START_PREVIEW)->EnableWindow(TRUE);
	GetDlgItem(IDC_STOP_PREVIEW)->EnableWindow(FALSE);
	GetDlgItem(IDC_START_RECORD)->EnableWindow(FALSE);
	GetDlgItem(IDC_STOP_RECORD)->EnableWindow(FALSE);

	CString info;
	info.Format(_T("Ready to record 1920x1080 @ %.0f FPS"), g_formats[g_selectedFormat].fps);
	SetDlgItemText(IDC_STATUS, info);
}

void CSimpleRecorderDlg::OnBnClickedStartRecord()
{
	if (g_isRecording) return;

	// 生成输出文件名
	auto now = system_clock::now();
	auto time = system_clock::to_time_t(now);
	tm tm_time;
	localtime_s(&tm_time, &time);

	char filename[256];
	strftime(filename, sizeof(filename), "recording_%Y%m%d_%H%M%S.raw", &tm_time);

	// 打开输出文件
	g_outputFile.open(filename, ios::binary);
	if (!g_outputFile.is_open())
	{
		MessageBox(_T("Failed to create output file!"), _T("Error"), MB_ICONERROR);
		return;
	}

	g_isRecording = true;

	// 更新UI
	GetDlgItem(IDC_START_RECORD)->EnableWindow(FALSE);
	GetDlgItem(IDC_STOP_RECORD)->EnableWindow(TRUE);

	CString status;
	status.Format(_T("Recording to %s..."), filename);
	SetDlgItemText(IDC_STATUS, status);
}

void CSimpleRecorderDlg::OnBnClickedStopRecord()
{
	if (!g_isRecording) return;

	g_isRecording = false;

	// 关闭文件
	if (g_outputFile.is_open())
	{
		g_outputFile.close();
	}

	// 更新UI
	GetDlgItem(IDC_START_RECORD)->EnableWindow(TRUE);
	GetDlgItem(IDC_STOP_RECORD)->EnableWindow(FALSE);

	CString info;
	info.Format(_T("Recording stopped. Ready to record 1920x1080 @ %.0f FPS"), g_formats[g_selectedFormat].fps);
	SetDlgItemText(IDC_STATUS, info);
}

void CSimpleRecorderDlg::OnTimer(UINT_PTR nIDEvent)
{
	if (nIDEvent == 1)
	{
		// 更新状态显示
		if (g_isRecording)
		{
			static int frameCount = 0;
			static auto startTime = steady_clock::now();

			frameCount++;

			auto currentTime = steady_clock::now();
			auto duration = duration_cast<seconds>(currentTime - startTime).count();

			if (duration > 0)
			{
				float fps = static_cast<float>(frameCount) / duration;
				CString status;
				status.Format(_T("Recording... %d frames, %.1f FPS"), frameCount, fps);
				SetDlgItemText(IDC_STATUS, status);
			}
		}
	}

	CDialog::OnTimer(nIDEvent);
}
