// SimpleRecorderDlg.h: 头文件
//

#pragma once

#include "cap_card.h"
using namespace CameraCAP;

// CSimpleRecorderDlg 对话框
class CSimpleRecorderDlg : public CDialog
{
// 构造
public:
	CSimpleRecorderDlg(CWnd* pParent = nullptr);	// 标准构造函数

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_SIMPLERECORDER_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持


// 实现
protected:
	HICON m_hIcon;
	CStatic m_previewWindow;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg void OnClose();
	afx_msg void OnBnClickedStartPreview();
	afx_msg void OnBnClickedStopPreview();
	afx_msg void OnBnClickedStartRecord();
	afx_msg void OnBnClickedStopRecord();
	afx_msg void OnTimer(UINT_PTR nIDEvent);

	DECLARE_MESSAGE_MAP()
};
