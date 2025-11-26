#ifndef  CAP_CARD_H
#define  CAP_CARD_H


#include <stdio.h>
#include <string>
#include <windows.h>
//#include <wincodec.h>

namespace CameraCAP
{


#ifndef WinAPI
#define WinAPI __stdcall
#endif

//#ifndef WinAPI
//#define WinAPI	__stdcall
//#endif

//#ifdef DLL_EXPORTS
//#define DLLAPI __declspec(dllexport)
//#else
//#define DLLAPI __declspec(dllimport)
//#endif

	// 图像编码类型
	typedef enum class VideoColorSpace_t
	{
		Unknow,
		//H264,
		Mjpg,
		Yuy2
	}vcVideoColorSpace;

	// 图像信息
	typedef struct vcVideoFormat_t
	{
		vcVideoColorSpace vcs; // 图像编码类型
		uint16_t w, h; // 图像宽高
		float fps;	// 帧率
		GUID vcs_guid;
		vcVideoFormat_t() {
			vcs = vcVideoColorSpace::Unknow;
			w = 0;
			h = 0;
			fps = 0.0;
			memset(&vcs_guid, 0, sizeof(vcs_guid));
		}
	}vcVideoFormat;

	// 设备信息
	typedef struct CaptureDevice_t
	{
		char vidpid[9];
		wchar_t devName[256];
		CaptureDevice_t() {
			memset(vidpid, 0, 9);
			wmemset(devName, 0, 256);
		}
	}CaptureDevice;

	// 相机属性值ID
	enum Property
	{
		PowerLineFrequency = 13,
		Exposure = 14,
		Zoom = (Exposure + 1),
		Focus = (Zoom + 1),
		Pan = (Focus + 1),
		Tilt = (Pan + 1)
	};
	
	// 相机属性范围
	class VideoPropertyRange1 {
	public:
		long Min;
		long Max;
		long Step;
		long Default;
		long Flags;

		VideoPropertyRange1() : Min(0), Max(0), Step(0), Default(0), Flags(0) {}

		// 
		VideoPropertyRange1(long min, long max, long step, long def, long flags)
			: Min(min), Max(max), Step(step), Default(def), Flags(flags) {}
	};
	// 相机属性值
	class VideoProperty1 {
	public:
		long Value;
		long Flags;  // 

		VideoProperty1() : Value(0), Flags(0) {}
		VideoProperty1(long value, long flags) : Value(value), Flags(flags) {}
	};


	typedef int(WinAPI* CapGrabber)(double sampleTime, uint8_t* buf, long bufSize, void* ptrClass);


	bool  InitLib();

	bool  WinAPI UnInitLib();
	
	int getFrameRate(uint32_t devIndex);

	int getBitrate(uint32_t devIndex);

	bool  EnumCameras(CaptureDevice devs[], uint32_t arrSize_devs, uint32_t* devCount);

	bool  OpenCamera(uint32_t devIndex, HWND preWndHandle = 0, CapGrabber grab = 0, void* ptrClass = 0);


	
	bool Getprops(uint32_t devIndex, long prop, VideoProperty1& b);

	bool Getrange(uint32_t devIndex,long prop, VideoPropertyRange1 &b);

	bool Setprops(uint32_t devIndex, long props, int value,int flag);

	
	bool SaveImage(uint8_t* imageBuf, long imgSize, vcVideoFormat format, wchar_t* photo_path);

	bool SaveFrameAsJPG(uint8_t* pBuffer, long BufferLen, wchar_t* filename);

	bool compress_yuy2_to_jpeg(uint8_t* pBuffer, long BufferLen, int width, int height, char const* filename);

	bool decompress_jpeg_to_yuy2(uint8_t* jpeg_buffer, long jpeg_size, uint8_t* yuy2_buffer, long& size, int& width, int& height);

	bool UpdateVideoWind(uint32_t devIndex, HWND preWndHandle);

	bool ClearVideoWind(uint32_t devIndex);

	bool  WinAPI CloseCamera(uint32_t devIndex);

	bool  WinAPI RunCamera(uint32_t devIndex);

	bool  WinAPI StopCamera(uint32_t devIndex);

	bool  WinAPI GetFormats(uint32_t devIndex, vcVideoFormat formats[], uint16_t* formatNum, uint16_t arrSize_Formats);

	bool  WinAPI CurFormat(uint32_t devIndex, uint16_t* formatIndex);

	bool  WinAPI SetFormat(uint32_t devIndex, uint16_t formatIndex, HWND preWndHandle = 0);

	bool  WinAPI SetFrameRate(uint32_t devIndex, uint16_t fps);

	
}

#endif // ! CAP_CARD_H






