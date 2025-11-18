"""
view_video.py
-------------

简单的视频查看工具，支持：
- 直接播放 capture.avi
- 逐帧查看
- 提取眼睛检测结果
- 导出为 MP4
"""

import os
import sys
import cv2
import argparse
from pathlib import Path
import mediapipe as mp

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    from PIL import Image, ImageTk
except ImportError:
    tk = None


class VideoViewer:
    """简单的视频查看工具。"""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            print(f"❌ 无法打开文件: {video_path}")
            sys.exit(1)
        
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n{'='*60}")
        print(f"📹 视频信息")
        print(f"{'='*60}")
        print(f"文件：{Path(video_path).name}")
        print(f"分辨率：{self.width}×{self.height}")
        print(f"帧率：{self.fps:.2f} FPS")
        print(f"总帧数：{self.frame_count}")
        print(f"时长：{self.frame_count / self.fps:.2f} 秒")
        print(f"{'='*60}\n")
    
    def play(self):
        """播放视频。"""
        print("🎬 开始播放")
        print("按键说明:")
        print("  Space: 暂停/播放")
        print("  q: 退出")
        print("  →: 下一帧")
        print("  ←: 上一帧")
        print("  e: 导出为 MP4\n")
        
        paused = False
        current_frame = 0
        frame_buffer = None
        
        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("✅ 播放完成")
                    break
                current_frame += 1
                frame_buffer = frame
            
            # 显示帧信息
            display_frame = frame.copy()
            info_text = f"Frame: {current_frame}/{self.frame_count} | {self.fps:.0f} FPS"
            cv2.putText(display_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Video Viewer", display_frame)
            
            key = cv2.waitKey(int(1000 / self.fps)) & 0xFF
            
            if key == ord('q'):
                print("❌ 用户退出")
                break
            elif key == ord(' '):
                paused = not paused
                status = "⏸️  暂停" if paused else "▶️  播放"
                print(status)
            elif key == ord('e'):
                print("\n🔄 导出为 MP4...")
                self.cap.release()
                output_path = self.video_path.replace('.avi', '_converted.mp4')
                self.export_to_mp4(output_path)
                cv2.destroyAllWindows()
                print("✅ 导出完成，退出播放")
                return
            elif key == 81:  # 左箭头
                current_frame = max(0, current_frame - 2)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                paused = True
            elif key == 83:  # 右箭头
                paused = False
        
        cv2.destroyAllWindows()
        self.cap.release()
    
    def export_to_mp4(self, output_path):
        """导出为 MP4 格式。"""
        print(f"\n🔄 正在转换为 MP4...")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        if not out.isOpened():
            print("❌ 无法创建输出文件")
            return False
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            out.write(frame)
            frame_idx += 1
            
            if frame_idx % 30 == 0:
                progress = (frame_idx / self.frame_count) * 100
                print(f"  进度: {progress:.1f}% ({frame_idx}/{self.frame_count})", end='\r')
        
        out.release()
        print(f"\n✅ 转换完成: {output_path}")
        print(f"   文件大小: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        return True


def main():
    parser = argparse.ArgumentParser(description="视频查看工具")
    parser.add_argument("video", nargs="?", help="视频文件路径")
    parser.add_argument("--export", "-e", help="导出为 MP4 格式")
    parser.add_argument("--info", "-i", action="store_true", help="仅显示文件信息")
    
    args = parser.parse_args()
    
    # 如果没指定文件，让用户选择
    if not args.video:
        if tk:
            root = tk.Tk()
            root.withdraw()
            video_path = filedialog.askopenfilename(
                title="选择视频文件",
                filetypes=[("视频文件", "*.avi *.mp4 *.mov"), ("所有文件", "*.*")]
            )
            root.destroy()
            
            if not video_path:
                print("❌ 未选择文件")
                sys.exit(1)
        else:
            print("❌ 请指定视频文件路径")
            print(f"   用法: python3 view_video.py <视频文件>")
            sys.exit(1)
    else:
        video_path = args.video
    
    if not os.path.exists(video_path):
        print(f"❌ 文件不存在: {video_path}")
        sys.exit(1)
    
    viewer = VideoViewer(video_path)
    
    if args.info:
        print("✅ 文件信息已显示")
        return
    
    if args.export:
        viewer.export_to_mp4(args.export)
        return
    
    # 播放视频
    viewer.play()


if __name__ == "__main__":
    main()

