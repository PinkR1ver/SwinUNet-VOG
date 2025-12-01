#!/usr/bin/env python3
"""
自定义 HTTP 服务器，正确设置 WASM 文件的 MIME 类型
"""

import http.server
import socketserver
import mimetypes

# 注册 WASM 和 MediaPipe 相关 MIME 类型
mimetypes.add_type('application/wasm', '.wasm')
mimetypes.add_type('application/octet-stream', '.data')
mimetypes.add_type('application/octet-stream', '.binarypb')
mimetypes.add_type('application/javascript', '.js')

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # 添加 SharedArrayBuffer 所需的安全头（用于多线程 WASM）
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        
        # 添加 CORS 头
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        
        # 缓存控制
        if self.path.endswith('.wasm'):
            self.send_header('Cache-Control', 'public, max-age=31536000')
        
        super().end_headers()
    
    def guess_type(self, path):
        # 确保 WASM 文件使用正确的 MIME 类型
        if path.endswith('.wasm'):
            return 'application/wasm'
        return super().guess_type(path)
    
    def log_message(self, format, *args):
        # 自定义日志格式
        print(f"[{self.log_date_time_string()}] {format % args}")

PORT = 8000

print("=" * 60)
print("SwinUNet-VOG Web Demo 服务器")
print("=" * 60)
print(f"\n服务器启动在端口 {PORT}")
print(f"\n请访问:")
print(f"  诊断工具: http://localhost:{PORT}/diagnose.html")
print(f"  主演示:   http://localhost:{PORT}/demo.html")
print(f"\n按 Ctrl+C 停止服务器")
print("=" * 60)
print()

with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n服务器已停止")

