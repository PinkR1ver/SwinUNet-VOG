#!/usr/bin/env python3
"""
下载 Web Demo 所需的所有依赖到本地
避免依赖外部 CDN，解决 VPN 环境下加载慢的问题
"""

import os
import urllib.request
import sys

def download_file(url, output_path):
    """下载文件并显示进度"""
    print(f"正在下载: {os.path.basename(output_path)}")
    print(f"  从: {url}")
    
    try:
        # 下载文件
        urllib.request.urlretrieve(url, output_path)
        
        # 显示文件大小
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  ✓ 完成 ({size_mb:.2f} MB)\n")
        return True
        
    except Exception as e:
        print(f"  ✗ 失败: {e}\n")
        return False

def main():
    print("=" * 60)
    print("下载 Web Demo 依赖文件")
    print("=" * 60)
    print()
    
    # 确保目录存在
    os.makedirs("js/lib", exist_ok=True)
    
    # 需要下载的文件列表
    # 使用 1.16.3（稳定版本）
    ort_version = "1.16.3"
    files = [
        # ONNX Runtime Web
        {
            "url": f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ort_version}/dist/ort.min.js",
            "output": "js/lib/ort.min.js"
        },
        {
            "url": f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ort_version}/dist/ort-wasm.wasm",
            "output": "js/lib/ort-wasm.wasm"
        },
        {
            "url": f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ort_version}/dist/ort-wasm-simd.wasm",
            "output": "js/lib/ort-wasm-simd.wasm"
        },
        {
            "url": f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ort_version}/dist/ort-wasm-threaded.wasm",
            "output": "js/lib/ort-wasm-threaded.wasm"
        },
        {
            "url": f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ort_version}/dist/ort-wasm-simd-threaded.wasm",
            "output": "js/lib/ort-wasm-simd-threaded.wasm"
        },
        
        # MediaPipe Face Mesh - Complete package
        {
            "url": "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/face_mesh.js",
            "output": "js/lib/face_mesh.js"
        },
        {
            "url": "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/face_mesh.binarypb",
            "output": "js/lib/face_mesh.binarypb"
        },
        {
            "url": "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/face_mesh_solution_packed_assets.data",
            "output": "js/lib/face_mesh_solution_packed_assets.data"
        },
        {
            "url": "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/face_mesh_solution_packed_assets_loader.js",
            "output": "js/lib/face_mesh_solution_packed_assets_loader.js"
        },
        {
            "url": "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/face_mesh_solution_simd_wasm_bin.js",
            "output": "js/lib/face_mesh_solution_simd_wasm_bin.js"
        },
        {
            "url": "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/face_mesh_solution_simd_wasm_bin.wasm",
            "output": "js/lib/face_mesh_solution_simd_wasm_bin.wasm"
        },
        {
            "url": "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/face_mesh_solution_wasm_bin.js",
            "output": "js/lib/face_mesh_solution_wasm_bin.js"
        },
        {
            "url": "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/face_mesh_solution_wasm_bin.wasm",
            "output": "js/lib/face_mesh_solution_wasm_bin.wasm"
        },
        
        # Chart.js
        {
            "url": "https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js",
            "output": "js/lib/chart.min.js"
        }
    ]
    
    # 下载所有文件
    success_count = 0
    fail_count = 0
    
    for file_info in files:
        if download_file(file_info["url"], file_info["output"]):
            success_count += 1
        else:
            fail_count += 1
    
    # 总结
    print("=" * 60)
    print(f"下载完成: {success_count} 成功, {fail_count} 失败")
    print("=" * 60)
    
    if fail_count > 0:
        print("\n⚠️ 部分文件下载失败，请检查网络连接后重试")
        sys.exit(1)
    else:
        print("\n✓ 所有依赖已下载到 js/lib/ 目录")
        print("✓ 现在可以在无网络环境下运行 demo.html")
        print("\n下一步：")
        print("  python -m http.server 8000")
        print("  然后访问: http://localhost:8000/demo.html")

if __name__ == "__main__":
    main()

