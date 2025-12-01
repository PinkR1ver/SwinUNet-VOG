#!/usr/bin/env python3
"""
将 PyTorch SwinUNet 模型导出为 ONNX 格式

ONNX 格式可以直接在浏览器中使用 ONNX Runtime Web 运行，
无需转换为 TensorFlow.js，避免依赖兼容性问题。
"""

import os
import torch
import numpy as np

from model import SwinUNet


def export_to_onnx(checkpoint_path, output_path='models/swinunet.onnx'):
    """
    导出 PyTorch 模型为 ONNX 格式
    """
    print("=" * 60)
    print("Exporting SwinUNet to ONNX")
    print("=" * 60)
    
    # 加载模型
    device = torch.device('cpu')
    model = SwinUNet(
        img_size=(36, 60),
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 2],
        num_heads=[3, 6, 12],
        window_size=7,
        drop_rate=0.0
    )
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建虚拟输入
    dummy_input = torch.randn(1, 3, 36, 60)
    
    # 测试模型
    print("Testing model...")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # 导出 ONNX
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    print(f"\nExporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,  # 使用 PyTorch 推荐的版本
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"[OK] ONNX model saved to: {output_path}")
    
    # 验证 ONNX 模型
    try:
        import onnx
        print("\nVerifying ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("[OK] ONNX model is valid")
        
        # 打印模型信息
        print(f"\nModel info:")
        print(f"  IR version: {onnx_model.ir_version}")
        print(f"  Opset version: {onnx_model.opset_import[0].version}")
        print(f"  Producer: {onnx_model.producer_name}")
        
    except ImportError:
        print("⚠ onnx package not found, skipping verification")
    except Exception as e:
        print(f"⚠ Verification warning: {e}")
    
    # 计算文件大小
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nFile size: {file_size:.2f} MB")
    
    return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Export SwinUNet to ONNX')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_best.pth',
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, default='models/swinunet.onnx',
                        help='Output ONNX file path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        print("\nPlease train the model first:")
        print("  python train.py --epochs 50 --batch_size 64")
        return 1
    
    try:
        output_path = export_to_onnx(args.checkpoint, args.output)
        
        print("\n" + "=" * 60)
        print("[OK] Export completed successfully!")
        print("=" * 60)
        
        print(f"\nONNX model: {output_path}")
        
        print("\nUsage in JavaScript (ONNX Runtime Web):")
        print("```javascript")
        print("const session = await ort.InferenceSession.create('models/swinunet.onnx');")
        print("const input = new ort.Tensor('float32', inputData, [1, 3, 36, 60]);")
        print("const results = await session.run({ input });")
        print("```")
        
        print("\nNext steps:")
        print("  1. Use models/swinunet.onnx with ONNX Runtime Web")
        print("  2. See demo_onnx.html for example usage")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Export failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())

