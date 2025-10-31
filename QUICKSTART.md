# Quick Start Guide

This guide will help you get started with training and evaluating the SwinUNet gaze estimation model.

## Step 1: Install Dependencies

```bash
pip install torch torchvision numpy scipy matplotlib pillow
```

Or install all at once:
```bash
pip install -r requirements.txt
```

## Step 2: Verify Dataset

Make sure you have the MPIIGaze dataset in the correct location:
```
SwinUNet-VOG/
└── MPIIGaze/
    └── Data/
        └── Normalized/
            ├── p00/
            │   ├── day01.mat
            │   ├── day02.mat
            │   └── ...
            ├── p01/
            └── ...
```

Test dataset loading:
```bash
python data.py
```

Expected output:
- Number of participants and samples loaded
- Sample batch shape: [batch_size, 3, 36, 60]
- Gaze shape: [batch_size, 3]

## Step 3: Test Model

Verify the model works:
```bash
python model.py
```

Expected output:
- Model created successfully
- Total parameters: ~7.6M
- Input shape: [2, 3, 36, 60]
- Output shape: [2, 3]

## Step 4: Train the Model

### Quick training (small test):
```bash
python train.py --epochs 5 --batch_size 32
```

### Full training (recommended):
```bash
python train.py --epochs 50 --batch_size 64 --lr 0.001
```

### Training with GPU:
```bash
python train.py --epochs 50 --batch_size 64 --device cuda
```

### Person-specific training:
```bash
python train.py --person_specific --person_id p00 --epochs 50
```

Training outputs:
- Checkpoints saved to `checkpoints/`:
  - `checkpoint_latest.pth` - Latest checkpoint
  - `checkpoint_best.pth` - Best model based on validation

## Step 5: Evaluate the Model

### Evaluate on test set:
```bash
python test.py --checkpoint checkpoints/checkpoint_best.pth
```

### Person-specific evaluation:
```bash
python test.py --checkpoint checkpoints/checkpoint_best.pth --person_specific --person_id p00
```

Evaluation outputs:
- Results saved to `results/`:
  - `results.json` - Numerical metrics
  - `error_distribution.png` - Error histogram
  - `cumulative_distribution.png` - Cumulative distribution
  - `predictions_vs_targets.png` - Prediction scatter plots

## Common Issues and Solutions

### Issue: Out of Memory Error
**Solution**: Reduce batch size
```bash
python train.py --batch_size 32
```

### Issue: Training is too slow
**Solution**: Reduce number of workers or disable GPU if slow
```bash
python train.py --num_workers 0 --device cpu
```

### Issue: CUDA not available
**Solution**: Install CUDA-enabled PyTorch or use CPU
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Or use CPU:
```bash
python train.py --device cpu
```

### Issue: Import errors
**Solution**: Make sure all dependencies are installed
```bash
pip install -r requirements.txt
```

## Training Progress

During training, you'll see:
```
Epoch: 1, Batch: 0/17526, Loss: 0.123456, Angle: 45.67°
Epoch: 1, Batch: 500/17526, Loss: 0.098765, Angle: 38.42°
...

Epoch 1/50 Summary:
  Train Loss: 0.095432, Train Angle: 36.23°
  Val Loss: 0.082345, Val Angle: 32.15°
  Time: 125.45s
```

## Evaluation Metrics

Evaluation outputs include:
- **Mean Angular Error**: Average error in degrees
- **Median Angular Error**: Middle error value
- **95th/99th Percentile**: Threshold error values
- **Visualizations**: Distribution plots and scatter plots

Good results typically show:
- Mean angular error < 10-15°
- 95th percentile < 20-30°

## Advanced Usage

### Resume training from checkpoint:
```bash
python train.py --resume checkpoints/checkpoint_latest.pth --epochs 50
```

### Custom configuration:
```bash
python train.py \
    --batch_size 128 \
    --epochs 100 \
    --lr 0.0001 \
    --save_dir my_checkpoints \
    --num_workers 8
```

### Skip plots in evaluation:
```bash
python test.py --checkpoint checkpoints/checkpoint_best.pth --no_plots
```

## Next Steps

1. Experiment with different model architectures in `model.py`
2. Try different data augmentation in `data.py`
3. Adjust hyperparameters in `train.py`
4. Implement additional evaluation metrics in `test.py`

## Getting Help

If you encounter issues:
1. Check that all dependencies are installed
2. Verify dataset is in correct location
3. Review error messages carefully
4. Try with smaller batch size or fewer workers
5. Check system resources (RAM, GPU memory)

For more information, see `README.md` and `mpiigaze_summary.md`.

