# SwinUNet-VOG: Gaze Estimation with SwinUNet

This project implements a SwinUNet-based model for gaze point estimation using the MPIIGaze dataset.

## Dataset

The project uses the MPIIGaze dataset, which contains:
- 15 participants with 521 days of recording
- Eye images: 36×60 pixels
- 3D gaze vectors
- Two evaluation protocols:
  - Cross-subject evaluation (default)
  - Person-specific evaluation

See `mpiigaze_summary.md` for detailed dataset information.

## Project Structure

```
SwinUNet-VOG/
├── data.py           # Dataset loading and preprocessing
├── model.py          # SwinUNet model architecture
├── train.py          # Training script
├── test.py           # Evaluation script
├── requirements.txt  # Dependencies
├── README.md         # This file
├── mpiigaze_summary.md  # Dataset documentation
└── MPIIGaze/         # Dataset folder (not included in repo)
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Place the MPIIGaze dataset in the project root:
```
SwinUNet-VOG/
└── MPIIGaze/
    ├── Data/
    │   └── Normalized/
    │       ├── p00/
    │       ├── p01/
    │       └── ...
    ├── Evaluation Subset/
    └── Annotation Subset/
```

## Usage

### Training

**Cross-subject evaluation (default):**
```bash
python train.py --data_dir MPIIGaze/Data/Normalized --batch_size 64 --epochs 50 --lr 0.001
```

**Person-specific evaluation:**
```bash
python train.py --person_specific --person_id p00 --batch_size 64 --epochs 50 --lr 0.001
```

**Training arguments:**
- `--data_dir`: Path to normalized data directory
- `--batch_size`: Batch size (default: 64)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--device`: Device to use (default: auto-detect)
- `--num_workers`: Number of data loading workers (default: 4)
- `--save_dir`: Directory to save checkpoints (default: checkpoints)
- `--person_specific`: Use person-specific protocol
- `--person_id`: Person ID for person-specific evaluation
- `--resume`: Path to checkpoint to resume training

### Testing

**Cross-subject evaluation:**
```bash
python test.py --checkpoint checkpoints/checkpoint_best.pth --data_dir MPIIGaze/Data/Normalized
```

**Person-specific evaluation:**
```bash
python test.py --checkpoint checkpoints/checkpoint_best.pth --person_specific --person_id p00
```

**Testing arguments:**
- `--checkpoint`: Path to model checkpoint (required)
- `--data_dir`: Path to normalized data directory
- `--batch_size`: Batch size for evaluation (default: 64)
- `--device`: Device to use (default: auto-detect)
- `--num_workers`: Number of data loading workers (default: 4)
- `--save_dir`: Directory to save results (default: results)
- `--person_specific`: Use person-specific protocol
- `--person_id`: Person ID for person-specific evaluation
- `--no_plots`: Skip saving evaluation plots

### Evaluation Metrics

The model is evaluated using:
- **Mean Angular Error**: Average angular error in degrees
- **Median Angular Error**: Median angular error in degrees
- **95th/99th Percentile**: Percentile values for error distribution

The test script generates:
- Error distribution histogram
- Cumulative error distribution
- Predictions vs ground truth scatter plots

## Model Architecture

The SwinUNet model consists of:
1. **Patch Embedding**: Convolutional patch embedding
2. **Encoder**: 3 Swin Transformer blocks with downsampling
3. **Bottleneck**: Additional Swin Transformer block
4. **Decoder**: 2 Swin Transformer blocks with upsampling
5. **Head**: Fully connected layers for 3D gaze regression

Key features:
- Simplified Swin Transformer blocks with depthwise separable convolution
- U-Net-like encoder-decoder structure
- Dropout and batch normalization for regularization
- ~7.6M parameters

## Results

Training produces:
- `checkpoint_best.pth`: Best model based on validation angular error
- `checkpoint_latest.pth`: Latest checkpoint

Evaluation produces:
- `results.json`: Numerical evaluation metrics
- `error_distribution.png`: Histogram of angular errors
- `cumulative_distribution.png`: Cumulative error distribution
- `predictions_vs_targets.png`: Scatter plots comparing predictions to ground truth

## Customization

### Changing Model Architecture

Edit `model.py` to modify:
- Embedding dimension: `embed_dim`
- Number of layers: `depths`
- Number of heads: `num_heads`
- Window size: `window_size`
- Dropout rate: `drop_rate`

### Changing Training Parameters

Edit `train.py` or use command-line arguments to modify:
- Learning rate and optimizer
- Loss function
- Learning rate scheduler
- Data augmentation

### Changing Dataset

Modify `data.py` to use a different dataset:
- Implement custom `Dataset` class
- Implement custom data loaders
- Adjust preprocessing/augmentation

## Citation

If you use this code, please cite:

```
@inproceedings{zhang2015appearance,
  title={Appearance-based gaze estimation in the wild},
  author={Zhang, Xucong and Sugano, Yusuke and Fritz, Mario and Bulling, Andreas},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2015}
}
```

## License

This project is for educational and research purposes. Please check the MPIIGaze dataset license:
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License

## Contact

For questions or issues, please open an issue on the repository.

