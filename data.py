"""
Dataset loader for MPIIGaze gaze estimation.
Builds training, validation, and test sets from normalized data.
"""

import os
import numpy as np
import scipy.io
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


class MPIIGazeDataset(Dataset):
    """MPIIGaze Dataset for Gaze Estimation."""
    
    def __init__(self, data_dir, participants, transform=None, augment=False, preprocessor=None):
        """
        Args:
            data_dir: Path to MPIIGaze/Data/Normalized/
            participants: List of participant IDs (e.g., ['p00', 'p01'])
            transform: Optional transform to apply to images (deprecated, use preprocessor)
            augment: Whether to apply data augmentation
            preprocessor: Optional image preprocessor for handling different camera standards
        """
        self.data_dir = Path(data_dir)
        self.participants = participants
        self.transform = transform
        self.augment = augment
        self.preprocessor = preprocessor
        
        # Load all data
        self.images = []
        self.gaze_vectors = []
        self.participant_ids = []
        
        self._load_data()
    
    def _load_data(self):
        """Load all data from specified participants."""
        for participant in self.participants:
            participant_dir = self.data_dir / participant
            
            if not participant_dir.exists():
                print(f"Warning: Participant {participant} directory not found")
                continue
            
            # Get all day mat files
            mat_files = sorted(participant_dir.glob("day*.mat"))
            
            for mat_file in mat_files:
                try:
                    data = scipy.io.loadmat(str(mat_file))
                    
                    # Load left and right eye data
                    left_data = data['data'][0, 0]['left'][0]
                    right_data = data['data'][0, 0]['right'][0]
                    
                    # Extract gaze and images
                    left_gaze = left_data['gaze'][0]
                    left_images = left_data['image'][0]
                    right_gaze = right_data['gaze'][0]
                    right_images = right_data['image'][0]
                    
                    # Add left eye data
                    for i in range(len(left_gaze)):
                        img = left_images[i]
                        gaze = left_gaze[i]
                        
                        # Convert grayscale to RGB if needed
                        if img.ndim == 2:
                            img = np.stack([img, img, img], axis=0)  # (36, 60) -> (3, 36, 60)
                        elif img.ndim == 3 and img.shape[2] == 3:
                            img = img.transpose(2, 0, 1)  # (36, 60, 3) -> (3, 36, 60)
                        else:
                            img = img[np.newaxis, ...]  # Add channel dimension
                        
                        self.images.append(img)
                        self.gaze_vectors.append(gaze.astype(np.float32))
                        self.participant_ids.append(participant)
                    
                    # Add right eye data (already flipped in normalized data)
                    for i in range(len(right_gaze)):
                        img = right_images[i]
                        gaze = right_gaze[i]
                        
                        # Convert grayscale to RGB if needed
                        if img.ndim == 2:
                            img = np.stack([img, img, img], axis=0)
                        elif img.ndim == 3 and img.shape[2] == 3:
                            img = img.transpose(2, 0, 1)
                        else:
                            img = img[np.newaxis, ...]
                        
                        self.images.append(img)
                        self.gaze_vectors.append(gaze.astype(np.float32))
                        self.participant_ids.append(participant)
                        
                except Exception as e:
                    print(f"Error loading {mat_file}: {e}")
                    continue
        
        print(f"Loaded {len(self.images)} samples from {len(self.participants)} participants")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].copy()
        gaze = self.gaze_vectors[idx].copy()
        
        # Apply preprocessor if provided (for handling different camera standards)
        if self.preprocessor is not None:
            # Preprocessor expects numpy array and returns torch tensor
            image = self.preprocessor(image)
            gaze = torch.from_numpy(gaze).float()
        else:
            # Convert to torch tensors
            image = torch.from_numpy(image).float()
            gaze = torch.from_numpy(gaze).float()
            
            # Normalize image to [0, 1]
            if image.max() > 1:
                image = image / 255.0
        
        # Apply data augmentation if enabled
        if self.augment and np.random.rand() > 0.5:
            # Random horizontal flip
            image = torch.flip(image, [2])
            # Flip gaze x-component
            gaze[0] = -gaze[0]
        
        if self.augment and np.random.rand() > 0.5:
            # Random brightness
            brightness_factor = np.random.uniform(0.8, 1.2)
            image = image * brightness_factor
            image = torch.clamp(image, 0, 1)
        
        return image, gaze, self.participant_ids[idx]


def split_participants(all_participants, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """
    Split participants into train, validation, and test sets.
    
    Args:
        all_participants: List of all participant IDs
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
    
    Returns:
        train_participants, val_participants, test_participants
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    np.random.seed(42)  # For reproducibility
    shuffled = np.random.permutation(all_participants)
    
    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_participants = shuffled[:n_train].tolist()
    val_participants = shuffled[n_train:n_train+n_val].tolist()
    test_participants = shuffled[n_train+n_val:].tolist()
    
    return train_participants, val_participants, test_participants


def get_data_loaders(data_dir, batch_size=32, num_workers=4, train_augment=True, preprocessor=None):
    """
    Get data loaders for train, validation, and test sets.
    
    Args:
        data_dir: Path to MPIIGaze/Data/Normalized/
        batch_size: Batch size for training
        num_workers: Number of worker threads for data loading
        train_augment: Whether to use data augmentation for training
        preprocessor: Optional image preprocessor for handling different camera standards
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Get all participants
    all_participants = sorted([f"p{i:02d}" for i in range(15)])
    
    # Split participants
    train_participants, val_participants, test_participants = split_participants(
        all_participants, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2
    )
    
    print(f"\nData splits:")
    print(f"  Train participants ({len(train_participants)}): {train_participants}")
    print(f"  Val participants ({len(val_participants)}): {val_participants}")
    print(f"  Test participants ({len(test_participants)}): {test_participants}")
    
    # Create datasets
    train_dataset = MPIIGazeDataset(
        data_dir, train_participants, augment=train_augment, preprocessor=preprocessor
    )
    val_dataset = MPIIGazeDataset(
        data_dir, val_participants, augment=False, preprocessor=preprocessor
    )
    test_dataset = MPIIGazeDataset(
        data_dir, test_participants, augment=False, preprocessor=preprocessor
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_person_specific_loaders(data_dir, participant_id, batch_size=32, num_workers=4, preprocessor=None):
    """
    Get data loaders for person-specific evaluation.
    Uses one participant for test, rest for train/val.
    
    Args:
        data_dir: Path to MPIIGaze/Data/Normalized/
        participant_id: Participant ID to use as test set (e.g., 'p00')
        batch_size: Batch size
        num_workers: Number of worker threads
        preprocessor: Optional image preprocessor for handling different camera standards
    
    Returns:
        train_loader, val_loader, test_loader
    """
    all_participants = sorted([f"p{i:02d}" for i in range(15)])
    
    # Split: one participant for test, rest for train/val
    test_participants = [participant_id]
    remaining = [p for p in all_participants if p != participant_id]
    
    train_participants, val_participants, _ = split_participants(
        remaining, train_ratio=0.85, val_ratio=0.15, test_ratio=0.0
    )
    
    print(f"\nPerson-specific evaluation for {participant_id}:")
    print(f"  Train participants ({len(train_participants)}): {train_participants[:3]}...")
    print(f"  Val participants ({len(val_participants)}): {val_participants}")
    print(f"  Test participant: {test_participants}")
    
    # Create datasets
    train_dataset = MPIIGazeDataset(data_dir, train_participants, augment=True, preprocessor=preprocessor)
    val_dataset = MPIIGazeDataset(data_dir, val_participants, augment=False, preprocessor=preprocessor)
    test_dataset = MPIIGazeDataset(data_dir, test_participants, augment=False, preprocessor=preprocessor)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing MPIIGaze Dataset Loading...")
    
    data_dir = "MPIIGaze/Data/Normalized"
    
    # Test basic loading
    test_participants = ['p00', 'p01']
    dataset = MPIIGazeDataset(data_dir, test_participants)
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Get a sample
    if len(dataset) > 0:
        image, gaze, participant = dataset[0]
        print(f"\nSample info:")
        print(f"  Image shape: {image.shape}")
        print(f"  Gaze vector: {gaze}")
        print(f"  Participant: {participant}")
    
    # Test data loaders
    print("\n" + "="*80)
    print("Testing Data Loaders...")
    
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir, batch_size=16, num_workers=0
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get a batch
    if len(train_loader) > 0:
        images, gazes, participants = next(iter(train_loader))
        print(f"\nBatch info:")
        print(f"  Images shape: {images.shape}")
        print(f"  Gazes shape: {gazes.shape}")
        print(f"  Participants: {participants[:5]}")
    
    print("\nDataset loading test completed!")

