"""
features.py - Feature extraction and preprocessing for chest X-ray project.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image

from .data import get_image_paths, load_image


def load_dataset(
    datasets: Dict[str, Dict[str, Path]],
    splits: List[str],
    size: Tuple[int, int] = (128, 128),
    grayscale: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load all images from specified splits.
    
    Args:
        datasets: Dataset structure from discover_datasets()
        splits: List of split names to load (e.g., ['train', 'val'])
        size: (width, height) to resize images
        grayscale: Convert to grayscale if True
        verbose: Print progress if True
    
    Returns:
        X: Array of images (n_samples, height, width) or (n_samples, height, width, 3)
        y: Array of labels (0=NORMAL, 1=PNEUMONIA)
        class_names: ['NORMAL', 'PNEUMONIA']
    """
    class_names = ['NORMAL', 'PNEUMONIA']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    images = []
    labels = []
    
    for split in splits:
        if split not in datasets:
            print(f"Warning: split '{split}' not found in datasets")
            continue
            
        for class_name, class_path in datasets[split].items():
            if class_name not in class_to_idx:
                continue
                
            label = class_to_idx[class_name]
            paths = get_image_paths(class_path)
            
            if verbose:
                print(f"Loading {split}/{class_name}: {len(paths)} images...")
            
            for path in paths:
                img = load_image(path, size=size, grayscale=grayscale)
                images.append(img)
                labels.append(label)
    
    X = np.array(images)
    y = np.array(labels)
    
    if verbose:
        print(f"Loaded {len(X)} images total")
        print(f"  Shape: {X.shape}")
        print(f"  Labels: {np.bincount(y)} (NORMAL, PNEUMONIA)")
    
    return X, y, class_names


def flatten_images(X: np.ndarray) -> np.ndarray:
    """
    Flatten 2D/3D images to 1D vectors.
    
    Args:
        X: Array of shape (n_samples, height, width) or (n_samples, height, width, channels)
    
    Returns:
        X_flat: Array of shape (n_samples, n_features)
    """
    n_samples = X.shape[0]
    return X.reshape(n_samples, -1)


def prepare_data(
    datasets: Dict[str, Dict[str, Path]],
    splits: List[str] = ['train'],
    size: Tuple[int, int] = (128, 128),
    grayscale: bool = True,
    flatten: bool = True,
    normalize: bool = False,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Complete data preparation pipeline.
    
    Args:
        datasets: Dataset structure from discover_datasets()
        splits: List of split names to load
        size: (width, height) to resize images
        grayscale: Convert to grayscale if True
        flatten: Flatten images to 1D if True
        normalize: Normalize pixel values to [0, 1] if True
        verbose: Print progress if True
    
    Returns:
        X: Prepared feature array
        y: Label array
        class_names: ['NORMAL', 'PNEUMONIA']
    """
    X, y, class_names = load_dataset(
        datasets, splits, size, grayscale, verbose
    )
    
    if normalize:
        X = X.astype(np.float32) / 255.0
    
    if flatten:
        X = flatten_images(X)
        if verbose:
            print(f"Flattened to shape: {X.shape}")
    
    return X, y, class_names
