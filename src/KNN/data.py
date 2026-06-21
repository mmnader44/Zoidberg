"""
data.py - Data loading and exploration utilities for chest X-ray project.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image


def discover_datasets(root_path: str | Path) -> Dict[str, Dict[str, Path]]:
    """
    Discover train/val/test splits and their class folders.
    
    Args:
        root_path: Path to the data root (e.g., 'data/chest_Xray')
    
    Returns:
        Dictionary with structure: {split_name: {class_name: path}}
    """
    root = Path(root_path)
    datasets = {}
    
    for split_dir in root.iterdir():
        if not split_dir.is_dir() or split_dir.name.startswith('.'):
            continue
        
        split_name = split_dir.name
        classes = {}
        
        # Handle nested structure (train/train/)
        check_dir = split_dir
        nested = split_dir / split_name
        if nested.exists() and nested.is_dir():
            check_dir = nested
        
        for class_dir in check_dir.iterdir():
            if class_dir.is_dir() and not class_dir.name.startswith('.'):
                classes[class_dir.name] = class_dir
        
        if classes:
            datasets[split_name] = classes
    
    return datasets


def count_images_per_class(dataset_path: Path) -> Dict[str, int]:
    """
    Count images in each class folder.
    
    Args:
        dataset_path: Path to a dataset split containing class folders
    
    Returns:
        Dictionary {class_name: count}
    """
    counts = {}
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    for class_dir in dataset_path.iterdir():
        if class_dir.is_dir() and not class_dir.name.startswith('.'):
            count = sum(
                1 for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in valid_extensions
            )
            counts[class_dir.name] = count
    
    return counts


def get_image_paths(class_path: Path) -> List[Path]:
    """
    Get all image paths from a class folder.
    
    Args:
        class_path: Path to a class folder
    
    Returns:
        List of image paths
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    return [
        f for f in class_path.iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]


def load_image(
    path: str | Path,
    size: Optional[Tuple[int, int]] = None,
    grayscale: bool = True
) -> np.ndarray:
    """
    Load an image as numpy array.
    
    Args:
        path: Path to the image
        size: Optional (width, height) to resize
        grayscale: If True, convert to grayscale
    
    Returns:
        Image as numpy array
    """
    img = Image.open(path)
    
    if grayscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    
    if size is not None:
        img = img.resize(size, Image.Resampling.LANCZOS)
    
    return np.array(img)


def get_image_info(path: str | Path) -> Dict:
    """
    Get metadata about an image file.
    
    Args:
        path: Path to the image
    
    Returns:
        Dictionary with format, size, mode
    """
    path = Path(path)
    with Image.open(path) as img:
        return {
            'filename': path.name,
            'format': img.format,
            'size': img.size,  # (width, height)
            'mode': img.mode,
            'file_size_kb': path.stat().st_size / 1024
        }


def sample_images(
    class_path: Path,
    n: int = 5,
    seed: int = 42
) -> List[Path]:
    """
    Randomly sample n images from a class folder.
    
    Args:
        class_path: Path to class folder
        n: Number of images to sample
        seed: Random seed for reproducibility
    
    Returns:
        List of sampled image paths
    """
    rng = np.random.default_rng(seed)
    all_images = get_image_paths(class_path)
    n = min(n, len(all_images))
    indices = rng.choice(len(all_images), size=n, replace=False)
    return [all_images[i] for i in indices]
