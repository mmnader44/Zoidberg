"""
KNN - Data loading, features and metrics for the KNN notebook.
"""

from .data import (
    discover_datasets,
    get_image_paths,
    load_image,
    count_images_per_class,
    get_image_info,
    sample_images,
)
from .metrics import plot_confusion_matrix, print_metrics, compare_models
from .features import load_dataset, flatten_images, prepare_data

__all__ = [
    'discover_datasets',
    'get_image_paths',
    'load_image',
    'count_images_per_class',
    'get_image_info',
    'sample_images',
    'plot_confusion_matrix',
    'print_metrics',
    'compare_models',
    'load_dataset',
    'flatten_images',
    'prepare_data',
]
