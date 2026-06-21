"""
metrics.py - Evaluation metrics and visualization for chest X-ray project.
"""

from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ['NORMAL', 'PNEUMONIA'],
    title: str = 'Confusion Matrix',
    figsize: tuple = (6, 5),
    cmap: str = 'Blues',
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix with counts and percentages.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        save_path: Path to save figure (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='Vraie classe',
        xlabel='Classe predite'
    )
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})',
                   ha='center', va='center',
                   color='white' if cm[i, j] > thresh else 'black',
                   fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def print_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = 'Model'
) -> dict:
    """
    Print classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model for display
    
    Returns:
        Dictionary with all metrics
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"=== {model_name} ===")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    return {
        'model': model_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }


def compare_models(results: List[dict]) -> None:
    """
    Print comparison table of multiple models.
    
    Args:
        results: List of dictionaries from print_metrics()
    """
    print("\n" + "="*60)
    print("COMPARAISON DES MODELES")
    print("="*60)
    print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-"*60)
    
    for r in results:
        print(f"{r['model']:<20} {r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f}")
    
    print("="*60)
