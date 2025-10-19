"""Plot training curves from CSV log."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_log(log_file: str):
    """Plot training curves from log file."""
    df = pd.read_csv(log_file)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot total loss
    axes[0].plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Training Progress')
    
    # Plot component losses
    axes[1].plot(df['epoch'], df['train_splice_loss'], label='Train Splice Loss', linestyle='--')
    axes[1].plot(df['epoch'], df['train_usage_loss'], label='Train Usage Loss', linestyle='--')
    axes[1].plot(df['epoch'], df['val_splice_loss'], label='Val Splice Loss')
    axes[1].plot(df['epoch'], df['val_usage_loss'], label='Val Usage Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Component Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(log_file).parent / 'training_curves.png', dpi=150)
    print(f"Saved plot to: {Path(log_file).parent / 'training_curves.png'}")
    plt.show()

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        plot_training_log(sys.argv[1])
    else:
        log_file = '/home/elek/projects/splicing/results/models/checkpoints/training_log.csv'
        plot_training_log(log_file)
