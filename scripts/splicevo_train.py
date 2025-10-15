"""Simple training script for Splicevo model using the training module."""

import numpy as np
import torch
from pathlib import Path
import os
import sys

from splicevo.model import SplicevoModel
from splicevo.training import SpliceTrainer, SpliceDataset
from torch.utils.data import DataLoader


def main():
    # Set resource limits for shared system
    # Limit CPU threads to be considerate of other users
    torch.set_num_threads(4)  # Limit PyTorch CPU threads
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # GPU memory optimization
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        # Set memory fraction to leave room for other users
        mem_frac = 0.6
        torch.cuda.set_per_process_memory_fraction(mem_frac)
        print(f"GPU memory limit set to {mem_frac*100}% of available memory")
    
    # Load data
    print("Loading data...")
    data_path = '/home/elek/projects/splicing/results/data_processing_subset/processed_data.npz'
    data = np.load(data_path)
    
    sequences = data['sequences']
    labels = data['labels']
    usage_arrays = {
        'alpha': data['usage_alpha'],
        'beta': data['usage_beta'],
        'sse': data['usage_sse']
    }
    
    print(f"Loaded {len(sequences)} samples")
    print(f"  Sequence shape: {sequences.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Usage shape: {usage_arrays['alpha'].shape}")
    
    # Split train/val (80/20)
    n_samples = len(sequences)
    n_train = int(0.8 * n_samples)
    
    # Create datasets
    train_dataset = SpliceDataset(
        sequences[:n_train],
        labels[:n_train],
        {k: v[:n_train] for k, v in usage_arrays.items()}
    )
    
    val_dataset = SpliceDataset(
        sequences[n_train:],
        labels[n_train:],
        {k: v[n_train:] for k, v in usage_arrays.items()}
    )

    # Create dataloaders 
    batch_size = 16  # be conservative for shared system
    num_workers = 8 # reduced to leave CPU for others

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,  
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"CPU workers: {num_workers}")

    # Initialize model with small size for shared system
    print("\nInitializing model...")
    model = SplicevoModel(
        embed_dim=256,   # keep small for shared system
        num_resblocks=4, # reduce for shared system
        dilation_strategy='exponential',
        num_classes=3,
        n_conditions=usage_arrays['alpha'].shape[2],
        dropout=0.1,
        context_len=4500
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Where to save checkpoints
    checkpoint_dir='/home/elek/projects/splicing/results/models/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = SpliceTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-4,
        weight_decay=1e-5,
        splice_weight=1.0,
        usage_weight=0.5,
        checkpoint_dir=checkpoint_dir
    )
    
    # Train
    print("\nStarting training...")
    trainer.train(
        n_epochs=10,
        verbose=True,
        save_best=True,
        early_stopping_patience=5
    )
    
    print("\nTraining completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}/")


if __name__ == '__main__':
    main()