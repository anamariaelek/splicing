"""Simple training script for Splicevo model."""

import numpy as np
import torch
from pathlib import Path
import os
import sys
import time

from splicevo.model import SplicevoModel
from splicevo.training import SpliceTrainer, SpliceDataset
from splicevo.training.normalization import (
    normalize_usage_arrays,
    save_normalization_stats
)
from torch.utils.data import DataLoader


def main():
    script_start = time.time()
    
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
    load_start = time.time()
    data_path = '/home/elek/projects/splicing/results/data_processing_subset/processed_data.npz'
    data = np.load(data_path)
    
    sequences = data['sequences']
    labels = data['labels']
    usage_arrays = {
        'alpha': data['usage_alpha'],
        'beta': data['usage_beta'],
        'sse': data['usage_sse']
    }
    
    load_time = time.time() - load_start
    print(f"Loaded {len(sequences)} samples in {load_time:.2f}s")
    print(f"  Sequence shape: {sequences.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Usage shape: {usage_arrays['alpha'].shape}")

    print("Original usage array statistics:")
    for key, arr in usage_arrays.items():
        valid = ~np.isnan(arr)
        print(f"  '{key}': [{arr[valid].min():.1f}, {arr[valid].max():.1f}] "
              f"(mean={arr[valid].mean():.1f}, std={arr[valid].std():.1f})")
    
    normalized_usage, usage_stats = normalize_usage_arrays(
        usage_arrays,
        method='per_sample_cpm' 
    )
    
    print("Normalized statistics:")
    for key in ['alpha', 'beta', 'sse']:
        arr = normalized_usage[key]
        valid = ~np.isnan(arr)
        print(f"  '{key}': [{arr[valid].min():.3f}, {arr[valid].max():.3f}] "
              f"(mean={arr[valid].mean():.3f}, std={arr[valid].std():.3f})")
    
    # Save normalization stats
    checkpoint_dir = '/home/elek/projects/splicing/results/models/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    stats_path = Path(checkpoint_dir) / 'usage_normalization_stats.json'
    save_normalization_stats(usage_stats, stats_path)
    print(f"Saved normalization stats to: {stats_path}")

    # Split train/val (80/20)
    n_samples = len(sequences)
    n_train = int(0.8 * n_samples)
    
    # Create datasets
    dataset_start = time.time()
    train_dataset = SpliceDataset(
        sequences[:n_train],
        labels[:n_train],
        {k: v[:n_train] for k, v in normalized_usage.items()}
    )
    
    val_dataset = SpliceDataset(
        sequences[n_train:],
        labels[n_train:],
        {k: v[n_train:] for k, v in normalized_usage.items()}
    )
    dataset_time = time.time() - dataset_start
    print(f"Created datasets in {dataset_time:.2f}s")

    # Calculate class distribution in training data
    print("\nCalculating class weights...")
    
    train_labels_flat = labels[:n_train].flatten()
    unique_classes, class_counts = np.unique(train_labels_flat, return_counts=True)
    print(f"Class distribution in training data:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  Class {cls}: {count:,} ({100*count/len(train_labels_flat):.2f}%)")
    
    # Compute inverse frequency weights
    total_samples = len(train_labels_flat)
    class_weights = total_samples / (len(unique_classes) * class_counts)
    class_weights = class_weights / class_weights.mean()
    class_weights = torch.FloatTensor(class_weights)
    print(f"Class weights (inverse frequency): {class_weights}")

    # Create dataloaders 
    batch_size = 64  # be conservative for shared system
    num_workers = 32 # reduce to leave CPU for others

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

    # Initialize model
    print("\nInitializing model...")
    model_start = time.time()
    model = SplicevoModel(
        embed_dim=256,   # keep this small for shared system
        num_resblocks=8, # reduce for shared system
        dilation_strategy='exponential',
        num_classes=3,
        n_conditions=usage_arrays['alpha'].shape[2],
        context_len=4500
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    model_time = time.time() - model_start
    print(f"Model parameters: {n_params:,}")
    print(f"Model initialized in {model_time:.2f}s")
    
    # Where to save checkpoints
    checkpoint_dir='/home/elek/projects/splicing/results/models/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer_start = time.time()
    trainer = SpliceTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-4,
        weight_decay=1e-5,
        splice_weight=0.5,
        usage_weight=0.5, 
        class_weights=class_weights,
        checkpoint_dir=checkpoint_dir,
        use_tensorboard=True
    )
    trainer_time = time.time() - trainer_start
    print(f"Trainer initialized in {trainer_time:.2f}s")
    
    # Train
    print("\nStarting training...")
    train_start = time.time()
    trainer.train(
        n_epochs=10,
        verbose=True,
        save_best=True,
        early_stopping_patience=5
    )
    train_time = time.time() - train_start
    
    total_time = time.time() - script_start
    
    print("\nTraining completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}/")
    print(f"\nTiming Summary:")
    print(f"  Data loading:     {load_time:8.2f}s ({100*load_time/total_time:5.1f}%)")
    print(f"  Dataset creation: {dataset_time:8.2f}s ({100*dataset_time/total_time:5.1f}%)")
    print(f"  Model init:       {model_time:8.2f}s ({100*model_time/total_time:5.1f}%)")
    print(f"  Trainer init:     {trainer_time:8.2f}s ({100*trainer_time/total_time:5.1f}%)")
    print(f"  Training:         {train_time:8.2f}s ({100*train_time/total_time:5.1f}%)")
    print(f"  Total time:       {total_time:8.2f}s")


if __name__ == '__main__':
    main()