"""Simple training script for Splicevo model."""

import numpy as np
import torch
from pathlib import Path
import os
import sys
import time
import json
from datetime import datetime

from splicevo.model import SplicevoModel
from splicevo.training import SpliceTrainer, SpliceDataset
from splicevo.training.normalization import (
    normalize_usage_arrays,
    save_normalization_stats
)
from torch.utils.data import DataLoader

# Use memory-mapped data? (if it doesn't exist, it will be created)
USE_MEMMAP = True 

def main():
    script_start = time.time()
    
    # Set resource limits for shared system
    # Limit CPU threads to be considerate of other users
    torch.set_num_threads(8)
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['MKL_NUM_THREADS'] = '8'
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # GPU memory optimization
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        # Set memory fraction to leave room for other users
        mem_frac = 0.7
        torch.cuda.set_per_process_memory_fraction(mem_frac)
        print(f"GPU memory limit set to {mem_frac*100}% of available memory")
    
    # Where to save checkpoints and logs
    checkpoint_dir = '/home/elek/projects/splicing/results/models/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(checkpoint_dir, f'training_log_{timestamp}.txt')
    
    def log_print(msg):
        """Print and write to log file."""
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')
    
    log_print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"Using device: {device}")
    log_print(f"Log file: {log_file}")
    
    # Load data
    data_path = '/home/elek/projects/splicing/results/data_processing_subset/processed_data.npz'
    log_print(f"\nLoading data {data_path}")
    load_start = time.time()
    
    # Check if memmap files exist
    memmap_dir = Path('/home/elek/projects/splicing/results/data_processing_subset/memmap')
    memmap_exists = (memmap_dir / 'sequences.mmap').exists() if USE_MEMMAP else False
    
    if USE_MEMMAP and memmap_exists:
        log_print("Loading from memory-mapped files...")
        # Load metadata
        with open(memmap_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Create memmap arrays
        sequences = np.memmap(
            memmap_dir / 'sequences.mmap',
            dtype=np.float32,
            mode='r',
            shape=tuple(metadata['sequences_shape'])
        )
        labels = np.memmap(
            memmap_dir / 'labels.mmap',
            dtype=np.int64,
            mode='r',
            shape=tuple(metadata['labels_shape'])
        )
        usage_alpha = np.memmap(
            memmap_dir / 'usage_alpha.mmap',
            dtype=np.float32,
            mode='r',
            shape=tuple(metadata['usage_shape'])
        )
        usage_beta = np.memmap(
            memmap_dir / 'usage_beta.mmap',
            dtype=np.float32,
            mode='r',
            shape=tuple(metadata['usage_shape'])
        )
        usage_sse = np.memmap(
            memmap_dir / 'usage_sse.mmap',
            dtype=np.float32,
            mode='r',
            shape=tuple(metadata['usage_shape'])
        )
        
        usage_arrays = {
            'alpha': usage_alpha,
            'beta': usage_beta,
            'sse': usage_sse
        }
        
        log_print("Using existing normalized memmap data")
        normalized_usage = usage_arrays
        usage_stats = metadata.get('normalization_stats', {})
        
    else:
        # Original loading path
        if USE_MEMMAP:
            log_print("Memory-mapped files not found, loading from .npz and creating memmap...")
        
        data = np.load(data_path)
        
        sequences = data['sequences']
        labels = data['labels']
        usage_arrays = {
            'alpha': data['usage_alpha'],
            'beta': data['usage_beta'],
            'sse': data['usage_sse']
        }
        
        log_print("\nOriginal usage array statistics:")
        for key, arr in usage_arrays.items():
            valid = ~np.isnan(arr)
            log_print(f"  '{key}': [{arr[valid].min():.1f}, {arr[valid].max():.1f}] "
                      f"(mean={arr[valid].mean():.1f}, std={arr[valid].std():.1f})")
        
        normalized_usage, usage_stats = normalize_usage_arrays(
            usage_arrays,
            method='per_sample_cpm' 
        )
        
        log_print("\nNormalized statistics:")
        for key in ['alpha', 'beta', 'sse']:
            arr = normalized_usage[key]
            valid = ~np.isnan(arr)
            log_print(f"  '{key}': [{arr[valid].min():.3f}, {arr[valid].max():.3f}] "
                      f"(mean={arr[valid].mean():.3f}, std={arr[valid].std():.3f})")
        
        # Create memmap files if enabled
        if USE_MEMMAP:
            log_print("\nCreating memory-mapped files for future use...")
            memmap_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as memmap
            seq_mmap = np.memmap(
                memmap_dir / 'sequences.mmap',
                dtype=np.float32,
                mode='w+',
                shape=sequences.shape
            )
            seq_mmap[:] = sequences[:]
            seq_mmap.flush()
            
            labels_mmap = np.memmap(
                memmap_dir / 'labels.mmap',
                dtype=np.int64,
                mode='w+',
                shape=labels.shape
            )
            labels_mmap[:] = labels[:]
            labels_mmap.flush()
            
            for key in ['alpha', 'beta', 'sse']:
                usage_mmap = np.memmap(
                    memmap_dir / f'usage_{key}.mmap',
                    dtype=np.float32,
                    mode='w+',
                    shape=normalized_usage[key].shape
                )
                usage_mmap[:] = normalized_usage[key][:]
                usage_mmap.flush()
            
            # Save metadata
            metadata = {
                'sequences_shape': sequences.shape,
                'labels_shape': labels.shape,
                'usage_shape': normalized_usage['alpha'].shape,
                'normalization_stats': usage_stats
            }
            with open(memmap_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            log_print(f"Memory-mapped files saved to: {memmap_dir}")
            
            # Switch to memmap for training
            sequences = seq_mmap
            labels = labels_mmap
            normalized_usage = {
                'alpha': usage_mmap,
                'beta': np.memmap(memmap_dir / 'usage_beta.mmap', dtype=np.float32, mode='r', shape=metadata['usage_shape']),
                'sse': np.memmap(memmap_dir / 'usage_sse.mmap', dtype=np.float32, mode='r', shape=metadata['usage_shape'])
            }
    
    load_time = time.time() - load_start
    log_print(f"Loaded {len(sequences)} samples in {load_time:.2f}s")
    log_print(f"  Sequence shape: {sequences.shape}")
    log_print(f"  Labels shape: {labels.shape}")
    log_print(f"  Usage shape: {normalized_usage['alpha'].shape}")
    log_print(f"  Using memory-mapped data: {USE_MEMMAP and isinstance(sequences, np.memmap)}")
    
    # Save normalization stats
    stats_path = Path(checkpoint_dir) / 'usage_normalization_stats.json'
    save_normalization_stats(usage_stats, stats_path)
    log_print(f"\nSaved normalization stats to: {stats_path}")

    # Split train/val (80/20)
    n_samples = len(sequences)
    n_train = int(0.8 * n_samples)
    
    # Create datasets - memmap slicing is efficient
    dataset_start = time.time()
    train_dataset = SpliceDataset(
        sequences[:n_train],
        labels[:n_train],
        {k: v[:n_train] for k, v in normalized_usage.items()},
        use_memmap=USE_MEMMAP and isinstance(sequences, np.memmap)
    )
    
    val_dataset = SpliceDataset(
        sequences[n_train:],
        labels[n_train:],
        {k: v[n_train:] for k, v in normalized_usage.items()},
        use_memmap=USE_MEMMAP and isinstance(sequences, np.memmap)
    )
    dataset_time = time.time() - dataset_start
    log_print(f"\nCreated datasets in {dataset_time:.2f}s")

    # Calculate class distribution in training data
    log_print("\nCalculating class weights...")
    
    train_labels_flat = labels[:n_train].flatten()
    unique_classes, class_counts = np.unique(train_labels_flat, return_counts=True)
    log_print("Class distribution in training data:")
    for cls, count in zip(unique_classes, class_counts):
        log_print(f"  Class {cls}: {count:,} ({100*count/len(train_labels_flat):.2f}%)")
    
    # Compute inverse frequency weights
    total_samples = len(train_labels_flat)
    class_weights = total_samples / (len(unique_classes) * class_counts)
    class_weights = class_weights / class_weights.mean()
    class_weights = torch.FloatTensor(class_weights)
    log_print(f"Class weights (inverse frequency): {class_weights.tolist()}")

    # Model parameters
    model_params = {
        'embed_dim': 128,
        'num_resblocks': 6,
        'dilation_strategy': 'exponential',
        'num_classes': 3,
        'n_conditions': usage_arrays['alpha'].shape[2],
        'context_len': 4500,
        'dropout': 0.5
    }
    
    # Training parameters
    training_params = {
        'learning_rate': 1e-5,
        'weight_decay': 1e-3,
        'splice_weight': 0.5,
        'usage_weight': 0.5,
        'batch_size': 128,
        'num_workers': 32,
        'n_epochs': 100,
        'early_stopping_patience': 5,
        'device': device
    }
    
    # Combined configuration
    config = {
        'timestamp': timestamp,
        'data_path': data_path,
        'checkpoint_dir': checkpoint_dir,
        'n_train_samples': len(train_dataset),
        'n_val_samples': len(val_dataset),
        'class_weights': class_weights.tolist(),
        'normalization_method': 'per_sample_cpm',
        'model_params': model_params,
        'training_params': training_params
    }
    
    # Save configuration
    config_path = os.path.join(checkpoint_dir, f'config_{timestamp}.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    log_print(f"\nSaved configuration to: {config_path}")
    
    # Create dataloaders 

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_params['batch_size'], 
        shuffle=True,
        num_workers=training_params['num_workers'],  
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_params['batch_size'],
        shuffle=False,
        num_workers=training_params['num_workers'],
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    log_print(f"\nTraining samples: {len(train_dataset)}")
    log_print(f"Validation samples: {len(val_dataset)}")

    # Log all parameters
    log_print("\n" + "="*60)
    log_print("MODEL PARAMETERS:")
    log_print("="*60)
    for key, value in model_params.items():
        log_print(f"  {key}: {value}")
    
    log_print("\n" + "="*60)
    log_print("TRAINING PARAMETERS:")
    log_print("="*60)
    for key, value in training_params.items():
        log_print(f"  {key}: {value}")
    log_print("="*60 + "\n")

    # Initialize model
    log_print("Initializing model...")
    model_start = time.time()
    model = SplicevoModel(**model_params)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    model_time = time.time() - model_start
    log_print(f"Model parameters: {n_params:,}")
    log_print(f"Model initialized in {model_time:.2f}s")

    # Initialize trainer
    log_print("\nInitializing trainer...")
    trainer_start = time.time()
    trainer = SpliceTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=training_params['learning_rate'],
        weight_decay=training_params['weight_decay'],
        splice_weight=training_params['splice_weight'],
        usage_weight=training_params['usage_weight'],
        class_weights=class_weights,
        checkpoint_dir=checkpoint_dir,
        use_tensorboard=True
    )
    trainer_time = time.time() - trainer_start
    log_print(f"Trainer initialized in {trainer_time:.2f}s")
    
    # Train
    log_print("\nStarting training...")
    train_start = time.time()
    trainer.train(
        n_epochs=training_params['n_epochs'],
        verbose=True,
        save_best=True,
        early_stopping_patience=training_params['early_stopping_patience']
    )
    train_time = time.time() - train_start
    
    total_time = time.time() - script_start
    
    log_print("\nTraining completed!")
    log_print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    log_print(f"Checkpoints saved to: {checkpoint_dir}/")
    log_print(f"\nTiming Summary:")
    log_print(f"  Data loading:     {load_time:8.2f}s ({100*load_time/total_time:5.1f}%)")
    log_print(f"  Dataset creation: {dataset_time:8.2f}s ({100*dataset_time/total_time:5.1f}%)")
    log_print(f"  Model init:       {model_time:8.2f}s ({100*model_time/total_time:5.1f}%)")
    log_print(f"  Trainer init:     {trainer_time:8.2f}s ({100*trainer_time/total_time:5.1f}%)")
    log_print(f"  Training:         {train_time:8.2f}s ({100*train_time/total_time:5.1f}%)")
    log_print(f"  Total time:       {total_time:8.2f}s")
    log_print(f"\nTraining ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()