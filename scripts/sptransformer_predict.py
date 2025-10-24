import numpy as np
import tqdm
import os
import torch
import argparse
import sys

# os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
torch.set_num_threads(8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SpliceTransformer prediction script\n"
                    "Example:\n"
                    "python scripts/sptransformer_predict.py --input /path/to/input.npz --output /path/to/output.npz --weights /path/to/weights.ckpt"
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input .npz file")
    parser.add_argument("--output", type=str, required=True, help="Path to output .npz file")
    parser.add_argument("--weights", type=str, default="./model/weights/SpTransformer_pytorch.ckpt", help="Path to model weights (.ckpt)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for prediction")
    args = parser.parse_args()

    sys.path.append('/home/elek/projects/SpliceTransformer')
    from tasks_annotate_mutations import SpTransformerDriver

    # The model will automatically use GPU if available
    model = SpTransformerDriver(ref_fasta='', load_db=False, context=4500, weights=args.weights)

    # Check which device is being used
    print(f"Using device: {model.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # GPU memory optimization
    if str(model.device) == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        # Set memory fraction to leave room for other users
        mem_frac = 0.7
        torch.cuda.set_per_process_memory_fraction(mem_frac)
        print(f"GPU memory limit set to {mem_frac*100}% of available memory")

    def main(input_path, output_path, batch_size):
        # Load test data
        test_path = input_path
        test_data = np.load(test_path)
        test_sequences = test_data['sequences']
        test_labels = test_data['labels']
        test_usage = test_data['usage_sse']
        print(f"Test sequences shape: {test_sequences.shape}")
        print(f"Test labels shape: {test_labels.shape}")
        print(f"Using batch size: {batch_size}")

        # Calculate batched predictions (automatically uses GPU if available)
        # For large datasets, process in batches to avoid memory issues
        all_predictions = []

        print("Running predictions...")
        for i in tqdm.tqdm(range(0, len(test_sequences), batch_size)):
            batch = test_sequences[i:i+batch_size]
            output = model.calc_batched_sequence(batch, encode=False)
            all_predictions.append(output)

        # Concatenate all predictions
        output = np.concatenate(all_predictions, axis=0)
        splice_probs = output[:, :, :3]
        splice_preds = splice_probs.argmax(axis=-1)
        splice_usage = output[:, :, 3:]
        print(f"Output shape: {output.shape}")
        print(f"Splice probs shape: {splice_probs.shape}")
        print(f"Tissue usage shape: {splice_usage.shape}")

        # Save predictions
        np.savez_compressed(output_path, 
                 splice_preds=splice_preds, 
                 splice_probs=splice_probs, 
                 splice_usage=splice_usage,         
                 labels_true=test_labels,
                 usage_true=test_usage
                 )
        print(f"Predictions saved to {output_path}")

    main(args.input, args.output, args.batch_size)

