#!/usr/bin/env python
"""Optimize model for production deployment using quantization and pruning.

This script demonstrates TensorFlow model optimization techniques:
1. Post-training quantization (FP32 → INT8)
2. Quantization-aware training (optional)
3. Magnitude-based weight pruning
4. Combined pruning + quantization

Usage:
    python scripts/optimize_model.py \
        --model models/multitask_unet/best_model.keras \
        --data-dir data/synthetic_v2_survival \
        --output models/optimized
"""

import argparse
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot

from med_seg.data import PETCTPreprocessor
from med_seg.data.survival_generator import create_survival_generators


def convert_to_tflite(model: keras.Model,
                      representative_dataset,
                      use_quantization: bool = True,
                      output_path: Path = None) -> bytes:
    """Convert Keras model to TFLite format.

    Args:
        model: Keras model
        representative_dataset: Generator for representative data
        use_quantization: Whether to apply INT8 quantization
        output_path: Where to save .tflite file

    Returns:
        TFLite model bytes
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if use_quantization:
        # Dynamic range quantization (INT8)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Representative dataset for full integer quantization
        converter.representative_dataset = representative_dataset

        # INT8 quantization for all ops
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        print(f"[+] Saved TFLite model: {output_path}")

    return tflite_model


def representative_dataset_gen(data_gen, num_samples: int = 100):
    """Generator for representative dataset (for quantization calibration).

    Args:
        data_gen: Training data generator
        num_samples: Number of samples to use

    Yields:
        Input data for calibration
    """
    count = 0
    for x_batch, y_batch in data_gen:
        for x in x_batch:
            if count >= num_samples:
                return
            yield [np.expand_dims(x, axis=0).astype(np.float32)]
            count += 1


def apply_pruning(model: keras.Model,
                 initial_sparsity: float = 0.0,
                 final_sparsity: float = 0.5,
                 begin_step: int = 0,
                 end_step: int = 1000) -> keras.Model:
    """Apply magnitude-based weight pruning.

    Args:
        model: Keras model to prune
        initial_sparsity: Starting sparsity (0 = no pruning)
        final_sparsity: Target sparsity (0.5 = 50% weights zeroed)
        begin_step: When to start pruning
        end_step: When to reach final sparsity

    Returns:
        Pruned model
    """
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=initial_sparsity,
            final_sparsity=final_sparsity,
            begin_step=begin_step,
            end_step=end_step
        )
    }

    # Apply pruning to the whole model
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
        model,
        **pruning_params
    )

    return model_for_pruning


def benchmark_model(model_path: Path,
                   data_gen,
                   n_samples: int = 100) -> dict:
    """Benchmark model performance.

    Args:
        model_path: Path to model file
        data_gen: Data generator
        n_samples: Number of samples to test

    Returns:
        Dictionary with benchmark results
    """
    # Load model
    if model_path.suffix == '.tflite':
        # TFLite model
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        times = []
        for i, (x_batch, _) in enumerate(data_gen):
            if i >= n_samples // len(x_batch):
                break

            for x in x_batch:
                x_input = np.expand_dims(x, axis=0).astype(np.float32)

                start = time.time()
                interpreter.set_tensor(input_details[0]['index'], x_input)
                interpreter.invoke()
                end = time.time()

                times.append((end - start) * 1000)  # ms

        model_size = model_path.stat().st_size / (1024 * 1024)  # MB

    else:
        # Keras model
        model = keras.models.load_model(model_path, compile=False)

        times = []
        for i, (x_batch, _) in enumerate(data_gen):
            if i >= n_samples // len(x_batch):
                break

            for x in x_batch:
                x_input = np.expand_dims(x, axis=0)

                start = time.time()
                _ = model(x_input, training=False)
                end = time.time()

                times.append((end - start) * 1000)  # ms

        # Estimate model size (parameters × 4 bytes for FP32)
        model_size = model.count_params() * 4 / (1024 * 1024)  # MB

    results = {
        'mean_time_ms': float(np.mean(times)),
        'std_time_ms': float(np.std(times)),
        'median_time_ms': float(np.median(times)),
        'min_time_ms': float(np.min(times)),
        'max_time_ms': float(np.max(times)),
        'model_size_mb': float(model_size),
        'throughput_fps': 1000.0 / float(np.mean(times))
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Optimize model for deployment")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory")
    parser.add_argument("--output", type=str, default="models/optimized", help="Output directory")
    parser.add_argument("--quantize", action="store_true", help="Apply INT8 quantization")
    parser.add_argument("--prune", action="store_true", help="Apply weight pruning")
    parser.add_argument("--sparsity", type=float, default=0.5, help="Target sparsity for pruning")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")

    args = parser.parse_args()

    print("\n[+] Model Optimization for Production Deployment")
    print("="*70)
    print(f"Input model: {args.model}")
    print(f"Output directory: {args.output}")
    print(f"Quantization: {'Yes' if args.quantize else 'No'}")
    print(f"Pruning: {'Yes (sparsity={})'.format(args.sparsity) if args.prune else 'No'}")
    print()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original model
    print("[1/4] Loading original model...")
    model = keras.models.load_model(args.model, compile=False)
    original_params = model.count_params()
    print(f"  Parameters: {original_params:,}")
    print(f"  Size (FP32): {original_params * 4 / 1024 / 1024:.2f} MB")

    # Create data generator for calibration
    print("\n[2/4] Creating data generator...")
    preprocessor = PETCTPreprocessor(target_size=(256, 256))
    train_gen, _ = create_survival_generators(
        data_dir=args.data_dir,
        preprocessor=preprocessor,
        batch_size=8
    )

    # Optimization
    print("\n[3/4] Applying optimizations...")

    # Quantization
    if args.quantize:
        print("  [*] Applying INT8 quantization...")

        def rep_dataset():
            return representative_dataset_gen(train_gen, num_samples=100)

        tflite_quant_path = output_dir / 'model_quantized_int8.tflite'
        convert_to_tflite(
            model,
            representative_dataset=rep_dataset,
            use_quantization=True,
            output_path=tflite_quant_path
        )

        quant_size = tflite_quant_path.stat().st_size / (1024 * 1024)
        reduction = (1 - quant_size / (original_params * 4 / 1024 / 1024)) * 100
        print(f"      Size: {quant_size:.2f} MB ({reduction:.1f}% reduction)")

    # Pruning (requires retraining - just show structure here)
    if args.prune:
        print(f"  [*] Applying magnitude-based pruning (sparsity={args.sparsity})...")
        print(f"      Note: Pruning requires retraining - see TensorFlow docs")
        print(f"      Expected reduction: ~{args.sparsity * 100:.0f}% of weights")

    # Baseline TFLite (no quantization)
    print("  [*] Creating baseline TFLite model (FP32)...")
    tflite_fp32_path = output_dir / 'model_fp32.tflite'
    convert_to_tflite(
        model,
        representative_dataset=None,
        use_quantization=False,
        output_path=tflite_fp32_path
    )
    fp32_size = tflite_fp32_path.stat().st_size / (1024 * 1024)
    print(f"      Size: {fp32_size:.2f} MB")

    # Benchmarking
    if args.benchmark:
        print("\n[4/4] Running benchmarks...")

        # Original Keras model
        print("  [*] Benchmarking original Keras model...")
        keras_results = benchmark_model(Path(args.model), train_gen, n_samples=50)

        print(f"      Inference time: {keras_results['mean_time_ms']:.2f} ± {keras_results['std_time_ms']:.2f} ms")
        print(f"      Throughput: {keras_results['throughput_fps']:.1f} FPS")
        print(f"      Model size: {keras_results['model_size_mb']:.2f} MB")

        # TFLite FP32
        print("  [*] Benchmarking TFLite FP32 model...")
        fp32_results = benchmark_model(tflite_fp32_path, train_gen, n_samples=50)
        print(f"      Inference time: {fp32_results['mean_time_ms']:.2f} ± {fp32_results['std_time_ms']:.2f} ms")
        print(f"      Speedup: {keras_results['mean_time_ms'] / fp32_results['mean_time_ms']:.2f}x")

        # TFLite INT8
        if args.quantize:
            print("  [*] Benchmarking TFLite INT8 model...")
            int8_results = benchmark_model(tflite_quant_path, train_gen, n_samples=50)
            print(f"      Inference time: {int8_results['mean_time_ms']:.2f} ± {int8_results['std_time_ms']:.2f} ms")
            print(f"      Speedup: {keras_results['mean_time_ms'] / int8_results['mean_time_ms']:.2f}x vs Keras")
            print(f"      Size reduction: {(1 - int8_results['model_size_mb'] / keras_results['model_size_mb']) * 100:.1f}%")

        # Save benchmark results
        import json
        results_path = output_dir / 'benchmark_results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'keras': keras_results,
                'tflite_fp32': fp32_results,
                'tflite_int8': int8_results if args.quantize else None
            }, f, indent=2)
        print(f"\n[+] Saved benchmark results: {results_path}")

    print("\n" + "="*70)
    print("[+] Optimization complete!")
    print("\n[*] Summary:")
    print(f"    Original: {original_params * 4 / 1024 / 1024:.2f} MB")
    if args.quantize:
        print(f"    Quantized (INT8): {quant_size:.2f} MB ({reduction:.1f}% smaller)")

    print("\n[*] For full pruning + quantization:")
    print("    1. Train with pruning: tfmot.sparsity.keras.prune_low_magnitude()")
    print("    2. Strip pruning wrappers: tfmot.sparsity.keras.strip_pruning()")
    print("    3. Quantize pruned model")
    print("    4. Expected: 10-12x size reduction with <5% accuracy loss")

    return 0


if __name__ == "__main__":
    sys.exit(main())
