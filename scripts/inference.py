#!/usr/bin/env python3
"""Inference script for making predictions on new medical images.

This script loads a trained model and performs inference on new medical images,
saving the predicted segmentation masks.

Usage:
    python scripts/inference.py --model models/brain-growth/expert_01/best_model.keras --input data/new_images --output predictions/
    python scripts/inference.py --config configs/kidney.yaml --ensemble --input data/kidney/new --output predictions/kidney
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import SimpleITK as sitk
from tensorflow import keras

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from med_seg.data import MedicalImageLoader
from med_seg.training import dice_loss, dice_coefficient, precision, recall
from med_seg.utils import load_config


def save_prediction_as_nifti(
    prediction: np.ndarray,
    output_path: str,
    reference_image_path: Optional[str] = None
):
    """Save prediction as NIfTI file.

    Args:
        prediction: Predicted mask array (H, W) or (H, W, 1)
        output_path: Path to save the prediction
        reference_image_path: Optional reference image to copy metadata from
    """
    # Remove channel dimension if present
    if prediction.ndim == 3 and prediction.shape[-1] == 1:
        prediction = prediction[:, :, 0]

    # Convert to SimpleITK image
    sitk_prediction = sitk.GetImageFromArray(prediction.astype(np.float32))

    # Copy metadata from reference if provided
    if reference_image_path and Path(reference_image_path).exists():
        reference = sitk.ReadImage(reference_image_path)
        sitk_prediction.CopyInformation(reference)

    # Save
    sitk.WriteImage(sitk_prediction, output_path)


def perform_inference_single_model(
    model_path: str,
    input_dir: str,
    output_dir: str,
    image_size: int = 256,
    num_channels: int = 1,
    use_intensity_windowing: bool = False,
    threshold: float = 0.5,
    save_probability: bool = True
):
    """Perform inference using a single model.

    Args:
        model_path: Path to trained model
        input_dir: Directory containing input images
        output_dir: Directory to save predictions
        image_size: Input image size
        num_channels: Number of image channels
        use_intensity_windowing: Whether to apply intensity windowing
        threshold: Threshold for binary segmentation
        save_probability: Whether to save probability maps in addition to binary masks
    """
    print(f"\n{'='*80}")
    print(" Single Model Inference")
    print(f"{'='*80}\n")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"[i] Loading model from {model_path}")
    custom_objects = {
        'dice_loss': dice_loss,
        'dice_coefficient': dice_coefficient,
        'precision': precision,
        'recall': recall
    }
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"[+] Model loaded successfully")

    # Find all images in input directory
    input_path = Path(input_dir)
    image_files = list(input_path.rglob('*.nii.gz'))

    if len(image_files) == 0:
        print(f"[!] No .nii.gz files found in {input_dir}")
        return

    print(f"[i] Found {len(image_files)} images")

    # Create loader
    loader = MedicalImageLoader(
        data_dir=str(input_path.parent),  # Parent since we'll use full paths
        image_size=image_size,
        num_channels=num_channels,
        use_intensity_windowing=use_intensity_windowing
    )

    # Process each image
    for i, img_file in enumerate(image_files, 1):
        print(f"\n[i] Processing image {i}/{len(image_files)}: {img_file.name}")

        # Load image
        image = loader.load_single_image(str(img_file))
        image_batch = np.expand_dims(image, axis=0)  # Add batch dimension

        # Predict
        prediction = model.predict(image_batch, verbose=0)[0]  # Remove batch dimension

        # Save probability map
        if save_probability:
            prob_output = output_dir / f"{img_file.stem}_probability.nii.gz"
            save_prediction_as_nifti(
                prediction,
                str(prob_output),
                str(img_file)
            )
            print(f"[+] Probability map saved to {prob_output.name}")

        # Save binary mask
        binary_prediction = (prediction >= threshold).astype(np.float32)
        binary_output = output_dir / f"{img_file.stem}_segmentation.nii.gz"
        save_prediction_as_nifti(
            binary_prediction,
            str(binary_output),
            str(img_file)
        )
        print(f"[+] Binary mask saved to {binary_output.name}")

    print(f"\n[+] Inference complete! Results saved to {output_dir}")


def perform_ensemble_inference(
    config_path: str,
    model_dir: str,
    input_dir: str,
    output_dir: str,
    threshold: float = 0.5,
    ensemble_method: str = 'mean'
):
    """Perform inference using ensemble of expert models.

    Args:
        config_path: Path to configuration file
        model_dir: Directory containing expert models
        input_dir: Directory containing input images
        output_dir: Directory to save predictions
        threshold: Threshold for binary segmentation
        ensemble_method: Method for ensembling ('mean', 'median', 'max')
    """
    print(f"\n{'='*80}")
    print(" Ensemble Model Inference")
    print(f"{'='*80}\n")

    # Load configuration
    config = load_config(config_path)
    dataset_config = config['dataset']
    data_config = config['data']

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all expert models
    model_dir = Path(model_dir)
    num_experts = dataset_config['num_experts']

    print(f"[i] Loading {num_experts} expert models...")
    models = []
    custom_objects = {
        'dice_loss': dice_loss,
        'dice_coefficient': dice_coefficient,
        'precision': precision,
        'recall': recall
    }

    for expert_id in range(1, num_experts + 1):
        expert_dir = model_dir / f"expert_{expert_id:02d}"
        model_path = expert_dir / 'best_model.keras'

        if not model_path.exists():
            model_path = expert_dir / 'final_model.keras'

        if not model_path.exists():
            print(f"[!] Warning: Model not found for expert {expert_id}")
            continue

        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        models.append(model)
        print(f"[+] Loaded model for expert {expert_id}")

    if len(models) == 0:
        print("[!] No models loaded. Exiting.")
        return

    # Find images
    input_path = Path(input_dir)
    image_files = list(input_path.rglob('*.nii.gz'))

    if len(image_files) == 0:
        print(f"[!] No .nii.gz files found in {input_dir}")
        return

    print(f"[i] Found {len(image_files)} images")

    # Create loader
    loader = MedicalImageLoader(
        data_dir=str(input_path.parent),
        image_size=data_config['image_size'],
        num_channels=dataset_config['num_channels'],
        use_intensity_windowing=data_config.get('use_intensity_windowing', False)
    )

    # Process each image
    for i, img_file in enumerate(image_files, 1):
        print(f"\n[i] Processing image {i}/{len(image_files)}: {img_file.name}")

        # Load image
        image = loader.load_single_image(str(img_file))
        image_batch = np.expand_dims(image, axis=0)

        # Get predictions from all models
        expert_predictions = []
        for model in models:
            pred = model.predict(image_batch, verbose=0)[0]
            expert_predictions.append(pred)

        # Ensemble predictions
        from med_seg.evaluation import ensemble_predictions
        ensembled_pred = ensemble_predictions(expert_predictions, method=ensemble_method)

        # Save probability map
        prob_output = output_dir / f"{img_file.stem}_ensemble_probability.nii.gz"
        save_prediction_as_nifti(
            ensembled_pred,
            str(prob_output),
            str(img_file)
        )
        print(f"[+] Ensemble probability saved to {prob_output.name}")

        # Save binary mask
        binary_pred = (ensembled_pred >= threshold).astype(np.float32)
        binary_output = output_dir / f"{img_file.stem}_ensemble_segmentation.nii.gz"
        save_prediction_as_nifti(
            binary_pred,
            str(binary_output),
            str(img_file)
        )
        print(f"[+] Ensemble segmentation saved to {binary_output.name}")

    print(f"\n[+] Ensemble inference complete! Results saved to {output_dir}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description='Perform inference on medical images'
    )

    # Common arguments
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing images'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for predictions'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for binary segmentation (default: 0.5)'
    )
    parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help='GPU ID to use'
    )

    # Single model or ensemble
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--model',
        type=str,
        help='Path to single trained model'
    )
    mode_group.add_argument(
        '--ensemble',
        action='store_true',
        help='Use ensemble of expert models (requires --config and --model-dir)'
    )

    # Ensemble-specific arguments
    parser.add_argument(
        '--config',
        type=str,
        help='Configuration file (required for ensemble mode)'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        help='Directory containing expert models (required for ensemble mode)'
    )
    parser.add_argument(
        '--ensemble-method',
        type=str,
        default='mean',
        choices=['mean', 'median', 'max'],
        help='Ensemble method (default: mean)'
    )

    # Single model-specific arguments
    parser.add_argument(
        '--image-size',
        type=int,
        default=256,
        help='Input image size (single model mode, default: 256)'
    )
    parser.add_argument(
        '--channels',
        type=int,
        default=1,
        help='Number of image channels (single model mode, default: 1)'
    )
    parser.add_argument(
        '--intensity-windowing',
        action='store_true',
        help='Apply intensity windowing (single model mode)'
    )

    args = parser.parse_args()

    # Set GPU
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    try:
        if args.ensemble:
            # Ensemble mode
            if not args.config or not args.model_dir:
                print("[!] Error: --config and --model-dir required for ensemble mode")
                sys.exit(1)

            perform_ensemble_inference(
                config_path=args.config,
                model_dir=args.model_dir,
                input_dir=args.input,
                output_dir=args.output,
                threshold=args.threshold,
                ensemble_method=args.ensemble_method
            )
        else:
            # Single model mode
            perform_inference_single_model(
                model_path=args.model,
                input_dir=args.input,
                output_dir=args.output,
                image_size=args.image_size,
                num_channels=args.channels,
                use_intensity_windowing=args.intensity_windowing,
                threshold=args.threshold
            )

    except Exception as e:
        print(f"[!] Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
