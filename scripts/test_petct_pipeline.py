#!/usr/bin/env python
"""Test PET/CT data loading and preprocessing pipeline.

This script verifies that the complete data pipeline works correctly
with synthetic data before using it for training.

Usage:
    python scripts/test_petct_pipeline.py --data-dir data/synthetic
"""

import argparse
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from med_seg.data import PETCTLoader, PETCTPreprocessor
from med_seg.data.petct_generator import PETCTDataGenerator


def test_loader(data_dir):
    """Test PETCTLoader functionality."""
    print("\n[TEST 1] PETCTLoader")
    print("=" * 70)

    loader = PETCTLoader(data_dir)
    print(f"Loader: {loader}")
    print(f"Number of patients: {len(loader)}")

    # Test loading single patient
    print("\n[*] Loading patient 0...")
    data = loader.load_patient_3d(0)

    print(f"CT shape: {data['ct'].shape}")
    print(f"PET shape: {data['pet'].shape}")
    print(f"SEG shape: {data['seg'].shape}")
    print(f"Spacing: {data['spacing']}")
    print(f"CT range: [{data['ct'].min():.1f}, {data['ct'].max():.1f}]")
    print(f"PET range: [{data['pet'].min():.2f}, {data['pet'].max():.2f}]")
    print(f"Tumor voxels: {data['seg'].sum()}")

    # Test 2D slice extraction
    print("\n[*] Extracting 2D slices (axial)...")
    ct_slices, pet_slices, seg_slices = loader.extract_2d_slices(0, axis=0)

    print(f"CT slices shape: {ct_slices.shape}")
    print(f"PET slices shape: {pet_slices.shape}")
    print(f"SEG slices shape: {seg_slices.shape}")

    # Test tumor slice finding
    print("\n[*] Finding tumor slices...")
    tumor_slices = loader.get_tumor_slices(0, axis=0)
    print(f"Tumor slices: {tumor_slices}")
    print(f"Number of tumor slices: {len(tumor_slices)}")

    # Test statistics
    print("\n[*] Computing dataset statistics...")
    stats = loader.get_statistics()
    print(f"CT range: [{stats['ct_range']['min']:.1f}, {stats['ct_range']['max']:.1f}]")
    print(f"PET range: [{stats['pet_range']['min']:.2f}, {stats['pet_range']['max']:.2f}]")
    print(f"Avg tumor prevalence: {stats['avg_tumor_prevalence']:.3f}%")

    print("\n[+] PETCTLoader test PASSED")
    return loader


def test_preprocessor(loader):
    """Test PETCTPreprocessor functionality."""
    print("\n[TEST 2] PETCTPreprocessor")
    print("=" * 70)

    preprocessor = PETCTPreprocessor(
        target_size=(256, 256), ct_window_center=0, ct_window_width=400, suv_max=15
    )
    print(f"Preprocessor: {preprocessor}")

    # Load one slice
    data = loader.load_patient_3d(0)
    ct_slice = data["ct"][75]  # Middle slice
    pet_slice = data["pet"][75]
    seg_slice = data["seg"][75]

    print("\n[*] Input shapes:")
    print(f"  CT: {ct_slice.shape}")
    print(f"  PET: {pet_slice.shape}")
    print(f"  SEG: {seg_slice.shape}")

    # Test 2D preprocessing
    print("\n[*] Preprocessing single slice...")
    input_processed, seg_processed = preprocessor.preprocess_2d_slice(
        ct_slice, pet_slice, seg_slice
    )

    print(f"  Processed input shape: {input_processed.shape}")
    print(f"  Processed input range: [{input_processed.min():.3f}, {input_processed.max():.3f}]")
    print(
        f"  CT channel range: [{input_processed[:,:,0].min():.3f}, {input_processed[:,:,0].max():.3f}]"
    )
    print(
        f"  PET channel range: [{input_processed[:,:,1].min():.3f}, {input_processed[:,:,1].max():.3f}]"
    )
    print(f"  Processed seg shape: {seg_processed.shape}")
    print(f"  Processed seg range: [{seg_processed.min():.0f}, {seg_processed.max():.0f}]")

    # Test batch preprocessing
    print("\n[*] Preprocessing batch of slices...")
    ct_batch, pet_batch, seg_batch = loader.extract_2d_slices(0, axis=0, slice_indices=[70, 75, 80])

    inputs_batch, segs_batch = preprocessor.preprocess_batch_2d(ct_batch, pet_batch, seg_batch)

    print(f"  Batch input shape: {inputs_batch.shape}")
    print(f"  Batch seg shape: {segs_batch.shape}")

    # Test CT window presets
    print("\n[*] Testing CT window presets...")
    presets = preprocessor.get_ct_window_presets()
    print(f"  Available presets: {list(presets.keys())}")

    preprocessor.set_ct_window("lung")
    print(f"  Changed to lung window: {preprocessor}")

    print("\n[+] PETCTPreprocessor test PASSED")
    return preprocessor


def test_generator(loader, preprocessor):
    """Test PETCTDataGenerator functionality."""
    print("\n[TEST 3] PETCTDataGenerator")
    print("=" * 70)

    generator = PETCTDataGenerator(
        loader=loader,
        preprocessor=preprocessor,
        batch_size=4,
        axis=0,
        shuffle=True,
        augment=True,
        tumor_only=True,
        min_tumor_voxels=10,
    )

    print(f"Generator: {generator}")
    print(f"Number of batches: {len(generator)}")
    print(f"Total slices: {len(generator.slice_index)}")

    # Test getting a batch
    print("\n[*] Getting first batch...")
    inputs, targets = generator[0]

    print(f"  Inputs shape: {inputs.shape}")
    print(f"  Inputs dtype: {inputs.dtype}")
    print(f"  Inputs range: [{inputs.min():.3f}, {inputs.max():.3f}]")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Targets dtype: {targets.dtype}")
    print(f"  Targets range: [{targets.min():.0f}, {targets.max():.0f}]")

    # Check tumor presence
    num_tumor_pixels = targets.sum()
    print(f"  Tumor pixels in batch: {int(num_tumor_pixels)}")

    # Test class weights
    print("\n[*] Computing class weights...")
    class_weights = generator.get_class_weights()
    print(f"  Class weights: {class_weights}")
    print(f"  Background weight: {class_weights[0]:.2f}")
    print(f"  Tumor weight: {class_weights[1]:.2f}")

    # Test epoch end
    print("\n[*] Testing epoch end shuffle...")
    first_indices = generator.slice_index[:5].copy()
    generator.on_epoch_end()
    second_indices = generator.slice_index[:5]
    indices_changed = not all(first_indices[i] == second_indices[i] for i in range(5))
    print(f"  Indices shuffled: {indices_changed}")

    print("\n[+] PETCTDataGenerator test PASSED")
    return generator


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test PET/CT data pipeline")
    parser.add_argument(
        "--data-dir", type=str, default="data/synthetic", help="Directory containing patient data"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"[!] Error: Data directory not found: {data_dir}")
        print("[*] Generate synthetic data first:")
        print(f"    python scripts/create_synthetic_petct.py --output {data_dir} --num-patients 3")
        return 1

    print("[+] PET/CT Data Pipeline Test Suite")
    print("=" * 70)
    print(f"Data directory: {data_dir}")

    try:
        # Test 1: Loader
        loader = test_loader(str(data_dir))

        # Test 2: Preprocessor
        preprocessor = test_preprocessor(loader)

        # Test 3: Generator
        test_generator(loader, preprocessor)

        print("\n" + "=" * 70)
        print("[+] ALL TESTS PASSED")
        print("=" * 70)
        print("\n[*] Data pipeline is ready for training!")
        print("[*] You can now train models using:")
        print("    - PETCTLoader for data loading")
        print("    - PETCTPreprocessor for preprocessing")
        print("    - PETCTDataGenerator for batch generation")

        return 0

    except Exception as e:
        print(f"\n[!] TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
