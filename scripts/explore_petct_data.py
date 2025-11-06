#!/usr/bin/env python
"""Explore PET/CT dataset structure and statistics.

This script loads PET/CT data, prints metadata, and displays statistics
to help understand the data before training.

Usage:
    python scripts/explore_petct_data.py --data-dir data/synthetic
"""

import argparse
from pathlib import Path
import numpy as np
import SimpleITK as sitk


def load_nifti_info(file_path):
    """Load NIfTI file and extract metadata.

    Args:
        file_path: Path to NIfTI file

    Returns:
        Dictionary with image info
    """
    image = sitk.ReadImage(str(file_path))
    array = sitk.GetArrayFromImage(image)

    return {
        "path": file_path,
        "shape": image.GetSize(),  # (x, y, z)
        "spacing": image.GetSpacing(),  # (x, y, z) in mm
        "origin": image.GetOrigin(),
        "direction": image.GetDirection(),
        "array_shape": array.shape,  # (z, y, x) - numpy convention
        "dtype": array.dtype,
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "array": array,
    }


def print_volume_info(name, info):
    """Print formatted volume information.

    Args:
        name: Volume name (e.g., "CT", "PET")
        info: Dictionary from load_nifti_info()
    """
    print(f"\n  [{name}]")
    print(f"    File: {info['path'].name}")
    print(f"    Shape (ITK): {info['shape'][0]}x{info['shape'][1]}x{info['shape'][2]}")
    print(f"    Shape (numpy): {info['array_shape']}")
    print(
        f"    Spacing: {info['spacing'][0]:.2f}x{info['spacing'][1]:.2f}x{info['spacing'][2]:.2f} mm"
    )
    print(f"    Data type: {info['dtype']}")
    print(f"    Value range: [{info['min']:.2f}, {info['max']:.2f}]")
    print(f"    Mean +/- Std: {info['mean']:.2f} +/- {info['std']:.2f}")


def analyze_segmentation(seg_array):
    """Analyze segmentation mask statistics.

    Args:
        seg_array: Segmentation mask array

    Returns:
        Dictionary with segmentation stats
    """
    total_voxels = seg_array.size
    tumor_voxels = np.sum(seg_array > 0)
    tumor_percent = (tumor_voxels / total_voxels) * 100

    # Find bounding box of tumors
    tumor_coords = np.where(seg_array > 0)
    if len(tumor_coords[0]) > 0:
        bbox = {
            "z_min": int(tumor_coords[0].min()),
            "z_max": int(tumor_coords[0].max()),
            "y_min": int(tumor_coords[1].min()),
            "y_max": int(tumor_coords[1].max()),
            "x_min": int(tumor_coords[2].min()),
            "x_max": int(tumor_coords[2].max()),
        }
    else:
        bbox = None

    return {
        "total_voxels": total_voxels,
        "tumor_voxels": tumor_voxels,
        "tumor_percent": tumor_percent,
        "bbox": bbox,
    }


def explore_patient(patient_dir):
    """Explore one patient's data.

    Args:
        patient_dir: Path to patient directory
    """
    print(f"\n{'='*70}")
    print(f"Patient: {patient_dir.name}")
    print("=" * 70)

    # Check what files exist
    files = {
        "CT": patient_dir / "CT.nii.gz",
        "CTres": patient_dir / "CTres.nii.gz",
        "SUV": patient_dir / "SUV.nii.gz",
        "SEG": patient_dir / "SEG.nii.gz",
    }

    available_files = {k: v for k, v in files.items() if v.exists()}

    if not available_files:
        print(f"  [!] No NIfTI files found in {patient_dir}")
        return

    print(f"\n  Available files: {', '.join(available_files.keys())}")

    # Load and analyze each volume
    volumes = {}
    for name, path in available_files.items():
        try:
            volumes[name] = load_nifti_info(path)
            print_volume_info(name, volumes[name])
        except Exception as e:
            print(f"  [!] Error loading {name}: {e}")

    # Special analysis for segmentation
    if "SEG" in volumes:
        seg_stats = analyze_segmentation(volumes["SEG"]["array"])
        print("\n  [Segmentation Analysis]")
        print(f"    Total voxels: {seg_stats['total_voxels']:,}")
        print(
            f"    Tumor voxels: {seg_stats['tumor_voxels']:,} ({seg_stats['tumor_percent']:.3f}%)"
        )

        if seg_stats["bbox"]:
            bbox = seg_stats["bbox"]
            print("    Tumor bounding box (z,y,x):")
            print(
                f"      Z: {bbox['z_min']} - {bbox['z_max']} (size: {bbox['z_max']-bbox['z_min']+1})"
            )
            print(
                f"      Y: {bbox['y_min']} - {bbox['y_max']} (size: {bbox['y_max']-bbox['y_min']+1})"
            )
            print(
                f"      X: {bbox['x_min']} - {bbox['x_max']} (size: {bbox['x_max']-bbox['x_min']+1})"
            )

    # Cross-modality analysis
    if "SUV" in volumes and "SEG" in volumes:
        suv_array = volumes["SUV"]["array"]
        seg_array = volumes["SEG"]["array"]

        # SUV values within tumors
        tumor_suv = suv_array[seg_array > 0]
        background_suv = suv_array[seg_array == 0]

        print("\n  [PET/Segmentation Analysis]")
        if len(tumor_suv) > 0:
            print(f"    Tumor SUV: {tumor_suv.mean():.2f} +/- {tumor_suv.std():.2f}")
            print(f"    Tumor SUV range: [{tumor_suv.min():.2f}, {tumor_suv.max():.2f}]")
        if len(background_suv) > 0:
            print(f"    Background SUV: {background_suv.mean():.2f} +/- {background_suv.std():.2f}")

    # Shape consistency check
    if len(volumes) > 1:
        shapes = {name: vol["array_shape"] for name, vol in volumes.items()}
        unique_shapes = set(shapes.values())

        print("\n  [Shape Consistency Check]")
        if len(unique_shapes) == 1:
            print(f"    [+] All volumes have matching shapes: {unique_shapes.pop()}")
        else:
            print("    [!] WARNING: Volumes have different shapes!")
            for name, shape in shapes.items():
                print(f"        {name}: {shape}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Explore PET/CT dataset structure")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Directory containing patient folders"
    )
    parser.add_argument("--patient", type=str, help="Specific patient to explore (default: all)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"[!] Error: Directory not found: {data_dir}")
        return

    print("[+] PET/CT Data Explorer")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print()

    # Find patient directories
    if args.patient:
        patient_dirs = [data_dir / args.patient]
    else:
        patient_dirs = sorted(
            [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("patient")]
        )

    if not patient_dirs:
        print(f"[!] No patient directories found in {data_dir}")
        return

    print(f"[*] Found {len(patient_dirs)} patient(s)")

    # Explore each patient
    for patient_dir in patient_dirs:
        explore_patient(patient_dir)

    print("\n" + "=" * 70)
    print(f"[+] Exploration complete! Analyzed {len(patient_dirs)} patient(s)")


if __name__ == "__main__":
    main()
