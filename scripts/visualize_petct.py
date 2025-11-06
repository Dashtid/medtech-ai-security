#!/usr/bin/env python
"""Visualize PET/CT data with segmentation overlays.

This script creates slice visualizations of PET/CT data with tumor
segmentation overlays for quality inspection.

Usage:
    python scripts/visualize_petct.py --data-dir data/synthetic --patient patient_001 --output visualizations/
"""

import argparse
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def load_nifti(file_path):
    """Load NIfTI file as numpy array.

    Args:
        file_path: Path to NIfTI file

    Returns:
        Numpy array (z, y, x)
    """
    image = sitk.ReadImage(str(file_path))
    array = sitk.GetArrayFromImage(image)
    return array


def normalize_ct(ct_array, window_center=0, window_width=400):
    """Normalize CT with windowing.

    Args:
        ct_array: CT volume
        window_center: Window center in HU
        window_width: Window width in HU

    Returns:
        Normalized array [0, 1]
    """
    ct_min = window_center - window_width / 2
    ct_max = window_center + window_width / 2
    ct_norm = np.clip(ct_array, ct_min, ct_max)
    ct_norm = (ct_norm - ct_min) / (ct_max - ct_min)
    return ct_norm


def normalize_suv(suv_array, max_suv=15):
    """Normalize SUV for visualization.

    Args:
        suv_array: SUV volume
        max_suv: Maximum SUV for clipping

    Returns:
        Normalized array [0, 1]
    """
    suv_norm = np.clip(suv_array, 0, max_suv)
    suv_norm = suv_norm / max_suv
    return suv_norm


def create_pet_colormap():
    """Create PET-style colormap (hot colors).

    Returns:
        Matplotlib colormap
    """
    colors = ["black", "darkblue", "blue", "cyan", "green", "yellow", "orange", "red", "white"]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list("pet", colors, N=n_bins)
    return cmap


def visualize_slice(ct_slice, suv_slice, seg_slice, slice_idx, output_path=None):
    """Visualize single slice with CT, PET, and overlay.

    Args:
        ct_slice: CT slice (2D array)
        suv_slice: SUV slice (2D array)
        seg_slice: Segmentation slice (2D array)
        slice_idx: Slice index
        output_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # CT
    ax = axes[0, 0]
    ct_norm = normalize_ct(ct_slice)
    ax.imshow(ct_norm, cmap="gray", origin="lower")
    ax.set_title(f"CT (Slice {slice_idx})", fontsize=12, weight="bold")
    ax.axis("off")

    # PET/SUV
    ax = axes[0, 1]
    suv_norm = normalize_suv(suv_slice)
    pet_cmap = create_pet_colormap()
    im_pet = ax.imshow(suv_norm, cmap=pet_cmap, origin="lower")
    ax.set_title(f"PET/SUV (Slice {slice_idx})", fontsize=12, weight="bold")
    ax.axis("off")
    plt.colorbar(im_pet, ax=ax, fraction=0.046, pad=0.04, label="SUV (norm)")

    # CT + Segmentation overlay
    ax = axes[1, 0]
    ax.imshow(ct_norm, cmap="gray", origin="lower")
    # Overlay segmentation in red
    seg_overlay = np.ma.masked_where(seg_slice == 0, seg_slice)
    ax.imshow(seg_overlay, cmap="Reds", alpha=0.5, origin="lower")
    ax.set_title("CT + Segmentation", fontsize=12, weight="bold")
    ax.axis("off")

    # PET + Segmentation overlay
    ax = axes[1, 1]
    ax.imshow(suv_norm, cmap=pet_cmap, origin="lower")
    # Overlay segmentation contours

    if seg_slice.max() > 0:
        # Draw contours
        from scipy import ndimage

        contours = ndimage.binary_dilation(seg_slice) ^ seg_slice
        contours_overlay = np.ma.masked_where(contours == 0, contours)
        ax.imshow(contours_overlay, cmap="spring", alpha=1.0, origin="lower")
    ax.set_title("PET + Segmentation Contours", fontsize=12, weight="bold")
    ax.axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"    [+] Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def find_tumor_slices(seg_volume, num_slices=5):
    """Find slices with tumor for visualization.

    Args:
        seg_volume: Segmentation volume (z, y, x)
        num_slices: Number of slices to select

    Returns:
        List of slice indices
    """
    # Count tumor voxels per slice
    tumor_counts = np.sum(seg_volume, axis=(1, 2))

    # Find slices with tumors
    tumor_slices = np.where(tumor_counts > 0)[0]

    if len(tumor_slices) == 0:
        # No tumors, return middle slices
        z_dim = seg_volume.shape[0]
        return list(range(z_dim // 2 - num_slices // 2, z_dim // 2 + num_slices // 2 + 1))

    # Select slices with most tumor content
    top_slices = tumor_slices[np.argsort(tumor_counts[tumor_slices])[-num_slices:]]
    return sorted(top_slices)


def visualize_patient(patient_dir, output_dir=None, num_slices=5):
    """Visualize patient PET/CT data.

    Args:
        patient_dir: Path to patient directory
        output_dir: Optional output directory for saving figures
        num_slices: Number of slices to visualize
    """
    print(f"\n[*] Visualizing {patient_dir.name}...")

    # Load volumes
    ct = load_nifti(patient_dir / "CT.nii.gz")
    suv = load_nifti(patient_dir / "SUV.nii.gz")
    seg = load_nifti(patient_dir / "SEG.nii.gz")

    print(f"    Loaded volumes: CT {ct.shape}, SUV {suv.shape}, SEG {seg.shape}")

    # Find interesting slices (with tumors)
    slice_indices = find_tumor_slices(seg, num_slices)
    print(f"    Visualizing slices: {slice_indices}")

    # Create output directory if saving
    if output_dir:
        patient_output_dir = output_dir / patient_dir.name
        patient_output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize each slice
    for slice_idx in slice_indices:
        ct_slice = ct[slice_idx, :, :]
        suv_slice = suv[slice_idx, :, :]
        seg_slice = seg[slice_idx, :, :]

        output_path = None
        if output_dir:
            output_path = patient_output_dir / f"slice_{slice_idx:03d}.png"

        visualize_slice(ct_slice, suv_slice, seg_slice, slice_idx, output_path)

    print("    [+] Visualization complete!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Visualize PET/CT data with segmentation")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Directory containing patient folders"
    )
    parser.add_argument("--patient", type=str, help="Specific patient to visualize (default: all)")
    parser.add_argument(
        "--output", type=str, help="Output directory for saving figures (default: display only)"
    )
    parser.add_argument(
        "--num-slices",
        type=int,
        default=5,
        help="Number of slices to visualize per patient (default: 5)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output) if args.output else None

    if not data_dir.exists():
        print(f"[!] Error: Directory not found: {data_dir}")
        return

    print("[+] PET/CT Visualization Tool")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    if output_dir:
        print(f"Output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        print("Output: Display only (not saving)")
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

    # Visualize each patient
    for patient_dir in patient_dirs:
        if not patient_dir.exists():
            print(f"[!] Patient directory not found: {patient_dir}")
            continue

        visualize_patient(patient_dir, output_dir, args.num_slices)

    print("\n" + "=" * 70)
    print(f"[+] Complete! Visualized {len(patient_dirs)} patient(s)")
    if output_dir:
        print(f"[+] Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
