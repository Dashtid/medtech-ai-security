#!/usr/bin/env python
"""Create synthetic PET/CT data for testing the pipeline.

This generates realistic-looking PET/CT volumes with tumors for development
and testing before working with real patient data.

Usage:
    python scripts/create_synthetic_petct.py --output data/synthetic --num-patients 3
"""

import argparse
from pathlib import Path
import numpy as np
import SimpleITK as sitk


def create_synthetic_ct_volume(shape=(200, 200, 150)):
    """Create synthetic CT volume with realistic Hounsfield Units.

    Args:
        shape: Volume shape (width, height, depth)

    Returns:
        CT volume as numpy array
    """
    print("    [*] Generating CT volume...")

    # Initialize with air (-1000 HU)
    ct = np.ones(shape, dtype=np.float32) * -1000

    # Create body outline (ellipse in axial plane)
    w, h, d = shape
    for z in range(d):
        for y in range(h):
            for x in range(w):
                # Ellipse equation for body
                if ((x - w / 2) ** 2 / (w / 2.5) ** 2 + (y - h / 2) ** 2 / (h / 2.8) ** 2) < 1:
                    # Soft tissue: -50 to +50 HU
                    ct[x, y, z] = np.random.normal(0, 20)

    # Add some organs with different densities
    # Liver region (40-60 HU)
    liver_z = slice(int(d * 0.4), int(d * 0.6))
    liver_x = slice(int(w * 0.4), int(w * 0.7))
    liver_y = slice(int(h * 0.3), int(h * 0.6))
    ct[liver_x, liver_y, liver_z] = np.random.normal(
        50, 10, size=ct[liver_x, liver_y, liver_z].shape
    )

    # Lungs (with some air, -800 to -600 HU)
    lung_z = slice(int(d * 0.5), int(d * 0.8))
    lung_left_x = slice(int(w * 0.25), int(w * 0.45))
    lung_right_x = slice(int(w * 0.55), int(w * 0.75))
    lung_y = slice(int(h * 0.2), int(h * 0.7))

    ct[lung_left_x, lung_y, lung_z] = np.random.normal(
        -700, 50, size=ct[lung_left_x, lung_y, lung_z].shape
    )
    ct[lung_right_x, lung_y, lung_z] = np.random.normal(
        -700, 50, size=ct[lung_right_x, lung_y, lung_z].shape
    )

    # Add some bones (200-400 HU)
    # Spine
    spine_x = slice(int(w * 0.45), int(w * 0.55))
    spine_y = slice(int(h * 0.6), int(h * 0.8))
    ct[spine_x, spine_y, :] = np.random.normal(300, 50, size=ct[spine_x, spine_y, :].shape)

    # Add noise
    ct += np.random.normal(0, 5, shape)

    return ct


def create_synthetic_pet_volume(shape=(200, 200, 150), ct_volume=None):
    """Create synthetic PET/SUV volume with tumor lesions.

    Args:
        shape: Volume shape (width, height, depth)
        ct_volume: Optional CT volume to guide PET generation

    Returns:
        Tuple of (PET volume, segmentation mask)
    """
    print("    [*] Generating PET/SUV volume...")

    # Background SUV (normal tissue: 1-2)
    pet = np.random.normal(1.5, 0.3, shape).astype(np.float32)
    pet = np.clip(pet, 0, None)

    # Create segmentation mask
    mask = np.zeros(shape, dtype=np.uint8)

    # Add 2-4 random tumor lesions with high SUV
    num_tumors = np.random.randint(2, 5)
    print(f"    [*] Adding {num_tumors} tumor lesions...")

    w, h, d = shape
    for i in range(num_tumors):
        # Random tumor location (avoid edges)
        cx = np.random.randint(int(w * 0.2), int(w * 0.8))
        cy = np.random.randint(int(h * 0.2), int(h * 0.8))
        cz = np.random.randint(int(d * 0.3), int(d * 0.7))

        # Random tumor size (5-15mm diameter)
        radius = np.random.randint(3, 8)

        # Random tumor SUV (typically 4-12 for malignant lesions)
        tumor_suv = np.random.uniform(5, 12)

        print(f"        Tumor {i+1}: center=({cx},{cy},{cz}), radius={radius}, SUV={tumor_suv:.1f}")

        # Create spherical tumor
        for x in range(max(0, cx - radius), min(w, cx + radius + 1)):
            for y in range(max(0, cy - radius), min(h, cy + radius + 1)):
                for z in range(max(0, cz - radius), min(d, cz + radius + 1)):
                    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
                    if dist <= radius:
                        # Gaussian falloff from center
                        intensity = np.exp(-(dist**2) / (2 * (radius / 2) ** 2))
                        pet[x, y, z] = tumor_suv * intensity + pet[x, y, z] * (1 - intensity)
                        mask[x, y, z] = 1

    # Add some physiological uptake (brain, heart, bladder)
    # Brain (SUV 5-8)
    brain_z = slice(int(d * 0.85), int(d * 0.95))
    brain_region = (
        slice(int(w * 0.35), int(w * 0.65)),
        slice(int(h * 0.35), int(h * 0.65)),
        brain_z,
    )
    pet[brain_region] = np.random.normal(6.5, 1, size=pet[brain_region].shape)

    # Heart (SUV 3-5)
    heart_region = (
        slice(int(w * 0.45), int(w * 0.60)),
        slice(int(h * 0.40), int(h * 0.55)),
        slice(int(d * 0.60), int(d * 0.70)),
    )
    pet[heart_region] = np.random.normal(4, 0.5, size=pet[heart_region].shape)

    # Ensure non-negative
    pet = np.clip(pet, 0, None)

    # Add Poisson noise (PET is count-based)
    pet = np.random.poisson(pet * 10) / 10.0

    return pet.astype(np.float32), mask


def save_as_nifti(volume, output_path, spacing=(2.0, 2.0, 3.0)):
    """Save numpy array as NIfTI file.

    Args:
        volume: Numpy array
        output_path: Output file path
        spacing: Voxel spacing in mm (x, y, z)
    """
    # Convert numpy to SimpleITK image
    image = sitk.GetImageFromArray(volume.transpose(2, 1, 0))  # ITK uses (z, y, x)
    image.SetSpacing(spacing)

    # Write to file
    sitk.WriteImage(image, str(output_path))
    print(f"        [+] Saved: {output_path}")


def create_synthetic_patient(patient_id, output_dir, shape=(200, 200, 150)):
    """Create one synthetic patient with PET, CT, and segmentation.

    Args:
        patient_id: Patient identifier
        output_dir: Output directory
        shape: Volume shape
    """
    print(f"\n[*] Creating Patient {patient_id}...")

    # Create patient directory
    patient_dir = output_dir / f"patient_{patient_id:03d}"
    patient_dir.mkdir(parents=True, exist_ok=True)

    # Generate volumes
    ct = create_synthetic_ct_volume(shape)
    suv, seg = create_synthetic_pet_volume(shape, ct)

    # Resample CT to match PET resolution (for CTres)
    # In reality, CT would be higher res, but for simplicity we'll just copy
    ct_res = ct.copy()

    # Save as NIfTI files
    print("    [*] Saving NIfTI files...")
    save_as_nifti(ct, patient_dir / "CT.nii.gz", spacing=(1.0, 1.0, 2.0))
    save_as_nifti(ct_res, patient_dir / "CTres.nii.gz", spacing=(2.0, 2.0, 3.0))
    save_as_nifti(suv, patient_dir / "SUV.nii.gz", spacing=(2.0, 2.0, 3.0))
    save_as_nifti(seg, patient_dir / "SEG.nii.gz", spacing=(2.0, 2.0, 3.0))

    # Print statistics
    print(f"    [*] CT range: [{ct.min():.1f}, {ct.max():.1f}] HU")
    print(f"    [*] SUV range: [{suv.min():.2f}, {suv.max():.2f}]")
    print(f"    [*] Tumor voxels: {seg.sum()} ({seg.sum()/seg.size*100:.2f}%)")
    print(f"    [+] Patient {patient_id} complete!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create synthetic PET/CT data for testing")
    parser.add_argument("--output", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument(
        "--num-patients", type=int, default=3, help="Number of synthetic patients to create"
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        default=[200, 200, 150],
        help="Volume shape (width height depth)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[+] Synthetic PET/CT Data Generator")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Number of patients: {args.num_patients}")
    print(f"Volume shape: {args.shape[0]}x{args.shape[1]}x{args.shape[2]}")
    print()

    # Create synthetic patients
    for i in range(1, args.num_patients + 1):
        create_synthetic_patient(i, output_dir, tuple(args.shape))

    print("\n" + "=" * 60)
    print(f"[+] Complete! Created {args.num_patients} synthetic patients")
    print(f"[+] Data saved to: {output_dir}")
    print()
    print("[*] Each patient contains:")
    print("    - CT.nii.gz: High-resolution CT volume")
    print("    - CTres.nii.gz: CT resampled to PET resolution")
    print("    - SUV.nii.gz: PET in SUV units")
    print("    - SEG.nii.gz: Binary tumor segmentation mask")


if __name__ == "__main__":
    main()
