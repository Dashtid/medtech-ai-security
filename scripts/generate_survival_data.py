#!/usr/bin/env python
"""Generate realistic survival data for PET/CT patients.

This script augments the synthetic PET/CT dataset with realistic survival times
based on tumor characteristics (volume, SUV intensity, location).

The survival model is inspired by clinical observations:
- Larger tumor volume → shorter survival
- Higher SUV (metabolic activity) → shorter survival
- Random baseline hazard with realistic censoring

Usage:
    python scripts/generate_survival_data.py --data-dir data/synthetic_v2 --output data/synthetic_v2_survival
"""

import argparse
from pathlib import Path
import sys
import json
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import SimpleITK as sitk
from typing import Dict


def calculate_tumor_features(seg_path: Path, suv_path: Path) -> Dict[str, float]:
    """Calculate tumor features for survival prediction.

    Args:
        seg_path: Path to segmentation mask
        suv_path: Path to PET SUV volume

    Returns:
        Dictionary of tumor features
    """
    # Load volumes
    seg = sitk.ReadImage(str(seg_path))
    suv = sitk.ReadImage(str(suv_path))

    seg_array = sitk.GetArrayFromImage(seg)
    suv_array = sitk.GetArrayFromImage(suv)

    # Calculate features
    tumor_mask = seg_array > 0

    # Tumor volume (in cm³)
    spacing = seg.GetSpacing()  # mm
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    voxel_volume_cm3 = voxel_volume_mm3 / 1000.0
    tumor_volume_cm3 = np.sum(tumor_mask) * voxel_volume_cm3

    # SUV statistics
    tumor_suvs = suv_array[tumor_mask]
    if len(tumor_suvs) > 0:
        suv_mean = float(np.mean(tumor_suvs))
        suv_max = float(np.max(tumor_suvs))
        suv_std = float(np.std(tumor_suvs))
    else:
        suv_mean = suv_max = suv_std = 0.0

    # Tumor location (centroid in normalized coords)
    if np.sum(tumor_mask) > 0:
        coords = np.where(tumor_mask)
        centroid_z = float(np.mean(coords[0])) / seg_array.shape[0]
        centroid_y = float(np.mean(coords[1])) / seg_array.shape[1]
        centroid_x = float(np.mean(coords[2])) / seg_array.shape[2]
    else:
        centroid_z = centroid_y = centroid_x = 0.5

    return {
        "tumor_volume_cm3": tumor_volume_cm3,
        "suv_mean": suv_mean,
        "suv_max": suv_max,
        "suv_std": suv_std,
        "centroid_z": centroid_z,
        "centroid_y": centroid_y,
        "centroid_x": centroid_x,
    }


def generate_survival_time(
    features: Dict[str, float],
    base_rate: float = 60.0,
    volume_effect: float = 0.5,
    suv_effect: float = 2.0,
    noise_std: float = 12.0,
) -> float:
    """Generate realistic survival time based on tumor features.

    Uses a Cox-like proportional hazards model:
    survival_time = baseline * exp(-beta1*volume - beta2*SUVmax) + noise

    Args:
        features: Dictionary of tumor features
        base_rate: Baseline survival time in months
        volume_effect: Beta coefficient for tumor volume
        suv_effect: Beta coefficient for SUVmax
        noise_std: Standard deviation of random noise

    Returns:
        Survival time in months
    """
    # Normalize features
    volume_norm = features["tumor_volume_cm3"] / 100.0  # Normalize to ~1.0
    suv_norm = features["suv_max"] / 10.0  # Normalize to ~1.0

    # Cox-like hazard model
    hazard = np.exp(-(volume_effect * volume_norm + suv_effect * suv_norm))

    # Baseline survival time modified by hazard
    expected_survival = base_rate * hazard

    # Add random noise (but keep positive)
    survival = expected_survival + np.random.normal(0, noise_std)
    survival = max(survival, 1.0)  # Minimum 1 month

    return survival


def generate_censoring(
    survival_time: float, study_duration: float = 60.0, censoring_prob: float = 0.3
) -> tuple:
    """Generate censoring information.

    In clinical studies, not all patients reach the endpoint (death).
    Some are "censored" because:
    - Study ends before event
    - Patient lost to follow-up
    - Patient withdraws

    Args:
        survival_time: True survival time
        study_duration: Study follow-up duration in months
        censoring_prob: Probability of random censoring

    Returns:
        (observed_time, event_occurred)
    """
    # Random censoring (lost to follow-up)
    if np.random.random() < censoring_prob:
        # Censored at random time before true survival
        observed_time = np.random.uniform(0, min(survival_time, study_duration))
        event_occurred = False
    # Censored by study end
    elif survival_time > study_duration:
        observed_time = study_duration
        event_occurred = False
    # Event observed
    else:
        observed_time = survival_time
        event_occurred = True

    return observed_time, event_occurred


def process_patient(
    patient_dir: Path, study_duration: float = 60.0, censoring_prob: float = 0.3
) -> Dict:
    """Process one patient to generate survival data.

    Args:
        patient_dir: Path to patient directory
        study_duration: Study follow-up duration
        censoring_prob: Probability of censoring

    Returns:
        Dictionary with survival information
    """
    patient_id = patient_dir.name

    # Paths
    seg_path = patient_dir / "SEG.nii.gz"
    suv_path = patient_dir / "SUV.nii.gz"

    if not seg_path.exists() or not suv_path.exists():
        print(f"[!] Skipping {patient_id}: missing files")
        return None

    # Calculate tumor features
    try:
        features = calculate_tumor_features(seg_path, suv_path)
    except Exception as e:
        print(f"[!] Error processing {patient_id}: {e}")
        return None

    # Generate survival time
    true_survival = generate_survival_time(features)

    # Generate censoring
    observed_time, event = generate_censoring(true_survival, study_duration, censoring_prob)

    # Create survival record
    survival_data = {
        "patient_id": patient_id,
        "observed_time_months": float(observed_time),
        "event_occurred": bool(event),
        "true_survival_months": float(true_survival),  # For analysis only
        "tumor_features": features,
    }

    return survival_data


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate survival data for PET/CT patients")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/synthetic_v2",
        help="Input directory with patient data",
    )
    parser.add_argument(
        "--output", type=str, help="Output directory (default: <data-dir>_survival)"
    )
    parser.add_argument(
        "--study-duration", type=float, default=60.0, help="Study follow-up duration in months"
    )
    parser.add_argument(
        "--censoring-prob", type=float, default=0.3, help="Probability of random censoring (0-1)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(str(args.data_dir) + "_survival")

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"[!] Error: Data directory not found: {data_dir}")
        return 1

    print("\n[+] PET/CT Survival Data Generation")
    print("=" * 70)
    print(f"Input directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Study duration: {args.study_duration} months")
    print(f"Censoring probability: {args.censoring_prob}")
    print(f"Random seed: {args.seed}")
    print()

    # Find all patient directories
    patient_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"[*] Found {len(patient_dirs)} patient directories")

    # Process each patient
    survival_records = []

    for patient_dir in patient_dirs:
        survival_data = process_patient(patient_dir, args.study_duration, args.censoring_prob)

        if survival_data:
            survival_records.append(survival_data)

            status = "EVENT" if survival_data["event_occurred"] else "CENSORED"
            print(
                f"  {survival_data['patient_id']}: "
                f"{survival_data['observed_time_months']:.1f} months ({status})"
            )

    if not survival_records:
        print("[!] No survival data generated")
        return 1

    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy original data (symbolic links or copy)
    print(f"\n[*] Copying patient data to {output_dir}...")
    import shutil

    for patient_dir in patient_dirs:
        dest_dir = output_dir / patient_dir.name
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(patient_dir, dest_dir)

    # Save survival data as JSON
    survival_json_path = output_dir / "survival_data.json"
    with open(survival_json_path, "w") as f:
        json.dump(survival_records, f, indent=2)

    print(f"[+] Saved survival data: {survival_json_path}")

    # Generate summary statistics
    print("\n" + "=" * 70)
    print("SURVIVAL DATA SUMMARY")
    print("=" * 70)

    observed_times = [r["observed_time_months"] for r in survival_records]
    events = [r["event_occurred"] for r in survival_records]

    print(f"Total patients: {len(survival_records)}")
    print(f"Events observed: {sum(events)} ({100*sum(events)/len(events):.1f}%)")
    print(
        f"Censored: {len(events) - sum(events)} ({100*(len(events)-sum(events))/len(events):.1f}%)"
    )
    print("\nObserved time (months):")
    print(f"  Mean: {np.mean(observed_times):.2f}")
    print(f"  Median: {np.median(observed_times):.2f}")
    print(f"  Range: [{np.min(observed_times):.2f}, {np.max(observed_times):.2f}]")

    # Tumor characteristics
    volumes = [r["tumor_features"]["tumor_volume_cm3"] for r in survival_records]
    suvs = [r["tumor_features"]["suv_max"] for r in survival_records]

    print("\nTumor characteristics:")
    print(f"  Volume (cm³): {np.mean(volumes):.2f} ± {np.std(volumes):.2f}")
    print(f"  SUVmax: {np.mean(suvs):.2f} ± {np.std(suvs):.2f}")

    print("\n[+] Survival data generation complete!")
    print("[*] Use this directory for multi-task training:")
    print(f"    python scripts/train_multitask.py --data-dir {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
