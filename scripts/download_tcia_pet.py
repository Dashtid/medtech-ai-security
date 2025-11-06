#!/usr/bin/env python
"""Download FDG-PET/CT dataset from The Cancer Imaging Archive (TCIA).

This script downloads a subset of the FDG-PET-CT-Lesions dataset for
tumor segmentation in nuclear medicine imaging.

Usage:
    # Download small subset for testing (10 patients)
    python scripts/download_tcia_pet.py --output data/tcia --max-patients 10

    # Download larger subset (50 patients)
    python scripts/download_tcia_pet.py --output data/tcia --max-patients 50
"""

import argparse
from pathlib import Path
import sys

try:
    from tcia_utils import nbia
except ImportError:
    print("[!] tcia-utils not installed. Installing...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "tcia-utils"])
    from tcia_utils import nbia


def explore_tcia_collection(collection_name: str = "FDG-PET-CT-Lesions"):
    """Explore TCIA collection metadata.

    Args:
        collection_name: TCIA collection name
    """
    print(f"[+] Exploring TCIA collection: {collection_name}")
    print("=" * 60)

    # Get collection info
    try:
        # Get patient count
        patients = nbia.getPatient(collection=collection_name)
        print(f"[*] Total patients in collection: {len(patients)}")

        # Get series info (imaging series)
        series = nbia.getSeries(collection=collection_name)
        print(f"[*] Total imaging series: {len(series)}")

        # Show modalities available
        if series:
            modalities = set(s["Modality"] for s in series if "Modality" in s)
            print(f"[*] Available modalities: {', '.join(modalities)}")

        return patients, series

    except Exception as e:
        print(f"[!] Error exploring collection: {e}")
        print("[!] Collection may require access request or have different name")
        print("[*] Available collections:")
        try:
            collections = nbia.getCollections()
            for col in collections[:20]:  # Show first 20
                print(f"    - {col}")
        except Exception as e:
            print(f"    [!] Could not fetch collections: {e}")
        return None, None


def download_patient_subset(collection_name: str, output_dir: Path, max_patients: int = 10):
    """Download subset of patients from TCIA.

    Args:
        collection_name: TCIA collection name
        output_dir: Output directory for downloaded data
        max_patients: Maximum number of patients to download
    """
    print(f"\n[+] Downloading {max_patients} patients from {collection_name}")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get patient list
    try:
        patients = nbia.getPatient(collection=collection_name)

        if not patients:
            print("[!] No patients found in collection")
            return

        # Limit to requested number
        patients_to_download = patients[:max_patients]

        print(f"[*] Downloading {len(patients_to_download)} patients...")

        for idx, patient in enumerate(patients_to_download, 1):
            patient_id = patient.get("PatientId", "unknown")
            print(f"\n[{idx}/{len(patients_to_download)}] Patient: {patient_id}")

            # Get series for this patient
            series = nbia.getSeries(collection=collection_name, patientId=patient_id)

            if not series:
                print(f"    [!] No series found for patient {patient_id}")
                continue

            print(f"    [*] Found {len(series)} series")

            # Download each series
            for series_idx, s in enumerate(series, 1):
                series_uid = s.get("SeriesInstanceUID")
                modality = s.get("Modality", "unknown")

                if not series_uid:
                    continue

                print(f"    [{series_idx}/{len(series)}] Downloading {modality} series...")

                # Create patient-specific directory
                patient_dir = output_dir / f"patient_{patient_id}"
                patient_dir.mkdir(exist_ok=True)

                try:
                    # Download series
                    nbia.downloadSeries(series_uid, path=str(patient_dir), input_type="list")
                    print(f"        [+] Downloaded to {patient_dir}")

                except Exception as e:
                    print(f"        [!] Error downloading series: {e}")

        print("\n[+] Download complete!")
        print(f"    Data saved to: {output_dir}")

    except Exception as e:
        print(f"[!] Error during download: {e}")
        print("\n[*] Troubleshooting:")
        print("    1. Check if collection name is correct")
        print("    2. Some collections require data access request")
        print("    3. Try with a different collection first")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download FDG-PET/CT data from TCIA")
    parser.add_argument(
        "--output", type=str, default="data/tcia", help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=10,
        help="Maximum number of patients to download (default: 10)",
    )
    parser.add_argument(
        "--collection", type=str, default="FDG-PET-CT-Lesions", help="TCIA collection name"
    )
    parser.add_argument(
        "--explore-only",
        action="store_true",
        help="Only explore collection metadata, don't download",
    )

    args = parser.parse_args()

    print("[+] TCIA FDG-PET/CT Download Script")
    print("=" * 60)
    print(f"Collection: {args.collection}")
    print(f"Output: {args.output}")
    print(f"Max patients: {args.max_patients}")
    print()

    # Explore collection
    patients, series = explore_tcia_collection(args.collection)

    if args.explore_only:
        print("\n[*] Exploration complete (--explore-only mode)")
        return

    if patients is None:
        print("\n[!] Cannot proceed with download - collection not accessible")
        return

    # Download subset
    download_patient_subset(
        collection_name=args.collection,
        output_dir=Path(args.output),
        max_patients=args.max_patients,
    )


if __name__ == "__main__":
    main()
