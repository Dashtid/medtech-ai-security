#!/usr/bin/env python
"""Download medical imaging datasets for training and evaluation.

Supported datasets:
- Medical Segmentation Decathlon (MSD)
- MedMNIST v2 (lightweight, good for quick testing)

Usage:
    python scripts/download_data.py --dataset msd --task liver --output data/
    python scripts/download_data.py --dataset medmnist --task pathmnist --output data/
"""

import argparse
import os
from pathlib import Path
import sys


def download_medmnist(task: str, output_dir: Path) -> None:
    """Download MedMNIST dataset.

    Args:
        task: MedMNIST task name (e.g., 'pathmnist', 'chestmnist')
        output_dir: Output directory for downloaded data
    """
    try:
        import medmnist
        from medmnist import INFO
    except ImportError:
        print("[!] MedMNIST not installed. Installing...")
        os.system(f"{sys.executable} -m pip install medmnist")
        import medmnist
        from medmnist import INFO

    print(f"[+] Downloading MedMNIST: {task}")

    if task not in INFO:
        available = ", ".join(INFO.keys())
        raise ValueError(f"Unknown MedMNIST task: {task}. Available: {available}")

    # Get dataset class
    DataClass = getattr(medmnist, INFO[task]["python_class"])

    # Download train/val/test splits
    output_dir = output_dir / "medmnist" / task
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Downloading train split...")
    train_dataset = DataClass(split="train", download=True, root=str(output_dir))

    print("[*] Downloading validation split...")
    val_dataset = DataClass(split="val", download=True, root=str(output_dir))

    print("[*] Downloading test split...")
    test_dataset = DataClass(split="test", download=True, root=str(output_dir))

    print(f"[+] Downloaded {task} to {output_dir}")
    print(f"    Train: {len(train_dataset)} samples")
    print(f"    Val: {len(val_dataset)} samples")
    print(f"    Test: {len(test_dataset)} samples")


def download_msd_task(task: str, output_dir: Path) -> None:
    """Download Medical Segmentation Decathlon task.

    Args:
        task: MSD task name (e.g., 'liver', 'brain', 'heart')
        output_dir: Output directory for downloaded data
    """
    # Map task names to MSD URLs
    MSD_TASKS = {
        "liver": "Task03_Liver",
        "brain": "Task01_BrainTumour",
        "heart": "Task02_Heart",
        "hippocampus": "Task04_Hippocampus",
        "prostate": "Task05_Prostate",
        "lung": "Task06_Lung",
        "pancreas": "Task07_Pancreas",
        "hepatic": "Task08_HepaticVessel",
        "spleen": "Task09_Spleen",
        "colon": "Task10_Colon",
    }

    if task not in MSD_TASKS:
        available = ", ".join(MSD_TASKS.keys())
        raise ValueError(f"Unknown MSD task: {task}. Available: {available}")

    task_name = MSD_TASKS[task]
    base_url = f"https://msd-for-monai.s3-us-west-2.amazonaws.com/{task_name}.tar"

    output_dir = output_dir / "msd"
    output_dir.mkdir(parents=True, exist_ok=True)

    tar_path = output_dir / f"{task_name}.tar"

    print(f"[+] Downloading Medical Segmentation Decathlon: {task_name}")
    print(f"[*] URL: {base_url}")
    print(f"[*] Destination: {tar_path}")

    # Download using wget or curl
    if os.system(f"wget {base_url} -O {tar_path}") != 0:
        # Try curl if wget fails
        if os.system(f"curl -L {base_url} -o {tar_path}") != 0:
            # Fall back to Python requests
            print("[*] wget/curl not available, using Python requests...")
            import requests

            response = requests.get(base_url, stream=True)
            total_size = int(response.headers.get("content-length", 0))

            with open(tar_path, "wb") as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r[*] Progress: {percent:.1f}%", end="")
            print()

    print("[*] Extracting archive...")
    import tarfile

    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(output_dir)

    print(f"[+] Downloaded and extracted {task_name} to {output_dir / task_name}")
    print(f"[*] You can now remove the tar file: {tar_path}")


def main():
    """Main entry point for dataset downloader."""
    parser = argparse.ArgumentParser(
        description="Download medical imaging datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Medical Segmentation Decathlon liver task
  python scripts/download_data.py --dataset msd --task liver --output data/

  # Download MedMNIST PathMNIST task (lightweight, good for testing)
  python scripts/download_data.py --dataset medmnist --task pathmnist --output data/

  # List available MedMNIST tasks
  python scripts/download_data.py --dataset medmnist --list
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["msd", "medmnist"],
        help="Dataset to download",
    )
    parser.add_argument("--task", type=str, help="Specific task/subset to download")
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output directory for downloaded data (default: data/)",
    )
    parser.add_argument("--list", action="store_true", help="List available tasks and exit")

    args = parser.parse_args()
    output_dir = Path(args.output)

    if args.list:
        if args.dataset == "medmnist":
            try:
                from medmnist import INFO

                print("[+] Available MedMNIST tasks:")
                for key, value in INFO.items():
                    print(f"    {key}: {value['task']}")
            except ImportError:
                print("[!] MedMNIST not installed. Install with: pip install medmnist")
        elif args.dataset == "msd":
            print("[+] Available Medical Segmentation Decathlon tasks:")
            tasks = [
                "liver",
                "brain",
                "heart",
                "hippocampus",
                "prostate",
                "lung",
                "pancreas",
                "hepatic",
                "spleen",
                "colon",
            ]
            for task in tasks:
                print(f"    {task}")
        return

    if not args.task:
        parser.error("--task is required (use --list to see available tasks)")

    # Download specified dataset
    if args.dataset == "medmnist":
        download_medmnist(args.task, output_dir)
    elif args.dataset == "msd":
        download_msd_task(args.task, output_dir)

    print("\n[+] Download complete!")
    print(f"[+] Data saved to: {output_dir}")


if __name__ == "__main__":
    main()
