"""
Dataset Downloader for LENS-XAI Project
This script automates downloading the primary datasets required for the project:
- Edge-IIoTset
- UKM-IDS20
- CTU-13
- NSL-KDD

Prerequisites:
If downloading from Kaggle, ensure 'kaggle.json' is placed in '~/.kaggle/' or 'C:\\Users\\<User>\\.kaggle\\'
"""

import os
import subprocess
import zipfile
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "NSL-KDD": {
        "source": "kaggle",
        "identifier": "hassan06/nslkdd",
        "folder": "nsl-kdd"
    },
    "UKM-IDS20": {
        "source": "kaggle",
        "identifier": "muatazsalam/ukm-ids20",
        "folder": "ukm-ids20"
    },
    # Edge-IIoTset and CTU-13 might require manual steps if Kaggle mirrors don't match paper exactly.
    "Edge-IIoTset": {
        "source": "manual",
        "url": "https://ieee-dataport.org/documents/edge-iiotset",
        "folder": "edge-iiotset"
    },
    "CTU-13": {
        "source": "manual",
        "url": "https://stratosphereips.org/datasets-ctu13",
        "folder": "ctu-13"
    }
}

def download_kaggle_dataset(identifier: str, dest_folder: Path):
    """Download and extract a dataset from Kaggle."""
    print(f"Downloading Kaggle dataset: {identifier}")
    try:
        env = os.environ.copy()
        env['KAGGLE_CONFIG_DIR'] = str(Path.home() / '.kaggle')
        subprocess.run(["kaggle", "datasets", "download", "-d", identifier, "-p", str(dest_folder)], check=True, env=env)
        
        # Extract zip files
        for item in dest_folder.glob("*.zip"):
            print(f"Extracting {item.name}...")
            with zipfile.ZipFile(item, 'r') as zip_ref:
                zip_ref.extractall(dest_folder)
            os.remove(item)  # Clean up zip file
            
        print(f"Successfully downloaded and extracted {identifier} into {dest_folder}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {identifier}. Please ensure your Kaggle API key is configured properly. ({e})")
    except FileNotFoundError:
        print("Kaggle CLI not found. Please run 'pip install kaggle'.")

def main():
    print(f"Initializing dataset downloads into: {RAW_DATA_DIR}\n")
    
    for name, info in DATASETS.items():
        dataset_folder = RAW_DATA_DIR / info["folder"]
        dataset_folder.mkdir(exist_ok=True)
        
        if info["source"] == "kaggle":
            download_kaggle_dataset(info["identifier"], dataset_folder)
        elif info["source"] == "manual":
            print(f"Dataset {name} requires manual download.")
            print(f"Please visit: {info['url']}")
            print(f"And extract the contents into: {dataset_folder}\n")

if __name__ == "__main__":
    main()
