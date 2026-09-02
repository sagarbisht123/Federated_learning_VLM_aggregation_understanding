#!/usr/bin/env python3
"""
Kaggle Dataset Downloader WITH PROGRESS BAR
Dataset: Chest X-rays (Indiana University) - ~14.13 GB
"""

import os
import json
import requests
from tqdm import tqdm
import zipfile

# ========================= CONFIG =========================
OUTPUT_FOLDER = "/scratch/dharmendra.rs.phy23.itbhu/datasets"   # ← CHANGE ONLY IF YOU WANT
DATASET_SLUG = "raddar/chest-xrays-indiana-university"
# ========================================================

# Create folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load your kaggle.json
kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
with open(kaggle_json) as f:
    creds = json.load(f)

print(f"✅ Starting download to: {OUTPUT_FOLDER}")
print("📥 Downloading zip with progress bar (this will take 30-90 min)...")

# Download URL
url = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_SLUG}"
zip_path = os.path.join(OUTPUT_FOLDER, "chest-xrays-indiana-university.zip")

# Download with tqdm progress bar
with requests.get(url, auth=(creds["username"], creds["key"]), stream=True) as response:
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    
    with open(zip_path, "wb") as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
        colour="green"
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)

print(" Download finished! Now unzipping...")

# Unzip
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(OUTPUT_FOLDER)

# Cleanup
os.remove(zip_path)


print(f" Everything is ready in: {OUTPUT_FOLDER}")
print("You can now run: ls", OUTPUT_FOLDER)
