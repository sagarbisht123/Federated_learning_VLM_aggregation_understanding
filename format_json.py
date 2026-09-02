# -*- coding: utf-8 -*-
"""
SCRIPT 1 — Indiana University Chest X-Ray → JSONL Converter
============================================================
Reads the raw Indiana dataset and converts it into a clean
annotations.jsonl file ready for training.

Input structure expected:
    <dataset_root>/
        images/
            images_normalized/
                CXR1_1_IM-0001-3001.png
                ...
        indiana_projections.csv
        indiana_reports.csv

Output structure created:
    data/
        images/         
        annotations/
            annotations.jsonl

Each JSONL line:
    {"image": "images/CXR1_1_IM-0001-3001.png", "suffix": "findings. impression", "prefix": "<CAPTION>"}

Caption = findings + ". " + impression (whichever fields are available)
Both Frontal and Lateral views are included.

NOTE: Image paths in annotations.jsonl are stored as paths RELATIVE to
      output_dir (e.g. "images/CXR1_1_IM-0001-3001.png"), not absolute paths.
"""

import os
import json
import argparse
import shutil
import pandas as pd

# ──────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────
TASK_PREFIX    = "<CAPTION>"
COPY_IMAGES    = False   # True  = physically copy images into data/images/
                         # False = store relative paths (no copy, saves disk space)

# ──────────────────────────────────────────────────────────
# ARGS
# ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Convert Indiana dataset to JSONL format")
parser.add_argument(
    "--dataset_root",
    type=str,
    required=True,
    help="Path to the raw downloaded Indiana dataset folder"
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./data",
    help="Where to create the output data/ folder (default: ./data)"
)
parser.add_argument(
    "--copy_images",
    action="store_true",
    default=True,
    help="Physically copy images into output_dir/images/ (default: use relative paths)"
)
args = parser.parse_args()

DATASET_ROOT = args.dataset_root
OUTPUT_DIR   = os.path.abspath(args.output_dir)  # resolve once for consistent relative path calc
COPY_IMAGES  = args.copy_images

# ──────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────
IMAGES_DIR       = os.path.join(DATASET_ROOT, "images", "images_normalized")
REPORTS_CSV      = os.path.join(DATASET_ROOT, "indiana_reports.csv")
PROJECTIONS_CSV  = os.path.join(DATASET_ROOT, "indiana_projections.csv")

OUT_IMAGES_DIR   = os.path.join(OUTPUT_DIR, "images")
OUT_ANNOT_DIR    = os.path.join(OUTPUT_DIR, "annotations")
OUT_JSONL        = os.path.join(OUT_ANNOT_DIR, "annotations.jsonl")

# ──────────────────────────────────────────────────────────
# VALIDATE INPUTS
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("SCRIPT 1 — Indiana Dataset → JSONL")
print("=" * 60)

assert os.path.isdir(IMAGES_DIR),       f"Images folder not found: {IMAGES_DIR}"
assert os.path.isfile(REPORTS_CSV),     f"Reports CSV not found:   {REPORTS_CSV}"
assert os.path.isfile(PROJECTIONS_CSV), f"Projections CSV not found: {PROJECTIONS_CSV}"

print(f"Dataset root : {DATASET_ROOT}")
print(f"Output dir   : {OUTPUT_DIR}")
print(f"Copy images  : {COPY_IMAGES}")

# ──────────────────────────────────────────────────────────
# CREATE OUTPUT DIRS
# ──────────────────────────────────────────────────────────
os.makedirs(OUT_ANNOT_DIR, exist_ok=True)
if COPY_IMAGES:
    os.makedirs(OUT_IMAGES_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────
# LOAD CSVs
# ──────────────────────────────────────────────────────────
print("\nLoading CSVs...")
reports     = pd.read_csv(REPORTS_CSV)
projections = pd.read_csv(PROJECTIONS_CSV)

print(f"  Reports     : {len(reports)} rows   | columns: {list(reports.columns)}")
print(f"  Projections : {len(projections)} rows | columns: {list(projections.columns)}")

# ──────────────────────────────────────────────────────────
# MERGE on uid
# ──────────────────────────────────────────────────────────
df = projections.merge(reports, on="uid", how="inner")
print(f"\nAfter merge  : {len(df)} rows")

# ──────────────────────────────────────────────────────────
# BUILD CAPTION = findings + impression
# Handle NaN gracefully — use whichever part is available
# ──────────────────────────────────────────────────────────
def build_caption(row):
    findings   = str(row.get("findings",   "") or "").strip()
    impression = str(row.get("impression", "") or "").strip()

    # remove literal "nan" strings that come from pd NaN → str conversion
    if findings   == "nan": findings   = ""
    if impression == "nan": impression = ""

    if findings and impression:
        return findings + ". " + impression
    elif findings:
        return findings
    elif impression:
        return impression
    else:
        return None   # both empty — will be dropped

df["caption"] = df.apply(build_caption, axis=1)

# drop rows with no caption at all
before = len(df)
df = df[df["caption"].notna()].reset_index(drop=True)
print(f"Dropped {before - len(df)} rows with empty findings AND impression")

# ──────────────────────────────────────────────────────────
# RESOLVE IMAGE PATHS
# Returns the full absolute path for disk operations (copy/exists checks).
# The annotation will later store only the relative path.
# ──────────────────────────────────────────────────────────
def resolve_image_path(filename: str) -> str | None:
    """
    Returns the full ABSOLUTE path to the image on disk (used internally for
    copy operations and existence checks).  The JSONL will store only the
    relative path — see make_relative_path() below.

    Handles:
      - filename only
      - filename + .png
      - old .dcm files
      - old full paths (strips them to basename first)
    """
    if not filename:
        return None

    # Clean: remove any old absolute path, keep only basename
    filename = os.path.basename(filename)

    candidates = [
        filename,                                   # 1_IM-0001-4001.dcm.png
        filename + ".png",                          # just in case
        filename.replace(".dcm.png", ".png"),       # strip .dcm if present
        filename.replace(".dcm", "") + ".png",      # common case
        filename.replace(".dcm", ""),               # without extension
    ]

    for cand in candidates:
        full_path = os.path.join(IMAGES_DIR, cand)
        if os.path.isfile(full_path):
            return full_path

    print(f"⚠️  Image NOT FOUND: {filename}  (searched in {IMAGES_DIR})")
    return None


def make_relative_path(abs_image_path: str) -> str:
    """
    Converts an absolute image path to a path RELATIVE to OUTPUT_DIR.

    If images were copied:   images/<filename>          (relative to OUTPUT_DIR)
    If images were NOT copied, we still build a relative path from OUTPUT_DIR
    to wherever the source image lives.  If the source is outside OUTPUT_DIR
    the result will start with '../...' — which is still a valid relative path
    that tools like HuggingFace datasets can resolve given the base directory.
    """
    return os.path.relpath(abs_image_path, start=OUTPUT_DIR)


df["image_path_abs"] = df["filename"].apply(resolve_image_path)

missing = df["image_path_abs"].isna().sum()
print(f"Images found : {df['image_path_abs'].notna().sum()} / {len(df)}  ({missing} missing)")

# drop rows where image file doesn't exist on disk
df = df[df["image_path_abs"].notna()].reset_index(drop=True)
print(f"Final rows   : {len(df)}")

# ──────────────────────────────────────────────────────────
# OPTIONALLY COPY IMAGES
# ──────────────────────────────────────────────────────────
if COPY_IMAGES:
    print(f"\nCopying {len(df)} images to {OUT_IMAGES_DIR} ...")
    copied = 0
    for src in df["image_path_abs"]:
        dst = os.path.join(OUT_IMAGES_DIR, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        copied += 1
        if copied % 500 == 0:
            print(f"  {copied}/{len(df)} copied...")
    print(f"  Done. {copied} images copied.")
    # Point absolute paths to the NEW location inside OUTPUT_DIR
    df["image_path_abs"] = df["filename"].apply(
        lambda fn: os.path.join(OUT_IMAGES_DIR, os.path.basename(fn))
    )

# ──────────────────────────────────────────────────────────
# BUILD RELATIVE PATHS  (used in the JSONL)
# When images are copied  → "images/<filename>"
# When images are NOT copied → relative path from OUTPUT_DIR to source
# ──────────────────────────────────────────────────────────
df["image_path_rel"] = df["image_path_abs"].apply(make_relative_path)

# ──────────────────────────────────────────────────────────
# WRITE JSONL
# ──────────────────────────────────────────────────────────
print(f"\nWriting JSONL → {OUT_JSONL}")

written = 0
skipped = 0

with open(OUT_JSONL, "w") as f:
    for _, row in df.iterrows():
        entry = {
            "image":  row["image_path_rel"],   # ← relative path, not absolute
            "suffix": row["caption"],
            "prefix": TASK_PREFIX
        }
        f.write(json.dumps(entry, separators=(",", ":")) + "\n")
        written += 1

print(f"Written : {written} entries")
print(f"Skipped : {skipped} entries")

# ──────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"\nOutput structure:")
print(f"  {OUTPUT_DIR}/")
print(f"  ├── images/          {'(copied)' if COPY_IMAGES else '(not copied — relative paths point to source)'}")
print(f"  └── annotations/")
print(f"        └── annotations.jsonl   ({written} entries)")
print(f"\nNOTE: Image paths in annotations.jsonl are relative to: {OUTPUT_DIR}")
print(f"\nSample entries:")

with open(OUT_JSONL, "r") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        d = json.loads(line)
        print(f"  [{i+1}] image  : {d['image']}")
        print(f"       suffix : {d['suffix'][:80]}{'...' if len(d['suffix']) > 80 else ''}")
        print(f"       prefix : {d['prefix']}")
        print()

"""
python format_json.py --dataset_root /scratch/dharmendra.rs.phy23.itbhu/datasets --output_dir /scratch/dharmendra.rs.phy23.itbhu/datasets/florence_data
"""
