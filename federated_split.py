# -*- coding: utf-8 -*-
"""
SCRIPT 2 — Federated Data Splitter
====================================
Takes the clean data/ folder (output of Script 1) and splits it
into federated client folders using either:
  (a) explicit --client_ratios (deterministic, exact percentages), or
  (b) Dirichlet distribution + alpha labeled data control (old behavior)

Input structure:
    data/
        images/
            CXR1_1_IM-0001-3001.png
            ...
        annotations/
            annotations.jsonl        ← "image" field is just the filename

Output structure:
    data/
        fed_input_data/
            client_01_data/
                images/              (physically copied)
                annotations/
                    annotations.jsonl
            client_02_data/  ...
            client_03_data/  ...
            test_data/       ...
            split_summary.json

Usage:
    python federated_split.py --data_dir ./data
    python federated_split.py --data_dir ./data --alpha 100 --dirichlet_alpha 10

    # Explicit ratios instead of Dirichlet (must match --num_clients, sum to 100):
    python federated_split.py --data_dir ./data --client_ratios 10,25,65
"""

import os
import json
import shutil
import argparse
import numpy as np

# ──────────────────────────────────────────────────────────
# DEFAULTS
# ──────────────────────────────────────────────────────────
NUM_CLIENTS     = 3
DIRICHLET_ALPHA = 100.0   # high = uniform split, low = skewed sample-count split
TEST_SPLIT      = 0.2     # 20% held out as global test set
MIN_SAMPLES     = 10      # minimum samples per client
RANDOM_SEED     = 42
TASK_PREFIX     = "<CAPTION>"
DATA_DIR=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ──────────────────────────────────────────────────────────
# ARGS
# ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Federated data splitter for chest X-ray captioning")
parser.add_argument("--data_dir", type=str,
                    default=DATA_DIR,
                    help="Path to the data/ folder (must contain images/ and annotations/)")
parser.add_argument("--alpha",           type=int,   default=100,
                    help="Percentage of labeled data to use (1-100). Default: 100")
parser.add_argument("--num_clients",     type=int,   default=NUM_CLIENTS,
                    help=f"Number of federated clients (default: {NUM_CLIENTS})")
parser.add_argument("--dirichlet_alpha", type=float, default=DIRICHLET_ALPHA,
                    help=f"Dirichlet concentration param (default: {DIRICHLET_ALPHA}). Higher = more uniform. "
                         f"Ignored if --client_ratios is set.")
parser.add_argument("--client_ratios",   type=str,   default=None,
                    help="Comma-separated percentages for each client, e.g. '10,25,65'. "
                         "Must have exactly --num_clients values and sum to 100 (±0.5 tolerance "
                         "for rounding). When set, this OVERRIDES --dirichlet_alpha entirely — "
                         "the split becomes a deterministic slice by these exact proportions "
                         "instead of a random Dirichlet draw.")
parser.add_argument("--seed",            type=int,   default=RANDOM_SEED,
                    help=f"Random seed (default: {RANDOM_SEED})")
args = parser.parse_args()

DATA_DIR        = os.path.abspath(args.data_dir)
ALPHA           = args.alpha
NUM_CLIENTS     = args.num_clients
DIRICHLET_ALPHA = args.dirichlet_alpha
RANDOM_SEED     = args.seed

# ──────────────────────────────────────────────────────────
# PARSE + VALIDATE --client_ratios (if provided)
# ──────────────────────────────────────────────────────────
CLIENT_RATIOS = None
if args.client_ratios is not None:
    try:
        CLIENT_RATIOS = [float(x.strip()) for x in args.client_ratios.split(",")]
    except ValueError:
        raise ValueError(
            f"--client_ratios must be comma-separated numbers, got: '{args.client_ratios}'"
        )

    assert len(CLIENT_RATIOS) == NUM_CLIENTS, (
        f"--client_ratios has {len(CLIENT_RATIOS)} values but --num_clients is "
        f"{NUM_CLIENTS} — these must match exactly."
    )
    assert all(r > 0 for r in CLIENT_RATIOS), \
        "All --client_ratios values must be positive."
    ratio_sum = sum(CLIENT_RATIOS)
    assert abs(ratio_sum - 100.0) <= 0.5, (
        f"--client_ratios must sum to 100 (got {ratio_sum}). "
        f"Example: --client_ratios 10,25,65"
    )

np.random.seed(RANDOM_SEED)

# ──────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────
SRC_IMAGES_DIR = os.path.join(DATA_DIR, "images")
ANNOT_PATH     = os.path.join(DATA_DIR, "annotations", "annotations.jsonl")
FED_DIR        = os.path.join(DATA_DIR, "fed_input_data")

# ──────────────────────────────────────────────────────────
# VALIDATE
# ──────────────────────────────────────────────────────────
assert os.path.isdir(SRC_IMAGES_DIR), \
    f"images/ folder not found at: {SRC_IMAGES_DIR}"
assert os.path.isfile(ANNOT_PATH), \
    f"annotations.jsonl not found at: {ANNOT_PATH}"

# ──────────────────────────────────────────────────────────
# PRINT CONFIG
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("SCRIPT 2 — Federated Data Splitter")
print("=" * 60)
print(f"  Data dir        : {DATA_DIR}")
print(f"  Source images   : {SRC_IMAGES_DIR}")
print(f"  Annotations     : {ANNOT_PATH}")
print(f"  Alpha           : {ALPHA}%  (labeled data used)")
print(f"  Num clients     : {NUM_CLIENTS}")
if CLIENT_RATIOS is not None:
    print(f"  Split mode      : explicit client_ratios = {CLIENT_RATIOS}")
else:
    print(f"  Split mode      : Dirichlet (alpha={DIRICHLET_ALPHA})")
print(f"  Test split      : {TEST_SPLIT*100:.0f}%")
print(f"  Random seed     : {RANDOM_SEED}")
print(f"  Copy images     : always (hardcoded)")

# ──────────────────────────────────────────────────────────
# LOAD ANNOTATIONS
# "image" field may be a bare filename, relative, or absolute path.
# We always resolve to just the basename for source lookup.
# ──────────────────────────────────────────────────────────
print(f"\nLoading annotations...")
entries = []
missing_images = []

with open(ANNOT_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        assert "image"  in data, f"Missing 'image' key in: {line}"
        assert "suffix" in data, f"Missing 'suffix' key in: {line}"

        # resolve the source image path — always look in DATA_DIR/images/
        filename = os.path.basename(data["image"])
        src_path = os.path.join(SRC_IMAGES_DIR, filename)

        if not os.path.isfile(src_path):
            missing_images.append(filename)
            continue

        entries.append({
            "filename": filename,       # just the bare name
            "src_path": src_path,       # full absolute path to source
            "suffix":   data["suffix"],
            "prefix":   data.get("prefix", TASK_PREFIX),
        })

print(f"  Loaded   : {len(entries)} entries with valid images")
if missing_images:
    print(f"  ⚠️  Skipped: {len(missing_images)} entries — image file not found in {SRC_IMAGES_DIR}")
    for m in missing_images[:5]:
        print(f"      - {m}")
    if len(missing_images) > 5:
        print(f"      ... and {len(missing_images)-5} more")

assert len(entries) > 0, "No valid entries found! Check that images/ folder contains the right files."

# ──────────────────────────────────────────────────────────
# SHUFFLE
# ──────────────────────────────────────────────────────────
indices = np.random.permutation(len(entries))
entries = [entries[i] for i in indices]

# ──────────────────────────────────────────────────────────
# GLOBAL TRAIN / TEST SPLIT
# ──────────────────────────────────────────────────────────
n_test     = int(len(entries) * TEST_SPLIT)
test_data  = entries[:n_test]
train_data = entries[n_test:]

print(f"\nGlobal train : {len(train_data)}")
print(f"Global test  : {len(test_data)}")

# ──────────────────────────────────────────────────────────
# APPLY ALPHA — keep only ALPHA% of train as labeled
# ──────────────────────────────────────────────────────────
n_labeled    = int(len(train_data) * ALPHA / 100.0)
labeled_data = train_data[:n_labeled]
n_dropped    = len(train_data) - n_labeled

print(f"\nAlpha = {ALPHA}%")
print(f"  Labeled (used)     : {n_labeled}")
print(f"  Unlabeled (dropped): {n_dropped}")

# ──────────────────────────────────────────────────────────
# DIRICHLET SPLIT (unchanged — used when --client_ratios is NOT set)
# ──────────────────────────────────────────────────────────
def dirichlet_split(n, n_clients, alpha, min_samples=10, max_attempts=200):
    for _ in range(max_attempts):
        proportions = np.random.dirichlet([alpha] * n_clients)
        counts = (proportions * n).astype(int)
        if np.all(counts >= min_samples):
            counts[-1] = n - counts[:-1].sum()   # fix rounding remainder
            perm = np.random.permutation(n)
            splits, start = [], 0
            for c in counts:
                splits.append(perm[start:start + c].tolist())
                start += c
            return splits

    # fallback: equal split
    print(f"  ⚠️  Dirichlet failed after {max_attempts} attempts → falling back to equal split")
    base   = n // n_clients
    counts = [base] * n_clients
    counts[-1] = n - base * (n_clients - 1)
    perm = np.random.permutation(n)
    splits, start = [], 0
    for c in counts:
        splits.append(perm[start:start + c].tolist())
        start += c
    return splits


# ──────────────────────────────────────────────────────────
# EXPLICIT-RATIO SPLIT (new — used when --client_ratios IS set)
# Deterministic: no random search, no rejection sampling. Slices the
# already-shuffled `labeled_data` directly by the given percentages,
# so the exact split you asked for is the exact split you get.
# ──────────────────────────────────────────────────────────
def ratio_split(n, n_clients, ratios, min_samples=10):
    proportions = np.array(ratios) / sum(ratios)   # normalize, in case of float rounding
    counts = (proportions * n).astype(int)
    counts[-1] = n - counts[:-1].sum()              # fix rounding remainder onto last client

    if not np.all(counts >= min_samples):
        under = [f"client_{i+1:02d}: {c}" for i, c in enumerate(counts) if c < min_samples]
        raise ValueError(
            f"--client_ratios produced a client below MIN_SAMPLES={min_samples}: "
            f"{', '.join(under)}. Either raise that client's ratio or lower MIN_SAMPLES."
        )

    perm = np.random.permutation(n)
    splits, start = [], 0
    for c in counts:
        splits.append(perm[start:start + c].tolist())
        start += c
    return splits


if CLIENT_RATIOS is not None:
    print(f"\nExplicit ratio split ({CLIENT_RATIOS}) across {NUM_CLIENTS} clients...")
    client_splits = ratio_split(
        n           = len(labeled_data),
        n_clients   = NUM_CLIENTS,
        ratios      = CLIENT_RATIOS,
        min_samples = MIN_SAMPLES,
    )
else:
    print(f"\nDirichlet split (alpha={DIRICHLET_ALPHA}) across {NUM_CLIENTS} clients...")
    client_splits = dirichlet_split(
        n           = len(labeled_data),
        n_clients   = NUM_CLIENTS,
        alpha       = DIRICHLET_ALPHA,
        min_samples = MIN_SAMPLES,
    )

client_data = []
for i, split_indices in enumerate(client_splits):
    chunk = [labeled_data[idx] for idx in split_indices]
    client_data.append(chunk)
    pct = len(chunk) / len(labeled_data) * 100
    print(f"  Client {i+1:02d}: {len(chunk)} samples ({pct:.1f}%)")

# ──────────────────────────────────────────────────────────
# HELPER — write one client/test folder
# Always copies images from SRC_IMAGES_DIR into the folder.
# JSONL stores absolute path to the copied image.
# ──────────────────────────────────────────────────────────
def write_folder(folder_name, data_entries):
    client_dir = os.path.join(FED_DIR, folder_name)
    annot_dir  = os.path.join(client_dir, "annotations")
    images_dir = os.path.join(client_dir, "images")

    os.makedirs(annot_dir,  exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    jsonl_path     = os.path.join(annot_dir, "annotations.jsonl")
    copied         = 0
    already_exists = 0
    not_found      = 0

    with open(jsonl_path, "w") as f:
        for entry in data_entries:
            src = entry["src_path"]
            dst = os.path.join(images_dir, entry["filename"])

            if os.path.isfile(src):
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    copied += 1
                else:
                    already_exists += 1
            else:
                not_found += 1
                print(f"  ⚠️  Source missing at copy time: {src}")

            record = {
                "prefix": entry["prefix"],
                "suffix": entry["suffix"],
                "image":  os.path.abspath(dst),   # absolute path to the copied image
            }
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    return jsonl_path, len(data_entries), copied, already_exists, not_found


# ──────────────────────────────────────────────────────────
# WRITE ALL FOLDERS
# ──────────────────────────────────────────────────────────
print(f"\nWriting federated folders to {FED_DIR} ...")
os.makedirs(FED_DIR, exist_ok=True)

for i, chunk in enumerate(client_data):
    folder = f"client_{i+1:02d}_data"
    path, total, copied, exists, missing = write_folder(folder, chunk)
    print(f"  ✓ {folder}: {total} entries | copied {copied} | already existed {exists} | missing {missing}")

path, total, copied, exists, missing = write_folder("test_data", test_data)
print(f"  ✓ test_data    : {total} entries | copied {copied} | already existed {exists} | missing {missing}")

# ──────────────────────────────────────────────────────────
# SPLIT SUMMARY JSON
# ──────────────────────────────────────────────────────────
summary = {
    "alpha":             ALPHA,
    "num_clients":       NUM_CLIENTS,
    "split_mode":        "client_ratios" if CLIENT_RATIOS is not None else "dirichlet",
    "client_ratios":     CLIENT_RATIOS,       # None if Dirichlet mode was used
    "dirichlet_alpha":   DIRICHLET_ALPHA if CLIENT_RATIOS is None else None,
    "random_seed":       RANDOM_SEED,
    "total_entries":     len(entries),
    "test_entries":      len(test_data),
    "train_total":       len(train_data),
    "labeled_used":      n_labeled,
    "unlabeled_dropped": n_dropped,
    "client_sizes": {
        f"client_{i+1:02d}": len(client_data[i]) for i in range(NUM_CLIENTS)
    },
}

summary_path = os.path.join(FED_DIR, "split_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n  ✓ Split summary → {summary_path}")

# ──────────────────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"\n  {DATA_DIR}/")
print(f"  └── fed_input_data/")
for i in range(NUM_CLIENTS):
    print(f"        ├── client_{i+1:02d}_data/   ({len(client_data[i])} samples)")
print(f"        ├── test_data/          ({len(test_data)} samples)")
print(f"        └── split_summary.json")
print(f"\nExample commands:")
print(f"  python federated_split.py --data_dir {DATA_DIR} --alpha 25")
print(f"  python federated_split.py --data_dir {DATA_DIR} --alpha 50 --dirichlet_alpha 10")
print(f"  python federated_split.py --data_dir {DATA_DIR} --alpha 100 --dirichlet_alpha 100")
print(f"  python federated_split.py --data_dir {DATA_DIR} --alpha 100 --client_ratios 10,25,65")

