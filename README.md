# Florence-2 Federated Learning for Captioning

Fine-tunes [Microsoft Florence-2](https://huggingface.co/microsoft/Florence-2-base) for detailed image captioning using Federated Learning (FedProx + FedAvg) with LoRA adapters in fp16.

---

## Repository Structure

```
Florence/
├── data/
│   ├── images/                  ← put your images here
│   └── annotations/
│       └── annotations.jsonl    ← one JSON object per line
├── federated_split.py           ← Script 1: splits data into federated clients
├── florence_fed_captioning.py   ← Script 2: federated training
├── requirements.txt
└── Output/                      ← training runs saved here (auto-created)
```

---


Place your images in `data/images/` and create `data/annotations/annotations.jsonl`. Each line is one JSON object:

```json
{"image": "img1.jpg", "suffix": "A dog playing in the park"}
{"image": "img2.jpg", "suffix": "A cat sitting on a chair"}
```

- `image` — just the filename, not a full path. The scripts resolve it automatically from the `images/` folder.
- `suffix` — the ground truth caption for that image.
- `prefix` *(optional)* — task token. Defaults to `<CAPTION>` if not provided.

---



**Create and activate a conda environment:**

```bash
conda create -n florence_fl python=3.10 -y
conda activate florence_fl
```

**Install PyTorch first** — pick the command that matches your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/). For CUDA 12.4:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Then install the rest of the dependencies:**

```bash
pip install -r requirements.txt
```

---

### Split Data into Federated Clients

`federated_split.py` takes your `data/` folder and splits it into 3 client folders + a held-out test set using a Dirichlet distribution.

**Basic usage (uses `data/` inside the repo by default):**

```bash
python federated_split.py
```

**With a custom data path:**

```bash
python federated_split.py --data_dir /path/to/your/data
```

**All options:**

```bash
python federated_split.py \
  --data_dir   ./data          \  # path to data/ folder
  --alpha      100             \  # % of labeled data to use (1–100)
  --num_clients 3              \  # number of federated clients
  --dirichlet_alpha 100        \  # Dirichlet concentration (higher = more uniform split)
  --seed       42
```

### About `--alpha`

If you only want to train on a **subset** of your labeled data (useful for low-data experiments), set `--alpha` to a value less than 100. For example `--alpha 10` uses only 10% of the training data. Default is `100` (all data).

### About `--dirichlet_alpha`

Controls how evenly data is distributed across clients. High value (e.g. `100`) → near-uniform distribution across clients (IID). Low value (e.g. `1`) → skewed, non-IID distribution where some clients get much more data than others.

**Output structure after running:**

```
data/
└── fed_input_data/
    ├── client_01_data/
    │   ├── images/
    │   └── annotations/annotations.jsonl
    ├── client_02_data/  ...
    ├── client_03_data/  ...
    ├── test_data/       ...
    └── split_summary.json
```

`split_summary.json` records the exact split configuration (alpha, client sizes, seed) so runs are reproducible.

---

##  Run Federated Training

`florence_fed_captioning.py` loads the split data, fine-tunes Florence-2 with LoRA across 3 clients for 20 rounds using FedProx + FedAvg aggregation.

**Basic usage (uses `data/fed_input_data/` and `Output/` inside the repo by default):**

```bash
python florence_fed_captioning.py
```

**With custom paths:**

```bash
python florence_fed_captioning.py \
  --data_dir   /path/to/fed_input_data \
  --output_dir /path/to/Output
```

Each run creates a timestamped subdirectory so nothing gets overwritten:

```
Output/
└── run_2025-01-15_10-30-00/
    ├── run_config.json          ← full config snapshot for this run
    ├── model_checkpoints/
    │   └── best/                ← best checkpoint by CIDEr score
    └── scores/
        ├── round_metrics.csv    ← live-updated every round
        └── final_results.json   ← final evaluation on full test set
```

---

## Evaluation

After every federated round, the global model is evaluated on the held-out test set. Four metrics are computed:

| Metric | What it measures |
|--------|-----------------|
| **BLEU** | N-gram precision between prediction and reference |
| **METEOR** | Unigram F-score with stemming and synonym matching |
| **ROUGE-L** | Longest common subsequence overlap |
| **CIDEr** | Consensus-based image description evaluation — primary metric used for best-checkpoint selection and early stopping |

Results are saved to `scores/round_metrics.csv` (updated live after each round) and `scores/final_results.json` (full test set evaluation using the best checkpoint).

Early stopping triggers if CIDEr does not improve for 6 consecutive rounds.

---

## Credits

Training pipeline adapted from the Microsoft DSToolkit fine-tuning accelerator:

> **[microsoft/dstoolkit-finetuning-florence-2](https://github.com/microsoft/dstoolkit-finetuning-florence-2)**
> Accelerator for fine-tuning Microsoft's Florence-2 model across a variety of computer vision use cases.
