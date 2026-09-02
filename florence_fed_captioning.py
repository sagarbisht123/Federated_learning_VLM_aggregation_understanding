# -*- coding: utf-8 -*-
"""
Florence-2 Federated Learning for Detailed Captioning

Model  : microsoft/Florence-2-base (LoRA + fp16, NO quantization)
Task   : <DETAILED_CAPTION>
FL     : N clients, FedAvg aggregation (no FedProx/MU)
Freq.  : aggregation frequency controlled via --local_steps
Eval   : BLEU / METEOR / ROUGE-L / CIDEr (generation quality)
         + CE loss per client and aggregated model (for drift analysis)
Output : <output_dir>/run_YYYY-MM-DD_HH-MM-SS/
             ├── run_config.json
             ├── training.log
             ├── model_checkpoints/best/
             └── scores/
                 ├── round_metrics.csv     (live update every round)
                 ├── final_results.json
                 ├── client_losses.png     (live update every round)
                 ├── aggregated_loss.png   (live update every round)
                 └── combined_loss.png     (live update every round)

python florence_fed_captioning.py --local_steps 4 --rounds 30
"""

import os
import json
import gc
import logging
import sys
import argparse
import itertools
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    get_scheduler,
)
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider

# Plotting logic lives in its own module — see utils/metrics_plotting.py
from utils.metrics_plotting import plot_client_losses, plot_aggregated_loss, plot_combined

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ========================= NLTK DOWNLOADS =========================
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ========================= ARGUMENTS =========================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/fed_input_data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Output")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default=DATA_DIR,
    help="Path to fed_input_data/")
parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
    help="Base path — each run creates output_dir/run_YYYY-MM-DD_HH-MM-SS/")
parser.add_argument("--local_steps", type=int, default=None,
    help="Number of local optimizer.step() updates per client per round. "
         "This is the aggregation-frequency knob: low = frequent "
         "aggregation, high = infrequent aggregation. If omitted, falls "
         "back to one full local epoch per round (old behavior).")
parser.add_argument("--local_epochs", type=int, default=1,
    help="Used only when --local_steps is not set.")
parser.add_argument("--rounds", type=int, default=30,
    help="Total number of FL rounds (= number of aggregation events). "
         "When sweeping --local_steps, set this to "
         "total_step_budget // local_steps to hold total local "
         "computation constant across the sweep.")
args = parser.parse_args()

DATA_DIR = args.data_dir
LOCAL_STEPS = args.local_steps
LOCAL_EPOCHS = args.local_epochs
ROUNDS = args.rounds

# ── Timestamped run directory ─────────────────────────────────────────────
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = os.path.join(args.output_dir, f"run_{RUN_TIMESTAMP}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Logger setup ────────────────────────────────────────────────────────
LOG_FILE = os.path.join(OUTPUT_DIR, "training.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger()
log.info("=" * 70)
log.info(f"Run directory : {OUTPUT_DIR}")
log.info("=" * 70)

# ========================= CONFIGURATION =========================
MODEL_NAME = "microsoft/Florence-2-base"
CAPTION_TASK = "<DETAILED_CAPTION>"
RESIZE_IMAGES = False
RESIZE_SCALE_FACTOR = 0.5

# LoRA
LORA_R = 16
LORA_ALPHA = 32          # 2x of r — standard practice
LORA_DROPOUT = 0.05

# FL
NUM_CLIENTS = 3
LEARNING_RATE = 5e-5
BATCH_SIZE = 6
PATIENCE = 30

# Memory Optimization
GRADIENT_CHECKPOINTING = True
GRADIENT_ACCUMULATION = 4   # effective batch = BATCH_SIZE * GRADIENT_ACCUMULATION

# Device (NO quantization — fp16 only, no bitsandbytes needed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda:0")
torch_dtype = torch.float16
torch.cuda.set_device(0)

log.info(f"Device        : {DEVICE} ({torch.cuda.get_device_name(0)})")
log.info(f"Model         : {MODEL_NAME}")
log.info(f"Task          : {CAPTION_TASK}")
log.info(f"Precision     : fp16 (no quantization)")
log.info(f"Batch size    : {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION})")
log.info(f"LoRA r/alpha  : {LORA_R}/{LORA_ALPHA}")
if LOCAL_STEPS is not None:
    log.info(f"Aggregation   : every {LOCAL_STEPS} local steps "
             f"({LOCAL_STEPS * GRADIENT_ACCUMULATION * BATCH_SIZE} image-text pairs/client/round)")
else:
    log.info(f"Aggregation   : every {LOCAL_EPOCHS} local epoch(s) (old default behavior)")
log.info(f"LR            : {LEARNING_RATE} | Rounds: {ROUNDS} | Patience: {PATIENCE}")

# ========================= OUTPUT DIRS =========================
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "model_checkpoints")
SCORES_DIR = os.path.join(OUTPUT_DIR, "scores")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SCORES_DIR, exist_ok=True)

# ── Persist run config ──────────────────────────────────────────────────
run_config = {
    "run_timestamp"    : RUN_TIMESTAMP,
    "model"             : MODEL_NAME,
    "task"              : CAPTION_TASK,
    "precision"         : "fp16_no_quantization",
    "lora_r"            : LORA_R,
    "lora_alpha"        : LORA_ALPHA,
    "lora_dropout"      : LORA_DROPOUT,
    "num_clients"       : NUM_CLIENTS,
    "rounds"            : ROUNDS,
    "local_steps"       : LOCAL_STEPS,
    "local_epochs"      : LOCAL_EPOCHS if LOCAL_STEPS is None else None,
    "learning_rate"     : LEARNING_RATE,
    "batch_size"        : BATCH_SIZE,
    "effective_batch"   : BATCH_SIZE * GRADIENT_ACCUMULATION,
    "patience"          : PATIENCE,
    "data_dir"          : DATA_DIR,
    "output_dir"        : OUTPUT_DIR,
    "log_file"          : LOG_FILE,
}
with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w") as f:
    json.dump(run_config, f, indent=2)
log.info(f"Config saved  → {os.path.join(OUTPUT_DIR, 'run_config.json')}\n")

# ========================= HELPERS =========================
def gpu_mem_str():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(DEVICE) / 1024**3
        reserved = torch.cuda.memory_reserved(DEVICE) / 1024**3
        return f"{alloc:.1f}/{reserved:.1f}GB"
    return "N/A"


def trainable_checksum(model):
    """
    Debug helper: sums every trainable parameter's values into one number.
    If this number is IDENTICAL across rounds, the weights are not
    actually changing — this is the first thing to check when loss
    curves look suspiciously flat.
    """
    with torch.no_grad():
        return sum(p.sum().item() for p in model.parameters() if p.requires_grad)

# ========================= DATA =========================
log.info("=" * 70)
log.info("LOADING FEDERATED DATA")
log.info("=" * 70)

CLIENT_NAMES = ["client_01_data", "client_02_data", "client_03_data"]
TEST_NAME = "test_data"


def resize_images(image_dir, output_dir, scale):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(image_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue
        img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
        new_size = (int(img.width * scale), int(img.height * scale))
        img.resize(new_size, Image.LANCZOS).save(os.path.join(output_dir, fname))
    log.info(f"Resized → {output_dir}")


class JSONLDataset:
    def __init__(self, jsonl_path, image_dir):
        self.image_dir = image_dir
        self.entries = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.entries.append(json.loads(line))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        image_field = e["image"]

        primary_path = os.path.join(self.image_dir, image_field)
        if os.path.isfile(primary_path):
            return Image.open(primary_path).convert("RGB"), e

        filename = os.path.basename(image_field)
        fallback_path = os.path.join(self.image_dir, filename)
        if os.path.isfile(fallback_path):
            return Image.open(fallback_path).convert("RGB"), e

        raise FileNotFoundError(
            f"Image not found.\n"
            f"  Tried (primary)  : {primary_path}\n"
            f"  Tried (fallback) : {fallback_path}\n"
            f"  Check that images exist in: {self.image_dir}"
        )


class CaptionDataset(Dataset):
    def __init__(self, jsonl_path, image_dir, task_prefix):
        self.dataset = JSONLDataset(jsonl_path, image_dir)
        self.task_prefix = task_prefix

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, data = self.dataset[idx]
        return self.task_prefix, data["suffix"], image


def get_active_image_dir(base_dir):
    if not RESIZE_IMAGES:
        return base_dir
    resized = base_dir.rstrip("/") + "_resized"
    if not os.path.exists(resized):
        resize_images(base_dir, resized, RESIZE_SCALE_FACTOR)
    return resized


# Load client datasets
client_datasets = []
client_sizes = []
for cid, name in enumerate(CLIENT_NAMES):
    path = os.path.join(DATA_DIR, name)
    img_dir = get_active_image_dir(os.path.join(path, "images"))
    ann_path = os.path.join(path, "annotations", "annotations.jsonl")
    ds = CaptionDataset(ann_path, img_dir, CAPTION_TASK)
    if len(ds) == 0:
        raise ValueError(
            f"Client {cid+1} ({name}) has 0 samples — check the data split. "
            f"An empty dataset will hang forever with itertools.cycle()."
        )
    client_datasets.append(ds)
    client_sizes.append(len(ds))
    log.info(f"  Client {cid+1} ({name}): {len(ds)} samples")

# Load test dataset
test_path = os.path.join(DATA_DIR, TEST_NAME)
test_img_dir = get_active_image_dir(os.path.join(test_path, "images"))
test_ann = os.path.join(test_path, "annotations", "annotations.jsonl")
test_dataset = CaptionDataset(test_ann, test_img_dir, CAPTION_TASK)
log.info(f"  Test set      : {len(test_dataset)} samples\n")

# ========================= MODEL + LoRA =========================
log.info("=" * 70)
log.info("LOADING FLORENCE-2 + LoRA (fp16, no quantization)")
log.info("=" * 70)

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch_dtype,
    device_map={"": DEVICE},
    trust_remote_code=True,
)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj",
                     "linear", "Conv2d", "lm_head", "fc2"],
    task_type="CAUSAL_LM",
    bias="none",
    inference_mode=False,
    use_rslora=True,
    init_lora_weights="gaussian",
)

global_model = get_peft_model(base_model, lora_config)
global_model.enable_input_require_grads()
global_model.gradient_checkpointing_enable()
global_model.print_trainable_parameters()
log.info(f"GPU after model load : {gpu_mem_str()}")
log.info(f"[DEBUG] Initial trainable-weight checksum: {trainable_checksum(global_model):.6f}")
log.info("LoRA applied — only adapter weights are trainable\n")

# ========================= COLLATE =========================
def collate_fn(batch):
    prefixes, captions, images = zip(*batch)
    inputs = processor(
        text=list(prefixes),
        images=list(images),
        return_tensors="pt",
        padding=True,
    )
    # NOTE: do NOT call .to(DEVICE) here — causes CUDA error in forked workers
    return inputs, captions

# ========================= LOCAL TRAINING =========================
def local_train(model, dataset, lr, round_num, client_id,
                 local_steps=None, epochs=1):
    """
    Trains `model` on `dataset` either for a fixed number of optimizer
    steps (local_steps, the aggregation-frequency knob) or for a fixed
    number of epochs (old behavior, used only if local_steps is None).

    No FedProx term — this is plain local SGD/AdamW training. Weights
    are reset to the global snapshot by the caller before the next client.
    """
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=6,
        pin_memory=True, prefetch_factor=2,
    )

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    if local_steps is not None:
        num_steps = local_steps
    else:
        num_steps = max(1, epochs * (len(loader) // GRADIENT_ACCUMULATION))

    lr_scheduler = get_scheduler(
        "linear", optimizer,
        num_warmup_steps=min(2, num_steps),
        num_training_steps=num_steps,
    )

    model.train()
    optimizer.zero_grad()

    # itertools.cycle re-starts the loader once exhausted, so num_steps can
    # exceed what one pass over this client's data provides — this is what
    # decouples "amount of local computation" from "amount of local data".
    data_iter = itertools.cycle(loader)

    global_step = 0
    accum_counter = 0
    running_loss, n_batches = 0.0, 0
    grad_norm_logged = False  # only print the debug grad-norm line once per call

    bar = tqdm(
        total=num_steps,
        desc=f"  [R{round_num:02d}|C{client_id}] steps",
        dynamic_ncols=True,
    )

    while global_step < num_steps:
        inputs, captions = next(data_iter)

        inputs = {
            k: v.to(device=DEVICE, dtype=torch_dtype) if v.dtype.is_floating_point else v.to(DEVICE)
            for k, v in inputs.items()
        }
        labels = processor.tokenizer(
            list(captions), padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
        ).input_ids.to(DEVICE)

        outputs = model(**inputs, labels=labels)
        ce_loss = outputs.loss

        loss = ce_loss / GRADIENT_ACCUMULATION
        loss.backward()

        # DEBUG: confirm gradients are actually non-zero on the very first
        # backward pass of this call. If total_grad_norm prints ~0.0, the
        # loss isn't flowing into the LoRA adapter parameters at all —
        # that's the root cause to chase (requires_grad wiring or a
        # gradient-checkpointing / enable_input_require_grads() ordering
        # issue), not the FedAvg or aggregation-frequency logic.
        if not grad_norm_logged:
            total_grad_norm = sum(
                p.grad.norm().item()
                for p in model.parameters()
                if p.requires_grad and p.grad is not None
            )
            log.info(f"    [DEBUG] Grad norm after first backward: {total_grad_norm:.6f}")
            grad_norm_logged = True

        running_loss += ce_loss.item()
        n_batches += 1
        accum_counter += 1

        if accum_counter == GRADIENT_ACCUMULATION:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            accum_counter = 0
            global_step += 1
            bar.update(1)
            bar.set_postfix(
                ce=f"{ce_loss.item():.4f}",
                avg_ce=f"{running_loss / n_batches:.4f}",
                lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                gpu=gpu_mem_str(),
            )

    bar.close()
    avg_ce = running_loss / max(n_batches, 1)
    log.info(f"    ↳ avg CE={avg_ce:.4f} | steps={global_step}/{num_steps} | GPU {gpu_mem_str()}")
    return avg_ce


# ========================= AGGREGATED-MODEL LOSS =========================
def compute_aggregated_loss(model, dataset, max_samples=200):
    """
    Teacher-forced CE loss of the current (post-aggregation) global model
    on held-out data. Cheap proxy for "how good is the merged model" —
    no beam search, just a forward pass with labels supplied.
    """
    model.eval()
    n = min(len(dataset), max_samples)
    losses = []
    with torch.no_grad():
        for i in range(n):
            prefix, caption, img = dataset[i]
            inputs = processor(text=prefix, images=img, return_tensors="pt").to(DEVICE)
            inputs = {
                k: v.to(dtype=torch.float16) if v.dtype.is_floating_point else v
                for k, v in inputs.items()
            }
            labels = processor.tokenizer(
                caption, return_tensors="pt", return_token_type_ids=False
            ).input_ids.to(DEVICE)
            out = model(**inputs, labels=labels)
            losses.append(out.loss.item())
    model.train()
    return float(np.mean(losses))


# ========================= EVALUATION (generation metrics) =========================
def generate_predictions(model, dataset, processor, max_samples=200):
    model.eval()
    preds, refs = [], []
    n = min(len(dataset), max_samples)
    bar = tqdm(range(n), desc="  Generating predictions", dynamic_ncols=True)
    with torch.no_grad():
        for i in bar:
            prefix, gt, img = dataset[i]
            inputs = processor(text=prefix, images=img, return_tensors="pt").to(DEVICE)
            inputs = {
                k: v.to(dtype=torch.float16) if v.dtype.is_floating_point else v
                for k, v in inputs.items()
            }
            generated = model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=3,
                length_penalty=1.0,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
            pred = processor.decode(generated[0], skip_special_tokens=True).strip()
            preds.append(pred)
            refs.append(gt)
            if i % 50 == 0:
                torch.cuda.empty_cache()
            bar.set_postfix(sample=f"{i+1}/{n}", gpu=gpu_mem_str())
    return preds, refs


def evaluate_captions(preds, refs):
    smooth = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    pt = [nltk.word_tokenize(p.lower()) for p in preds]
    rt = [[nltk.word_tokenize(r.lower())] for r in refs]       # corpus_bleu format
    rt_flat = [nltk.word_tokenize(r.lower()) for r in refs]    # meteor_score format

    bleu = corpus_bleu(rt, pt, smoothing_function=smooth)
    meteor = float(np.mean([meteor_score([rt_flat[i]], pt[i]) for i in range(len(preds))]))
    rouge_l = float(np.mean([
        rouge.score(preds[i], refs[i])["rougeL"].fmeasure for i in range(len(preds))
    ]))

    cider_scorer = Cider()
    gts = {i: [refs[i]] for i in range(len(refs))}
    res = {i: [preds[i]] for i in range(len(preds))}
    cs, _ = cider_scorer.compute_score(gts, res)

    return {"BLEU": bleu, "METEOR": meteor, "ROUGE-L": rouge_l, "CIDEr": float(cs)}


# ========================= MAIN FL LOOP =========================
log.info("\n" + "=" * 70)
log.info("STARTING FLORENCE-2 FEDERATED TRAINING (FedAvg, no FedProx)")
log.info(f"Run ID : run_{RUN_TIMESTAMP}")
log.info("=" * 70)

BEST_CIDER = -1.0
BEST_ROUND = -1
patience_counter = 0
all_metrics = []

# Loss histories for plotting — filled in every round, plotted every round
client_loss_history = {cid + 1: [] for cid in range(NUM_CLIENTS)}
aggregated_loss_history = []

for round_num in range(ROUNDS):
    log.info(f"\n{'='*70}")
    log.info(
        f"Round {round_num+1}/{ROUNDS} | LR: {LEARNING_RATE:.2e} | "
        f"Best CIDEr: {BEST_CIDER:.4f} | GPU: {gpu_mem_str()}"
    )
    log.info("=" * 70)

    client_updates = []

    for cid, client_ds in enumerate(client_datasets):
        log.info(
            f"\n  ── Client {cid+1}/{NUM_CLIENTS} "
            f"({CLIENT_NAMES[cid]}, {len(client_ds)} samples) ──"
        )

        # Snapshot global LoRA weights on CPU
        original_trainable = {
            name: param.clone().cpu()
            for name, param in global_model.named_parameters()
            if param.requires_grad
        }

        avg_ce = local_train(
            global_model, client_ds, LEARNING_RATE,
            round_num=round_num + 1,
            client_id=cid + 1,
            local_steps=LOCAL_STEPS,
            epochs=LOCAL_EPOCHS,
        )
        client_loss_history[cid + 1].append(avg_ce)

        # DEBUG: confirm this client's local training actually moved the
        # weights away from the snapshot taken before local_train() ran.
        with torch.no_grad():
            post_local_checksum = trainable_checksum(global_model)
        log.info(f"    [DEBUG] Post-local-train checksum: {post_local_checksum:.6f}")

        # Collect updated weights
        new_trainable = {
            name: param.clone().cpu()
            for name, param in global_model.named_parameters()
            if param.requires_grad
        }
        client_updates.append((new_trainable, len(client_ds)))

        # Reset to global before next client
        for name, param in global_model.named_parameters():
            if param.requires_grad:
                param.data.copy_(original_trainable[name].to(param.device))

        gc.collect()
        torch.cuda.empty_cache()
        log.info(f"  Client {cid+1} done | GPU after cleanup: {gpu_mem_str()}")

    # ── FedAvg ───────────────────────────────────────────────────────────
    log.info("\n  ── FedAvg aggregation ──")
    total_size = sum(size for _, size in client_updates)
    first_keys = list(client_updates[0][0].keys())

    matched = 0  # DEBUG: counts how many parameter names actually got overwritten
    agg_bar = tqdm(first_keys, desc="  Aggregating weights",
                    dynamic_ncols=True, leave=False)
    for key in agg_bar:
        avg_tensor = sum(
            state[key] * (size / total_size) for state, size in client_updates
        )
        for name, param in global_model.named_parameters():
            if name == key:
                param.data.copy_(avg_tensor.to(param.device))
                matched += 1
                break
        agg_bar.set_postfix(param=key[-35:])
    log.info(f"  [DEBUG] Aggregation matched {matched}/{len(first_keys)} parameter names")
    log.info(f"  Aggregation done | GPU: {gpu_mem_str()}")

    # DEBUG: this is the key check from the flat-loss investigation.
    # If this number is identical round to round, the aggregated model's
    # weights genuinely are not changing between rounds.
    post_agg_checksum = trainable_checksum(global_model)
    log.info(f"  [DEBUG] Post-aggregation trainable-weight checksum: {post_agg_checksum:.6f}")

    # ── Post-aggregation CE loss (cheap, teacher-forced) ────────────────
    agg_loss = compute_aggregated_loss(global_model, test_dataset)
    aggregated_loss_history.append(agg_loss)
    log.info(f"  Post-aggregation CE loss: {agg_loss:.4f}")

    # ── Update loss plots every round (mirrors round_metrics.csv live update) ──
    plot_client_losses(client_loss_history, SCORES_DIR)
    plot_aggregated_loss(aggregated_loss_history, SCORES_DIR)
    plot_combined(client_loss_history, aggregated_loss_history, SCORES_DIR)

    # ── Generation-quality evaluation ───────────────────────────────────
    log.info("\n  ── Evaluation on test set ──")
    preds, refs = generate_predictions(global_model, test_dataset, processor, max_samples=200)
    metrics = evaluate_captions(preds, refs)
    all_metrics.append({
        "round": round_num + 1,
        "local_steps": LOCAL_STEPS,
        "aggregated_ce_loss": agg_loss,
        "trainable_checksum": post_agg_checksum,   # DEBUG column — track/plot vs. round to spot a stuck model
        **{f"client_{cid+1}_ce_loss": client_loss_history[cid + 1][-1] for cid in range(NUM_CLIENTS)},
        **metrics,
    })

    pd.DataFrame(all_metrics).to_csv(
        os.path.join(SCORES_DIR, "round_metrics.csv"), index=False
    )

    log.info(f"\n  ── Round {round_num+1} Results ──")
    log.info(f"    BLEU    = {metrics['BLEU']:.4f}")
    log.info(f"    METEOR  = {metrics['METEOR']:.4f}")
    log.info(f"    ROUGE-L = {metrics['ROUGE-L']:.4f}")
    log.info(f"    CIDEr   = {metrics['CIDEr']:.4f}")

    # ── Save best / early stopping ──────────────────────────────────────
    if metrics["CIDEr"] > BEST_CIDER:
        BEST_CIDER = metrics["CIDEr"]
        BEST_ROUND = round_num + 1
        patience_counter = 0
        best_path = os.path.join(CHECKPOINT_DIR, "best")
        os.makedirs(best_path, exist_ok=True)
        global_model.save_pretrained(best_path)
        processor.save_pretrained(best_path)
        log.info(f"\n  *** NEW BEST CIDEr={BEST_CIDER:.4f} (round {BEST_ROUND}) → {best_path}")
    else:
        patience_counter += 1
        log.info(f"\n  No improvement ({patience_counter}/{PATIENCE})")
        if patience_counter >= PATIENCE:
            log.info("  Early stopping triggered.")
            break

    gc.collect()
    torch.cuda.empty_cache()

# ========================= FINAL EVALUATION =========================
log.info("\n" + "=" * 70)
log.info("FINAL EVALUATION (full test set)")
log.info("=" * 70)
log.info("Loading best checkpoint...")

base_final = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch_dtype,
    device_map={"": DEVICE},
    trust_remote_code=True,
)
best_path = os.path.join(CHECKPOINT_DIR, "best")
final_model = PeftModel.from_pretrained(base_final, best_path, is_trainable=False)

preds, refs = generate_predictions(
    final_model, test_dataset, processor, max_samples=len(test_dataset)
)
final_metrics = evaluate_captions(preds, refs)

log.info(f"\n  BLEU    = {final_metrics['BLEU']:.4f}")
log.info(f"  METEOR  = {final_metrics['METEOR']:.4f}")
log.info(f"  ROUGE-L = {final_metrics['ROUGE-L']:.4f}")
log.info(f"  CIDEr   = {final_metrics['CIDEr']:.4f}")

# ── Save results ─────────────────────────────────────────────────────────
results = {
    "run_id"            : f"run_{RUN_TIMESTAMP}",
    "model"              : MODEL_NAME,
    "task"               : CAPTION_TASK,
    "precision"          : "fp16_no_quantization",
    "local_steps"        : LOCAL_STEPS,
    "rounds"             : ROUNDS,
    "best_round"         : BEST_ROUND,
    "best_cider"         : BEST_CIDER,
    "final_metrics"      : final_metrics,
    "all_round_metrics"  : all_metrics,
}
with open(os.path.join(SCORES_DIR, "final_results.json"), "w") as f:
    json.dump(results, f, indent=2)

log.info(f"\n{'='*70}")
log.info("TRAINING COMPLETE!")
log.info(f"Run ID          : run_{RUN_TIMESTAMP}")
log.info(f"Run directory   : {OUTPUT_DIR}")
log.info(f"Log file        : {LOG_FILE}")
log.info(f"Best checkpoint : {os.path.join(CHECKPOINT_DIR, 'best')}")
log.info(f"Scores          : {SCORES_DIR}")
log.info(f"  * round_metrics.csv (per-client + aggregated CE loss, generation metrics)")
log.info(f"  * client_losses.png / aggregated_loss.png / combined_loss.png")
log.info(f"  * final_results.json")
log.info(f"  * run_config.json")
log.info(f"Best CIDEr      : {BEST_CIDER:.4f} (round {BEST_ROUND})")
log.info("=" * 70)

"""
Example usage:

python florence_fed_captioning.py \
    --data_dir /path/to/fed_input_data \
    --output_dir /path/to/Output \
    --local_steps 8 \
    --rounds 15

# Sweep aggregation frequency while holding TOTAL local step budget
# constant (here: 120 steps per client total across the whole run):
for steps in 1 2 4 8 15; do
    rounds=$((120 / steps))
    python florence_fed_captioning.py --local_steps $steps --rounds $rounds
done
"""