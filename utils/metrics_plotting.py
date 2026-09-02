# -*- coding: utf-8 -*-
"""
metrics_plotting.py

Plotting utilities for the Florence-2 federated learning experiment.

Kept separate from florence_fed_captioning.py so plotting logic can be
reused, tweaked, or unit-tested without touching training code, and so
other scripts (analysis notebooks, sweep-comparison scripts) can import
the same functions and get visually consistent figures.
"""

import os
import matplotlib

# "Agg" is a non-interactive, headless-safe backend. This matters because
# HPC/Slurm jobs usually have no display server attached — without this,
# matplotlib can raise errors trying to open a GUI window that doesn't exist.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_client_losses(client_loss_history, save_dir, filename="client_losses.png"):
    """
    Plot each client's local training loss (avg CE per round) as its own line.

    Parameters
    ----------
    client_loss_history : dict[int, list[float]]
        Maps client_id -> list of avg local CE loss values, one entry per round.
    save_dir : str
        Directory to save the PNG into (pass SCORES_DIR from the main script).
    filename : str
        Output filename.

    Returns
    -------
    str
        Full path to the saved plot, so the caller can log it.
    """
    plt.figure(figsize=(8, 5))
    for cid, losses in sorted(client_loss_history.items()):
        rounds = range(1, len(losses) + 1)
        plt.plot(rounds, losses, marker="o", label=f"Client {cid}")
    plt.xlabel("Round")
    plt.ylabel("Local CE loss")
    plt.title("Per-client local training loss")
    plt.legend()
    plt.grid(alpha=0.3)

    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_aggregated_loss(aggregated_loss_history, save_dir, filename="aggregated_loss.png"):
    """
    Plot the global (post-FedAvg) model's held-out CE loss across rounds.

    Parameters
    ----------
    aggregated_loss_history : list[float]
        One value per round — CE loss of the aggregated model on the test set.
    save_dir : str
    filename : str

    Returns
    -------
    str
        Full path to the saved plot.
    """
    plt.figure(figsize=(8, 5))
    rounds = range(1, len(aggregated_loss_history) + 1)
    plt.plot(rounds, aggregated_loss_history, marker="o", color="black")
    plt.xlabel("Round")
    plt.ylabel("Aggregated model CE loss (test set)")
    plt.title("Post-aggregation loss")
    plt.grid(alpha=0.3)

    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_combined(client_loss_history, aggregated_loss_history, save_dir,
                   filename="combined_loss.png"):
    """
    Overlay per-client local loss curves with the aggregated loss curve on
    one figure. Useful for eyeballing the local-vs-global gap — if client
    lines keep dropping while the black aggregated line stalls, that's a
    sign local_steps is too high relative to your aggregation frequency.

    Parameters
    ----------
    client_loss_history : dict[int, list[float]]
    aggregated_loss_history : list[float]
    save_dir : str
    filename : str

    Returns
    -------
    str
        Full path to the saved plot.
    """
    plt.figure(figsize=(9, 6))
    for cid, losses in sorted(client_loss_history.items()):
        rounds = range(1, len(losses) + 1)
        plt.plot(rounds, losses, marker="o", alpha=0.6, label=f"Client {cid} (local)")

    rounds = range(1, len(aggregated_loss_history) + 1)
    plt.plot(rounds, aggregated_loss_history, marker="s", color="black",
              linewidth=2.5, label="Aggregated (global)")

    plt.xlabel("Round")
    plt.ylabel("CE loss")
    plt.title("Local vs. aggregated loss")
    plt.legend()
    plt.grid(alpha=0.3)

    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path