
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Adjust these imports if your committed filenames differ.
from src.models.paper_unet import PaperUNet
from src.losses.dice_loss import PaperDiceLoss
from src.datasets.radar_h1_dataset import RadarH1Dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the paper-aligned Dice baseline."
    )

    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr-start", type=float, default=7e-3)
    parser.add_argument("--lr-end", type=float, default=5e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best_val_loss: float,
    history: dict,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
        },
        path,
    )


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_scenes = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = x_batch.shape[0]
            total_loss += loss.item() * batch_size
            total_scenes += batch_size

    if total_scenes == 0:
        raise RuntimeError("DataLoader contains no scenes.")

    return total_loss / total_scenes


def main():
    args = parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")

    if args.lr_start <= 0 or args.lr_end <= 0:
        raise ValueError("Learning rates must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------------------------------------
    # Dataset and frozen split
    # --------------------------------------------------------

    dataset = RadarH1Dataset(args.data_dir)

    train_indices = np.load(
        args.split_dir / "h1_train_indices.npy"
    )
    val_indices = np.load(
        args.split_dir / "h1_val_indices.npy"
    )

    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError("Training and validation splits must be non-empty.")

    if (
        np.intersect1d(train_indices, val_indices).size > 0
        or np.any(train_indices < 0)
        or np.any(val_indices < 0)
        or np.any(train_indices >= len(dataset))
        or np.any(val_indices >= len(dataset))
    ):
        raise ValueError("Invalid or overlapping split indices.")

    train_dataset = Subset(dataset, train_indices.tolist())
    val_dataset = Subset(dataset, val_indices.tolist())

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # --------------------------------------------------------
    # Model, loss, optimizer, scheduler
    # --------------------------------------------------------

    model = PaperUNet().to(device)
    criterion = PaperDiceLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr_start,
    )

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args.lr_end / args.lr_start,
        total_iters=args.epochs,
    )

    best_path = args.output_dir / "dice_baseline_best.pt"
    latest_path = args.output_dir / "dice_baseline_latest.pt"
    history_path = args.output_dir / "dice_baseline_history.json"
    summary_path = args.output_dir / "dice_baseline_summary.json"

    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "epoch_time": [],
    }

    best_val_loss = float("inf")
    start_epoch = 0

    # --------------------------------------------------------
    # Resume from latest checkpoint
    # --------------------------------------------------------

    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(
                f"Cannot resume: checkpoint not found at {latest_path}"
            )

        checkpoint = torch.load(
            latest_path,
            map_location=device,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = int(checkpoint["epoch"])
        best_val_loss = float(checkpoint["best_val_loss"])
        history = checkpoint["history"]

        if start_epoch >= args.epochs:
            print("Training has already reached the requested epoch count.")
            return

        print(f"Resuming from epoch {start_epoch + 1}")

    # --------------------------------------------------------
    # Experiment configuration
    # --------------------------------------------------------

    print("\n=== Dice baseline ===")
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Training scenes:", len(train_dataset))
    print("Validation scenes:", len(val_dataset))
    print("Batch size:", args.batch_size)
    print("Epochs:", args.epochs)
    print("Initial LR:", args.lr_start)
    print("Final LR:", args.lr_end)
    print("Loss:", type(criterion).__name__)

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )

        val_loss = run_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        history["epoch_time"].append(epoch_time)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        # Advance the scheduler before saving, so the checkpoint
        # contains the state for the next epoch.
        scheduler.step()

        checkpoint_args = {
            "epoch": epoch + 1,
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "best_val_loss": best_val_loss,
            "history": history,
        }

        if is_best:
            save_checkpoint(
                best_path,
                **checkpoint_args,
            )

        save_checkpoint(
            latest_path,
            **checkpoint_args,
        )

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        marker = " <-- BEST" if is_best else ""

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"Train {train_loss:.6f} | "
            f"Val {val_loss:.6f} | "
            f"LR {current_lr:.6e} | "
            f"{epoch_time:.1f}s{marker}"
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    summary = {
        "experiment": "Dice baseline",
        "architecture": "PaperUNet",
        "loss": type(criterion).__name__,
        "dataset": args.data_dir.name,
        "train_scenes": len(train_dataset),
        "validation_scenes": len(val_dataset),
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history["train_loss"]),
        "lr_start": args.lr_start,
        "lr_end": args.lr_end,
        "best_validation_loss": best_val_loss,
        "best_checkpoint": str(best_path),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete.")
    print("Best validation loss:", best_val_loss)
    print("Best checkpoint:", best_path)
    print("History:", history_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
