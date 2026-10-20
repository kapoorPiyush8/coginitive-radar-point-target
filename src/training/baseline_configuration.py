# ============================================================
# DICE BASELINE — TRAINING CONFIGURATION
# ============================================================

import torch
from pathlib import Path

# Fresh model for the actual experiment
model = PaperUNet().to(device)

# Adam optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=7e-3
)

# Linear learning-rate schedule:
# 7e-3 -> 5e-5 over 150 epochs
EPOCHS = 150
LR_START = 7e-3
LR_END = 5e-5

scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=1.0,
    end_factor=LR_END / LR_START,
    total_iters=EPOCHS
)

# Loss
criterion = DiceLoss()

# Checkpoint location
CHECKPOINT_DIR = Path(
    "/content/drive/MyDrive/radar_project/experiment_state/checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR / "dice_baseline_best.pt"
)

print("Model parameters:",
      sum(p.numel() for p in model.parameters()))

print("Optimizer       :", type(optimizer).__name__)
print("Initial LR      :", optimizer.param_groups[0]["lr"])
print("Final LR        :", LR_END)
print("Epochs          :", EPOCHS)
print("Loss            :", type(criterion).__name__)
print("Best checkpoint :", BEST_CHECKPOINT)
