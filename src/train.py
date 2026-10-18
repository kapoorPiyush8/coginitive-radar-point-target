import time
import torch

NUM_EPOCHS = 150
LR_START = 7e-3
LR_END = 5e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = PaperUNet().to(device)
criterion = PaperDiceLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR_START
)

history = {
    "epoch": [],
    "learning_rate": [],
    "train_loss": [],
    "val_loss": [],
    "epoch_time_sec": []
}

best_val_loss = float("inf")
best_epoch = None

def get_epoch_lr(epoch):
    if NUM_EPOCHS <= 1:
        return LR_END
    return LR_START + (
        epoch / (NUM_EPOCHS - 1)
    ) * (LR_END - LR_START)

for epoch in range(NUM_EPOCHS):

    start_time = time.time()

    lr = get_epoch_lr(epoch)

    for group in optimizer.param_groups:
        group["lr"] = lr

    # ---------------- TRAIN ----------------
    model.train()

    train_loss_sum = 0.0
    train_samples = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        predictions = model(images)
        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()

        n = images.size(0)

        train_loss_sum += loss.item() * n
        train_samples += n

    train_loss = train_loss_sum / train_samples

    # ---------------- VALIDATION ----------------
    model.eval()

    val_loss_sum = 0.0
    val_samples = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            loss = criterion(predictions, labels)

            n = images.size(0)

            val_loss_sum += loss.item() * n
            val_samples += n

    val_loss = val_loss_sum / val_samples

    elapsed = time.time() - start_time

    history["epoch"].append(epoch + 1)
    history["learning_rate"].append(lr)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["epoch_time_sec"].append(elapsed)

    # ---------------- BEST MODEL ----------------
    marker = ""

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch + 1

        torch.save(
            {
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": lr,
            },
            CHECKPOINT_DIR / "dice_paper_baseline_best.pth"
        )

        marker = " <-- BEST"

    print(
        f"Epoch {epoch+1:03d}/150 | "
        f"LR {lr:.6e} | "
        f"Train {train_loss:.6f} | "
        f"Val {val_loss:.6f} | "
        f"{elapsed:.1f}s"
        f"{marker}"
    )

print("\nTraining complete.")
print("Best epoch:", best_epoch)
print("Best validation loss:", best_val_loss)
