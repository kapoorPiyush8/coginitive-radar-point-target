import os
import glob
import numpy as np
import torch
import torch.nn.functional as F

BASE = "/content/drive/MyDrive/radar_project"
H0_PATH = os.path.join(
    BASE, "experiment_state", "h0_confidence",
    "h0_confidence_maps.npy"
)
H1_DIR = os.path.join(
    BASE, "experiment_state", "h1_confidence"
)
H1_DATA_DIR = os.path.join(
    BASE, "radar_dataset",
    "single_target_exoclutter_H1_thermal_noise"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Evaluation device:", device)

thresholds = torch.tensor([
    1e-12, 1e-11, 1e-10, 1e-9, 1e-8,
    5e-8, 1e-7, 1e-6, 1e-5, 1e-4,
    1e-3, 1e-2, 0.1, 0.5
], device=device)

# 3x3 local maxima, using max pooling.
# Inputs are (batch, 1, height, width).
def local_maxima(x):
    pooled = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    return (x == pooled) & (x > 0)

# H0: count scenes with at least one candidate at each threshold.
h0 = np.load(H0_PATH, mmap_mode="r")
n_h0 = h0.shape[0]
h0_flagged = torch.zeros(len(thresholds), dtype=torch.int64, device=device)

batch_size = 100

for start in range(0, n_h0, batch_size):
    end = min(start + batch_size, n_h0)
    x = torch.from_numpy(np.array(h0[start:end], copy=True)).to(device)
    x = x.unsqueeze(1)

    peaks = local_maxima(x)

    for j, tau in enumerate(thresholds):
        candidates = peaks & (x >= tau)
        h0_flagged[j] += candidates.flatten(1).any(dim=1).sum()

    if end % 2000 == 0 or end == n_h0:
        print(f"H0: {end}/{n_h0}")

pfa = (h0_flagged.float() / n_h0).cpu().numpy()

# H1: use the saved labels for target matching.
conf_files = sorted(glob.glob(os.path.join(H1_DIR, "h1_conf_*.npy")))
label_files = sorted(glob.glob(os.path.join(H1_DATA_DIR, "labels_*.npy")))

assert len(conf_files) == 400
assert len(label_files) == 400

# Counts for a 3-pixel Euclidean tolerance.
radius = 3
offsets = [
    (dy, dx)
    for dy in range(-radius, radius + 1)
    for dx in range(-radius, radius + 1)
    if dy * dy + dx * dx <= radius * radius
]

n_h1 = 0
h1_detected = torch.zeros(len(thresholds), dtype=torch.int64, device=device)

for batch_idx, (conf_path, label_path) in enumerate(zip(conf_files, label_files)):
    conf = np.load(conf_path)
    labels = np.load(label_path)

    x = torch.from_numpy(conf).to(device).unsqueeze(1)
    y = torch.from_numpy(labels).to(device).unsqueeze(1)

    assert x.shape == y.shape
    assert torch.all(y.sum(dim=(1, 2, 3)) == 1)

    # Candidate peaks.
    peaks = local_maxima(x)

    # Dilate target pixels by the fixed 3-pixel matching radius.
    # A peak inside this region counts as a target-associated detection.
    target_neighborhood = torch.zeros_like(y, dtype=torch.bool)
    for dy, dx in offsets:
        shifted = torch.roll(y > 0.5, shifts=(dy, dx), dims=(2, 3))
        if dy > 0:
            shifted[:, :, :dy, :] = False
        elif dy < 0:
            shifted[:, :, dy:, :] = False
        if dx > 0:
            shifted[:, :, :, :dx] = False
        elif dx < 0:
            shifted[:, :, :, dx:] = False
        target_neighborhood |= shifted

    for j, tau in enumerate(thresholds):
        candidates = peaks & (x >= tau)
        detected = (candidates & target_neighborhood).flatten(1).any(dim=1)
        h1_detected[j] += detected.sum()

    n_h1 += len(conf)

    if (batch_idx + 1) % 40 == 0:
        print(f"H1: {batch_idx + 1}/400")

pd = (h1_detected.float() / n_h1).cpu().numpy()

print("\nPreliminary GPU detector operating points")
print(f"{'Threshold':>12} | {'P_D (3px)':>12} | {'H0 flagged rate':>16}")
for j, tau in enumerate(thresholds.cpu().numpy()):
    print(f"{tau:12.1e} | {pd[j]:12.6f} | {pfa[j]:16.6e}")

out_path = os.path.join(
    BASE, "experiment_state", "gpu_operating_points.npz"
)
np.savez(
    out_path,
    thresholds=thresholds.cpu().numpy(),
    pd=pd,
    pfa=pfa,
    n_h0=n_h0,
    n_h1=n_h1,
    matching_radius_pixels=radius
)
print("\nSaved:", out_path)
