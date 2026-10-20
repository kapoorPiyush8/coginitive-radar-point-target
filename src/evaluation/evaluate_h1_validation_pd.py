
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter

state_dir = Path(
    "/content/drive/MyDrive/radar_project/experiment_state"
)
conf_dir = state_dir / "h1_confidence"
label_dir = Path(
    "/content/drive/MyDrive/radar_project/radar_dataset/"
    "single_target_exoclutter_H1_thermal_noise"
)

val_idx = np.load(state_dir / "h1_val_indices.npy")
thresholds = np.array([
    1e-12, 1e-11, 1e-10, 1e-9,
    1e-8, 5e-8, 1e-7, 1e-6,
    1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5
])

# Load only validation scenes, preserving global scene-index order.
val_scores = []
val_labels = []

for batch_id in range(400):
    start = batch_id * 50
    stop = start + 50

    selected = val_idx[(val_idx >= start) & (val_idx < stop)]
    if len(selected) == 0:
        continue

    conf = np.load(conf_dir / f"h1_conf_{batch_id:04d}.npy")
    labels = np.load(
        label_dir / f"labels_{start:05d}_{stop-1:05d}.npy"
    )

    local_idx = selected - start
    val_scores.extend(conf[local_idx])
    val_labels.extend(labels[local_idx])

val_scores = np.asarray(val_scores, dtype=np.float32)
val_labels = np.asarray(val_labels, dtype=np.uint8)

print("Validation score shape:", val_scores.shape)
print("Validation label shape:", val_labels.shape)
assert len(val_scores) == len(val_idx) == 4000
assert len(val_labels) == 4000

# Extract positive ground-truth pixel for each scene.
target_yx = []
for lab in val_labels:
    pos = np.argwhere(lab > 0)
    assert len(pos) == 1, f"Expected one target pixel, got {len(pos)}"
    target_yx.append(pos[0])
target_yx = np.asarray(target_yx)

# Diagnostic 3x3 local maxima, followed by 3-pixel target matching.
local_max = maximum_filter(
    val_scores, size=(1, 3, 3), mode="constant", cval=0
)
is_peak = (val_scores == local_max) & (val_scores > 0)

yy, xx = np.ogrid[:256, :256]
dy = yy[None, :, :] - target_yx[:, 0, None, None]
dx = xx[None, :, :] - target_yx[:, 1, None, None]
near_target = (dy * dy + dx * dx) <= 9

print("\nThreshold       Validation PD (3-pixel tolerance)")
for threshold in thresholds:
    candidates = is_peak & (val_scores >= threshold)
    detected = np.any(candidates & near_target, axis=(1, 2))
    print(f"{threshold:10.1e}      {detected.mean():.6f}")
