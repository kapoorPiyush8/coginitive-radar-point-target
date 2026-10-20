import os
import glob
import numpy as np
from scipy.ndimage import maximum_filter

BASE = "/content/drive/MyDrive/radar_project"

CONF_DIR = os.path.join(
    BASE, "experiment_state", "h1_confidence"
)

H1_DIR = os.path.join(
    BASE, "radar_dataset",
    "single_target_exoclutter_H1_thermal_noise"
)

conf_files = sorted(glob.glob(os.path.join(CONF_DIR, "h1_conf_*.npy")))
label_files = sorted(glob.glob(os.path.join(H1_DIR, "labels_*.npy")))

assert len(conf_files) == 400, f"Expected 400 confidence files, got {len(conf_files)}"
assert len(label_files) == 400, f"Expected 400 label files, got {len(label_files)}"

# Same thresholds as the H0 diagnostic
thresholds = np.array([
    1e-12, 2e-12, 5e-12,
    1e-11, 2e-11, 5e-11,
    1e-10, 2e-10, 5e-10,
    1e-9, 2e-9, 5e-9,
    1e-8, 2e-8, 5e-8,
    1e-7, 1e-6, 1e-5,
    1e-4, 1e-3, 1e-2, 0.1, 0.5
])

# Pixel-distance tolerances from the labelled target
tolerances = [0, 1, 3, 5, 10, 20]

n_scenes = 0
detected_counts = {
    radius: np.zeros(len(thresholds), dtype=np.int64)
    for radius in tolerances
}

for batch_idx, (conf_path, label_path) in enumerate(zip(conf_files, label_files)):

    conf = np.load(conf_path).astype(np.float32)
    labels = np.load(label_path).astype(np.float32)

    assert conf.shape == (50, 256, 256), (conf_path, conf.shape)
    assert labels.shape == (50, 256, 256), (label_path, labels.shape)

    # Verify that each scene has exactly one positive target pixel.
    positive_counts = (labels > 0.5).sum(axis=(1, 2))
    assert np.all(positive_counts == 1), (
        f"Unexpected target-label counts in {label_path}: "
        f"{np.unique(positive_counts, return_counts=True)}"
    )

    # Local maxima, consistent with the H0 diagnostic
    local_max = maximum_filter(
        conf,
        size=(1, 3, 3),
        mode="constant",
        cval=0.0
    )

    is_local_max = (conf == local_max)

    # Candidate detection at each threshold
    for j, tau in enumerate(thresholds):

        candidates = is_local_max & (conf >= tau)

        for radius in tolerances:
            for scene_idx in range(len(conf)):

                ys, xs = np.where(labels[scene_idx] > 0.5)
                ty, tx = int(ys[0]), int(xs[0])

                cy, cx = np.where(candidates[scene_idx])

                if len(cx) == 0:
                    continue

                # Euclidean distance from each candidate to target
                distance_sq = (cy - ty)**2 + (cx - tx)**2

                if np.any(distance_sq <= radius**2):
                    detected_counts[radius][j] += 1

    n_scenes += len(conf)

    if (batch_idx + 1) % 40 == 0:
        print(f"Processed {batch_idx + 1}/400 batches")

assert n_scenes == 20000

print("\nH1 detection probability")
print("Scenes:", n_scenes)

for radius in tolerances:
    print(f"\nSpatial tolerance: {radius} pixels")
    print(f"{'Threshold':>12} | {'Detected':>10} | {'P_D':>12}")

    for j, tau in enumerate(thresholds):
        count = detected_counts[radius][j]
        pd = count / n_scenes

        print(f"{tau:12.1e} | {count:10d} | {pd:12.6f}")

# Save results so they survive runtime restarts
result_path = os.path.join(
    BASE, "experiment_state", "h1_pd_threshold_sweep.npz"
)

np.savez(
    result_path,
    thresholds=thresholds,
    tolerances=np.array(tolerances),
    n_scenes=n_scenes,
    **{
        f"detected_radius_{r}": detected_counts[r]
        for r in tolerances
    }
)

print("\nSaved P_D results:", result_path)
