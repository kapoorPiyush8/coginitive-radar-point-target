import numpy as np
from scipy.ndimage import maximum_filter

H0_CONF_PATH = (
    "/content/drive/MyDrive/radar_project/"
    "experiment_state/h0_confidence/h0_confidence_maps.npy"
)

h0 = np.load(H0_CONF_PATH, mmap_mode="r")

thresholds = np.array([
    1e-12, 2e-12, 5e-12,
    1e-11, 2e-11, 5e-11,
    1e-10, 2e-10, 5e-10,
    1e-9, 2e-9, 5e-9,
    1e-8, 2e-8, 5e-8,
    1e-7, 1e-6, 1e-5,
    1e-4, 1e-3, 1e-2, 0.1, 0.5
])

n_images = h0.shape[0]
chunk_size = 100

false_detections = np.zeros(len(thresholds), dtype=np.int64)
images_with_detection = np.zeros(len(thresholds), dtype=np.int64)

for start in range(0, n_images, chunk_size):
    end = min(start + chunk_size, n_images)
    maps = np.asarray(h0[start:end])

    # Diagnostic 3x3 local-maximum operation
    local_max = maximum_filter(
        maps,
        size=(1, 3, 3),
        mode="constant",
        cval=0.0
    )
    is_local_max = (maps == local_max)

    for j, tau in enumerate(thresholds):
        detections = is_local_max & (maps >= tau)

        false_detections[j] += np.count_nonzero(detections)
        images_with_detection[j] += np.count_nonzero(
            np.any(detections, axis=(1, 2))
        )

    if end % 2000 == 0 or end == n_images:
        print(f"Processed {end}/{n_images} scenes")

print("\nH0 candidate-detection diagnostic")
print(f"{'Threshold':>12} | {'Candidates':>12} | {'Scenes flagged':>14} | {'Flagged-scene rate':>19}")

for j, tau in enumerate(thresholds):
    rate = images_with_detection[j] / n_images
    print(
        f"{tau:12.1e} | "
        f"{false_detections[j]:12d} | "
        f"{images_with_detection[j]:14d} | "
        f"{rate:19.6e}"
    )
