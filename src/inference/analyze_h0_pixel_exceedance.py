import numpy as np

H0_CONF_PATH = "/content/drive/MyDrive/radar_project/experiment_state/h0_confidence/h0_confidence_maps.npy"

h0_confidence = np.load(H0_CONF_PATH, mmap_mode="r")

print("Shape:", h0_confidence.shape)
print("dtype:", h0_confidence.dtype)

thresholds = np.array([
    1e-12,
    2e-12,
    5e-12,
    1e-11,
    2e-11,
    5e-11,
    1e-10,
    2e-10,
    5e-10,
    1e-9,
    2e-9,
    5e-9,
    1e-8,
    2e-8,
    5e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    0.1,
    0.5,
])

total_pixels = h0_confidence.size

print("\nH0 pixel exceedance:")
print(f"{'Threshold':>12} | {'Pixels >= threshold':>20} | {'Pixel exceedance':>18}")

for tau in thresholds:

    count = np.count_nonzero(h0_confidence >= tau)
    p = count / total_pixels

    print(f"{tau:12.1e} | {count:20d} | {p:18.6e}")
