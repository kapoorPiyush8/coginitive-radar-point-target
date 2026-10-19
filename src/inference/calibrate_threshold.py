# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PROJECT_PATH = Path("/content/drive/MyDrive/radar_project")

CONF_DIR = (
    PROJECT_PATH
    / "results"
    / "S1_confidence_maps"
)

CAL_DIR = (
    PROJECT_PATH
    / "results"
    / "S1_threshold_calibration"
)

CAL_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Load all S1 confidence maps
# ------------------------------------------------------------
confidence_files = sorted(CONF_DIR.glob("*.npy"))

print("Confidence-map files:", len(confidence_files))

all_confidence = []

for f in confidence_files:
    data = np.load(f)

    print(
        f"{f.name:35s} "
        f"shape={data.shape} "
        f"range=[{data.min():.6f}, {data.max():.6f}]"
    )

    all_confidence.append(data)

# Combine all 1000 scenes
all_confidence = np.concatenate(all_confidence, axis=0)

print("\nCombined shape:", all_confidence.shape)
print("Total pixels:", all_confidence.size)
print("Global min:", all_confidence.min())
print("Global max:", all_confidence.max())

# ------------------------------------------------------------
# Threshold sweep
# ------------------------------------------------------------
thresholds = np.linspace(
    0.45,
    0.56,
    111
)

pfa_values = np.zeros_like(thresholds)
false_pixels = np.zeros_like(thresholds, dtype=np.int64)

total_pixels = all_confidence.size

for k, threshold in enumerate(thresholds):

    count = np.count_nonzero(
        all_confidence >= threshold
    )

    false_pixels[k] = count

    pfa_values[k] = count / total_pixels

# ------------------------------------------------------------
# Save calibration curve
# ------------------------------------------------------------
calibration = np.column_stack(
    [
        thresholds,
        false_pixels,
        pfa_values
    ]
)

calibration_file = CAL_DIR / "S1_threshold_Pfa_curve.npy"

np.save(
    calibration_file,
    calibration
)

# Also save as CSV for easy inspection
csv_file = CAL_DIR / "S1_threshold_Pfa_curve.csv"

np.savetxt(
    csv_file,
    calibration,
    delimiter=",",
    header="threshold,false_positive_pixels,Pfa",
    comments=""
)

# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("S1 THRESHOLD CALIBRATION COMPLETE")
print("=" * 70)

print("Total S1 scenes :", all_confidence.shape[0])
print("Total pixels    :", total_pixels)

print("\nSelected sweep:")
print("Threshold range :", thresholds[0], "to", thresholds[-1])
print("Number of points :", len(thresholds))

print("\nSample results:")

for idx in np.linspace(
    0,
    len(thresholds) - 1,
    12,
    dtype=int
):
    print(
        f"T={thresholds[idx]:.4f} | "
        f"false pixels={false_pixels[idx]:8d} | "
        f"Pfa={pfa_values[idx]:.8e}"
    )

print("\nSaved:")
print(calibration_file)
print(csv_file)
