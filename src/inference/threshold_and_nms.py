PROJECT_PATH = Path("/content/drive/MyDrive/radar_project")

CONF_DIR = (
    PROJECT_PATH
    / "results"
    / "S1_confidence_maps"
)

NMS_DIR = (
    PROJECT_PATH
    / "results"
    / "S1_NMS"
)

NMS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Operating threshold
# ------------------------------------------------------------
THRESHOLD = 0.54

# NMS neighbourhood.
# A 3x3 neighbourhood is a conservative first implementation.
NMS_SIZE = 3

# ------------------------------------------------------------
# NMS function
# ------------------------------------------------------------
def threshold_and_nms(confidence_map, threshold=THRESHOLD, nms_size=NMS_SIZE):

    # Threshold
    candidates = confidence_map >= threshold

    # Local maximum in neighbourhood
    local_max = (
        confidence_map
        == maximum_filter(
            confidence_map,
            size=nms_size,
            mode="nearest"
        )
    )

    # Candidate must also be a local maximum
    detections = candidates & local_max

    # Detection coordinates and confidence
    rows, cols = np.where(detections)

    scores = confidence_map[rows, cols]

    return rows, cols, scores


# ------------------------------------------------------------
# Process all S1 confidence maps
# ------------------------------------------------------------
confidence_files = sorted(CONF_DIR.glob("*.npy"))

total_candidates = 0
total_detections = 0

for file_idx, file in enumerate(confidence_files):

    confidence_batch = np.load(file)

    batch_results = []

    for scene_idx in range(confidence_batch.shape[0]):

        confidence_map = confidence_batch[scene_idx]

        rows, cols, scores = threshold_and_nms(
            confidence_map,
            threshold=THRESHOLD,
            nms_size=NMS_SIZE
        )

        total_candidates += np.count_nonzero(
            confidence_map >= THRESHOLD
        )

        total_detections += len(scores)

        batch_results.append(
            {
                "rows": rows.astype(np.int16),
                "cols": cols.astype(np.int16),
                "scores": scores.astype(np.float32)
            }
        )

    output_file = (
        NMS_DIR
        / file.name.replace(
            ".npy",
            "_nms.npy"
        )
    )

    np.save(
        output_file,
        np.array(batch_results, dtype=object),
        allow_pickle=True
    )

    print(
        f"{file_idx+1:02d}/{len(confidence_files):02d} | "
        f"{file.name} | "
        f"scenes={len(batch_results)}"
    )


print("\n" + "=" * 70)
print("S1 THRESHOLD + NMS COMPLETE")
print("=" * 70)

print("Threshold       :", THRESHOLD)
print("NMS size        :", NMS_SIZE)
print("Raw candidates  :", total_candidates)
print("NMS detections  :", total_detections)
print("Saved to        :", NMS_DIR)
