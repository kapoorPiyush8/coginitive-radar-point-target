h1_max_scores = []
h1_max_positions = []
h1_target_positions = []

model.eval()

with torch.no_grad():

    for file_idx, (rdm_file, meta_file) in enumerate(zip(rdm_files, meta_files)):

        rdms = np.load(rdm_file).astype(np.float32)
        metadata = np.load(meta_file, allow_pickle=True)

        assert rdms.ndim == 3
        assert rdms.shape[1:] == (69, 1334)
        assert len(rdms) == len(metadata)

        # Same preprocessing as training
        rdms = np.clip(rdms, -40.0, 0.0)
        rdms = (rdms + 40.0) / 40.0

        x = torch.from_numpy(rdms).unsqueeze(1).to(device)

        x = F.interpolate(
            x,
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )

        pred = model(x)

        # Flatten spatial dimensions
        pred_flat = pred[:, 0].reshape(pred.shape[0], -1)

        max_scores, max_indices = torch.max(pred_flat, dim=1)

        max_scores = max_scores.cpu().numpy()
        max_indices = max_indices.cpu().numpy()

        # Convert flattened index -> (y, x)
        max_y, max_x = np.unravel_index(
            max_indices,
            (256, 256)
        )

        for i in range(len(metadata)):

            h1_max_scores.append(float(max_scores[i]))
            h1_max_positions.append(
                (int(max_y[i]), int(max_x[i]))
            )

            m = metadata[i].item() if hasattr(metadata[i], "item") else metadata[i]

            # Target physical coordinates
            r = float(m["range_m"])
            v = float(m["velocity_mps"])

            # Native ROI axes
            roi_range_axis = range_axis[range_mask]
            roi_velocity_axis = vel_axis[velocity_mask]

            native_x = int(np.argmin(np.abs(roi_range_axis - r)))
            native_y = int(np.argmin(np.abs(roi_velocity_axis - v)))

            # Same mapping used for labels
            target_x = int(
                round(native_x * 255 / (len(roi_range_axis) - 1))
            )

            target_y = int(
                round(native_y * 255 / (len(roi_velocity_axis) - 1))
            )

            h1_target_positions.append(
                (target_y, target_x)
            )

        if (file_idx + 1) % 40 == 0:
            print(f"Processed {file_idx + 1}/{len(rdm_files)} files")


h1_max_scores = np.array(h1_max_scores)
h1_max_positions = np.array(h1_max_positions)
h1_target_positions = np.array(h1_target_positions)

print("\nH1 inference complete.")
print("Number of H1 scenes:", len(h1_max_scores))

print("\nConfidence statistics:")
print("Mean:", h1_max_scores.mean())
print("Median:", np.median(h1_max_scores))
print("95th percentile:", np.percentile(h1_max_scores, 95))
print("99th percentile:", np.percentile(h1_max_scores, 99))
print("99.9th percentile:", np.percentile(h1_max_scores, 99.9))
print("Maximum:", h1_max_scores.max())

print("\nFirst 10 predictions:")
for i in range(10):
    print(
        f"Scene {i:5d} | "
        f"score={h1_max_scores[i]:.6e} | "
        f"pred=(y={h1_max_positions[i,0]}, x={h1_max_positions[i,1]}) | "
        f"target=(y={h1_target_positions[i,0]}, x={h1_target_positions[i,1]})"
    )
