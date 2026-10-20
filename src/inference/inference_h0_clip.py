# Collect maximum confidence per H0 image.
# We initially store only the maximum because this is enough
# to inspect the false-alarm confidence distribution.

h0_max_scores = []

model.eval()

with torch.no_grad():

    for file_idx, rdm_file in enumerate(rdm_files):

        rdms = np.load(rdm_file).astype(np.float32)

        # Expected shape: (100, 69, 1334)
        assert rdms.ndim == 3
        assert rdms.shape[1:] == (69, 1334)

        # Paper preprocessing:
        # clip [-40, 0] dB -> [0, 1]
        rdms = np.clip(rdms, -40.0, 0.0)
        rdms = (rdms + 40.0) / 40.0

        # Convert to torch: (N, 1, 69, 1334)
        x = torch.from_numpy(rdms).unsqueeze(1).to(device)

        # Resize to model input: (N, 1, 256, 256)
        x = F.interpolate(
            x,
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )

        # CNN inference
        pred = model(x)

        # Maximum confidence in each image
        batch_max = pred.amax(dim=(1, 2, 3)).cpu().numpy()

        h0_max_scores.append(batch_max)

        if (file_idx + 1) % 20 == 0:
            print(f"Processed {file_idx + 1}/{len(rdm_files)} files")

h0_max_scores = np.concatenate(h0_max_scores)

print("\nH0 inference complete.")
print("Number of H0 scenes:", len(h0_max_scores))
print("Min max-score:", h0_max_scores.min())
print("Mean max-score:", h0_max_scores.mean())
print("Median max-score:", np.median(h0_max_scores))
print("95th percentile:", np.percentile(h0_max_scores, 95))
print("99th percentile:", np.percentile(h0_max_scores, 99))
print("99.9th percentile:", np.percentile(h0_max_scores, 99.9))
print("Maximum:", h0_max_scores.max())
