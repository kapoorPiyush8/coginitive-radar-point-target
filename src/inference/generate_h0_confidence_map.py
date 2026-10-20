import os
import glob
import numpy as np
import torch
import torch.nn.functional as F

H0_DIR = "/content/drive/MyDrive/radar_project/radar_dataset/S1_targetless_thermal_noise"

H0_CONF_DIR = "/content/drive/MyDrive/radar_project/experiment_state/h0_confidence"
os.makedirs(H0_CONF_DIR, exist_ok=True)

rdm_files = sorted(glob.glob(os.path.join(H0_DIR, "rdm_*.npy")))

print("H0 files:", len(rdm_files))

model.eval()

all_confidence = []

with torch.no_grad():

    for file_idx, rdm_file in enumerate(rdm_files):

        rdms = np.load(rdm_file).astype(np.float32)

        # Same preprocessing used during training/inference
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

        # Move complete confidence maps to CPU
        all_confidence.append(
            pred[:, 0].cpu().numpy().astype(np.float32)
        )

        if (file_idx + 1) % 20 == 0:
            print(f"Processed {file_idx + 1}/{len(rdm_files)} files")

h0_confidence = np.concatenate(all_confidence, axis=0)

print("\nComplete.")
print("Confidence-map shape:", h0_confidence.shape)
print("dtype:", h0_confidence.dtype)
print("Minimum:", h0_confidence.min())
print("Maximum:", h0_confidence.max())
print("Mean:", h0_confidence.mean())

# Save for later threshold/NMS experiments
save_path = os.path.join(H0_CONF_DIR, "h0_confidence_maps.npy")
np.save(save_path, h0_confidence)

print("Saved:", save_path)
