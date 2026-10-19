# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PROJECT_PATH = Path("/content/drive/MyDrive/radar_project")

S1_DIR = PROJECT_PATH / "radar_dataset" / "S1_targetless_thermal_noise"
OUTPUT_DIR = PROJECT_PATH / "results" / "S1_confidence_maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = PROJECT_PATH / "checkpoints" / "dice_paper_baseline_current.pth"

# ------------------------------------------------------------
# Load saved network weights
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model = model.to(device)
model.eval()

print("Loaded:", CHECKPOINT)

# ------------------------------------------------------------
# Same preprocessing used during training
# ------------------------------------------------------------
RDM_MEAN = -33.711260
RDM_STD  = 29.247385

# ------------------------------------------------------------
# Find S1 RDM batches
# ------------------------------------------------------------
s1_files = sorted(S1_DIR.glob("*.npy"))

print("S1 files:", len(s1_files))

# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------
with torch.no_grad():

    for file_idx, s1_file in enumerate(s1_files):

        rdm_batch = np.load(s1_file, mmap_mode="r")

        # Expected: (100, 69, 1334)
        confidence_maps = []

        for i in range(len(rdm_batch)):

            rdm = np.array(
                rdm_batch[i],
                dtype=np.float32,
                copy=True
            )

            # Native RDM -> network input
            x = torch.from_numpy(rdm).unsqueeze(0).unsqueeze(0)

            x = F.interpolate(
                x,
                size=(256, 256),
                mode="bilinear",
                align_corners=False
            )

            # Same global standardization as training
            x = (x - RDM_MEAN) / RDM_STD

            x = x.to(device)

            # Raw sigmoid confidence map
            prediction = model(x)

            confidence_maps.append(
                prediction.squeeze().cpu().numpy().astype(np.float32)
            )

        confidence_maps = np.stack(confidence_maps)

        # Save one batch
        output_file = OUTPUT_DIR / s1_file.name
        np.save(output_file, confidence_maps)

        print(
            f"{file_idx+1:02d}/{len(s1_files):02d} | "
            f"{s1_file.name} | "
            f"shape={confidence_maps.shape} | "
            f"range=[{confidence_maps.min():.6f}, "
            f"{confidence_maps.max():.6f}]"
        )

print("\n" + "=" * 70)
print("S1 INFERENCE COMPLETE")
print("=" * 70)
print("Location:", OUTPUT_DIR)
