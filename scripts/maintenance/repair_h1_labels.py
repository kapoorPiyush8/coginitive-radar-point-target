
H1_DIR = Path(
    "/content/drive/MyDrive/radar_project/radar_dataset/"
    "single_target_exoclutter_H1_thermal_noise"
)

label_files = sorted(H1_DIR.glob("labels_*.npy"))
metadata_files = sorted(H1_DIR.glob("metadata_*.npy"))

assert len(label_files) == len(metadata_files)

roi_range_axis = range_axis[range_mask]
roi_vel_axis = vel_axis[velocity_mask]

for label_file, metadata_file in zip(label_files, metadata_files):

    labels = np.zeros(
        np.load(label_file, mmap_mode="r").shape,
        dtype=np.float32
    )

    metadata = np.load(
        metadata_file,
        allow_pickle=True
    )

    for i, target in enumerate(metadata):

        R = float(target["range_m"])
        v = float(target["velocity_mps"])

        # Nearest native ROI bin
        r_idx = np.argmin(
            np.abs(roi_range_axis - R)
        )

        v_idx = np.argmin(
            np.abs(roi_vel_axis - v)
        )

        # Native ROI -> 256x256
        x = int(round(
            r_idx * 255 / (len(roi_range_axis) - 1)
        ))

        y = int(round(
            v_idx * 255 / (len(roi_vel_axis) - 1)
        ))

        labels[i, y, x] = 1.0

    np.save(label_file, labels)

    print(
        f"Repaired {label_file.name} | "
        f"positive pixels: {labels.sum():.0f}"
    )

print("\nLabel repair complete.")
