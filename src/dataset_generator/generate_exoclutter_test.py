import numpy as np
from pathlib import Path

# ============================================================
# SINGLE-TARGET EXOCLUTTER EVALUATION DATASET
# PROJECT_PATH must be set for the local machine
# ============================================================

NUM_TEST_SCENES = 20_000
BATCH_SIZE = 100

TEST_DIR = (
    PROJECT_PATH
    / "radar_dataset"
    / "single_target_exoclutter_test"
)

TEST_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Same operational ROI as original dataset
# ------------------------------------------------------------

range_mask = (
    (range_axis >= 10) &
    (range_axis <= 60)
)

velocity_mask = (
    (vel_axis >= -10) &
    (vel_axis <= 10)
)

print("Range bins :", np.count_nonzero(range_mask))
print("Velocity bins:", np.count_nonzero(velocity_mask))

print("\n" + "=" * 70)
print("GENERATING SINGLE-TARGET EXOCLUTTER TEST DATASET")
print("=" * 70)

# ============================================================
# Generate in batches
# ============================================================

for batch_start in range(0, NUM_TEST_SCENES, BATCH_SIZE):

    batch_end = min(
        batch_start + BATCH_SIZE,
        NUM_TEST_SCENES
    )

    batch_rdm = []
    batch_metadata = []

    print(
        f"{batch_start:05d} - {batch_end-1:05d}",
        end=" | "
    )

    for scene_id in range(batch_start, batch_end):

        # ----------------------------------------------------
        # One target, exoclutter
        # ----------------------------------------------------

        rdm, targets = generate_scene(
            num_targets=1,
            range_limits=(10, 60),
            velocity_limits=(-10, 10),
            rcs_limits=(-10, 5),
            noise_power=0,
            seed=scene_id
        )

        # ----------------------------------------------------
        # Complex RDM -> magnitude dB
        # ----------------------------------------------------

        rdm_db = 20 * np.log10(
            np.abs(rdm) + 1e-12
        )

        # ----------------------------------------------------
        # Crop to operational ROI
        # ----------------------------------------------------

        rdm_roi = rdm_db[
            velocity_mask,
            :
        ][:, range_mask]

        batch_rdm.append(
            rdm_roi.astype(np.float32)
        )

        # ----------------------------------------------------
        # Store target metadata
        # ----------------------------------------------------

        batch_metadata.append(targets)

        #del rdm
        #del rdm_db
        #del rdm_roi

    # --------------------------------------------------------
    # Save batch
    # --------------------------------------------------------

    batch_rdm = np.stack(batch_rdm)

    np.save(
        TEST_DIR
        / f"rdm_{batch_start:05d}_{batch_end-1:05d}.npy",
        batch_rdm
    )

    np.save(
        TEST_DIR
        / f"metadata_{batch_start:05d}_{batch_end-1:05d}.npy",
        np.array(
            batch_metadata,
            dtype=object
        ),
        allow_pickle=True
    )

    print(f"shape={batch_rdm.shape}")

    del batch_rdm
    del batch_metadata

print("\n" + "=" * 70)
print("SINGLE-TARGET EXOCLUTTER TEST DATASET COMPLETE")
print("=" * 70)

print("Scenes :", NUM_TEST_SCENES)
print("Location:", TEST_DIR)
