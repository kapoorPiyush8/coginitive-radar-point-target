# ============================================================
# GPU-based H0 / S1 scene generator
# Confirm the preprocessing compatibility later, on the office machine. This generator saves native \(69 \times 1334\) RDMs, while the CNN uses \(256 \times 256\) inputs. Verify that the existing training preprocessing resizes and normalizes these RDMs consistently.
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants as torch tensors/scalars
N_SLOW = N_slow
N_FAST = N_fast

# Masks corresponding to the existing 69 x 1334 native ROI
range_mask_t = torch.tensor(
    (range_axis >= 10.0) & (range_axis <= 60.0),
    dtype=torch.bool,
    device=device
)

velocity_mask_t = torch.tensor(
    (vel_axis >= -10.0) & (vel_axis <= 10.0),
    dtype=torch.bool,
    device=device
)

print("Device:", device)
print("Velocity ROI bins:", velocity_mask_t.sum().item())
print("Range ROI bins:", range_mask_t.sum().item())

def generate_noise_scene_gpu(noise_power, seed=None):
    """
    GPU implementation of H0:

        y(t) = n(t)

    Returns:
        complex RDM ROI of shape (69, 1334)
    """

    if seed is not None:
        torch.manual_seed(int(seed))

    # --------------------------------------------------------
    # H0: complex circular Gaussian thermal noise
    # --------------------------------------------------------

    noise = (
        torch.randn(
            (N_SLOW, N_FAST),
            dtype=torch.float32,
            device=device
        )
        +
        1j * torch.randn(
            (N_SLOW, N_FAST),
            dtype=torch.float32,
            device=device
        )
    ) / np.sqrt(2)

    noise *= np.sqrt(noise_power)

    # --------------------------------------------------------
    # Range FFT
    # --------------------------------------------------------

    range_fft = torch.fft.fft(noise, dim=1)

    range_fft = range_fft[:, :N_FAST // 2]

    # Same stationary mean removal
    range_fft = range_fft - torch.mean(
        range_fft,
        dim=0,
        keepdim=True
    )

    # --------------------------------------------------------
    # Doppler FFT + shift
    # --------------------------------------------------------

    rdm = torch.fft.fftshift(
        torch.fft.fft(range_fft, dim=0),
        dim=0
    )

    # --------------------------------------------------------
    # Physical ROI
    # --------------------------------------------------------

    rdm_roi = rdm[
        velocity_mask_t
    ][:, range_mask_t]

    return rdm_roi

# ============================================================
# Generate S1 on GPU
# ============================================================


S1_DIR = Path(
    "/content/drive/MyDrive/radar_project/radar_dataset/S1_targetless_thermal_noise"
)

S1_DIR.mkdir(parents=True, exist_ok=True)

SCENES_PER_PROFILE = 4000
BATCH_SIZE = 100

scene_id = 0

for profile_idx, (offset_db, noise_power) in enumerate(
    zip(noise_offsets_db, noise_powers)
):

    print(
        f"\nProfile {profile_idx + 1}/5 | "
        f"noise offset = {offset_db:+.0f} dB | "
        f"noise power = {noise_power:.6e}"
    )

    profile_start = scene_id
    profile_end = scene_id + SCENES_PER_PROFILE

    while scene_id < profile_end:

        batch_end = min(
            scene_id + BATCH_SIZE,
            profile_end
        )

        batch_size = batch_end - scene_id

        # ----------------------------------------------------
        # Generate complex Gaussian noise directly on GPU
        # ----------------------------------------------------

        noise = (
            torch.randn(
                (batch_size, N_SLOW, N_FAST),
                dtype=torch.float32,
                device=device
            )
            +
            1j * torch.randn(
                (batch_size, N_SLOW, N_FAST),
                dtype=torch.float32,
                device=device
            )
        ) / np.sqrt(2)

        noise *= np.sqrt(noise_power)

        # ----------------------------------------------------
        # Range FFT
        # ----------------------------------------------------

        range_fft = torch.fft.fft(
            noise,
            dim=2
        )

        range_fft = range_fft[:, :, :N_FAST // 2]

        # ----------------------------------------------------
        # Stationary mean removal
        # ----------------------------------------------------

        range_fft = range_fft - torch.mean(
            range_fft,
            dim=1,
            keepdim=True
        )

        # ----------------------------------------------------
        # Doppler FFT
        # ----------------------------------------------------

        rdm = torch.fft.fftshift(
            torch.fft.fft(
                range_fft,
                dim=1
            ),
            dim=1
        )

        # ----------------------------------------------------
        # Extract native ROI
        # ----------------------------------------------------

        rdm_roi = rdm[
            :,
            velocity_mask_t,
            :
        ][:, :, range_mask_t]

        # ----------------------------------------------------
        # Convert to dB
        # ----------------------------------------------------

        rdm_db = 20 * torch.log10(
            torch.abs(rdm_roi) + 1e-12
        )

        # GPU -> CPU
        rdm_batch = (
            rdm_db
            .float()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Zero labels: H0 contains no targets
        # ----------------------------------------------------

        labels_batch = np.zeros(
            (batch_size, 256, 256),
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata_batch = []

        for i in range(batch_size):

            current_id = scene_id + i

            metadata_batch.append({
                "id": int(current_id),
                "hypothesis": "H0",
                "noise_profile": int(profile_idx),
                "noise_offset_db": float(offset_db),
                "noise_power": float(noise_power),
            })

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        start_id = scene_id
        end_id = batch_end - 1

        np.save(
            S1_DIR / f"rdm_{start_id:05d}_{end_id:05d}.npy",
            rdm_batch
        )

        np.save(
            S1_DIR / f"labels_{start_id:05d}_{end_id:05d}.npy",
            labels_batch
        )

        np.save(
            S1_DIR / f"metadata_{start_id:05d}_{end_id:05d}.npy",
            np.array(
                metadata_batch,
                dtype=object
            )
        )

        scene_id = batch_end

        # Release GPU tensors before next batch
        del noise, range_fft, rdm, rdm_roi, rdm_db

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if scene_id % 500 == 0:
            print(
                f"  Generated {scene_id}/{5 * SCENES_PER_PROFILE}"
            )

    print(
        f"Generated scenes "
        f"{profile_start}–{profile_end - 1}"
    )

print("\nS1 generation complete.")
print("Total scenes:", scene_id)
print("Location:", S1_DIR)
