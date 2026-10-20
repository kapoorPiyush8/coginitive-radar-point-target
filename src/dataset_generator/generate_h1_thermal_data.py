# ============================================================
# 1. PATHS
# ============================================================

OLD_H1_DIR = Path(
    "/content/drive/MyDrive/radar_project/radar_dataset/single_target_exoclutter_test"
)

H1_DIR = Path(
    "/content/drive/MyDrive/radar_project/radar_dataset/"
    "single_target_exoclutter_H1_thermal_noise"
)

H1_DIR.mkdir(parents=True, exist_ok=True)

print("Output:", H1_DIR)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 3. CONSTANT GPU TENSORS
# ============================================================

t_fast_t = torch.as_tensor(
    t_fast, dtype=torch.float32, device=device
)

t_slow_t = torch.as_tensor(
    t_slow, dtype=torch.float32, device=device
)

print("t_fast:", t_fast_t.shape)
print("t_slow:", t_slow_t.shape)


# ============================================================
# 4. NOISE PROFILES
# ============================================================

noise_offsets_db = np.array([-6, -3, 0, 3, 6], dtype=np.float32)

noise_powers = np.array([
    noise_power_ref * 10**(x / 10)
    for x in noise_offsets_db
], dtype=np.float64)

print("\nNoise profiles:")

for db, pn in zip(noise_offsets_db, noise_powers):
    print(f"{db:+.0f} dB : {pn:.9e}")


# ============================================================
# 5. ROBUST METADATA UNWRAPPER
# ============================================================

def unwrap_metadata_item(x):

    # Keep unwrapping single-element numpy containers
    while isinstance(x, np.ndarray) and x.size == 1:
        x = x.item()

    return x


# ============================================================
# 6. LOAD ALL EXISTING TARGET METADATA
# ============================================================

metadata_files = sorted(OLD_H1_DIR.glob("metadata_*.npy"))

print("\nMetadata files found:", len(metadata_files))

all_targets = []

for f in metadata_files:

    arr = np.load(f, allow_pickle=True)

    for x in arr.reshape(-1):
        x = unwrap_metadata_item(x)

        if isinstance(x, dict):
            all_targets.append(x)
        else:
            raise TypeError(
                f"Unexpected metadata type in {f}: {type(x)}"
            )

print("Total target scenes:", len(all_targets))

assert len(all_targets) == 20000, \
    f"Expected 20000 targets, found {len(all_targets)}"


# ============================================================
# 7. GPU H1 GENERATOR
# ============================================================

def generate_h1_scene_gpu(target_meta, noise_power, seed):

    R0 = float(target_meta["range_m"])
    v = float(target_meta["velocity_mps"])
    rcs_dbsm = float(target_meta["rcs_dbsm"])
    phase = float(target_meta["phase_rad"])

    # Target range versus slow time
    R = R0 + v * t_slow_t

    # Round-trip delay
    tau = 2.0 * R / c

    # Beat frequency
    fb = Slope * tau

    # Doppler frequency
    fd = 2.0 * v / lambda_c

    # RCS
    sigma_m2 = 10.0 ** (rcs_dbsm / 10.0)

    # Received target power
    Pr = (
        Ptx * Gtx * Grx * lambda_c**2 * sigma_m2
        / ((4 * np.pi)**3 * R**4 * L)
    )

    # Target amplitude
    A = torch.sqrt(Pr)

    # Target phase
    phase_fast = (
        2.0 * np.pi *
        (
            fb[:, None] * t_fast_t[None, :]
            + fd * t_slow_t[:, None]
        )
        + phase
    )

    beat = (
        A[:, None]
        * torch.exp(1j * phase_fast)
    )

    # --------------------------------------------------------
    # Thermal noise
    # --------------------------------------------------------

    torch.manual_seed(int(seed))

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
    ) / math.sqrt(2.0)

    noise *= math.sqrt(noise_power)

    # H1 = target + noise
    radar_cube = beat + noise

    # --------------------------------------------------------
    # Range FFT
    # --------------------------------------------------------

    range_fft = torch.fft.fft(
        radar_cube,
        dim=1
    )[:, :N_FAST // 2]

    # Remove stationary component
    range_fft = (
        range_fft
        - torch.mean(
            range_fft,
            dim=0,
            keepdim=True
        )
    )

    # --------------------------------------------------------
    # Doppler FFT
    # --------------------------------------------------------

    rdm = torch.fft.fftshift(
        torch.fft.fft(
            range_fft,
            dim=0
        ),
        dim=0
    )

    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    rdm_roi = rdm[
        velocity_mask_t
    ][:, range_mask_t]

    # --------------------------------------------------------
    # Raw-sample target SNR
    # --------------------------------------------------------

    target_power = torch.mean(
        torch.abs(beat)**2
    ).item()

    snr_db = 10.0 * np.log10(
        target_power / noise_power
    )

    return rdm_roi, snr_db


# ============================================================
# 8. GENERATE 20,000 H1 SCENES
# ============================================================

TOTAL = len(all_targets)

BATCH_SIZE = 50

print("\nStarting H1 generation...")
print("Total scenes:", TOTAL)
print("Batch size:", BATCH_SIZE)


for start in range(0, TOTAL, BATCH_SIZE):

    end = min(start + BATCH_SIZE, TOTAL)

    batch_rdms = []
    batch_metadata = []

    for idx in range(start, end):

        target = all_targets[idx]

        # ----------------------------------------------------
        # Assign noise profile
        # ----------------------------------------------------

        profile_idx = min(
            idx // 4000,
            4
        )

        noise_offset_db = float(
            noise_offsets_db[profile_idx]
        )

        noise_power = float(
            noise_powers[profile_idx]
        )

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        rdm_gpu, snr_db = generate_h1_scene_gpu(
            target_meta=target,
            noise_power=noise_power,
            seed=int(target["id"])
        )

        # Convert to dB
        rdm_db = (
            20.0 *
            torch.log10(
                torch.abs(rdm_gpu) + 1e-12
            )
        )

        batch_rdms.append(
            rdm_db.cpu().numpy().astype(np.float32)
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        meta = dict(target)

        meta["hypothesis"] = "H1"
        meta["noise_offset_db"] = noise_offset_db
        meta["noise_power"] = noise_power
        meta["snr_db_raw"] = float(snr_db)

        batch_metadata.append(meta)

        del rdm_gpu
        del rdm_db

    # --------------------------------------------------------
    # Save RDM
    # --------------------------------------------------------

    rdm_batch = np.stack(
        batch_rdms,
        axis=0
    )

    labels_batch = np.zeros(
        (len(batch_metadata), 256, 256),
        dtype=np.float32
    )

    np.save(
        H1_DIR /
        f"rdm_{start:05d}_{end-1:05d}.npy",
        rdm_batch
    )

    np.save(
        H1_DIR /
        f"labels_{start:05d}_{end-1:05d}.npy",
        labels_batch
    )

    np.save(
        H1_DIR /
        f"metadata_{start:05d}_{end-1:05d}.npy",
        np.array(
            batch_metadata,
            dtype=object
        )
    )

    del rdm_batch
    del labels_batch
    del batch_rdms
    del batch_metadata

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"Saved {start:05d}–{end-1:05d} "
        f"({end}/{TOTAL})"
    )


print("\n========================================")
print("H1 DATASET GENERATION COMPLETE")
print("========================================")
print("Scenes:", TOTAL)
print("Output:", H1_DIR)
