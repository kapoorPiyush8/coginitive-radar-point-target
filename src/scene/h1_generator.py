from pathlib import Path
import json
import numpy as np
import torch
import math
from google.colab import drive

# ============================================================
# 1. ROBUST RUNTIME RECOVERY BOOTSTRAP
# ============================================================

# Mount Google Drive securely
drive.mount("/content/drive", force_remount=True)

# Define directories
base_dir = Path("/content/drive/MyDrive/radar_project")
state_dir = base_dir / "experiment_state"
state_arrays_path = state_dir / "experiment_arrays.npz"

out_dir = state_dir / "independent_h1_test"
out_dir.mkdir(parents=True, exist_ok=True)

# Define/Load physical parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
c = 3e8
fc = 10e9
lambda_c = c / fc
Ptx = 1000.0
Gtx = 10**(11.6 / 10)
Grx = 10**(11.6 / 10)
L = 10**(2.5 / 10)
B = 4e9
Tchirp = 100e-6
Slope = B / Tchirp
PRI = 200e-6

if state_arrays_path.exists():
    with np.load(state_arrays_path, allow_pickle=True) as state_data:
        t_fast = state_data["t_fast"]
        t_slow = state_data["t_slow"]
        range_axis = state_data["range_axis"]
        vel_axis = state_data["vel_axis"]
        range_mask = state_data["range_mask"].astype(bool)
        velocity_mask = state_data["velocity_mask"].astype(bool)
        range_mask_t = torch.tensor(state_data["range_mask_t"], dtype=torch.bool, device=device)
        velocity_mask_t = torch.tensor(state_data["velocity_mask_t"], dtype=torch.bool, device=device)
        N_FAST = len(t_fast)
        N_SLOW = len(t_slow)
else:
    N_FAST = 6400
    N_SLOW = 256
    t_fast = np.arange(N_FAST) / 64e6
    t_slow = np.arange(N_SLOW) * PRI
    range_axis = np.linspace(10.0125, 60.0, 1334)  # Placeholder
    vel_axis = np.linspace(-9.9609375, 9.9609375, 69)  # Placeholder
    range_mask = np.ones(len(range_axis), dtype=bool)
    velocity_mask = np.ones(len(vel_axis), dtype=bool)
    range_mask_t = torch.ones(3200, dtype=torch.bool, device=device) # Placeholder
    velocity_mask_t = torch.ones(256, dtype=torch.bool, device=device) # Placeholder

# Restore missing noise profiles
noise_offsets_db = np.array([-6, -3, 0, 3, 6], dtype=np.float32)
noise_power_ref = 6.578515e-09
noise_powers = noise_power_ref * 10**(noise_offsets_db / 10)

# ============================================================
# 2. SIMULATOR GENERATION FUNCTION
# ============================================================

def generate_h1_from_metadata_gpu(target_meta, noise_power, seed):
    R0 = float(target_meta["range_m"])
    v = float(target_meta["velocity_mps"])
    rcs_dbsm = float(target_meta["rcs_dbsm"])
    phase = float(target_meta["phase_rad"])

    t_fast_t = torch.as_tensor(t_fast, dtype=torch.float32, device=device)
    t_slow_t = torch.as_tensor(t_slow, dtype=torch.float32, device=device)

    R = R0 + v * t_slow_t
    tau = 2.0 * R / c
    fb = Slope * tau
    fd = 2.0 * v / lambda_c

    sigma_m2 = 10.0 ** (rcs_dbsm / 10.0)
    Pr = (Ptx * Gtx * Grx * lambda_c**2 * sigma_m2) / ((4.0 * np.pi)**3 * R**4 * L)
    A = torch.sqrt(Pr)

    phase_fast = 2.0 * np.pi * (fb[:, None] * t_fast_t[None, :] + fd * t_slow_t[:, None]) + phase
    beat = A[:, None] * torch.exp(1j * phase_fast)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    noise = (torch.randn((N_SLOW, N_FAST), dtype=torch.float32, device=device, generator=generator) +
             1j * torch.randn((N_SLOW, N_FAST), dtype=torch.float32, device=device, generator=generator)) / math.sqrt(2.0)
    noise *= math.sqrt(noise_power)

    radar_cube = beat + noise

    range_fft = torch.fft.fft(radar_cube, dim=1)[:, :N_FAST // 2]
    range_fft = range_fft - torch.mean(range_fft, dim=0, keepdim=True)
    rdm = torch.fft.fftshift(torch.fft.fft(range_fft, dim=0), dim=0)
    rdm_roi = rdm[velocity_mask_t, :][:, range_mask_t]

    target_power = torch.mean(torch.abs(beat)**2).item()
    theoretical_snr_db = 10.0 * np.log10(target_power / noise_power)

    return rdm_roi, theoretical_snr_db, target_power

# ============================================================
# 3. EXPERIMENT MAIN GENERATION PASS
# ============================================================

N_PER_PROFILE = 4000
BATCH_SIZE = 50
R_MIN, R_MAX = 10.0, 60.0
V_MIN, V_MAX = -9.5, 9.5
RCS_MIN_DB, RCS_MAX_DB = -20.0, 10.0

rng = np.random.default_rng(20261009)

# Save experiment configuration before generation.
config = {
    "seed": 20261009,
    "n_per_noise_profile": N_PER_PROFILE,
    "batch_size": BATCH_SIZE,
    "noise_offsets_db": np.asarray(noise_offsets_db).tolist(),
    "noise_powers": np.asarray(noise_powers).tolist(),
    "range_bounds_m": [R_MIN, R_MAX],
    "velocity_bounds_mps": [V_MIN, V_MAX],
    "rcs_bounds_dbsm": [RCS_MIN_DB, RCS_MAX_DB],
    "generator": "generate_h1_from_metadata_gpu",
    "description": "Independent H1 diagnostic test; fresh targets and noise"
}
(out_dir / "config.json").write_text(json.dumps(config, indent=2))

total_generated = 0

for profile_idx, (offset_db, noise_power) in enumerate(
    zip(noise_offsets_db, noise_powers)
):
    profile_start = profile_idx * N_PER_PROFILE

    for local_start in range(0, N_PER_PROFILE, BATCH_SIZE):
        local_stop = min(local_start + BATCH_SIZE, N_PER_PROFILE)
        global_start = profile_start + local_start
        global_stop = profile_start + local_stop
        stem = f"h1_test_{global_start:05d}_{global_stop-1:05d}"

        rdm_path = out_dir / f"{stem}_rdm.npy"
        meta_path = out_dir / f"{stem}_metadata.npy"
        done_path = out_dir / f"{stem}.done"

        if done_path.exists() and rdm_path.exists() and meta_path.exists():
            continue

        batch_rdms = []
        batch_meta = []

        for scene_id in range(global_start, global_stop):
            target_meta = {
                "id": scene_id,
                "range_m": float(rng.uniform(R_MIN, R_MAX)),
                "velocity_mps": float(rng.uniform(V_MIN, V_MAX)),
                "rcs_dbsm": float(rng.uniform(RCS_MIN_DB, RCS_MAX_DB)),
                "phase_rad": float(rng.uniform(0.0, 2.0 * np.pi)),
            }

            rdm_roi, snr_db, target_power = generate_h1_from_metadata_gpu(
                target_meta=target_meta,
                noise_power=float(noise_power),
                seed=20261009 + scene_id
            )

            # Same dB preprocessing used for existing RDM datasets
            rdm_db = 20.0 * torch.log10(
                torch.abs(rdm_roi) / (torch.max(torch.abs(rdm_roi)) + 1e-12)
                + 1e-12
            )
            rdm_np = rdm_db.detach().cpu().numpy().astype(np.float32)

            # Map physical target position to the native ROI axes, then to 256x256 network grid
            range_axis_roi = range_axis[range_mask]
            vel_axis_roi = vel_axis[velocity_mask]

            ri = int(np.argmin(np.abs(range_axis_roi - target_meta["range_m"])))
            vi = int(np.argmin(np.abs(vel_axis_roi - target_meta["velocity_mps"])))

            y = int(np.rint(vi * 255 / max(len(vel_axis_roi) - 1, 1)))
            x = int(np.rint(ri * 255 / max(len(range_axis_roi) - 1, 1)))

            target_meta.update({
                "hypothesis": "H1",
                "noise_offset_db": float(offset_db),
                "noise_power": float(noise_power),
                "snr_db_raw": float(snr_db),
                "target_power": float(target_power),
                "range_bin_roi": ri,
                "velocity_bin_roi": vi,
                "label_y": y,
                "label_x": x,
            })

            batch_rdms.append(rdm_np)
            batch_meta.append(target_meta)

        tmp_rdm = out_dir / f"{stem}_rdm.tmp.npy"
        tmp_meta = out_dir / f"{stem}_metadata.tmp.npy"

        np.save(tmp_rdm, np.stack(batch_rdms))
        np.save(tmp_meta, np.array(batch_meta, dtype=object))

        tmp_rdm.replace(rdm_path)
        tmp_meta.replace(meta_path)
        done_path.write_text("complete\n")

        total_generated += len(batch_meta)

        if total_generated % 500 == 0:
            print(
                f"Generated this run: {total_generated}; "
                f"latest scene index: {global_stop - 1}"
            )

print("Generation pass finished.")
print("Output directory:", out_dir)
print("Completion markers:", len(list(out_dir.glob("*.done"))))
