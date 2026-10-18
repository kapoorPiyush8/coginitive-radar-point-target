# ============================================================
# S1 — TARGETLESS THERMAL-NOISE CALIBRATION DATASET
#
# IMPORTANT:
# This is an implementation approximation.
#
# The supplied paper specifies that S1 is a targetless
# thermal-noise dataset used for false-alarm threshold
# calibration, but does not specify the complete low-level
# thermal-noise generation procedure.
#
# Therefore this is NOT claimed to be an exact reproduction
# of the authors' S1 generator.
# ============================================================


# ============================================================
# S1 configuration
# ============================================================

S1_SCENES = 1000
S1_BATCH_SIZE = 100

S1_DIR = DATA_DIR / "S1_targetless_thermal_noise"
S1_DIR.mkdir(parents=True, exist_ok=True)

S1_SEED = 2026


# ============================================================
# Reuse radar configuration from our existing simulator
# ============================================================

c = 3e8
fc = 10e9
lam = c / fc

Gtx = 10**(11.6 / 10)
Grx = 10**(11.6 / 10)

Ptx = 1000.0
Ldb = 2.5
L = 10**(Ldb / 10)

B = 4e9
Tchirp = 100e-6

PRI = 200e-6
PRF = 1 / PRI

slope = B / Tchirp

fs = 64e6

Nfast = 6400
Nslow = 256


# ============================================================
# FFT axes
# ============================================================

# Native range FFT
freq_range = np.fft.fftfreq(
    Nfast,
    d=1 / fs
)

range_axis = (
    c * freq_range
    / (2 * slope)
)


# Native Doppler FFT
doppler_freq = np.fft.fftshift(
    np.fft.fftfreq(
        Nslow,
        d=PRI
    )
)

velocity_axis = (
    doppler_freq * lam / 2
)


# ============================================================
# Same ROI used for the existing dataset
# ============================================================

range_mask = (
    (range_axis >= 10.0)
    &
    (range_axis <= 60.0)
)

velocity_mask = (
    (velocity_axis >= -10.0)
    &
    (velocity_axis <= 10.0)
)

range_indices = np.where(range_mask)[0]
velocity_indices = np.where(velocity_mask)[0]

range_axis_roi = range_axis[range_indices]
velocity_axis_roi = velocity_axis[velocity_indices]


print("S1 native ROI:")
print("Velocity bins :", len(velocity_indices))
print("Range bins    :", len(range_indices))

assert len(velocity_indices) == 69
assert len(range_indices) == 1334


# ============================================================
# Thermal noise model
# ============================================================
#
# Approximation:
#
# Complex baseband receiver thermal noise is modeled as
#
#       n ~ CN(0, sigma^2)
#
# with independent Gaussian I/Q components.
#
# Since the exact receiver noise-temperature / noise-figure
# specification used by the paper is not supplied, we use
# normalized complex AWGN here.
#
# The absolute dB level is subsequently calibrated to the
# same RDM dynamic representation used by our dataset.
# ============================================================

rng = np.random.default_rng(S1_SEED)


# ============================================================
# Generate one targetless thermal-noise RDM
# ============================================================

def generate_s1_rdm():

    # Complex thermal receiver noise
    noise = (
        rng.standard_normal(
            (Nslow, Nfast)
        )
        +
        1j * rng.standard_normal(
            (Nslow, Nfast)
        )
    ) / np.sqrt(2.0)


    # Range FFT
    range_fft = np.fft.fft(
        noise,
        axis=1
    )


    # Doppler FFT
    rdm_complex = np.fft.fftshift(
        np.fft.fft(
            range_fft,
            axis=0
        ),
        axes=0
    )


    # Select exactly the same physical ROI
    rdm_roi = rdm_complex[
        velocity_indices,
        :
    ][:, range_indices]


    # Magnitude in dB
    #
    # Same convention used for our stored RDM representation.
    magnitude = np.abs(rdm_roi)

    rdm_db = 20.0 * np.log10(
        magnitude + 1e-12
    )


    return rdm_db.astype(
        np.float32
    )


# ============================================================
# Generate S1
# ============================================================

print()
print("=" * 70)
print("GENERATING S1 TARGETLESS THERMAL-NOISE DATASET")
print("=" * 70)

start_time = time.time()

for batch_start in range(
    0,
    S1_SCENES,
    S1_BATCH_SIZE
):

    batch_end = min(
        batch_start + S1_BATCH_SIZE,
        S1_SCENES
    )

    batch_size = (
        batch_end - batch_start
    )

    batch = np.empty(
        (
            batch_size,
            len(velocity_indices),
            len(range_indices)
        ),
        dtype=np.float32
    )


    for i in range(batch_size):

        batch[i] = generate_s1_rdm()


    filename = (
        f"s1_rdm_"
        f"{batch_start:05d}_"
        f"{batch_end - 1:05d}.npy"
    )

    np.save(
        S1_DIR / filename,
        batch
    )


    print(
        f"{batch_start:04d} - "
        f"{batch_end - 1:04d} | "
        f"shape={batch.shape}"
    )


# ============================================================
# Save S1 metadata
# ============================================================

metadata = {

    "dataset_name":
        "S1_targetless_thermal_noise",

    "num_scenes":
        S1_SCENES,

    "batch_size":
        S1_BATCH_SIZE,

    "native_shape":
        [
            len(velocity_indices),
            len(range_indices)
        ],

    "range_roi_m":
        [
            float(range_axis_roi.min()),
            float(range_axis_roi.max())
        ],

    "velocity_roi_mps":
        [
            float(velocity_axis_roi.min()),
            float(velocity_axis_roi.max())
        ],

    "seed":
        S1_SEED,

    "noise_model":
        "complex circular Gaussian receiver thermal noise",

    "normalization":
        "none",

    "representation":
        "20log10(abs(complex_RDM) + 1e-12)",

    "paper_status":
        (
            "implementation approximation; supplied paper "
            "specifies targetless thermal-noise S1 for threshold "
            "calibration but does not specify complete low-level "
            "thermal-noise generation parameters"
        ),

    "radar_parameters": {
        "fc_hz": fc,
        "bandwidth_hz": B,
        "fs_hz": fs,
        "Nfast": Nfast,
        "Nslow": Nslow,
        "PRI_s": PRI,
        "PRF_hz": PRF,
        "slope_hz_per_s": slope
    }
}


with open(
    S1_DIR / "s1_metadata.json",
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


elapsed = time.time() - start_time


print()
print("=" * 70)
print("S1 GENERATION COMPLETE")
print("=" * 70)

print("Scenes       :", S1_SCENES)
print("Native shape :", (
    len(velocity_indices),
    len(range_indices)
))

print("Location     :", S1_DIR)
print("Elapsed      :", f"{elapsed:.1f} s")
