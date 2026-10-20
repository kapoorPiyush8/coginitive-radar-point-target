# ============================================================
# GPU H1 generator: existing target + thermal noise
#
# H1:
#       y(t) = x(t) + n(t)
#
# Uses the EXISTING target metadata from the previous 20k
# zero-noise dataset.
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


def generate_h1_from_metadata_gpu(
    target_meta,
    noise_power,
    seed
):
    """
    Generate one H1 scene using an existing target.

    H1:
        y(t) = x(t) + n(t)

    Target parameters come directly from the previous
    single-target dataset metadata.
    """

    # --------------------------------------------------------
    # Target parameters
    # --------------------------------------------------------

    R0 = float(target_meta["range_m"])
    v = float(target_meta["velocity_mps"])
    rcs_dbsm = float(target_meta["rcs_dbsm"])
    phase = float(target_meta["phase_rad"])

    # --------------------------------------------------------
    # GPU tensors
    # --------------------------------------------------------

    t_fast_t = torch.as_tensor(
        t_fast,
        dtype=torch.float32,
        device=device
    )

    t_slow_t = torch.as_tensor(
        t_slow,
        dtype=torch.float32,
        device=device
    )

    # --------------------------------------------------------
    # Target trajectory
    # --------------------------------------------------------

    R = R0 + v * t_slow_t

    tau = 2.0 * R / c

    fb = Slope * tau

    fd = 2.0 * v / lambda_c

    # --------------------------------------------------------
    # Received target amplitude
    # --------------------------------------------------------

    sigma_m2 = 10.0 ** (rcs_dbsm / 10.0)

    Pr = (
        Ptx
        * Gtx
        * Grx
        * lambda_c**2
        * sigma_m2
        / (
            (4.0 * np.pi)**3
            * R**4
            * L
        )
    )

    A = torch.sqrt(Pr)

    # --------------------------------------------------------
    # Target beat signal
    # --------------------------------------------------------

    phase_fast = (
        2.0
        * np.pi
        * (
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

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    noise = (
        torch.randn(
            (N_SLOW, N_FAST),
            dtype=torch.float32,
            device=device,
            generator=generator
        )
        +
        1j * torch.randn(
            (N_SLOW, N_FAST),
            dtype=torch.float32,
            device=device,
            generator=generator
        )
    ) / np.sqrt(2.0)

    noise *= np.sqrt(noise_power)

    # --------------------------------------------------------
    # H1
    # --------------------------------------------------------

    radar_cube = beat + noise

    # --------------------------------------------------------
    # Range processing
    # --------------------------------------------------------

    range_fft = torch.fft.fft(
        radar_cube,
        dim=1
    )

    range_fft = range_fft[:, :N_FAST // 2]

    # Same stationary mean removal
    range_fft = (
        range_fft
        - torch.mean(
            range_fft,
            dim=0,
            keepdim=True
        )
    )

    # --------------------------------------------------------
    # Doppler processing
    # --------------------------------------------------------

    rdm = torch.fft.fftshift(
        torch.fft.fft(
            range_fft,
            dim=0
        ),
        dim=0
    )

    # --------------------------------------------------------
    # Native ROI
    # --------------------------------------------------------

    rdm_roi = rdm[
        velocity_mask_t,
        :
    ][:, range_mask_t]

    # --------------------------------------------------------
    # Theoretical raw-sample SNR
    # --------------------------------------------------------

    target_power = torch.mean(
        torch.abs(beat)**2
    ).item()

    theoretical_snr_db = 10.0 * np.log10(
        target_power / noise_power
    )

    return (
        rdm_roi,
        theoretical_snr_db,
        target_power
    )
