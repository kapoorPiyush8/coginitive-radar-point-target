def generate_noise_scene(noise_power, seed=None):
    """
    Generate one targetless H0 radar scene.

    H0:
        y(t) = n(t)

    Returns the same native ROI representation used by
    the existing radar datasets:

        velocity bins × range bins
        = (69, 1334)
    """

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # H0: thermal noise only
    # --------------------------------------------------------
    noise = (
        rng.standard_normal((N_slow, N_fast))
        + 1j * rng.standard_normal((N_slow, N_fast))
    ) / np.sqrt(2)

    noise *= np.sqrt(noise_power)

    # --------------------------------------------------------
    # Range processing
    # --------------------------------------------------------
    range_fft = np.fft.fft(noise, axis=1)

    # Positive-frequency half
    range_fft = range_fft[:, :N_fast // 2]

    # Same stationary mean removal
    range_fft -= np.mean(range_fft, axis=0)

    # --------------------------------------------------------
    # Doppler processing
    # --------------------------------------------------------
    rdm_complex = np.fft.fftshift(
        np.fft.fft(range_fft, axis=0),
        axes=0
    )

    # --------------------------------------------------------
    # Physical ROI
    # --------------------------------------------------------
    range_mask = (
        (range_axis >= 10.0) &
        (range_axis <= 60.0)
    )

    velocity_mask = (
        (vel_axis >= -10.0) &
        (vel_axis <= 10.0)
    )

    rdm_roi = rdm_complex[
        velocity_mask,
        :
    ][:, range_mask]

    return rdm_roi
