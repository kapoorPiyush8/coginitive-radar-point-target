def generate_noise_scene(noise_power, seed=None):
    rng = np.random.default_rng(seed)

    # H0: thermal noise only
    noise = (
        rng.standard_normal((N_slow, N_fast))
        + 1j * rng.standard_normal((N_slow, N_fast))
    ) / np.sqrt(2)

    noise *= np.sqrt(noise_power)

    # Range processing
    range_fft = np.fft.fft(noise, axis=1)
    range_fft = range_fft[:, :N_fast // 2]

    # Same stationary-mean removal as the target scenes
    range_fft -= np.mean(range_fft, axis=0)

    # Doppler processing
    rdm = np.fft.fftshift(
        np.fft.fft(range_fft, axis=0),
        axes=0
    )

    return rdm

# ============================================================
# S1 reference thermal-noise calibration
# ============================================================

R_ref = 30.0                 # m
rcs_ref_m2 = 1.0             # m^2
snr_ref_db = 10.0            # dB
c=3e8; fc=10e9; lam=c/fc
Gtx=10**(11.6/10); Ptx=1000; L=10**(2.5/10)
Grx=10**(11.6/10)
R=30; sigma=1
Ptx=1000

# Radar equation
Pr_ref = (
    Ptx
    * Gtx
    * Grx
    * lambda_c**2
    * rcs_ref_m2
    / ((4 * np.pi)**3 * R_ref**4 * L)
)

# Reference complex-sample noise power
noise_power_ref = Pr_ref / (10**(snr_ref_db / 10))

print(f"Reference range       : {R_ref:.1f} m")
print(f"Reference RCS         : {rcs_ref_m2:.2f} m^2")
print(f"Reference received P  : {Pr_ref:.6e}")
print(f"Reference received P  : {10*np.log10(Pr_ref):.3f} dB")
print(f"Reference SNR         : {snr_ref_db:.1f} dB")
print(f"Reference noise power : {noise_power_ref:.6e}")

noise_offsets_db = np.array([-6, -3, 0, 3, 6], dtype=np.float32)

noise_powers = noise_power_ref * 10**(noise_offsets_db / 10)

print("\nS1 noise profiles:")
for offset, pn in zip(noise_offsets_db, noise_powers):
    print(f"{offset:+.0f} dB : noise_power = {pn:.6e}")
