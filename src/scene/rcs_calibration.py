# ============================================================
# H1 SNR ↔ RCS calibration
# ============================================================

# Use the 0 dB S1 profile as the reference thermal
# noise condition for defining the H1 target SNR.
Pn_ref = noise_power_ref

# Candidate operational SNR range
snr_grid_db = np.array([
    -10, -5, 0, 5, 10, 15, 20, 25, 30
], dtype=float)

# Representative target ranges
range_grid_m = np.array([
    10, 20, 30, 40, 50, 60
], dtype=float)

def required_rcs_m2(range_m, snr_db, noise_power=Pn_ref):
    """
    RCS required to produce a specified raw-sample SNR:

        SNR = Pr / Pn

    where

        Pr = Pt Gt Gr lambda^2 sigma
             -------------------------
             (4 pi)^3 R^4 L
    """

    target_power = noise_power * 10**(snr_db / 10)

    rcs = (
        target_power
        * (4 * np.pi)**3
        * range_m**4
        * L
        / (
            Ptx
            * Gtx
            * Grx
            * lambda_c**2
        )
    )

    return rcs


print("Required RCS for target SNR")
print("=" * 65)

for R in range_grid_m:

    print(f"\nRange = {R:.0f} m")

    for snr_db in snr_grid_db:

        rcs_m2 = required_rcs_m2(R, snr_db)

        rcs_dbsm = 10 * np.log10(rcs_m2)

        print(
            f"  SNR = {snr_db:+3.0f} dB"
            f" -> RCS = {rcs_m2:.6e} m²"
            f" ({rcs_dbsm:+.2f} dBsm)"
        )
