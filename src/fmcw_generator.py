for target_id in range(num_targets):

        R0 = rng.uniform(range_limits[0],range_limits[1])
        v = rng.uniform(velocity_limits[0],velocity_limits[1])
        rcs_dbsm = rng.uniform(rcs_limits[0],rcs_limits[1])

        phase = rng.uniform(0,2 * np.pi)
        target = PointTarget(range=R0,velocity=v,rcs_dbsm=rcs_dbsm,phase=phase)

        targets.append(target)
        # ----------------------------------------------------
        # Doppler frequency
        # ----------------------------------------------------

        fd = 2 * v / lambda_c


        # ----------------------------------------------------
        # Generate all slow-time samples at once
        # ----------------------------------------------------

        R = (R0 + v * t_slow)
        tau = 2 * R / c
        fb = Slope * tau
        # ----------------------------------------------------
        # Instantaneous received amplitude
        # ----------------------------------------------------

        A = received_amplitude(R=R,rcs_dbsm=rcs_dbsm,P_tx=P_tx,G_tx=G_tx,G_rx=G_rx,lambda_c=lambda_c,L=L)
        # ----------------------------------------------------
        # FMCW target return
        #
        # Outer product gives:
        #
        # slow time × fast time
        # ----------------------------------------------------

        beat = (A[:, None]*np.exp(1j * 2 * np.pi *(fb[:, None] * t_fast[None, :]+fd * t_slow[:, None]+phase)))


        # ----------------------------------------------------
        # Coherent addition of this target
        # ----------------------------------------------------

        radar_cube += beat


    # ========================================================
    # 8. THERMAL NOISE
    # ========================================================

    if noise_power > 0:

        noise = (rng.standard_normal(radar_cube.shape)+1j * rng.standard_normal(radar_cube.shape)) / np.sqrt(2)

        noise *= np.sqrt(noise_power)

        radar_cube += noise


    # ========================================================
    # 9. RANGE FFT
    # ========================================================

    range_fft = np.fft.fft(radar_cube,axis=1)

    range_fft = range_fft[:, :N_fast // 2]


    # Remove stationary mean

    range_fft -= np.mean(range_fft,axis=0)


    # ========================================================
    # 10. DOPPLER FFT
    # ========================================================

    rdm = np.fft.fftshift(np.fft.fft(range_fft,axis=0),axes=0)


    # ========================================================
    # 11. GROUND TRUTH
    # ========================================================

    target_metadata = []

    for target_id, target in enumerate(targets):

        fd = 2 * target.velocity / lambda_c


        # Nearest physical RDM bins

        range_bin = np.argmin(np.abs(range_axis - target.range))

        velocity_bin = np.argmin(np.abs(vel_axis - target.velocity))


        target_metadata.append({

            "id": target_id,

            "range_m": target.range,

            "velocity_mps": target.velocity,

            "doppler_hz": fd,

            "rcs_dbsm": target.rcs_dbsm,

            "phase_rad": target.phase,

            "range_bin": range_bin,

            "velocity_bin": velocity_bin
        })


    return rdm, target_metadata
