def generate_scene(num_targets=5,range_limits=(10, 60),velocity_limits=(-10, 10),rcs_limits=(-10, 5),noise_power=0.0,seed=None):

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()


    # --------------------------------------------------------
    # Complex FMCW data cube
    # --------------------------------------------------------

    radar_cube = np.zeros((N_slow, N_fast),dtype=np.complex128)


    # --------------------------------------------------------
    # Ground-truth target information
    # --------------------------------------------------------

    targets = []


    # --------------------------------------------------------
    # Generate targets
    # --------------------------------------------------------

    for target_id in range(num_targets):

        R0 = rng.uniform(range_limits[0],range_limits[1])
        v = rng.uniform(velocity_limits[0],velocity_limits[1])
        rcs_dbsm = rng.uniform(rcs_limits[0],rcs_limits[1])

        phase = rng.uniform(0,2 * np.pi)
        target = PointTarget(range=R0,velocity=v,rcs_dbsm=rcs_dbsm,phase=phase)

        targets.append(target)
