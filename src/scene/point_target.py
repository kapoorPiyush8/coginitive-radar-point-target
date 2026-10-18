class PointTarget:

    def __init__(self, range, velocity, rcs_dbsm, phase):

        self.range = range
        self.velocity = velocity
        self.rcs_dbsm = rcs_dbsm
        self.phase = phase


def rcs_dbsm_to_linear(rcs_dbsm):

    return 10**(rcs_dbsm / 10)


def received_amplitude(R,rcs_dbsm,P_tx,G_tx,G_rx,lambda_c,L):

    sigma = rcs_dbsm_to_linear(rcs_dbsm)

    P_rx = (P_tx * G_tx * G_rx * lambda_c**2 * sigma / ((4 * np.pi)**3 * L * R**4))

    return np.sqrt(P_rx)


freq_fast = np.fft.fftfreq(N_fast, d=1/fs)[:N_fast // 2]

range_axis = (c * freq_fast/(2 * Slope))

freq_slow = np.fft.fftshift(np.fft.fftfreq(N_slow,d=PRI))

vel_axis = (freq_slow * lambda_c/ 2)
