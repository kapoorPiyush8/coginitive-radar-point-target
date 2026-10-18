"""
FMCW waveform generation module for 5 point targets 
"""

import numpy as np

c = 3e8

fc = 10e9
lambda_c = c / fc

G_tx = 10**(11.6 / 10)
G_rx = 10**(11.6 / 10)

P_tx = 1e3

L_db = 2.5
L = 10**(L_db / 10)

B = 4e9
T_chirp = 100e-6

PRI = 200e-6
PRF = 1 / PRI

Slope = B / T_chirp

fs = 64e6
N_fast = int(T_chirp * fs)

N_slow = 256

t_fast = np.arange(N_fast) / fs
t_slow = np.arange(N_slow) * PRI

fd_max = PRF / 2
v_max = lambda_c * PRF / 4

delta_R = c / (2 * B)

delta_fd = PRF / N_slow
delta_v = lambda_c * delta_fd / 2
