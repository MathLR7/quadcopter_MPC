import numpy as np

# Drone parameters
MASS = 0.0397
G = 9.81
F_MAX = 14.82
F_MIN = 4.0
OMEGA_MAX = 10.0

# MPC parameters
DT = 0.02
T_HORIZON = 1.5
N_STEPS = int(T_HORIZON / DT)

# Jerk limit
J_MAX = (1 / np.sqrt(3)) * F_MIN * OMEGA_MAX