import numpy as np
import matplotlib.pyplot as plt

from controllers.mpc_controller import QuadcopterMPC
from configs.parameters import DT
from configs.targets import DEFAULT_TARGET

def run_numerical_sim():

    mpc = QuadcopterMPC()

    x0 = np.array([
        [0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])

    target = DEFAULT_TARGET


    xh, yh, zh, jh = mpc.simulate(
        x0,
        target,
        720
    )


    t = np.arange(xh.shape[1]) * DT


    plt.figure()

    plt.plot(t, xh[0], label="x")
    plt.plot(t, yh[0], label="y")
    plt.plot(t, zh[0], label="z")

    plt.title("Position")
    plt.grid()
    plt.legend()
    plt.show()


    plt.figure()

    plt.plot(t, xh[1], label="vx")
    plt.plot(t, yh[1], label="vy")
    plt.plot(t, zh[1], label="vz")

    plt.title("Velocity")
    plt.grid()
    plt.legend()
    plt.show()


    plt.figure()

    plt.plot(t, xh[2], label="ax")
    plt.plot(t, yh[2], label="ay")
    plt.plot(t, zh[2], label="az")

    plt.title("Acceleration")
    plt.grid()
    plt.legend()
    plt.show()