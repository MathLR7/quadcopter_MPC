import numpy as np
import matplotlib.pyplot as plt

from controllers.mpc_controller import QuadcopterMPC
from simulation.numerical_plant import TripleIntegratorPlant, VerticalPlant

from configs.parameters import DT
from configs.targets import DEFAULT_TARGET


def run_numerical_sim():
    mpc = QuadcopterMPC()
    plant_x = TripleIntegratorPlant()
    plant_y = TripleIntegratorPlant()
    plant_z = VerticalPlant()

    x0 = np.array([
        [0.5,0,0],
        [0,0,0],
        [0,0,0]
    ])

    sim_steps = 360

    state_x = x0[0].copy()
    state_y = x0[1].copy()
    state_z = x0[2].copy()

    x_hist = np.zeros((3, sim_steps+1))
    y_hist = np.zeros((3, sim_steps+1))
    z_hist = np.zeros((3, sim_steps+1))

    j_hist = np.zeros((3, sim_steps))

    x_hist[:,0] = state_x
    y_hist[:,0] = state_y
    z_hist[:,0] = state_z



    for k in range(sim_steps):

        print(f"Step {k}")
        current_state = np.array([
            state_x,
            state_y,
            state_z
        ])

        acc, jerk = mpc.solve(
            current_state,
            DEFAULT_TARGET
        )

        print("acc:", acc)
        print("jerk:", jerk)

        if jerk is None:
            print("Solver failed")
            print(current_state)
            break

        j_hist[:,k] = jerk

        state_x = plant_x.step(
            state_x,
            jerk[0]
        )

        state_y = plant_y.step(
            state_y,
            jerk[1]
        )

        state_z = plant_z.step(
            state_z,
            jerk[2]
        )

        x_hist[:,k+1] = state_x
        y_hist[:,k+1] = state_y
        z_hist[:,k+1] = state_z

    print("FINAL")
    print("x:", x_hist[:, -1])
    print("y:", y_hist[:, -1])
    print("z:", z_hist[:, -1])

    return x_hist,y_hist,z_hist,j_hist

def plot_results(xh, yh, zh, jh):
    t = np.arange(xh.shape[1]) * DT

    # Position
    plt.figure()
    plt.plot(t, xh[0], label="x")
    plt.plot(t, yh[0], label="y")
    plt.plot(t, zh[0], label="z")
    plt.title("Position")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.grid()
    plt.legend()
    plt.show()

    # Velocity
    plt.figure()
    plt.plot(t, xh[1], label="vx")
    plt.plot(t, yh[1], label="vy")
    plt.plot(t, zh[1], label="vz")
    plt.title("Velocity")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.grid()
    plt.legend()
    plt.show()

    # Acceleration
    plt.figure()
    plt.plot(t, xh[2], label="ax")
    plt.plot(t, yh[2], label="ay")
    plt.plot(t, zh[2], label="az")
    plt.title("Acceleration")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s²]")
    plt.grid()
    plt.legend()
    plt.show()

    # Jerk command
    tj = np.arange(jh.shape[1]) * DT
    plt.figure()
    plt.plot(tj, jh[0], label="jx")
    plt.plot(tj, jh[1], label="jy")
    plt.plot(tj, jh[2], label="jz")
    plt.title("Jerk Command")
    plt.xlabel("Time [s]")
    plt.ylabel("Jerk [m/s³]")
    plt.grid()
    plt.legend()
    plt.show()