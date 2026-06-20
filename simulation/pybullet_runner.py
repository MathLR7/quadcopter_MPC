import numpy as np
import time

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from controllers.mpc_controller import QuadcopterMPC
from simulation.pybullet_sim import PyBulletSim
from simulation.simulated_cf import SimulatedCrazyflie

from configs.parameters import DT, G, MASS, OMEGA_MAX
from configs.targets import DEFAULT_TARGET

def run_pybullet_sim():

    env = HoverAviary(
        drone_model=DroneModel.CF2X,
        physics=Physics.PYB,
        gui=True
    )
    
    sim_core = PyBulletSim(env)

    env.reset()
    initial_state = env._getDroneStateVector(0)
    z_hover = initial_state[2]

    cf = SimulatedCrazyflie(sim_core)
    mpc = QuadcopterMPC()

    last_command = {
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "thrust": MASS*G
    }

    
    # target = np.array([
    #     [initial_state[0],0,0],
    #     [initial_state[1],0,0],
    #     [z_hover,0,0]
    # ])
    target = DEFAULT_TARGET

    while True:

        state = cf.sim.get_state()
        acc_plan, jerk_plan = mpc.solve(
            np.array([
                state["x"],
                state["y"],
                state["z"]
            ]),
            target
        )

        if acc_plan is not None:
            desired_acc = acc_plan

            f_acc = np.linalg.norm(desired_acc + np.array([0,0,G]))
            f_acc = max(f_acc, 1e-3)

            roll_rate = -jerk_plan[1]/f_acc
            pitch_rate = jerk_plan[0]/f_acc
            
            roll_rate = np.clip(roll_rate, -OMEGA_MAX, OMEGA_MAX)
            pitch_rate = np.clip(pitch_rate, -OMEGA_MAX, OMEGA_MAX)

            roll_cmd = np.degrees(roll_rate)
            pitch_cmd = np.degrees(pitch_rate)

            thrust_newtons = MASS * f_acc

            last_command["roll"] = roll_cmd
            last_command["pitch"] = pitch_cmd
            last_command["yaw"] = 0
            last_command["thrust"] = thrust_newtons

            print(
                "pos: {:.2f} {:.2f} {:.2f} |".format(state["x"], state["y"], state["z"]),
                "vel z: {:.2f} |".format(state["vz"]),
                "thrust (N): {:.3f}".format(thrust_newtons)
            )

            cf.commander.send_rate_setpoint(
                roll_cmd,
                pitch_cmd,
                0,
                thrust_newtons
            )

        else:

            # cf.commander.send_rate_setpoint(
            #     last_command["roll"],
            #     last_command["pitch"],
            #     last_command["yaw"],
            #     last_command["thrust"]
            # )

            print("Fallback activated, MPC failed")
            cf.commander.send_rate_setpoint(
                0.0, 
                0.0, 
                0.0, 
                MASS * G 
            )

        cf.step()

        time.sleep(DT)