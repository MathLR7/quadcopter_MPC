"""
Quadcopter Model Predictive Control (MPC) using CasADi.
Developed by Matheus Littig Radinz at RWTH Aachen (Started 06.05.2024).

This implementation uses a Minimum-Jerk formulation. It generates smooth trajectories 
by minimizing control effort (jerk) while enforcing strict terminal state constraints 
"""

from hardware.crazyflie_controller import CrazyflieMPCControl
from simulation.pybullet_runner import run_pybullet_sim
from simulation.numerical_sim import run_numerical_sim


if __name__ == "__main__":
    mode = "numerical" 

    if mode == "numerical": # simplified simulation to see convergences, only tests MPC
        run_numerical_sim()

    elif mode == "pybullet":
        run_pybullet_sim()

    elif mode == "drone":
        URI = 'radio://0/80/2M/E7E7E7E7E7'
        control = CrazyflieMPCControl(URI)
        control.run()
