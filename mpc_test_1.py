"""
Quadcopter Model Predictive Control (MPC) using CasADi.
Developed by Matheus Littig Radinz at RWTH Aachen (Started 06.05.2024).

This implementation uses a Minimum-Jerk formulation. It generates smooth trajectories 
by minimizing control effort (jerk) while enforcing strict terminal state constraints 
"""

import numpy as np
import casadi as ca
import time
import matplotlib.pyplot as plt
import pybullet as p
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

# PARAMETERS
MASS = 0.0397  # kg
G = 9.81       # m/s²
F_MAX = 14.82  # m/s²
F_MIN = 4.0    # m/s²
OMEGA_MAX = 10.0 # rad/s
DT = 0.02      # 50Hz
T_HORIZON = 1.5
N_STEPS = int(T_HORIZON / DT)
# Max jerk derived from coupling between thrust magnitude and max body angular rates
J_MAX = (1 / np.sqrt(3)) * F_MIN * OMEGA_MAX


class QuadcopterMPC:
    def __init__(self):
        self.N = N_STEPS
        self.dt = DT
        
        # State Space Matrices for a Triple Integrator
        # State vector: [position, velocity, acceleration]^T, Input: jerk
        self.Ad = np.array([
            [1, DT, 0.5*DT**2],
            [0, 1, DT],
            [0, 0, 1]
        ])
        self.Bd = np.array([[(DT**3)/6], 
                            [(DT**2)/2], 
                            [DT]])

        self.opti = ca.Opti()
        
        # Decision Variables: State trajectories over the horizon
        self.x = self.opti.variable(3, self.N + 1)
        self.y = self.opti.variable(3, self.N + 1)
        self.z = self.opti.variable(3, self.N + 1)
        # Decision Variables: Control inputs (jerk)
        self.jx = self.opti.variable(1, self.N)
        self.jy = self.opti.variable(1, self.N)
        self.jz = self.opti.variable(1, self.N)

        # Parameters (Initial states and targets for real-time updates)
        self.z0_x = self.opti.parameter(3, 1)
        self.z0_y = self.opti.parameter(3, 1)
        self.z0_z = self.opti.parameter(3, 1)
        self.target_x = self.opti.parameter(3, 1)
        self.target_y = self.opti.parameter(3, 1)
        self.target_z = self.opti.parameter(3, 1)

        # Cost Function: Minimum Jerk Trajectory

        cost = ca.sumsqr(self.jx) + ca.sumsqr(self.jy) + ca.sumsqr(self.jz)
        self.opti.minimize(cost)

        # Dynamics Constraints
        for k in range(self.N):
            self.opti.subject_to(self.x[:, k+1] == ca.mtimes(self.Ad, self.x[:, k]) + ca.mtimes(self.Bd, self.jx[k]))
            self.opti.subject_to(self.y[:, k+1] == ca.mtimes(self.Ad, self.y[:, k]) + ca.mtimes(self.Bd, self.jy[k]))
            self.opti.subject_to(self.z[:, k+1] == ca.mtimes(self.Ad, self.z[:, k]) + ca.mtimes(self.Bd, self.jz[k]))

        # Initial and Final Conditions
        self.opti.subject_to(self.x[:, 0] == self.z0_x)
        self.opti.subject_to(self.y[:, 0] == self.z0_y)
        self.opti.subject_to(self.z[:, 0] == self.z0_z)
        self.opti.subject_to(self.x[:, self.N] == self.target_x)
        self.opti.subject_to(self.y[:, self.N] == self.target_y)
        self.opti.subject_to(self.z[:, self.N] == self.target_z)

        # Limits (Inequality Constraints)
        x_min, x_max = -0.22, 1.53  # cage limit X 
        y_min, y_max = -0.25, 0.25  # cage limit y 
        z_min, z_max =  -0.05, 1.25  # cage limit Z
        
        for k in range(self.N + 1):
            self.opti.subject_to(self.opti.bounded(x_min, self.x[0, k], x_max))
            self.opti.subject_to(self.opti.bounded(y_min, self.y[0, k], y_max))
            self.opti.subject_to(self.opti.bounded(z_min, self.z[0, k], z_max))

        # Physical limits: Acceleration and Jerk constraints
        acc_min_z, acc_max_z = F_MIN - G, F_MAX - G
        acc_lim_xy = 4.0
        for k in range(self.N + 1):
            self.opti.subject_to(self.opti.bounded(-acc_lim_xy, self.x[2, k], acc_lim_xy))
            self.opti.subject_to(self.opti.bounded(-acc_lim_xy, self.y[2, k], acc_lim_xy))
            self.opti.subject_to(self.opti.bounded(acc_min_z, self.z[2, k], acc_max_z))
        for k in range(self.N):
            self.opti.subject_to(self.opti.bounded(-J_MAX, self.jx[k], J_MAX))
            self.opti.subject_to(self.opti.bounded(-J_MAX, self.jy[k], J_MAX))
            self.opti.subject_to(self.opti.bounded(-J_MAX, self.jz[k], J_MAX))

        # Configure solver settings (silence output for real-time performance)
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 100}
        self.opti.solver('ipopt', opts)

    def solve(self, current_state, target_state):
        # Update initial conditions and targets for the current optimization step
        self.opti.set_value(self.z0_x, current_state[0])
        self.opti.set_value(self.z0_y, current_state[1])
        self.opti.set_value(self.z0_z, current_state[2])
        self.opti.set_value(self.target_x, target_state[0])
        self.opti.set_value(self.target_y, target_state[1])
        self.opti.set_value(self.target_z, target_state[2])
        
        try:
            sol = self.opti.solve()
            # Return the next planned acceleration and jerk
            return (np.array([sol.value(self.x[2, 1]), sol.value(self.y[2, 1]), sol.value(self.z[2, 1])]),
                    np.array([sol.value(self.jx[0]), sol.value(self.jy[0]), sol.value(self.jz[0])]))
        except:
            return None, None
            # TODO: Handle solver failure (e.g., return hover command)

    def simulate(self, x0, target, sim_steps=100):
        x_hist = np.zeros((3, sim_steps+1))
        y_hist = np.zeros((3, sim_steps+1))
        z_hist = np.zeros((3, sim_steps+1))

        j_hist = np.zeros((3, sim_steps))

        x_hist[:, 0] = x0[0]
        y_hist[:, 0] = x0[1]
        z_hist[:, 0] = x0[2]

        state_x = x0[0].copy()
        state_y = x0[1].copy()
        state_z = x0[2].copy()

        for t in range(sim_steps):

            acc_plan, jerk_plan = self.solve(
                np.array([state_x, state_y, state_z]),
                target
            )

            if jerk_plan is None:
                break

            j = jerk_plan
            j_hist[:, t] = j

            # dinâmica discreta (triple integrator)
            state_x = self.Ad @ state_x + self.Bd.flatten() * j[0]
            state_y = self.Ad @ state_y + self.Bd.flatten() * j[1]
            state_z = self.Ad @ state_z + self.Bd.flatten() * j[2]

            x_hist[:, t+1] = state_x
            y_hist[:, t+1] = state_y
            z_hist[:, t+1] = state_z

        return x_hist, y_hist, z_hist, j_hist

def map_to_cf_thrust(f_desired):
    # Maps desired acceleration/force to the Crazyflie's 16-bit PWM command (0 to 65535)
    f_clamped = max(F_MIN, min(f_desired, F_MAX))
    return int((f_clamped / F_MAX) * 65535)

class CrazyflieMPCControl:

    def __init__(self, uri):
        self.uri = uri
        self.mpc = QuadcopterMPC()
        self.state = np.zeros((3, 3)) # Rows: X, Y, Z | Cols: pos, vel, acc

    def _log_callback(self, timestamp, data, logconf):
        self.state[0, :] = [data['stateEstimate.x'], data['stateEstimate.vx'], data['stateEstimate.ax']]
        self.state[1, :] = [data['stateEstimate.y'], data['stateEstimate.vy'], data['stateEstimate.ay']]
        self.state[2, :] = [data['stateEstimate.z'], data['stateEstimate.vz'], data['stateEstimate.az']]

    def run(self):
        import cflib.crtp
        cflib.crtp.init_drivers()
        with SyncCrazyflie(self.uri) as scf:
            
            # Hard constraints on the cage in case the MPC code fails
            scf.cf.param.set_value('geofence.minX', '-0.24')
            scf.cf.param.set_value('geofence.maxX', '1.55')
            scf.cf.param.set_value('geofence.minY', '-0.28')
            scf.cf.param.set_value('geofence.maxY', '0.28')
            scf.cf.param.set_value('geofence.minZ', '-0.05')
            scf.cf.param.set_value('geofence.maxZ', '1.30')
            scf.cf.param.set_value('geofence.enable', '1')

            # Log configuration for onboard EKF
            logconf = LogConfig(name='State', period_in_ms=20)
            for ax in ['x', 'y', 'z']:
                logconf.add_variable(f'stateEstimate.{ax}', 'float')
                logconf.add_variable(f'stateEstimate.v{ax}', 'float')
                logconf.add_variable(f'stateEstimate.a{ax}', 'float')
            scf.cf.log.add_config(logconf)
            logconf.data_received_cb.add_callback(self._log_callback)
            logconf.start()

            # Target states for X, Y, Z axes respectively 
            target = np.array([[1.0, 0, 0],   # X: [pos, vel, acc]
                               [0.15, 0, 0],  # Y: [pos, vel, acc]
                               [1.0, 0, 0]])  # Z: [pos, vel, acc]

            print("Starting MPC Control Loop.")
            try:
                while True:
                    start_time = time.time()
                    acc_plan, jerk_plan = self.mpc.solve(self.state, target)

                    if acc_plan is not None:
                        # Convert planned acceleration into thrust magnitude and direction
                        f_vec = acc_plan - np.array([0, 0, -G])
                        f_mag = np.linalg.norm(f_vec)

                        # Map jerk and thrust to roll and pitch rate commands
                        roll_rate = -jerk_plan[1] / f_mag
                        pitch_rate = jerk_plan[0] / f_mag
                        thrust = map_to_cf_thrust(f_mag)
                        
                        scf.cf.commander.send_rate_setpoint(np.degrees(roll_rate), np.degrees(pitch_rate), 0.0, thrust)

                    else:
                        # Hover fallback if solver fails
                        thrust_hover = map_to_cf_thrust(MASS * G) 
                        scf.cf.commander.send_rate_setpoint(0.0, 0.0, 0.0, thrust_hover)
                    
                    # Maintain the loop frequency based on DT
                    solve_time = time.time() - start_time
                    if solve_time < DT:
                        time.sleep(DT - solve_time)
                    else:
                        print(f"Warning: Solver took {solve_time*1000:.2f}ms")

            except KeyboardInterrupt:
                print("Interruption detected. Starting emergency landing sequence with PID controller.")
                current_thrust_cmd = map_to_cf_thrust(G)
                # Ramp down thrust for a soft landing
                while current_thrust_cmd > 10000:
                    loop_start = time.time()
                    current_thrust_cmd -= 800
                    if current_thrust_cmd < 0:
                        current_thrust_cmd = 0
                    scf.cf.commander.send_setpoint(0.0, 0.0, 0, current_thrust_cmd)
                    elapsed = time.time() - loop_start
                    if elapsed < DT:
                        time.sleep(DT - elapsed)
                for i in range(5):
                    scf.cf.commander.send_setpoint(0, 0, 0, 0)
                    time.sleep(0.01)

class PyBulletSim:
    def __init__(self, env):
        self.env = env
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.thrust = 0.0
    
    def apply_control(self, roll_rate, pitch_rate, yaw_rate, thrust):
        self.roll_rate = roll_rate
        self.pitch_rate = pitch_rate
        self.yaw_rate = yaw_rate
        self.thrust = thrust
        
    def step(self):
        self.roll += self.roll_rate * DT
        self.pitch += self.pitch_rate * DT
        self.yaw += self.yaw_rate * DT

        R = self._rotation_matrix(self.roll, self.pitch, self.yaw)

        thrust = np.array([0, 0, self.thrust])
        force_world = R @ thrust

        p.applyExternalForce(
            self.env.droneIds[0],
            -1,
            force_world.tolist(),
            [0, 0, 0],
            p.WORLD_FRAME
        )

        self.env.stepSimulation()
    
    def _rotation_matrix(self, roll, pitch, yaw):
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        return R
    
    def get_state(self):
        s = self.env._getDroneStateVector(0)

        return {
            "x": s[0],
            "y": s[1],
            "z": s[2],
            "vx": s[10],
            "vy": s[11],
            "vz": s[12],
            "ax": 0.0,
            "ay": 0.0,
            "az": 0.0
        }
    

class SimulatedCrazyflie:
    def __init__(self, sim):
        self.sim = sim
        self.commander = SimulatedCommander(sim)
        self.log = SimulatedLogger(sim)
        self.param = self._Param()

    class _Param:
        def set_value(self, k, v):
            pass

    def apply_control(self, roll_rate, pitch_rate, yaw_rate, thrust):
        self.sim.apply_control(roll_rate, pitch_rate, yaw_rate, thrust)
    
    def step(self):
        self.sim.step()
        self.log.update()

class SimulatedCommander:
    def __init__(self, sim):
        self.sim = sim  

    def send_rate_setpoint(self, roll, pitch, yaw, thrust):
        self.sim.apply_control(roll, pitch, yaw, thrust)

    def send_setpoint(self, roll, pitch, yaw, thrust):
        self.sim.apply_control(roll, pitch, yaw, thrust)


class SimulatedLogger:
    def __init__(self, sim):
        self.sim = sim
        self.callbacks = []

    def add_config(self, logconf):
        pass  

    def add_callback(self, cb):
        self.callbacks.append(cb)

    def update(self):
        state = self.sim.get_state()

        data = {
            "stateEstimate.x": state["x"],
            "stateEstimate.y": state["y"],
            "stateEstimate.z": state["z"],
            "stateEstimate.vx": state["vx"],
            "stateEstimate.vy": state["vy"],
            "stateEstimate.vz": state["vz"],
            "stateEstimate.ax": state["ax"],
            "stateEstimate.ay": state["ay"],
            "stateEstimate.az": state["az"],
        }

        for cb in self.callbacks:
            cb(0, data, None)


if __name__ == "__main__":
    mode = "simPybullet" 

    if mode == "sim": # simplified simulation to see convergences, only tests MPC
        mpc = QuadcopterMPC()

        x0 = np.array([
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])

        target = np.array([
            [1.0, 0, 0],
            [0.15, 0, 0],
            [1.0, 0, 0]
        ])

        xh, yh, zh, jh = mpc.simulate(x0, target, 720)
        

        import matplotlib.pyplot as plt
        t = np.arange(xh.shape[1]) * DT

        plt.plot(t, xh[0], label="x")
        plt.plot(t, yh[0], label="y")
        plt.plot(t, zh[0], label="z")
        plt.legend()
        plt.grid()
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

    elif mode == "simPybullet":

        env = HoverAviary(
            drone_model=DroneModel.CF2X,
            physics=Physics.PYB,
            gui=True
        )

        sim_core = PyBulletSim(env)
        cf = SimulatedCrazyflie(sim_core)

        mpc = QuadcopterMPC()

        target = np.array([
            [1.0, 0, 0],
            [0.15, 0, 0],
            [1.0, 0, 0]
        ])

        obs, _ = env.reset()

        while True:
            state = cf.sim.get_state()

            acc_plan, jerk_plan = mpc.solve(
                np.array([state["x"], state["y"], state["z"]]),
                target
            )

            if acc_plan is not None:
                f_vec = acc_plan - np.array([0, 0, -G])
                f_mag = np.linalg.norm(f_vec)

                f_mag = max(f_mag, 1e-3)
                roll_rate = -jerk_plan[1] / f_mag
                pitch_rate = jerk_plan[0] / f_mag
                thrust = f_mag

                cf.commander.send_rate_setpoint(
                    np.degrees(roll_rate),
                    np.degrees(pitch_rate),
                    0.0,
                    thrust
                )

            cf.step()
            sim_core.step()
            time.sleep(DT)

    elif mode == "drone":
        URI = 'radio://0/80/2M/E7E7E7E7E7'
        control = CrazyflieMPCControl(URI)
        control.run()
