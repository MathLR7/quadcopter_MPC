import numpy as np
import casadi as ca

from configs.parameters import *

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
        acc_lim_xy = 7.0
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