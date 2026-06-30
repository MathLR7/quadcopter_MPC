import numpy as np

from configs.parameters import (
    DT,
    MASS,
    G,
    F_MIN,
    F_MAX
)


class TripleIntegratorPlant:

    def __init__(self):

        self.Ad = np.array([
            [1, DT, 0.5*DT**2],
            [0, 1, DT],
            [0, 0, 1]
        ])

        self.Bd = np.array([
            [DT**3/6],
            [DT**2/2],
            [DT]
        ])


    def step(self, state, acc_cmd):

        return (
            self.Ad @ state
            +
            self.Bd.flatten()*acc_cmd
        )


class VerticalPlant:

    def __init__(self):

        self.Ad = np.array([
            [1, DT, 0.5*DT**2],
            [0, 1, DT],
            [0, 0, 1]
        ])

        self.Bd = np.array([
            [DT**3/6],
            [DT**2/2],
            [DT]
        ])

        self.acc_real = 0


    def step(self,state,jerk):

        # nominal triple integrator first
        acc_cmd = (
            state[2]
            +
            DT*jerk
        )


        # convert acceleration request to thrust
        thrust = MASS*(acc_cmd + G)


        thrust = np.clip(
            thrust,
            F_MIN,
            F_MAX
        )


        acc_real = thrust/MASS - G


        new_state = (
            self.Ad @ state
            +
            self.Bd.flatten()*jerk
        )

        # replace ideal acceleration
        new_state[2] = acc_real


        return new_state