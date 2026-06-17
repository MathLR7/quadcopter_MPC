import numpy as np
import pybullet as p

from configs.parameters import DT

class PyBulletSim:

    def __init__(self, env):
        self.env = env
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll_rate = 0.0
        self.pitch_rate = 0.0
        self.yaw_rate = 0.0
        self.thrust = 0.0
        self.prev_v = np.zeros(3)


    def apply_control(self, roll_rate, pitch_rate, yaw_rate, thrust):
        self.roll_rate = roll_rate
        self.pitch_rate = pitch_rate
        self.yaw_rate = yaw_rate
        self.thrust = thrust



    def step(self):
        self.roll += self.roll_rate * DT
        self.pitch += self.pitch_rate * DT
        self.yaw += self.yaw_rate * DT

        R = self._rotation_matrix(
            self.roll,
            self.pitch,
            self.yaw
        )

        thrust = np.array([
            0,
            0,
            self.thrust
        ])

        force_world = R @ thrust

        p.applyExternalForce(
            self.env.droneIds[0],
            -1,
            force_world.tolist(),
            [0,0,0],
            p.WORLD_FRAME
        )

        self.env.stepSimulation()



    def _rotation_matrix(self, roll, pitch, yaw):
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        return np.array([
            [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
            [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
            [-sp, cp*sr, cp*cr]
        ])



    def get_state(self):
        s = self.env._getDroneStateVector(0)

        v = np.array([
            s[10],
            s[11],
            s[12]
        ])
        a = (v - self.prev_v)/DT
        self.prev_v = v.copy()

        return {

            "x": s[0],
            "y": s[1],
            "z": s[2],

            "vx": v[0],
            "vy": v[1],
            "vz": v[2],

            "ax": a[0],
            "ay": a[1],
            "az": a[2]

        }