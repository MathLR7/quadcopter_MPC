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
        self.roll += np.radians(self.roll_rate) * DT
        self.pitch += np.radians(self.pitch_rate) * DT
        self.yaw += np.radians(self.yaw_rate) * DT

        self.roll = np.clip(self.roll, -1.0, 1.0)
        self.pitch = np.clip(self.pitch, -1.0, 1.0)
        self.yaw = np.clip(self.yaw, -np.pi, np.pi)

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

        pos, _ = p.getBasePositionAndOrientation(self.env.getDroneIds()[0])
        
        p.applyExternalForce(
            self.env.getDroneIds()[0],
            -1,
            force_world.tolist(),
            pos, 
            p.WORLD_FRAME
        )
        p.stepSimulation()



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

        pos, orn = p.getBasePositionAndOrientation(
            self.env.getDroneIds()[0]
        )

        vel, ang_vel = p.getBaseVelocity(
            self.env.getDroneIds()[0]
        )

        current_vel = np.array(vel)
        acc = (current_vel - self.prev_v) / DT
        self.prev_v = current_vel

        return {
            "x": pos[0],
            "y": pos[1],
            "z": pos[2],

            "vx": vel[0],
            "vy": vel[1],
            "vz": vel[2],

            "ax": acc[0],
            "ay": acc[1],
            "az": acc[2]
        }
