import numpy as np
import time

import cflib.crtp

from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig

from controllers.thrust_mapping import map_to_cf_thrust
from controllers.mpc_controller import QuadcopterMPC
from configs.parameters import *

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
            scf.cf.param.set_value('colAv.bboxMinX', '-0.24')
            scf.cf.param.set_value('colAv.bboxMaxX', '1.55')
            scf.cf.param.set_value('colAv.bboxMinY', '-0.28')
            scf.cf.param.set_value('colAv.bboxMaxY', '0.28')
            scf.cf.param.set_value('colAv.bboxMinZ', '-0.05')
            scf.cf.param.set_value('colAv.bboxMaxZ', '1.30')
            scf.cf.param.set_value('colAv.enable', '1')

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
                        f_vec = acc_plan + np.array([0,0,G])
                        f_mag = np.linalg.norm(f_vec)
                        if f_mag < 0.1:
                            f_mag = 0.1
  
                        # Map jerk and thrust to roll and pitch rate commands
                        roll_rate = -jerk_plan[1] / f_mag
                        pitch_rate = jerk_plan[0] / f_mag
                        roll_rate = np.clip(roll_rate,-OMEGA_MAX,OMEGA_MAX)
                        pitch_rate = np.clip(pitch_rate,-OMEGA_MAX,OMEGA_MAX)

                        thrust_raw = map_to_cf_thrust(f_mag)
                        thrust_percentage = (thrust_raw / 65535.0) * 100.0
                        
                        scf.cf.commander.send_setpoint_manual(np.degrees(roll_rate), np.degrees(pitch_rate), 0.0, thrust_percentage, True)

                    else:
                        # Hover fallback if solver fails
                        thrust_hover = map_to_cf_thrust(G)
                        th_percentage = (thrust_hover / 65535.0) * 100.0
                        scf.cf.commander.send_setpoint_manual(0.0, 0.0, 0.0, th_percentage, True)
                    
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
