class SimulatedCrazyflie:

    def __init__(self, sim):

        self.sim = sim

        self.commander = SimulatedCommander(sim)

        self.log = SimulatedLogger(sim)

        self.param = self._Param()


    def get_state(self):

        return self.sim.get_state()


    class _Param:

        def set_value(self, k, v):
            pass



    def step(self):

        self.sim.step()

        self.log.update()



class SimulatedCommander:

    def __init__(self, sim):

        self.sim = sim



    def send_rate_setpoint(self, roll, pitch, yaw, thrust):

        self.sim.apply_control(
            roll,
            pitch,
            yaw,
            thrust
        )



    def send_setpoint(self, roll, pitch, yaw, thrust):

        self.sim.apply_control(
            roll,
            pitch,
            yaw,
            thrust
        )




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