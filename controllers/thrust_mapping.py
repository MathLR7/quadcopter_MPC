from configs.parameters import F_MIN, F_MAX

def map_to_cf_thrust(f_desired):
    # Maps desired acceleration/force to the Crazyflie's 16-bit PWM command (0 to 65535)
    f_clamped = max(F_MIN, min(f_desired, F_MAX))
    return int((f_clamped / F_MAX) * 65535)