from configs.parameters import F_MIN, F_MAX

def map_to_cf_thrust(f_desired):

    f_clamped = max(F_MIN, min(f_desired, F_MAX))

    return int((f_clamped / F_MAX) * 65535)