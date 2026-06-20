from configs.parameters import MAX_THRUST_NEWTONS, MAX_COMMAND, MASS

def map_to_cf_thrust(f_desired):
    
    desired_newtons = f_desired * MASS
    
    desired_newtons = max(0.0, desired_newtons)
    
    if desired_newtons >= MAX_THRUST_NEWTONS:
        return int(MAX_COMMAND)

    thrust_ratio = desired_newtons / MAX_THRUST_NEWTONS
    final_command = thrust_ratio * MAX_COMMAND
    
    return int(final_command)