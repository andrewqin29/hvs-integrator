import numpy as np

def leapfrog_step(pos, vel, dt, acceleration_func):
    """
    Performs a single step of the Leapfrog integration procedure given initial position and acceleration functions.
    
    Parameters:
        - pos: The 3d position vector [x, y, z]
        - vel: The 3d velocity vector [u, v, w]
        - dt: Timestep in myr (negative for backwards)
        - acceleration_func (function): A function that takes a position vector and returns acceleration vector
    """
    # calculate acceleration at current position
    accel = acceleration_func(pos)
    # gives us velocity at current position + half-step
    vel_half = vel + accel * (dt / 2.0)
    # update position by full timestep using velocity at half time step
    pos_new = pos + vel_half * dt
    # calculate acceleration at pos_new
    accel_new = acceleration_func(pos_new)
    # obtain final velocity by adding new acceleration * dt/2 to velocity at half step
    vel_new = vel_half * accel_new * (dt / 2.0)

    return pos_new, vel_new