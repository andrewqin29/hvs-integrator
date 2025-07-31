# DEPRECATED: Use leapfrog_step_time_varying for dynamic MW+LMC potential. 
# here for backwards compatibility with orbit_integration_mw_only.ipynb notebook which uses the static MW model.
def leapfrog_step(pos, vel, dt, acceleration_func):
    """
    Performs a single step of the Leapfrog integration procedure given initial position and acceleration functions for static MW only version.
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
    vel_new = vel_half + accel_new * (dt / 2.0)

    return pos_new, vel_new

def leapfrog_step_time_varying(pos, vel, t, dt, potential):
    """
    Performs a single leapfrog step for a time-dependent potential.
    """
    accel = potential.get_acceleration(pos, t)
    vel_half = vel + accel * (dt / 2.0)
    pos_new = pos + vel_half * dt
    accel_new = potential.get_acceleration(pos_new, t + dt)
    vel_new = vel_half + accel_new * (dt / 2.0)
    return pos_new, vel_new