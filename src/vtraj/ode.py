"""ODE functions to generate trajectory curves"""

import numpy as np
import pandas as pd

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field

def plot_traj(df):
    # Create a figure with two subplots: the main one and the smaller one below it
    fig = plt.figure(figsize=(5, 4.33))  # Adjust total figure height
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.33, 1])

    xlim = max(150, df['x'].max())
    ylim = max(100, df['y'].max())

    # Main plot: y vs x
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df['x'], df['y'], label='Projectile Path')
    ax1.set_xlabel('x (meters)')
    ax1.set_ylabel('y (meters)')
    ax1.set_ylim(0, ylim)
    ax1.set_xlim(0, xlim)
    ax1.set_title('Projectile Motion: y vs x')
    ax1.legend()
    ax1.grid(True)

    # Secondary plot: theta vs x
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(df['x'], df['theta'], label='proj')
    ax2.plot(df['x'], df['v_angle'], label='v')
    ax2.set_xlabel('x (meters)')
    ax2.set_ylabel('theta (degrees)')
    ax2.set_xlim(0, xlim)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def add_error(value, pstd):
    """
    Adds a Gaussian error to the input value.

    Parameters:
    value (float): The original value to be modified.
    pstd (float): The standard deviation of the Gaussian error as a proportion of the value.

    Returns:
    float: The value modified with Gaussian error.
    """
    std_dev = abs(pstd * value)
    error = np.random.normal(0, std_dev)
    return value + error

@dataclass
class Parameters:
    # Parameters
    Kp_theta: float = 0.1  # Parameter for change in projectile orientation

    C_d_1: float = 0.047  # Drag coefficient
    A_1: float = 0.045  # Cross-sectional area (m^2)

    C_d_2: float = 0.06  # Drag coefficient
    A_2: float = 0.08  # Cross-sectional area (m^2)

@dataclass
class Context:
    
    # Constants
    dt: float = 0.01  # Time step (s)
    t_max: float = 30 # Max time to integrate
    g: float = 9.81  # Gravitational acceleration (m/s^2)
    rho: float = 1.225  # Air density (kg/m^3)
    m: float = 0.145  # Mass (kg)
    error: float = 0.01

    # Data
    rows: list = field(default_factory=list)

    # Parameters
    params: Parameters = field(default_factory=Parameters)

def reeval(ctx, solution):
    t = solution.t

    x, y, theta, vx, vy = solution.y  # Unpack the states

    F_d = np.vectorize(drag)(t, vx, vy, ctx)

    v = np.sqrt(vx ** 2 + vy ** 2)
    ax = -F_d * vx / (ctx.m * v)
    ay = -ctx.g - (F_d * vy / (ctx.m * v))

    # Create a pandas DataFrame

    data = {
        't': t,
        'x': x,
        'y': y,
        'theta': np.degrees(theta),
        'vx': vx,
        'vy': vy,
        'v_angle': np.degrees(np.arctan2(vy, vx)),
        'ax': ax,
        'ay': ay,
        'F_d': F_d
    }

    return pd.DataFrame(data)

# Event function to stop integration when y < 0
def event_y_below_zero(t, state, ctx: Context ):
    return state[1]  # state[1] is y, so looking for y == 0

event_y_below_zero.terminal = True
event_y_below_zero.direction = -1  # When y == 0, look for transition from pos to neg.

def drag(t, vx, vy, ctx: Context):
    """Compute the drag force, using a combination
    of two different C_d*A, proportional to the angle of the velocity vector """
    v = np.sqrt(vx ** 2 + vy ** 2)  # Magnitude of velocity vector
    a = math.atan2(vy, vx)

    F_d_1 = 0.5 * ctx.params.C_d_1 * ctx.rho * ctx.params.A_1 * v ** 2
    F_d_2 = 0.5 * ctx.params.C_d_2 * ctx.rho * ctx.params.A_2 * v ** 2

    F_d = F_d_1 * math.cos(a) ** 2 + F_d_2 * math.sin(a) ** 2

    return F_d

def projectile_motion(t, y, ctx: Context):
    """Calculate dy/dt for the state y. This version includes all state
    values, and the input is the same structure as the output. """

    x, y, theta, vx, vy = y

    a = math.atan2(vy, vx)

    F_d = drag(t, vx, vy, ctx)

    ax = -F_d * math.cos(a) / ctx.m
    ay = -ctx.g - (F_d * math.sin(a) / ctx.m)

    #vx = vx + ax * t
    #vy = vy + ay * t

    # The projectile has a tendency to orient along the  velocity vector.
    dtheta = (a - theta) * ctx.params.Kp_theta

    #ax = add_error(ax, ctx.error)
    #ay = add_error(ay, ctx.error)

    row = {
        't': t,
        'x': x,
        'y': y,
        'theta': np.degrees(theta),
        'vx': vx,
        'vy': vy,
        'v_angle': np.degrees(np.arctan2(vy, vx)),
        'ax': ax,
        'ay': ay,
        'F_d': F_d
    }

    ctx.rows.append(row)

    return (vx, vy, dtheta, ax, ay)


def throw(v0, angle, ctx: Context, dense_output=False):
    """Solve a trajectory for an initial velocity and angle"""

    angle_rad = np.radians(angle)

    # Initial velocity components
    vx0 = v0 * np.cos(angle_rad)
    vy0 = v0 * np.sin(angle_rad)

    x0 = 0  # Initial x position (m)
    y0 = 0  # Initial y position (m)

    # Time span
    t_span = (0, ctx.t_max)
    t_eval = np.arange(0, ctx.t_max, ctx.dt)

    y0 = [
        x0,  # x
        y0,  # y
        angle_rad,  # theta
        vx0,  # vx
        vy0,  # vy,
    ]

    solution = solve_ivp(projectile_motion, t_span,
                         y0=y0,  # Initial State
                         t_eval=t_eval,
                         args=(ctx,), events=event_y_below_zero,
                         dense_output=dense_output)

    return solution, reeval(ctx, solution)

