import pytest

from vtraj.ode import *

__author__ = "Eric Busboom"
__copyright__ = "Eric Busboom"
__license__ = "MIT"


def test_basic_plot():
    """API Tests"""
    # Example of how to use the dataclass
    params = Parameters(C_d_1=0.02, C_d_2=0.03,Kp_theta=1)
    ctx = Context( error=0, dt=1 / 30, params=params)

    sol, df = throw(v0=100, angle=45, ctx=ctx)
    plot_traj(df)
