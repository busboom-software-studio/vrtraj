import pytest

from vtraj.ode import *

__author__ = "Eric Busboom"
__copyright__ = "Eric Busboom"
__license__ = "MIT"

def test_basic_plot():
    """API Tests"""
    # Example of how to use the dataclass
    consts = Constants(drag_f=drag)
    params = Parameters()
    params.Kp_theta = .1

    df = throw(v0=300, angle=45,
               consts=consts, params=params)
    plot_traj(df)
