import pytest
import numpy as np
from ai4pdes.operators import get_weights_linear_2D

# x-advection stencil/ filter with dx=0.5
ADV_FILTER = np.array([[
    [[ 0.1667],
    [ 0.0000],
    [-0.1667]],

    [[ 0.6667],
    [ 0.0000],
    [-0.6667]],

    [[ 0.1667],
    [ 0.0000],
    [-0.1667]]
]])

# Test x-advection (values)
def test_xadv_operator():
    dx = 0.5
    adv_operator = get_weights_linear_2D(dx)
    assert np.allclose(adv_operator[1].numpy(), ADV_FILTER)

    # # test that this is close to what numpy.gradient does
    # xfield = np.random.randn(16, 16)
    # numpyouts = np.gradient(xfield, dx)
    
    # assert np.allclose(numpyouts, adv_operator*xfield)
