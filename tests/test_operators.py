import pytest
import numpy as np
from ai4pdes.operators import get_weights_linear_2D

# Test x-advection (values)
def test_xadv_operator():
    dx = 0.5
    adv_operator = get_weights_linear_2D(dx)[2].numpy()[0, 0] # index 2 get x-adv

    # Assert sum to zero per row
    assert np.allclose(adv_operator.sum(axis=0), 0)
    assert np.allclose(adv_operator.sum(), 0, atol=np.finfo(adv_operator.dtype).eps)

    # # test that this is close to what numpy.gradient does
    # xfield = np.random.randn(16, 16)
    # numpyouts = np.gradient(xfield, dx)
    
    # assert np.allclose(numpyouts, adv_operator*xfield)
