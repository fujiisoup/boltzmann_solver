import os
import numpy as np
import pytest
from . import md_core


def test_colliding_time():
    rng = np.random.RandomState(0)
    n = 100
    x = rng.randn(n, 3)
    v = rng.randn(n, 3)

    # vector computation
    x_rel = x[:, np.newaxis] - x
    v_rel = v[:, np.newaxis] - v
    colliding_time = md_core.get_colliding_time(x_rel, v_rel)
    assert colliding_time.shape == (n, n)
    assert (colliding_time > 0).all()
    assert np.allclose(colliding_time, colliding_time.T)

    for i in range(n):
        x_rel = x[i] - x
        v_rel = v[i] - v
        colliding_time1d = md_core.get_colliding_time(x_rel, v_rel)
        assert colliding_time1d.shape == (n, )
        assert (colliding_time1d > 0).all()
        assert (colliding_time1d == colliding_time[i]).all()


def test_colliding_time_known_value():
    # with known values
    x = np.array([[0, 0, 0], [0, 0, 1.0]])
    v = np.array([[0, 0, 1e-10], [0, 0, -10.0]])
    x_rel = x[:, np.newaxis] - x
    v_rel = v[:, np.newaxis] - v
    actual = md_core.get_colliding_time(x_rel, v_rel)
    expected = 0.1
    assert actual.shape == (2, 2)
    assert np.allclose(actual[0, 1], expected)
    assert np.allclose(actual[1, 0], expected)

