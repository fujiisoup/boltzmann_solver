import pytest
import numpy as np
from . import core


def test_scattering_plot():
    diffsigma = core.DifferentialCrossSection(
        lam=0, 
        legendre_coefs=[0, 0, 0, 0, 0, 0, 0, 1])
    # consider the beam configuration
    n = 300
    u1 = np.zeros((n, 3))
    u1[:, -1] = 1
    u2 = np.zeros((n, 3))
    rng = np.random.RandomState(0)

    v1, v2 = core.scattering(
        m1=1.0, u1=u1, m2=1.0, u2=u2, rng=rng, 
        differential_crosssection=diffsigma, density=1, dt=1e4
    )

    # should be symmetric along x axis
    assert np.allclose(np.std(v1[:, 0]), np.std(v1[:, 1]), rtol=0.1)
    assert np.allclose(np.std(v2[:, 0]), np.std(v2[:, 1]), rtol=0.1)
    assert np.allclose(np.std(v1[:, 0]), np.std(v2[:, 0]), rtol=0.1)
    """
    import matplotlib.pyplot as plt
    plt.plot(v1[:, 0], v1[:, 1], '.')
    plt.plot(v2[:, 0], v2[:, 1], '.')
    plt.show()
    """

@pytest.mark.parametrize(('m1', 'm2'), [(1.0, 1.0), (10.0, 1.0)])
def test_scattering_conservation(m1, m2):
    diffsigma = core.DifferentialCrossSection(
        lam=0, 
        legendre_coefs=[0, 0, 0, 0, 0, 0, 0, 1])
    # random velicity
    n = 300
    rng = np.random.RandomState(0)
    u1 = rng.randn(n, 3)
    u2 = rng.randn(n, 3)
    
    v1, v2 = core.scattering(
        m1=m1, u1=u1, m2=m2, u2=u2, rng=rng, 
        differential_crosssection=diffsigma, density=1, dt=1e4
    )

    # total energy should be the same
    before = 0.5 * m1 * np.sum(u1**2, axis=-1) + 0.5 * m2 * np.sum(u2**2, axis=-1)
    after = 0.5 * m1 * np.sum(v1**2, axis=-1) + 0.5 * m2 * np.sum(v2**2, axis=-1)
    assert np.allclose(before, after)
    # total momentum should be conserved
    before = m1 * u1 + m2 * u2
    after = m1 * v1 + m2 * v2
    assert np.allclose(before, after)
