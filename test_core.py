import os
import numpy as np
from numpy.lib.function_base import diff
import pytest
import xarray as xr
from . import core


THIS_DIR = os.path.dirname(__file__)


def test_coulomb():
    def g(x):
        return 1 / (x + 1) ** 2

    differential_crosssection = core.CoulombCrossSection(g)
    print(differential_crosssection._total_crosssection)
    assert np.isfinite(differential_crosssection._total_crosssection)


@pytest.mark.parametrize(
    "diffsigma",
    [
        core.DifferentialCrossSection(
            lam=0, legendre_coefs=[0, 0, 0, 0, 0, 0, 0, 1, 1]
        ),
        core.CoulombCrossSection(core.g_Lewkow),
        core.TheoreticalCrossSections(
            xr.load_dataarray(THIS_DIR + "/crosssections/H-H/elastic_differential.nc")
        ),
        core.IsotropicCrossSections(),
    ],
)
def test_scattering_plot(diffsigma):

    # consider the beam configuration
    n = 300
    u1 = np.zeros((n, 3))
    u1[:, -1] = 3.0
    u2 = np.zeros((n, 3))
    rng = np.random.RandomState(0)

    v1, v2, n_collision = core.scattering(
        m1=1.0,
        u1=u1,
        m2=1.0,
        u2=u2,
        rng=rng,
        differential_crosssection=diffsigma,
        density=1,
        dt=1e4,
    )

    # should be symmetric along x axis
    assert np.allclose(np.std(v1[:, 0]), np.std(v1[:, 1]), rtol=0.1, atol=1e-10)
    assert np.allclose(np.std(v2[:, 0]), np.std(v2[:, 1]), rtol=0.1, atol=1e-10)
    assert np.allclose(np.std(v1[:, 0]), np.std(v2[:, 0]), rtol=0.1, atol=1e-10)
    assert n_collision == n

    # assure extreme values do not raise nan
    for u in [1e-3, 1e3]:
        u1_test = rng.randn(n, 3) * u
        v1, v2, n_collision = core.scattering(
            m1=1.0,
            u1=u1,
            m2=1.0,
            u2=u2,
            rng=rng,
            differential_crosssection=diffsigma,
            density=1,
            dt=1e4,
        )
        assert np.all(np.isfinite(v1))
        assert np.all(np.isfinite(v2))

    """
    import matplotlib.pyplot as plt

    plt.plot(v1[:, 0], v1[:, 1], ".")
    plt.plot(v2[:, 0], v2[:, 1], ".")
    plt.show()

    plt.plot(v1[:, 2], np.sqrt(v1[:, 0] ** 2 + v1[:, 1] ** 2), ".")
    plt.plot(v2[:, 2], np.sqrt(v2[:, 0] ** 2 + v2[:, 1] ** 2), ".")
    plt.show()
    """


@pytest.mark.parametrize(("m1", "m2"), [(1.0, 1.0), (10.0, 1.0)])
@pytest.mark.parametrize("restrict_2d", [False, True])
def test_scattering_conservation(m1, m2, restrict_2d):
    diffsigma = core.DifferentialCrossSection(
        lam=0, legendre_coefs=[0, 0, 0, 0, 0, 0, 0, 1]
    )
    # random velicity
    n = 300
    rng = np.random.RandomState(0)
    u1 = rng.randn(n, 3)
    u2 = rng.randn(n, 3)
    if restrict_2d:
        u1[:, -1] = 0
        u2[:, -1] = 0
        
    v1, v2, n_collision = core.scattering(
        m1=m1,
        u1=u1,
        m2=m2,
        u2=u2,
        rng=rng,
        differential_crosssection=diffsigma,
        density=1,
        dt=1e4,
        restrict_2d=restrict_2d
    )

    # total energy should be the conserved
    before = 0.5 * m1 * np.sum(u1 ** 2, axis=-1) + 0.5 * m2 * np.sum(u2 ** 2, axis=-1)
    after = 0.5 * m1 * np.sum(v1 ** 2, axis=-1) + 0.5 * m2 * np.sum(v2 ** 2, axis=-1)
    assert np.allclose(before, after)
    # total momentum should be conserved
    before = m1 * u1 + m2 * u2
    after = m1 * v1 + m2 * v2
    assert np.allclose(before, after)
    assert n_collision == n

    if restrict_2d:
        assert np.all(v1[:, -1] == 0)
        assert np.all(v2[:, -1] == 0)


@pytest.mark.parametrize("restrict_2d", [False, True])
def test_scattering_heavy_limit(restrict_2d):
    diffsigma = core.DifferentialCrossSection(
        lam=0, legendre_coefs=[0, 0, 0, 0, 0, 0, 0, 1]
    )
    # random velicity
    n = 300
    rng = np.random.RandomState(0)
    m1 = 1
    m2 = 1e6  # assumin very large mass for m2
    u1 = rng.randn(n, 3) / np.sqrt(m1)
    u2 = rng.randn(n, 3) / np.sqrt(m2)
    if restrict_2d:
        u1[:, -1] = 0
        u2[:, -1] = 0

    v1, v2, n_collision = core.scattering(
        m1=m1,
        u1=u1,
        m2=m2,
        u2=u2,
        rng=rng,
        differential_crosssection=diffsigma,
        density=1,
        dt=1e8,
        restrict_2d=restrict_2d
    )
    # make sure many of the particles change the velocity
    assert n_collision > n // 2

    # total energy should be the conserved
    before = 0.5 * m1 * np.sum(u1 ** 2, axis=-1) + 0.5 * m2 * np.sum(u2 ** 2, axis=-1)
    after = 0.5 * m1 * np.sum(v1 ** 2, axis=-1) + 0.5 * m2 * np.sum(v2 ** 2, axis=-1)
    assert np.allclose(before, after)
    # total momentum should be conserved
    before = m1 * u1 + m2 * u2
    after = m1 * v1 + m2 * v2
    assert np.allclose(before, after)
    assert n_collision == n

    # the velocity of particle 2 should not change very much
    assert np.all(np.sqrt(np.sum((u2 - v2)**2, axis=-1)) < 1e-2 / np.sqrt(m2))

    if restrict_2d:
        assert np.all(v1[:, -1] == 0)
        assert np.all(v2[:, -1] == 0)


def allclose_angle(x, y, *args, **kwargs):
    return (
        np.allclose(np.cos(x), np.cos(y), *args, **kwargs)
        and 
        np.allclose(np.sin(x), np.sin(y), *args, **kwargs)
    )


def test_HardSphere():
    elastic = core.HardSphereCrossSections(lam=0)

    restitution_coef = np.linspace(1e-2, 1, 30)
    # with phi = 0, go straight
    assert allclose_angle(0, core._inelastic_scattering_angle(0, restitution_coef))
    # with phi = np.pi, go backward
    print(core._inelastic_scattering_angle(np.pi, restitution_coef) - np.pi)
    assert allclose_angle(np.pi, core._inelastic_scattering_angle(np.pi, restitution_coef), atol=1e-6)

    # with restitution_coef=1, should be elastic
    theta2 = np.linspace(0, np.pi, 100)
    theta_inelastic = core._inelastic_scattering_angle(theta2, restitution_coef=1)
    assert np.allclose(theta2, theta_inelastic)


@pytest.mark.parametrize("restrict_2d", [False, True])
# @pytest.mark.parametrize("restrict_2d", [False])
@pytest.mark.parametrize("restitution_coef", [0, 0.5, 0.9, 0.99])
def test_inelastic_energy(restrict_2d, restitution_coef):
    diffsigma = core.HardSphereCrossSections(lam=0, restrict_2d=restrict_2d)
    # random velicity
    n = 10000
    rng = np.random.RandomState(0)
    u1 = rng.randn(n, 3)
    u2 = rng.randn(n, 3)
    if restrict_2d:
        u1[:, -1] = 0
        u2[:, -1] = 0
        
    v1, v2, n_collision = core.scattering(
        m1=1,
        u1=u1,
        m2=1,
        u2=u2,
        rng=rng,
        differential_crosssection=diffsigma,
        density=1,
        restitution_coef=restitution_coef,
        dt=1e4,
        restrict_2d=restrict_2d
    )
    assert n_collision == n
    if restrict_2d:
        assert np.allclose(v1[:, -1], 0)
        assert np.allclose(v2[:, -1], 0)

    # total energy should be the conserved or smaller
    before = 0.5 * np.sum(u1 ** 2, axis=-1) + 0.5 * np.sum(u2 ** 2, axis=-1)
    after = 0.5 * np.sum(v1 ** 2, axis=-1) + 0.5 * np.sum(v2 ** 2, axis=-1)
    assert np.all(before + 1e-10 >= after)
    # total momentum should be conserved
    before = u1 + u2
    after = v1 + v2
    assert np.allclose(before, after)
    assert n_collision == n

    if restrict_2d:
        assert np.all(v1[:, -1] == 0)
        assert np.all(v2[:, -1] == 0)

    # the energy-loss distribution of an individual particles
    before = np.sum(u1**2, axis=-1)
    after = np.sum(v1**2, axis=-1)
    dE = (before - after)
    dE_actual = np.mean(dE) / np.mean(before)
    if restrict_2d:
        dE_expected = (1 - restitution_coef**2) / 2 * 2 / 3
    else:
        dE_expected = (1 - restitution_coef**2) / 4
    assert np.allclose(dE_actual, dE_expected, atol=0.03)
    
    # the energy-loss distribution of colliding particle pairs
    before = np.sum(u1**2 + u2**2, axis=-1)
    after = np.sum(v1**2 + v2**2, axis=-1)
    dE = (before - after) / before
    dE_actual = np.mean(dE)
    if restrict_2d:
        dE_expected = (1 - restitution_coef**2) / 2 * 2 / 3
    else:
        dE_expected = (1 - restitution_coef**2) / 4
    assert np.allclose(dE_actual, dE_expected, atol=0.003)

    '''
    import matplotlib.pyplot as plt
    plt.hist(dE)
    plt.axvline(dE_expected, ls='--', color='k')
    plt.axvline(dE_actual, ls='--', color='r')
    plt.title('r: {}, '.format(restitution_coef) + ('2d' if restrict_2d else '3d'))
    plt.show()
    '''

@pytest.mark.parametrize(
    "diffsigma",
    [
        core.IsotropicCrossSections(lam=-1),
        core.HardSphereCrossSections(lam=-1),
    ],
)
def test_crosssection(diffsigma):
    # random velicity
    n = 100000
    rng = np.random.RandomState(0)
    r = rng.uniform(0, 1, size=n)
    theta = diffsigma.scattering_angle(1, r)
    
    # moementum transfer
    actual = np.mean((1 - np.cos(theta)))
    assert np.allclose(actual, diffsigma.momentum_transfer(1), rtol=0.01)

    # viscosity
    actual = np.mean(np.sin(theta)**2)
    assert np.allclose(actual, diffsigma.viscosity(1), rtol=0.01)


def _test_viscosity_hardsphere():
    diffsigma = core.HardSphereCrossSections(lam=0)
    # random velicity
    n = 1000000
    rng = np.random.RandomState(0)
    r = rng.uniform(0, 1, size=n)
    theta = diffsigma.scattering_angle(0, r)
    
    actual = np.mean(np.sin(theta)**2)
    assert np.allclose(actual, diffsigma.viscosity(0), rtol=0.01)

    # inelastic collision
    # rcoef = np.logspace(-3, 0, num=21, base=10)
    rcoef = np.linspace(0, 1, num=21)
    actuals = []
    for r in rcoef:
        phi = core._inelastic_scattering_angle(theta, r)
        actuals.append(np.mean(np.sin(phi)**2))
    expected = diffsigma.viscosity(0, restitution_coef=rcoef)
    """
    import matplotlib.pyplot as plt
    plt.plot(rcoef, np.array(actuals))
    plt.plot(rcoef, expected, '--')
    plt.show()
    """

def test_optimize_dt():
    diffsigma = core.DifferentialCrossSection(
        lam=0, legendre_coefs=[0, 0, 0, 0, 0, 0, 0, 1]
    )
    # random velicity
    n = 300
    rng = np.random.RandomState(0)
    u1 = rng.randn(n, 3)
    u2 = rng.randn(n, 3)
    change_rate = 0.01
    dt_inits = [1e1, 1e2, 1e3]
    dt_results = []
    for dt_init in dt_inits:
        dt = core.optimize_dt(
            u1,
            u2,
            diffsigma,
            density=1.0,
            rng=rng,
            change_rate=1.0 + change_rate,
            reaction_rate_per_step=0.3,
            dt_init=1.0,
        )
        dt_results.append(dt)

    assert np.allclose(dt_results, np.mean(dt_results), rtol=change_rate * 10)


def test_thermal_distribution():
    # make sure shape=1 is the same with shape=None
    n = 10000
    m = 3.0
    T = 30.0
    rng = np.random.RandomState(0)
    v_expected = core.thermal_distribution(n, m, T, rng)
    v_actual = core.thermal_distribution(n, m, T, rng, shape=1.5)
    # moments should be close
    v_scale = np.std(v_expected, axis=0)
    for i in [1, 2, 4, 6]:
        m_expected = np.mean(v_expected**i, axis=0)**(1/i)
        m_actual = np.mean(v_actual**i, axis=0)**(1/i)
        assert np.allclose(
            m_expected / v_scale, m_actual / v_scale, 
            atol=0.1
        )

@pytest.mark.parametrize("shape", [1.5, 3, 4])
def test_thermal_distribution_shape(shape):
    # make sure shape=1 is the same with shape=None
    n = 10000
    m = 3.0
    T = 30.0
    rng = np.random.RandomState(0)
    v_expected = core.thermal_distribution(n, m, T, rng)
    v_actual = core.thermal_distribution(n, m, T, rng, shape=shape)
    m_expected = np.mean(v_expected**2, axis=0)
    m_actual = np.mean(v_actual**2, axis=0)

    # moments should be close
    assert np.allclose(
        np.sqrt(m_expected), np.sqrt(m_actual), 
        atol=0.1
    )


def test_boltzman_mixture_with_zero_density():
    heating_rate = 0.01
    heating_temperature = 100.0

    m1 = 1.0
    vmax = heating_temperature / m1 * 10
    vmin = vmax * 1e-4
    v_bins = np.logspace(np.log10(vmin), np.log10(vmax), num=31, base=10)
    v_size = v_bins[1:] - v_bins[:-1]

    differential_crosssection = core.DifferentialCrossSection(
        lam=0.0, legendre_coefs=[0, 0, 0, 0, 0, 0, 0, 1]
    )
    model = core.BoltzmannMixture(
        n=1000, m1=m1, m2=1.0, differential_crosssection=differential_crosssection
    )
    result, _ = model.compute(
        heating_rate,
        heating_temperature,
        mixture=1e10,
        nsamples=1000,
        thin=1,
        burnin=1000,
    )
    vsq = np.sum(result ** 2, axis=-1)
    hist_mixture = np.histogram(vsq.ravel(), bins=v_bins)[0] / v_size

    model = core.BoltzmannLinear(
        n=1000, m1=m1, m2=1.0, differential_crosssection=differential_crosssection
    )
    result, _ = model.compute(
        heating_rate, heating_temperature, nsamples=1000, thin=1, burnin=1000
    )
    vsq = np.sum(result ** 2, axis=-1)
    hist_linear = np.histogram(vsq.ravel(), bins=v_bins)[0] / v_size

    assert np.allclose(hist_mixture, hist_linear, rtol=0.2)


def test_boltzman_dissipative():
    differential_crosssection = core.HardSphereCrossSections(lam=1.0)

    model = core.BoltzmannDissipative(
        n=1000, m=1, differential_crosssection=differential_crosssection
    )
    result, _ = model.compute(0.01, 100.0, 0.95, nsamples=1000, thin=1, burnin=5000)

    vsq = np.sum(result ** 2, axis=-1)
    '''
    import matplotlib.pyplot as plt

    _ = plt.hist(np.log10(vsq[:330].ravel()), bins=51, alpha=0.5)
    _ = plt.hist(np.log10(vsq[330:660].ravel()), bins=51, alpha=0.5)
    _ = plt.hist(np.log10(vsq[660:].ravel()), bins=51, alpha=0.5)
    plt.yscale('log')
    plt.grid()
    plt.show()
    '''

def test_boltzman_mixture():
    differential_crosssection = core.DifferentialCrossSection(
        lam=0.0, legendre_coefs=[0, 0, 0, 0, 0, 0, 0, 1]
    )

    model = core.BoltzmannMixture(
        n=1000, m1=1.0, m2=1.0, differential_crosssection=differential_crosssection
    )
    result, _ = model.compute(0.01, 100.0, 0.5, nsamples=1000, thin=1, burnin=5000)

    vsq = np.sum(result ** 2, axis=-1)

    """
    import matplotlib.pyplot as plt

    _ = plt.hist(np.log10(vsq[:330].ravel()), bins=51, alpha=0.5)
    _ = plt.hist(np.log10(vsq[330:660].ravel()), bins=51, alpha=0.5)
    _ = plt.hist(np.log10(vsq[660:].ravel()), bins=51, alpha=0.5)
    plt.yscale('log')
    plt.grid()
    plt.show()
    """


def test_boltzman_mixture2d():
    differential_crosssection = core.IsotropicCrossSections(lam=-1.0)

    model = core.BoltzmannMixture(
        n=1000, m1=1.0, m2=1.0, differential_crosssection=differential_crosssection
    )
    result, _ = model.compute(0.01, 100.0, 0.5, nsamples=2, thin=1, burnin=0, restrict_2d=True)
    assert (result[:, :, -1] == 0).all()


def test_boltzman_dissipative():
    differential_crosssection = core.IsotropicCrossSections(lam=-1.0)

    model = core.BoltzmannDissipative(
        n=1000, m=1.0, differential_crosssection=differential_crosssection
    )
    result, _ = model.compute(0.01, 100.0, 0.5, nsamples=2, thin=1, burnin=0, restrict_2d=True)
    assert (result[:, :, -1] == 0).all()


def test_boltzman_linear():
    differential_crosssection = core.DifferentialCrossSection(
        lam=0.0, legendre_coefs=[0, 0, 0, 0, 0, 0, 0, 1]
    )
    model = core.BoltzmannLinear(
        n=1000, m1=1.0, m2=2.0, differential_crosssection=differential_crosssection
    )
    result, _ = model.compute(0.1, 100.0, nsamples=1000, thin=1, burnin=1000)

    vsq = np.sum(result ** 2, axis=-1)

    """
    import matplotlib.pyplot as plt

    _ = plt.hist(np.log10(vsq.ravel()), bins=51)
    plt.yscale('log')
    plt.grid()
    plt.show()
    """
