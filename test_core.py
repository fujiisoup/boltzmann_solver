import pytest
import numpy as np
from . import core


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
    ],
)
def test_scattering_plot(diffsigma):

    # consider the beam configuration
    n = 30000
    u1 = np.zeros((n, 3))
    u1[:, -1] = 100.0
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
def test_scattering_conservation(m1, m2):
    diffsigma = core.DifferentialCrossSection(
        lam=0, legendre_coefs=[0, 0, 0, 0, 0, 0, 0, 1]
    )
    # random velicity
    n = 300
    rng = np.random.RandomState(0)
    u1 = rng.randn(n, 3)
    u2 = rng.randn(n, 3)

    v1, v2, n_collision = core.scattering(
        m1=m1,
        u1=u1,
        m2=m2,
        u2=u2,
        rng=rng,
        differential_crosssection=diffsigma,
        density=1,
        dt=1e4,
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
            target_fraction=0.3,
            dt_init=1.0,
        )
        dt_results.append(dt)

    assert np.allclose(dt_results, np.mean(dt_results), rtol=change_rate * 10)


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
        mixture=1e-10,
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


def test_boltzman_nonlinear():
    differential_crosssection = core.DifferentialCrossSection(
        lam=0.0, legendre_coefs=[0, 0, 0, 0, 0, 0, 0, 1]
    )
    model = core.BoltzmannNonlinear(
        n=1000, m=1.0, differential_crosssection=differential_crosssection
    )
    result, _ = model.compute(0.01, 100.0, 0.05, nsamples=1000, thin=1, burnin=5000)

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
