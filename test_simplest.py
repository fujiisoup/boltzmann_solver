import os
import numpy as np
import pytest
import xarray as xr
from . import simplest


def test_angular():
    rng = np.random.RandomState(0)
    v = simplest.angular_distribution(1000, 3, rng)
    assert np.allclose(np.sum(v**2, axis=-1), 1)


def test_simplest():
    model = simplest.SimplestBotlzmann(n=1000)
    result = model.compute(
        0.001, 100.0, eta=0.8, nsamples=1000, thin=1, burnin=1000)


def test_levy():
    model = simplest.Levy(n=10000)
    result = model.compute(
        heating_rate=0.0001, 
        dillute_coef=0.8, d=3, 
        nsamples=1000, thin=1, burnin=1000, mix_angular=True)

    '''
    import matplotlib.pyplot as plt

    # _ = plt.hist(np.log10(result.ravel()), bins=51)
    bins = np.logspace(-4, 1, base=10, num=101)
    bins_center = (bins[:-1] + bins[1:]) / 2
    bins_size = bins[1:] - bins[:-1]
    hist = np.histogram(np.abs(result).ravel(), bins=bins)[0]
    plt.loglog(bins_center, hist / bins_size)
    plt.grid()
    plt.show()

    print(result[-1])
    raise ValueError

    '''