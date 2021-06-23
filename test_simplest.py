import os
import numpy as np
import pytest
import xarray as xr
from . import simplest


def test_simplest():
    model = simplest.SimplestBotlzmann(n=1000)
    result = model.compute(
        0.001, 100.0, eta=0.8, nsamples=1000, thin=1, burnin=1000)

    import matplotlib.pyplot as plt

    # _ = plt.hist(np.log10(result.ravel()), bins=51)
    _ = plt.hist(result.ravel(), bins=51)
    plt.yscale('log')
    plt.grid()
    plt.show()
