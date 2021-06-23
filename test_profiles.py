import numpy as np
import pytest
from . import profiles


@pytest.mark.parametrize("df", [1, 2, 10])
@pytest.mark.parametrize("gamma", [0.1, 1, 10])
def test_generalized_voigt1(df, gamma):
    # compute the convolution numerically
    xmax = 300
    x = np.linspace(-xmax, xmax, num=100000)
    # semi-analytical formula
    actual = profiles.exp_pareto_conv(x, gamma=gamma, df=df)
    