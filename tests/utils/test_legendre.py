import pytest
import numpy as np
from scipy.special import lpmv
from quantum.utils.legendre import LegendreFunction

class TestLegendreFunction(object):

    @pytest.mark.parametrize('m', range(-10, 10))
    @pytest.mark.parametrize('v', range(-10, 10))
    def test_legendre(self, m, v):
        x = np.linspace(-0.95, 0.95)
        # compare to lpmv
        np.testing.assert_allclose(
            LegendreFunction(m, v)(x), 
            np.nan_to_num(lpmv(m, v, x), nan=0.0)
        )

    @pytest.mark.parametrize('m', range(-10, 10))
    @pytest.mark.parametrize('v', range(-10, 10))
    def _test_legendre_derivative(self, m, v):
        x = np.linspace(-0.95, 0.95)
        f = LegendreFunction(m, v)
        # approximate gradient by finite difference
        eps = 1e-5
        dpdx = (f(x + eps) - f(x - eps)) / (2.0 * eps)
        # compare
        np.testing.assert_allclose(f.deriv(x), dpdx, atol=1e-3)
