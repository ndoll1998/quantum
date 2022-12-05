import pytest
import numpy as np
from quantum.chemistry.integrals.mmd import (
    ExpansionCoefficients,
    HermiteIntegrals
)

class TestExpansionCoefficients(object):

    @pytest.mark.parametrize('Q', np.linspace(-1, 1, 10))
    @pytest.mark.parametrize('alpha', np.linspace(0.01, 0.5, 5))
    @pytest.mark.parametrize('beta', np.linspace(0.01, 0.5, 5))
    @pytest.mark.parametrize('i', range(0, 3))
    @pytest.mark.parametrize('j', range(0, 3))
    @pytest.mark.parametrize('t', range(0, 3))
    def test_derivative(self, Q, alpha, beta, i, j, t):
        eps=1e-5
        assert np.allclose(
            # derivative
            ExpansionCoefficients(alpha, beta, Q).deriv(i, j, t, 1),
            # finite difference approximation of derivative
            (
                ExpansionCoefficients(alpha, beta, Q+eps).compute(i, j, t) - \
                ExpansionCoefficients(alpha, beta, Q-eps).compute(i, j, t)
            ) / (2*eps),
            # 
            atol=eps
        ), (Q, alpha, beta, i, j, t)

