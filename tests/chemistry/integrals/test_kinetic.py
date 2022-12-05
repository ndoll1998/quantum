import pytest
import numpy as np
from quantum.chemistry.integrals.kinetic import Kinetic
from quantum.chemistry.integrals.helpers import create_expansion_coefficients

class TestKinetic(object):
    
    @pytest.mark.parametrize('num_runs', range(10))
    @pytest.mark.parametrize('i', range(0, 2))
    @pytest.mark.parametrize('j', range(0, 2))
    @pytest.mark.parametrize('m', range(0, 2))
    @pytest.mark.parametrize('k', range(0, 2))
    @pytest.mark.parametrize('l', range(0, 2))
    @pytest.mark.parametrize('n', range(0, 2))
    def test_gradient(self, num_runs, i, j, m, k, l, n):

        # create some random values for the two GTOs
        A_origin = np.random.uniform(-1, 1, size=3)
        B_origin = np.random.uniform(-1, 1, size=3)
        A_alpha = np.random.uniform(0.01, 0.5, size=2)
        B_alpha = np.random.uniform(0.01, 0.5, size=2)
        A_coeff = np.random.uniform(-0.5, 0.5, size=2)
        B_coeff = np.random.uniform(-0.5, 0.5, size=2)

        # create constant kwargs dict
        kwargs = dict(
            # GTO A
            A_alpha=A_alpha,
            A_coeff=A_coeff,
            A_angular=(i, j, m),
            A_norm=np.ones(1),
            # GTO B
            B_alpha=B_alpha,
            B_coeff=B_coeff,
            B_angular=(k, l, n),
            B_norm=np.ones(1)
        ) 
        
        # create expansion coefficients
        Ex, Ey, Ez = create_expansion_coefficients(
            A_alpha=A_alpha,
            A_origin=A_origin,
            B_alpha=B_alpha,
            B_origin=B_origin
        )
        
        # compute gradient
        dT_dA, dT_dB = Kinetic.gradient(Ex=Ex, Ey=Ey, Ez=Ez, **kwargs)

        eps = 1e-5
        # perturb origin of A, note that origin is only used in
        # expansion coefficients, we perturb all dimensions but
        # as the dimensions are indipendent the perturbed coefficients
        # can be combined with the original once to mimic pertubation
        # in only a single dimension at a time
        Ex_pos_pert, Ey_pos_pert, Ez_pos_pert = create_expansion_coefficients(
            A_alpha=A_alpha,
            A_origin=A_origin + eps,
            B_alpha=B_alpha,
            B_origin=B_origin
        )
        Ex_neg_pert, Ey_neg_pert, Ez_neg_pert = create_expansion_coefficients(
            A_alpha=A_alpha,
            A_origin=A_origin - eps,
            B_alpha=B_alpha,
            B_origin=B_origin
        )
        
        dT_dA_ft = np.empty_like(dT_dA)
        # finite difference for each dimension
        for d in range(3):
            # combine perturbed and original coefficients
            # to mimic pertubation in only a single dimension
            # this way cached values can be reused for the
            # unperturbed dimensions (runs a little faster)
            T_pos = Kinetic.compute(**kwargs,
                Ex=Ex_pos_pert if d == 0 else Ex,
                Ey=Ey_pos_pert if d == 1 else Ey,
                Ez=Ez_pos_pert if d == 2 else Ez
            )
            T_neg = Kinetic.compute(**kwargs,
                Ex=Ex_neg_pert if d == 0 else Ex,
                Ey=Ey_neg_pert if d == 1 else Ey,
                Ez=Ez_neg_pert if d == 2 else Ez
            )
            # approximate partial derivative of current dimension
            # by finite difference
            dT_dA_ft[d] = (T_pos - T_neg) / (2 * eps)
        
        # compare
        assert np.allclose(dT_dA, dT_dA_ft, atol=eps)
        assert np.allclose(dT_dB, -dT_dA)
