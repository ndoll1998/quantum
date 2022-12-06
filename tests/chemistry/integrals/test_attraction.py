import pytest
import numpy as np
from quantum.chemistry.integrals.attraction import ElectronNuclearAttraction
from quantum.chemistry.integrals.helpers import (
    create_expansion_coefficients,
    create_R_PC
)

class TestAttraction(object):
    
    @pytest.mark.parametrize('num_runs', range(10))
    @pytest.mark.parametrize('i', range(0, 2))
    @pytest.mark.parametrize('j', range(0, 2))
    @pytest.mark.parametrize('m', range(0, 2))
    @pytest.mark.parametrize('k', range(0, 2))
    @pytest.mark.parametrize('l', range(0, 2))
    @pytest.mark.parametrize('n', range(0, 2))
    def test_gradient(self, num_runs, i, j, m, k, l, n):
        
        # number of atoms
        n_atoms = np.random.randint(1, 10)
        # create random atom charges and positions
        Z = np.random.randint(0, 10, size=n_atoms)
        C = np.random.uniform(-1, 1, size=(n_atoms, 3))
        # create some random values for the two GTOs
        A_origin = np.random.uniform(-1, 1, size=3)
        B_origin = np.random.uniform(-1, 1, size=3)
        A_alpha = np.random.uniform(0.01, 0.5, size=2)
        B_alpha = np.random.uniform(0.01, 0.5, size=2)
        A_coeff = np.random.uniform(-0.5, 0.5, size=2)
        B_coeff = np.random.uniform(-0.5, 0.5, size=2)

        # create constant kwargs dict
        kwargs = dict(
            Z=Z,
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
        # create hermit integral instance
        R_PC = create_R_PC(
            C=C,
            A_origin=A_origin,
            A_alpha=A_alpha,
            B_origin=B_origin,
            B_alpha=B_alpha
        )        

        # compute gradient
        dV_dA, dV_dB = ElectronNuclearAttraction.gradient(Ex=Ex, Ey=Ey, Ez=Ez, R_PC=R_PC, **kwargs)

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
        
        dV_dA_ft = np.empty_like(dV_dA)
        # finite difference for each dimension
        for d in range(3):
            dx = np.eye(3)[d, :] * eps
            # create perturbed hermite integrals
            # note the quiet major approximation that becomes obvious
            # here, i.e. the atom positions are assumed constant (that
            # is not influencing the gradient) even though there is a
            # strong relation between atom positions and oribal origins
            # TODO: is this a result of the born-oppenheimer approximation
            R_PC_pos_pert = create_R_PC(
                C=C,
                A_origin=A_origin + dx,
                A_alpha=A_alpha,
                B_origin=B_origin,
                B_alpha=B_alpha
            )
            R_PC_neg_pert = create_R_PC(
                C=C,
                A_origin=A_origin - dx,
                A_alpha=A_alpha,
                B_origin=B_origin,
                B_alpha=B_alpha
            )
            # combine perturbed and original coefficients
            # to mimic pertubation in only a single dimension
            # this way cached values can be reused for the
            # unperturbed dimensions (runs a little faster)
            V_pos = ElectronNuclearAttraction.compute(**kwargs,
                Ex=Ex_pos_pert if d == 0 else Ex,
                Ey=Ey_pos_pert if d == 1 else Ey,
                Ez=Ez_pos_pert if d == 2 else Ez,
                R_PC=R_PC_pos_pert
            )
            V_neg = ElectronNuclearAttraction.compute(**kwargs,
                Ex=Ex_neg_pert if d == 0 else Ex,
                Ey=Ey_neg_pert if d == 1 else Ey,
                Ez=Ez_neg_pert if d == 2 else Ez,
                R_PC=R_PC_neg_pert
            )
            # approximate partial derivative of current dimension
            # by finite difference
            dV_dA_ft[d] = (V_pos - V_neg) / (2 * eps)
        
        # compare
        np.testing.assert_allclose(dV_dA, dV_dA_ft, atol=eps)
        

        # finite difference for GTO B

        # perturb origin of B
        Ex_pos_pert, Ey_pos_pert, Ez_pos_pert = create_expansion_coefficients(
            A_alpha=A_alpha,
            A_origin=A_origin,
            B_alpha=B_alpha,
            B_origin=B_origin + eps
        )
        Ex_neg_pert, Ey_neg_pert, Ez_neg_pert = create_expansion_coefficients(
            A_alpha=A_alpha,
            A_origin=A_origin,
            B_alpha=B_alpha,
            B_origin=B_origin - eps
        )
        
        dV_dB_ft = np.empty_like(dV_dB)
        # finite difference for each dimension
        for d in range(3):
            dx = np.eye(3)[d, :] * eps
            # create perturbed hermite integrals
            R_PC_pos_pert = create_R_PC(
                C=C,
                A_origin=A_origin,
                A_alpha=A_alpha,
                B_origin=B_origin + dx,
                B_alpha=B_alpha
            )
            R_PC_neg_pert = create_R_PC(
                C=C,
                A_origin=A_origin,
                A_alpha=A_alpha,
                B_origin=B_origin - dx,
                B_alpha=B_alpha
            )
            # combine perturbed and original coefficients
            # to mimic pertubation in only a single dimension
            # this way cached values can be reused for the
            # unperturbed dimensions (runs a little faster)
            V_pos = ElectronNuclearAttraction.compute(**kwargs,
                Ex=Ex_pos_pert if d == 0 else Ex,
                Ey=Ey_pos_pert if d == 1 else Ey,
                Ez=Ez_pos_pert if d == 2 else Ez,
                R_PC=R_PC_pos_pert
            )
            V_neg = ElectronNuclearAttraction.compute(**kwargs,
                Ex=Ex_neg_pert if d == 0 else Ex,
                Ey=Ey_neg_pert if d == 1 else Ey,
                Ez=Ez_neg_pert if d == 2 else Ez,
                R_PC=R_PC_neg_pert
            )
            # approximate partial derivative of current dimension
            # by finite difference
            dV_dB_ft[d] = (V_pos - V_neg) / (2 * eps)
        
        # compare
        np.testing.assert_allclose(dV_dB, dV_dB_ft, atol=eps)
    
    @pytest.mark.parametrize('num_runs', range(10))
    @pytest.mark.parametrize('i', range(0, 2))
    @pytest.mark.parametrize('j', range(0, 2))
    @pytest.mark.parametrize('m', range(0, 2))
    @pytest.mark.parametrize('k', range(0, 2))
    @pytest.mark.parametrize('l', range(0, 2))
    @pytest.mark.parametrize('n', range(0, 2))
    def test_symmetry(self, num_runs, i, j, m, k, l, n):

        # number of atoms
        n_atoms = np.random.randint(1, 10)
        # create random atom charges and positions
        Z = np.random.randint(0, 10, size=n_atoms)
        C = np.random.uniform(-1, 1, size=(n_atoms, 3))
        # create some random values for the two GTOs
        A_origin = np.random.uniform(-1, 1, size=3)
        B_origin = np.random.uniform(-1, 1, size=3)
        A_alpha = np.random.uniform(0.01, 0.5, size=2)
        B_alpha = np.random.uniform(0.01, 0.5, size=2)
        A_coeff = np.random.uniform(-0.5, 0.5, size=2)
        B_coeff = np.random.uniform(-0.5, 0.5, size=2)
        
        # create expansion coefficients
        Ex, Ey, Ez = create_expansion_coefficients(
            A_alpha=A_alpha,
            A_origin=A_origin,
            B_alpha=B_alpha,
            B_origin=B_origin
        )
        # create hermit integral instance
        R_PC = create_R_PC(
            C=C,
            A_origin=A_origin,
            A_alpha=A_alpha,
            B_origin=B_origin,
            B_alpha=B_alpha
        )        
        # compute
        V_AB = ElectronNuclearAttraction.compute(
            Z=Z,
            # GTO A
            A_alpha=A_alpha,
            A_coeff=A_coeff,
            A_angular=(i, j, m),
            A_norm=np.ones(1),
            # GTO B
            B_alpha=B_alpha,
            B_coeff=B_coeff,
            B_angular=(k, l, n),
            B_norm=np.ones(1),
            # GTO pair AB
            Ex=Ex, 
            Ey=Ey, 
            Ez=Ez,
            # global
            R_PC=R_PC
        )
        
        # swap GTOs A and B to test symmetry

        # create expansion coefficients
        Ex, Ey, Ez = create_expansion_coefficients(
            A_alpha=B_alpha,
            A_origin=B_origin,
            B_alpha=A_alpha,
            B_origin=A_origin
        )
        # create hermit integral instance
        R_PC = create_R_PC(
            C=C,
            A_origin=B_origin,
            A_alpha=B_alpha,
            B_origin=A_origin,
            B_alpha=A_alpha
        )
        # compute
        V_BA = ElectronNuclearAttraction.compute(
            Z=Z,
            # GTO A
            A_alpha=B_alpha,
            A_coeff=B_coeff,
            A_angular=(k, l, n),
            A_norm=np.ones(1),
            # GTO B
            B_alpha=A_alpha,
            B_coeff=A_coeff,
            B_angular=(i, j, m),
            B_norm=np.ones(1),
            # GTO pair AB
            Ex=Ex, 
            Ey=Ey, 
            Ez=Ez,
            R_PC=R_PC
        )

        # compare
        np.testing.assert_allclose(V_AB, V_BA)
