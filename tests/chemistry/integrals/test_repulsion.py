import pytest
import numpy as np
from quantum.chemistry.integrals.repulsion import ElectronElectronRepulsion
from quantum.chemistry.integrals.helpers import (
    create_expansion_coefficients,
    create_R_PP
)

class TestRepulsion(object):
 
    def random_kwargs_and_origins(self):
        
        # create random GTO origins
        origins = dict(
            A_origin=np.random.uniform(-1, 1, size=3),
            B_origin=np.random.uniform(-1, 1, size=3),
            C_origin=np.random.uniform(-1, 1, size=3),
            D_origin=np.random.uniform(-1, 1, size=3)
        )
        
        # build keyword args dict
        kwargs = dict(
            # GTO A
            A_coeff=np.random.uniform(-0.5, 0.5, size=2),
            A_alpha=np.random.uniform(0.01, 0.5, size=2),
            A_angular=np.random.randint(0, 2, size=3),
            A_norm=np.ones(1),
            # GTO B
            B_coeff=np.random.uniform(-0.5, 0.5, size=2),
            B_alpha=np.random.uniform(0.01, 0.5, size=2),
            B_angular=np.random.randint(0, 2, size=3),
            B_norm=np.ones(1),
            # GTO C
            C_coeff=np.random.uniform(-0.5, 0.5, size=2),
            C_alpha=np.random.uniform(0.01, 0.5, size=2),
            C_angular=np.random.randint(0, 2, size=3),
            C_norm=np.ones(1),
            # GTO D
            D_coeff=np.random.uniform(-0.5, 0.5, size=2),
            D_alpha=np.random.uniform(0.01, 0.5, size=2),
            D_angular=np.random.randint(0, 2, size=3),
            D_norm=np.ones(1)
        )

        # expansion coefficients for GTO pair AB
        kwargs['Ex_AB'], kwargs['Ey_AB'], kwargs['Ez_AB'] = create_expansion_coefficients(
            A_origin=origins['A_origin'],
            B_origin=origins['B_origin'],
            A_alpha=kwargs['A_alpha'],
            B_alpha=kwargs['B_alpha'],
        )
        # expansion coefficients for GTO pair CD
        kwargs['Ex_CD'], kwargs['Ey_CD'], kwargs['Ez_CD'] = create_expansion_coefficients(
            A_origin=origins['C_origin'],
            B_origin=origins['D_origin'],
            A_alpha=kwargs['C_alpha'],
            B_alpha=kwargs['D_alpha'],
        )
        # hermite integrals
        kwargs['R_PP'] = create_R_PP(
            **origins,
            A_alpha=kwargs['A_alpha'],
            B_alpha=kwargs['B_alpha'],
            C_alpha=kwargs['C_alpha'],
            D_alpha=kwargs['D_alpha']
        )

        return kwargs, origins

    @pytest.mark.parametrize('num_runs', range(128))
    def test_gradient_A(self, num_runs):
        
        kwargs, origins = self.random_kwargs_and_origins()
        # compute gradient
        dVdA, _, _, _ = ElectronElectronRepulsion.gradient(**kwargs)

        eps = 1e-7
        # fininte difference for A
        dVdA_fd = np.empty_like(dVdA)
        for d, n in enumerate('xyz'):

            dx = np.eye(3)[d, :] * eps
            # positive perturbation
            kwargs_pos = kwargs.copy()
            kwargs_pos['R_PP'] = create_R_PP(
                A_origin=origins['A_origin'] + dx,
                B_origin=origins['B_origin'],
                C_origin=origins['C_origin'],
                D_origin=origins['D_origin'],
                A_alpha=kwargs['A_alpha'],
                B_alpha=kwargs['B_alpha'],
                C_alpha=kwargs['C_alpha'],
                D_alpha=kwargs['D_alpha']
            )
            kwargs_pos['E%c_AB' % n] = create_expansion_coefficients(
                A_origin=origins['A_origin'] + dx,
                B_origin=origins['B_origin'],
                A_alpha=kwargs['A_alpha'],
                B_alpha=kwargs['B_alpha'],
            )[d]

            # negative perturbation
            kwargs_neg = kwargs.copy()
            kwargs_neg['R_PP'] = create_R_PP(
                A_origin=origins['A_origin'] - dx,
                B_origin=origins['B_origin'],
                C_origin=origins['C_origin'],
                D_origin=origins['D_origin'],
                A_alpha=kwargs['A_alpha'],
                B_alpha=kwargs['B_alpha'],
                C_alpha=kwargs['C_alpha'],
                D_alpha=kwargs['D_alpha']
            )
            kwargs_neg['E%c_AB' % n] = create_expansion_coefficients(
                A_origin=origins['A_origin'] - dx,
                B_origin=origins['B_origin'],
                A_alpha=kwargs['A_alpha'],
                B_alpha=kwargs['B_alpha'],
            )[d]
            # approximate partial derivative and save to array 
            dVdA_fd[d] = (
                ElectronElectronRepulsion.compute(**kwargs_pos) - \
                ElectronElectronRepulsion.compute(**kwargs_neg)
            ) / (2.0*eps)

        # check gradient
        np.testing.assert_allclose(dVdA, dVdA_fd, atol=1e-3)
    
    @pytest.mark.parametrize('num_runs', range(128))
    def test_gradient_B(self, num_runs):
        
        kwargs, origins = self.random_kwargs_and_origins()
        # compute gradient
        _, dVdB, _, _ = ElectronElectronRepulsion.gradient(**kwargs)

        eps = 1e-7
        # fininte difference for A
        dVdB_fd = np.empty_like(dVdB)
        for d, n in enumerate('xyz'):

            dx = np.eye(3)[d, :] * eps
            # positive perturbation
            kwargs_pos = kwargs.copy()
            kwargs_pos['R_PP'] = create_R_PP(
                A_origin=origins['A_origin'],
                B_origin=origins['B_origin'] + dx,
                C_origin=origins['C_origin'],
                D_origin=origins['D_origin'],
                A_alpha=kwargs['A_alpha'],
                B_alpha=kwargs['B_alpha'],
                C_alpha=kwargs['C_alpha'],
                D_alpha=kwargs['D_alpha']
            )
            kwargs_pos['E%c_AB' % n] = create_expansion_coefficients(
                A_origin=origins['A_origin'],
                B_origin=origins['B_origin'] + dx,
                A_alpha=kwargs['A_alpha'],
                B_alpha=kwargs['B_alpha'],
            )[d]

            # negative perturbation
            kwargs_neg = kwargs.copy()
            kwargs_neg['R_PP'] = create_R_PP(
                A_origin=origins['A_origin'],
                B_origin=origins['B_origin'] - dx,
                C_origin=origins['C_origin'],
                D_origin=origins['D_origin'],
                A_alpha=kwargs['A_alpha'],
                B_alpha=kwargs['B_alpha'],
                C_alpha=kwargs['C_alpha'],
                D_alpha=kwargs['D_alpha']
            )
            kwargs_neg['E%c_AB' % n] = create_expansion_coefficients(
                A_origin=origins['A_origin'],
                B_origin=origins['B_origin'] - dx,
                A_alpha=kwargs['A_alpha'],
                B_alpha=kwargs['B_alpha'],
            )[d]
            # approximate partial derivative and save to array 
            dVdB_fd[d] = (
                ElectronElectronRepulsion.compute(**kwargs_pos) - \
                ElectronElectronRepulsion.compute(**kwargs_neg)
            ) / (2.0*eps)

        # check gradient
        np.testing.assert_allclose(dVdB, dVdB_fd, atol=1e-3)
    
    @pytest.mark.parametrize('num_runs', range(128))
    def test_gradient_C(self, num_runs):
        
        kwargs, origins = self.random_kwargs_and_origins()
        # compute gradient
        _, _, dVdC, _ = ElectronElectronRepulsion.gradient(**kwargs)

        eps = 1e-7
        # fininte difference for A
        dVdC_fd = np.empty_like(dVdC)
        for d, n in enumerate('xyz'):

            dx = np.eye(3)[d, :] * eps
            # positive perturbation
            kwargs_pos = kwargs.copy()
            kwargs_pos['R_PP'] = create_R_PP(
                A_origin=origins['A_origin'],
                B_origin=origins['B_origin'],
                C_origin=origins['C_origin'] + dx,
                D_origin=origins['D_origin'],
                A_alpha=kwargs['A_alpha'],
                B_alpha=kwargs['B_alpha'],
                C_alpha=kwargs['C_alpha'],
                D_alpha=kwargs['D_alpha']
            )
            kwargs_pos['E%c_CD' % n] = create_expansion_coefficients(
                A_origin=origins['C_origin'] + dx,
                B_origin=origins['D_origin'],
                A_alpha=kwargs['C_alpha'],
                B_alpha=kwargs['D_alpha'],
            )[d]

            # negative perturbation
            kwargs_neg = kwargs.copy()
            kwargs_neg['R_PP'] = create_R_PP(
                A_origin=origins['A_origin'],
                B_origin=origins['B_origin'],
                C_origin=origins['C_origin'] - dx,
                D_origin=origins['D_origin'],
                A_alpha=kwargs['A_alpha'],
                B_alpha=kwargs['B_alpha'],
                C_alpha=kwargs['C_alpha'],
                D_alpha=kwargs['D_alpha']
            )
            kwargs_neg['E%c_CD' % n] = create_expansion_coefficients(
                A_origin=origins['C_origin'] - dx,
                B_origin=origins['D_origin'],
                A_alpha=kwargs['C_alpha'],
                B_alpha=kwargs['D_alpha'],
            )[d]
            # approximate partial derivative and save to array 
            dVdC_fd[d] = (
                ElectronElectronRepulsion.compute(**kwargs_pos) - \
                ElectronElectronRepulsion.compute(**kwargs_neg)
            ) / (2.0*eps)

        # check gradient
        np.testing.assert_allclose(dVdC, dVdC_fd, atol=1e-3)
    
    @pytest.mark.parametrize('num_runs', range(128))
    def test_gradient_D(self, num_runs):
        
        kwargs, origins = self.random_kwargs_and_origins()
        # compute gradient
        _, _, _, dVdD = ElectronElectronRepulsion.gradient(**kwargs)

        eps = 1e-7
        # fininte difference for A
        dVdD_fd = np.empty_like(dVdD)
        for d, n in enumerate('xyz'):

            dx = np.eye(3)[d, :] * eps
            # positive perturbation
            kwargs_pos = kwargs.copy()
            kwargs_pos['R_PP'] = create_R_PP(
                A_origin=origins['A_origin'],
                B_origin=origins['B_origin'],
                C_origin=origins['C_origin'],
                D_origin=origins['D_origin'] + dx,
                A_alpha=kwargs['A_alpha'],
                B_alpha=kwargs['B_alpha'],
                C_alpha=kwargs['C_alpha'],
                D_alpha=kwargs['D_alpha']
            )
            kwargs_pos['E%c_CD' % n] = create_expansion_coefficients(
                A_origin=origins['C_origin'],
                B_origin=origins['D_origin'] + dx,
                A_alpha=kwargs['C_alpha'],
                B_alpha=kwargs['D_alpha'],
            )[d]

            # negative perturbation
            kwargs_neg = kwargs.copy()
            kwargs_neg['R_PP'] = create_R_PP(
                A_origin=origins['A_origin'],
                B_origin=origins['B_origin'],
                C_origin=origins['C_origin'],
                D_origin=origins['D_origin'] - dx,
                A_alpha=kwargs['A_alpha'],
                B_alpha=kwargs['B_alpha'],
                C_alpha=kwargs['C_alpha'],
                D_alpha=kwargs['D_alpha']
            )
            kwargs_neg['E%c_CD' % n] = create_expansion_coefficients(
                A_origin=origins['C_origin'],
                B_origin=origins['D_origin'] - dx,
                A_alpha=kwargs['C_alpha'],
                B_alpha=kwargs['D_alpha'],
            )[d]
            # approximate partial derivative and save to array 
            dVdD_fd[d] = (
                ElectronElectronRepulsion.compute(**kwargs_pos) - \
                ElectronElectronRepulsion.compute(**kwargs_neg)
            ) / (2.0*eps)

        # check gradient
        np.testing.assert_allclose(dVdD, dVdD_fd, atol=1e-3)
    
