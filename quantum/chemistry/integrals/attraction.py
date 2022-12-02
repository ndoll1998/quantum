import numpy as np
from itertools import product
from typing import Tuple
from .mmd import (
    ExpansionCoefficients,
    HermiteIntegrals
)

class ElectronNuclearAttraction(object):

    @staticmethod
    def compute(
        Z:np.ndarray,
        # GTO A
        A_coeff:np.ndarray,
        A_alpha:np.ndarray,
        A_angular:np.ndarray,
        A_norm:np.ndarray,
        # GTO B
        B_coeff:np.ndarray,
        B_alpha:np.ndarray,
        B_angular:np.ndarray,
        B_norm:np.ndarray,
        # GTO pair AB
        Ex:ExpansionCoefficients,
        Ey:ExpansionCoefficients,
        Ez:ExpansionCoefficients,
        # global (also depends on nuclei positions)
        R_PC:HermiteIntegrals
    ) -> float:
        """ Compute electron-nuclear repulsion energy integral for
            a GTO pair by Equations (199) and (204) in Helgaker
            and Taylor. 

            Args:

            Returns:
                V_en (float):
        """
        
        # unpack angulars
        i, k, m = A_angular
        j, l, n = B_angular
        # reshape to allow broadcasting
        alpha = A_alpha.reshape(-1, 1, 1)
        beta = B_alpha.reshape(1, -1, 1)
        # compute pairwise products of coefficients and normalizers
        c1c2 = A_coeff.reshape(-1, 1, 1) * B_coeff.reshape(1, -1, 1)
        n1n2 = A_norm.reshape(-1, 1, 1) * B_norm.reshape(1, -1, 1)

        # compute
        return np.sum(
            2.0 * np.pi / (alpha + beta) * c1c2 * n1n2 * \
            -Z * sum((
                Ex.compute(i, j, t)[:, :, None] * \
                Ey.compute(k, l, u)[:, :, None] * \
                Ez.compute(m, n, v)[:, :, None] * \
                R_PC.compute(t, u, v, 0)
                for t, u, v in product(range(i+j+1), range(k+l+1), range(m+n+1))
            ))
        )

    @classmethod
    def gradient(
        Z:np.ndarray,
        A_coeff:np.ndarray,
        A_alpha:np.ndarray,
        A_angular:np.ndarray,
        A_norm:np.ndarray,
        B_coeff:np.ndarray,
        B_alpha:np.ndarray,
        B_angular:np.ndarray,
        B_norm:np.ndarray,
        Ex:ExpansionCoefficients,
        Ey:ExpansionCoefficients,
        Ez:ExpansionCoefficients,
        R_PC:HermiteIntegrals
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        # unpack angulars
        i, k, m = A_angular
        j, l, n = B_angular
        # reshape to allow broadcasting
        alpha = A_alpha.reshape(-1, 1)
        beta = B_alpha.reshape(1, -1)
        # compute pairwise products of coefficients and normalizers
        c1c2 = A_coeff.reshape(-1, 1) * B_coeff.reshape(1, -1)
        n1n2 = A_norm.reshape(-1, 1) * B_norm.reshape(1, -1)

        # compute all expansion coefficients
        E = (
            np.stack([Ex.compute(i, j, t) for t in range(i+j+1)], axis=0),
            np.stack([Ey.compute(k, l, u) for u in range(k+l+1)], axis=0),
            np.stack([Ez.compute(m, n, v) for v in range(m+n+1)], axis=0)
        )
        # compute all derivatives of expansion coefficients
        E_dx = (
            np.stack([Ex.deriv(i, j, t, 1) for t in range(i+j+1)], axis=0),
            np.stack([Ey.deriv(k, l, u, 1) for u in range(k+l+1)], axis=0),
            np.stack([Ez.deriv(m, n, v, 1) for v in range(m+n+1)], axis=0)
        )
        
        # reshape to compute outer products by broadcasting
        E = (
            E[0][:, None, None, ...],
            E[1][None, :, None, ...],
            E[2][None, None, :, ...],
        )
        E_dx = (
            E_dx[0][:, None, None, ...],
            E_dx[1][None, :, None, ...],
            E_dx[2][None, None, :, ...],
        )
        
        # compute all auxiliary hermite integras
        R = np.stack([
                R_PC.compute(t, u, v, 0)
                for t, u, v in product(range(i+j+2), range(k+l+2), range(m+n+2))
        ], axis=0)
        R = R.reshape(i+j+2, k+l+2, m+n+2, alpha.shape[0], beta.shape[1], C.shape[0])
        
        # get the derivatives from the hermite integrals
        R_dx = R[1:, :-1, :-1, ...]
        R_dy = R[:-1, 1:, :-1, ...]
        R_dz = R[:-1, :-1, 1:, ...]
        # get the hermite integrals
        R = R[:-1, :-1, :-1, ...]

        # compute the gradient w.r.t (A - B)
        dV_dR = np.stack([
            E_dx[0] * E[1] * E[2] * R,
            E[0] * E_dx[1] * E[2] * R,
            E[0] * E[1] * E_dx[2] * R,
        ], axis=0)
        # compute the gradient w.r.t. P
        dV_dP = np.stack([
            E[0] * E[1] * E[2] * R_dx,
            E[0] * E[1] * E[2] * R_dy,
            E[0] * E[1] * E[2] * R_dz
        ], axis=0)

        # prefactor
        f = 2.0 * np.pi / (alpha + beta) * c1c2 * n1n2 * -Z
        # combine to obtain gradient w.r.t A and B
        return (
            np.sum(f * (alpha / (alpha + beta) * dV_dP + dV_dR), axis=tuple(range(1, 7))),
            np.sum(f * (beta  / (alpha + beta) * dV_dP - dV_dR), axis=tuple(range(1, 7)))
        )
