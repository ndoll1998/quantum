import numpy as np
from typing import Tuple
from .mmd import ExpansionCoefficients

class Overlap(object):
    """ Compute Overlap Integral and the derivative of two GTOs """

    @staticmethod
    def compute(
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
        Ez:ExpansionCoefficients
    ) -> float:
        """ Compute the overlap integral of a pait of GTOs by
            Equations (100) and (101) in Helgaker and Taylor.
            
            Args:
            

            Returns:
                S (float)
        """

        # unpack angulars
        i, k, m = A_angular
        j, l, n = B_angular
        # compute pairwise sum of exponents
        a1a2 = A_alpha.reshape(-1, 1) + B_alpha.reshape(1, -1)
        # compute pairwise products of coefficients and normalizers
        c1c2 = A_coeff.reshape(-1, 1) * B_coeff.reshape(1, -1)
        n1n2 = A_norm.reshape(-1, 1) * B_norm.reshape(1, -1)

        return (
            # compute overlap over each pair of gaussian
            c1c2 * n1n2 * \
            Ex.compute(i, j, 0) * \
            Ey.compute(k, l, 0) * \
            Ez.compute(m, n, 0) * \
            (np.pi / a1a2) ** 1.5
        # sum up all overlaps of all gaussian pairs
        ).sum()

    @staticmethod
    def gradient(
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
        Ez:ExpansionCoefficients
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        # unpack angulars
        i, k, m = A_angular
        j, l, n = B_angular
        # compute pairwise sum of exponents
        a1a2 = A_alpha.reshape(-1, 1) + B_alpha.reshape(1, -1)
        # compute pairwise products of coefficients and normalizers
        c1c2 = A_coeff.reshape(-1, 1) * B_coeff.reshape(1, -1)
        n1n2 = A_norm.reshape(-1, 1) * B_norm.reshape(1, -1)
        
        # compute all expansion coefficients
        E = (
            Ex.compute(i, j, 0),
            Ey.compute(k, l, 0),
            Ez.compute(m, n, 0)
        )
        # compute all derivatives of the expansion coefficients
        E_dx = (
            Ex.deriv(i, j, 0, 1),
            Ey.deriv(k, l, 0, 1),
            Ez.deriv(m, n, 0, 1)
        )
        # apply product rule to compute final derivative
        dS_dAx = np.stack([
            (E_dx[0] * E[1] * E[2]),  # x-derivative
            (E[0] * E_dx[1] * E[2]),  # y-derivative
            (E[0] * E[1] * E_dx[2]),  # z-derivative
        ], axis=0)

        # scale and multiply with integral of hermitian
        # and sum over all pairs of gaussians
        dS_dAx *= c1c2 * n1n2 * (np.pi / a1a2) ** 1.5
        dS_dAx = np.sum(dS_dAx, axis=(1, 2))
        # return gradients
        return dS_dAx, -dS_dAx

