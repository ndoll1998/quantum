import numpy as np
from typing import Optional, Tuple
from .mmd import ExpansionCoefficients

class Kinetic(object):
    """ Compute kinetic energy integral and it's derivative
        for a pair of GTOs 
    """

    @staticmethod
    def compute(
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
        Ez:ExpansionCoefficients
    ) -> float:
        """ Compute kinetic energy integral and it's for a pair of GTOs
            by Equations (100) and (116) in Helgaker and Taylor.

            Returns:
                T (float):
        """
    
        # unpack angulars
        i, k, m = A_angular
        j, l, n = B_angular
        # reshape to allow broadcasting
        alpha = A_alpha.reshape(-1, 1)
        beta = B_alpha.reshape(1, -1)
        # compute pairwise products of coefficients and normalizers
        c1c2 = A_coeff.reshape(-1, 1) * B_coeff.reshape(1, -1)
        n1n2 = A_norm.reshape(-1, 1) * B_norm.reshape(1, -1)
   
        # compute directional overlaps - actually only the expansion
        # coefficients, sqrt(pi/p) is factored out of the equation
        # and added to the final expression
        Sx = Ex.compute(i, j, 0)
        Sy = Ey.compute(k, l, 0)
        Sz = Ez.compute(m, n, 0)
        # compute kinetic terms in each direction
        # similarly to the overlaps only using the expansion coefficients here
        Tx = j * (j - 1) * Ex.compute(i, j-2, 0) - \
            2.0 * beta * (2.0 * j + 1.0) * Sx + \
            4.0 * beta**2 * Ex.compute(i, j+2, 0)
        Ty = l * (l - 1) * Ey.compute(k, l-2, 0) - \
            2.0 * beta * (2.0 * l + 1.0) * Sy + \
            4.0 * beta**2 * Ey.compute(k, l+2, 0)
        Tz = n * (n - 1) * Ez.compute(m, n-2, 0) - \
            2.0 * beta * (2.0 * n + 1.0) * Sz + \
            4.0 * beta**2 * Ez.compute(m, n+2, 0)
        # compute final value
        return -0.5 * np.sum(
            c1c2 * n1n2 * \
            (Tx * Sy * Sz + Sx * Ty * Sz + Sx * Sy * Tz) * \
            (np.pi / (alpha + beta))**1.5
        )
 
    @staticmethod
    def gradient(
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
        Ez:ExpansionCoefficients
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
        
        # compute directional overlaps - actually only the expansion
        # coefficients, sqrt(pi/p) is factored out of the equation
        # and added to the final expression
        Sx = Ex.compute(i, j, 0)
        Sy = Ex.compute(k, l, 0)
        Sz = Ex.compute(m, n, 0)

        # compute kinetic terms in each direction, similarly to the
        # overlaps only using the expansion coefficients here
        Tx = j * (j - 1) * Ex.compute(i, j-2, 0) - \
            2.0 * beta * (2.0 * j + 1.0) * Sx + \
            4.0 * beta**2 * Ex.compute(i, j+2, 0)
        Ty = l * (l - 1) * Ey.compute(k, l-2, 0) - \
            2.0 * beta * (2.0 * l + 1.0) * Sy + \
            4.0 * beta**2 * Ey.compute(k, l+2, 0)
        Tz = n * (n - 1) * Ez(m, n-2, 0) - \
            2.0 * beta * (2.0 * n + 1.0) * Sz + \
            4.0 * beta**2 * Ez(m, n+2, 0)
        
        # compute all directional overlap derivatives
        Sx_dx = Ex.deriv(i, j, 0, 1)
        Sy_dy = Ey.deriv(k, l, 0, 1)
        Sz_dz = Ez.deriv(m, n, 0, 1)

        # compute partial derivatives of kinetic terms
        Tx_dx = j * (j - 1) * Ex.deriv(i, j-2, 0, 1) - \
            2.0 * beta * (2.0 * j + 1.0) * Ex.deriv(i, j, 0, 1) + \
            4.0 * beta**2 * Ex.deriv(i, j+2, 0, 1)
        Ty_dy = l * (l - 1) * Ey.deriv(k, l-2, 0, 1) - \
            2.0 * beta * (2.0 * l + 1.0) * Ey.deriv(k, l, 0, 1) + \
            4.0 * beta**2 * Ey.deriv(k, l+2, 0, 1)
        Tz_dz = n * (n - 1) * Ez.deriv(m, n-2, 0, 1) - \
            2.0 * beta * (2.0 * n + 1.0) * Ez.deriv(m, n, 0, 1) + \
            4.0 * beta**2 * Ez.deriv(m, n+2, 0, 1)

        # build gradient
        dT_dAx = np.stack((
            (Tx_dx * Sy * Sz + Sx_dx * Ty * Sz + Sx_dx * Sy * Tz),
            (Tx * Sy_dy * Sz + Sx * Ty_dy * Sz + Sx * Sy_dy * Tz),
            (Tx * Sy * Sz_dz + Sx * Ty * Sz_dz + Sx * Sy * Tz_dz)
        ), axis=0)
        # scale and multiply with integral of hermitian 
        dT_dAx = c1c2 * n1n2 * dT_dAx * (np.pi / (alpha + beta)) ** 1.5
        dT_dAx = -0.5 * np.sum(dT_dAx, axis=(1, 2))
        # return gradients
        return dT_dAx, -dT_dAx
