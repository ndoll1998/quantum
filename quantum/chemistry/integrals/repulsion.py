import numpy as np
from .mmd import (
    ExpansionCoefficients,
    HermiteIntegrals
)
from itertools import product
from typing import Tuple

class ElectronElectronRepulsion(object):

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
        # GTO C
        C_coeff:np.ndarray,
        C_alpha:np.ndarray,
        C_angular:np.ndarray,
        C_norm:np.ndarray,
        # GTO D
        D_coeff:np.ndarray,
        D_alpha:np.ndarray,
        D_angular:np.ndarray,
        D_norm:np.ndarray,
        # GTO pair AB
        Ex_AB:ExpansionCoefficients,
        Ey_AB:ExpansionCoefficients,
        Ez_AB:ExpansionCoefficients,
        # GTO pair CD
        Ex_CD:ExpansionCoefficients,
        Ey_CD:ExpansionCoefficients,
        Ez_CD:ExpansionCoefficients,
        # global (i.e. depends on all GTOs)
        R_PP:HermiteIntegrals
    ) -> float:
        """ Compute the electron-electron repulsion energy integral
            for a quatuple of GTOs by Equations (199) and (205) in
            Helgaker and Taylor.

            Args:

            Returns:
                V_ee (float)
        """
        
        # unpack angulars
        i1, k1, m1 = A_angular
        j1, l1, n1 = B_angular
        # reshape to allow broadcasting
        alpha = A_alpha[:, None]
        beta = B_alpha[None, :]
        # compute pairwise products of coefficients and normalizers
        c1c2 = A_coeff[:, None] * B_coeff[None, :]
        n1n2 = A_norm[:, None] * B_norm[None, :]
        
        # unpack angulars
        i2, k2, m2 = C_angular
        j2, l2, n2 = D_angular
        # reshape to allow broadcasting
        gamma = C_alpha[:, None]
        delta = D_alpha[None, :]
        # compute pairwise products of coefficients and normalizers
        c3c4 = C_coeff[:, None] * D_coeff[None, :]
        n3n4 = C_norm[:, None] * D_norm[None, :]

        # compute composit exponents
        p1 = (alpha + beta)[:, :, None, None]
        p2 = (gamma + delta)[None, None, :, :]

        # compute repulsion
        return np.sum(
            # scaling factor
            2.0 * np.pi**2.5 / (p1 * p2 * np.sqrt(p1 + p2)) * \
            # outer summation corresping to basis elements a and b
            sum((
                (
                    c1c2 * n1n2 * \
                    Ex_AB.compute(i1, j1, t1) * \
                    Ey_AB.compute(k1, l1, u1) * \
                    Ez_AB.compute(m1, n1, v1)
                )[:, :, None, None] * \
                # inner summation corresponding to c and d
                sum((
                    c3c4 * n3n4 * (-1)**(t2+u2+v2) * \
                    Ex_CD.compute(i2, j2, t2) * \
                    Ey_CD.compute(k2, l2, u2) * \
                    Ez_CD.compute(m2, n2, v2) * \
                    R_PP.compute(t1+t2, u1+u2, v1+v2, 0)
                    for t2, u2, v2 in product(range(i2+j2+1), range(k2+l2+1), range(m2+n2+1))
                ))
                for t1, u1, v1 in product(range(i1+j1+1), range(k1+l1+1), range(m1+n1+1))
            ))
        )
    
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
        # GTO C
        C_coeff:np.ndarray,
        C_alpha:np.ndarray,
        C_angular:np.ndarray,
        C_norm:np.ndarray,
        # GTO D
        D_coeff:np.ndarray,
        D_alpha:np.ndarray,
        D_angular:np.ndarray,
        D_norm:np.ndarray,
        # GTO pair AB
        Ex_AB:ExpansionCoefficients,
        Ey_AB:ExpansionCoefficients,
        Ez_AB:ExpansionCoefficients,
        # GTO pair CD
        Ex_CD:ExpansionCoefficients,
        Ey_CD:ExpansionCoefficients,
        Ez_CD:ExpansionCoefficients,
        # global (i.e. depends on all GTOs)
        R_PP:HermiteIntegrals
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        # unpack angulars
        i1, k1, m1 = A_angular
        j1, l1, n1 = B_angular
        # reshape to allow broadcasting
        alpha = A_alpha[:, None]
        beta = B_alpha[None, :]
        # compute pairwise products of coefficients and normalizers
        c1c2 = A_coeff[:, None] * B_coeff[None, :]
        n1n2 = A_norm[:, None] * B_norm[None, :]
        
        # unpack angulars
        i2, k2, m2 = C_angular
        j2, l2, n2 = D_angular
        # reshape to allow broadcasting
        gamma = C_alpha[:, None]
        delta = D_alpha[None, :]
        # compute pairwise products of coefficients and normalizers
        c3c4 = C_coeff[:, None] * D_coeff[None, :]
        n3n4 = C_norm[:, None] * D_norm[None, :]

        # compute composit exponents
        p1 = (alpha + beta)[:, :, None, None, None]
        p2 = (gamma + delta)[None, None, :, :, None]

        # compute the scaling prefactors
        F = 2.0 * np.pi**2.5 / (p1 * p2 * np.sqrt(p1 + p2))

        # gradient w.r.t. Q ~ composite center of AB
        dV_dQ = F * (
            # outer summation corresping to basis elements a and b
            sum((
                (
                    (c1c2 * n1n2)[:, :, None] * \
                    np.stack((
                        # partial derivatives dE/dAi
                        Ex_AB.deriv(i1, j1, t1, 1) * Ey_AB.compute(k1, l1, u1) * Ez_AB.compute(m1, n1, v1),
                        Ex_AB.compute(i1, j1, t1) * Ey_AB.deriv(k1, l1, u1, 1) * Ez_AB.compute(m1, n1, v1),
                        Ex_AB.compute(i1, j1, t1) * Ey_AB.compute(k1, l1, u1) * Ez_AB.deriv(m1, n1, v1, 1)
                    ), axis=-1)
                # reshape to match (a, b, 1, 1, 3)
                )[:, :, None, None, :] * \
                # inner summation corresponding to c and d
                sum((
                    c3c4 * n3n4 * (-1)**(t2+u2+v2) * \
                    Ex_CD.compute(i2, j2, t2) * \
                    Ey_CD.compute(k2, l2, u2) * \
                    Ez_CD.compute(m2, n2, v2) * \
                    R_PP.compute(t1+t2, u1+u2, v1+v2, 0)
                    for t2, u2, v2 in product(range(i2+j2+1), range(k2+l2+1), range(m2+n2+1))
                # add spacial dimension
                ))[..., None]
                for t1, u1, v1 in product(range(i1+j1+1), range(k1+l1+1), range(m1+n1+1))
            ))
        )
        
        # gradient w.r.t. R ~ composite center of CD
        dV_dR = F * (
            # outer summation corresping to basis elements a and b
            sum((
                (
                    c1c2 * n1n2 * \
                    Ex_AB.compute(i1, j1, t1) * \
                    Ey_AB.compute(k1, l1, u1) * \
                    Ez_AB.compute(m1, n1, v1)
                )[:, :, None, None, None] * \
                # inner summation corresponding to c and d
                sum((
                    (-1)**(t2+u2+v2) * \
                    (c3c4 * n3n4)[:, :, None] * \
                    np.stack((
                        Ex_CD.deriv(i2, j2, t2, 1) * Ey_CD.compute(k2, l2, u2) * Ez_CD.compute(m2, n2, v2),
                        Ex_CD.compute(i2, j2, t2) * Ey_CD.deriv(k2, l2, u2, 1) * Ez_CD.compute(m2, n2, v2),
                        Ex_CD.compute(i2, j2, t2) * Ey_CD.compute(k2, l2, u2) * Ez_CD.deriv(m2, n2, v2, 1)
                    ), axis=-1) * \
                    R_PP.compute(t1+t2, u1+u2, v1+v2, 0)[..., None]
                    for t2, u2, v2 in product(range(i2+j2+1), range(k2+l2+1), range(m2+n2+1))
                ))
                for t1, u1, v1 in product(range(i1+j1+1), range(k1+l1+1), range(m1+n1+1))
            ))
        )

        # gradient w.r.t. P ~ composite exponent of AB
        dV_dP = F * (
            # outer summation corresping to basis elements a and b
            sum((
                (
                    c1c2 * n1n2 * \
                    Ex_AB.compute(i1, j1, t1) * \
                    Ey_AB.compute(k1, l1, u1) * \
                    Ez_AB.compute(m1, n1, v1)
                )[:, :, None, None, None] * \
                # inner summation corresponding to c and d
                sum((
                    (
                        c3c4 * n3n4 * (-1)**(t2+u2+v2) * \
                        Ex_CD.compute(i2, j2, t2) * \
                        Ey_CD.compute(k2, l2, u2) * \
                        Ez_CD.compute(m2, n2, v2)
                    # add spacial dimension
                    )[:, :, None] * \
                    np.stack((
                        # partial derivatives dR/dPi
                        R_PP.compute(t1+t2+1, u1+u2, v1+v2, 0),
                        R_PP.compute(t1+t2, u1+u2+1, v1+v2, 0),
                        R_PP.compute(t1+t2, u1+u2, v1+v2+1, 0)
                    ), axis=-1)
                    for t2, u2, v2 in product(range(i2+j2+1), range(k2+l2+1), range(m2+n2+1))
                ))
                for t1, u1, v1 in product(range(i1+j1+1), range(k1+l1+1), range(m1+n1+1))
            ))
        )

        # combine to final gradient dV/dA, dV/dB, dV/dC, dV/dD
        return (
            np.sum(A_alpha[:, None, None, None, None]/p1 * dV_dP + dV_dQ, axis=(0, 1, 2, 3)),
            np.sum(B_alpha[None, :, None, None, None]/p1 * dV_dP - dV_dQ, axis=(0, 1, 2, 3)),
            np.sum(C_alpha[None, None, :, None, None]/p2 * -dV_dP + dV_dR, axis=(0, 1, 2, 3)),
            np.sum(D_alpha[None, None, None, :, None]/p2 * -dV_dP - dV_dR, axis=(0, 1, 2, 3))
        )
