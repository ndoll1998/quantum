import numpy as np
from typing import Tuple
from .mmd import (
    ExpansionCoefficients, 
    HermiteIntegrals
)

def create_expansion_coefficients(
    A_origin:np.ndarray,
    A_alpha:np.ndarray,
    B_origin:np.ndarray,
    B_alpha:np.ndarray
) -> Tuple[
    ExpansionCoefficients,
    ExpansionCoefficients,
    ExpansionCoefficients
]:
    """ Create the expansion coefficient instances for the
        given pair of GTOs. Creates one coefficient instance
        per dimension, i.e. three in total.

        Args:
            A_origin (np.ndarray): origin of the first GTO
            A_alpha (np.ndarray): exponents of the first GTO
            B_origin (np.ndarray): origin of the second GTO
            B_alpha (np.ndarray): exponents of the second GTO

        Returns:
            Ex (ExpansionCoefficients): expansion coefficients instance for x-dimension
            Ey (ExpansionCoefficients): expansion coefficients instance for y-dimension
            Ez (ExpansionCoefficients): expansion coefficients instance for z-dimension
    """
    # unpack origins
    Ax, Ay, Az = A_origin
    Bx, By, Bz = B_origin
    # reshape to allow broadcasting
    alpha = A_alpha.reshape(-1, 1)
    beta = B_alpha.reshape(1, -1)
    # create expansion coefficient instances for each dimension
    return (
        ExpansionCoefficients(alpha, beta, Ax, Bx),
        ExpansionCoefficients(alpha, beta, Ay, By),
        ExpansionCoefficients(alpha, beta, Az, Bz)
    )


def create_R_PC(
    C:np.ndarray,
    A_origin:np.ndarray,
    A_alpha:np.ndarray,
    B_origin:np.ndarray,
    B_alpha:np.ndarray,
):
    """ Create the hermite integrals instance for the electron-nuclear
        attraction energy. Here `P` refers to the composit center of a
        gaussian pair and `C` refers a nuclei center.
        
        Args:
            C (np.ndarray): 
                the nuclei positions in units of Bohr and shape
                of (n, 3) where n refers to the number of atoms.
            A_origin (np.ndarray): origin of the first GTO
            A_alpha (np.ndarray): exponents of the first GTO
            B_origin (np.ndarray): origin of the second GTO
            B_alpha (np.ndarray): exponents of the second GTO
            
        Returns:
            R_PC (HermiteIntegrals): the hermite integrals instance
    """

    # reshape for broadcasting
    alpha = A_alpha.reshape(-1, 1, 1)
    beta = B_alpha.reshape(1, -1, 1)
    # compute all gaussian composite centers
    # here last dimension is re-used for coordinates
    P = (alpha * A_origin + beta * B_origin) / (alpha + beta)
    # add dimension for nuclei, note that last dimension of coordinates
    # is reduced in computations of R and thus the remaining last dimension
    # refers to the number of nuclei again
    P = np.expand_dims(P, -2)
    # create hermite integral instance
    return HermiteIntegrals(alpha + beta, P, C)

def create_R_PP(
    A_origin:np.ndarray,
    A_alpha:np.ndarray,
    B_origin:np.ndarray,
    B_alpha:np.ndarray,
    C_origin:np.ndarray,
    C_alpha:np.ndarray,
    D_origin:np.ndarray,
    D_alpha:np.ndarray,
) -> HermiteIntegrals:

    # reshape for broadcasting
    alpha = A_alpha.reshape(-1, 1, 1)
    beta  = B_alpha.reshape(1, -1, 1)
    gamma = C_alpha.reshape(-1, 1, 1)
    delta = D_alpha.reshape(1, -1, 1)

    # compute composit centers of GTO pairs AB and CD
    P_ab = (alpha * A_origin + beta * B_origin) / (alpha + beta)
    P_cd = (gamma * C_origin + delta * D_origin) / (gamma + delta)
    # compute composite exponents
    p1 = (alpha + beta)[:, :, None, None, 0]
    p2 = (gamma + delta)[None, None, :, :, 0]

    # create hermite integral instance
    return HermiteIntegrals(
        p1 * p2 / (p1 + p2), 
        P_ab[:, :, None, None, :], 
        P_cd[None, None, :, :, :]
    )
