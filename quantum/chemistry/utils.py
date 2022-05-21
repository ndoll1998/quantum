import numpy as np
from scipy.special import hyp1f1

def MMD_E(
    i:int,
    j:int,
    t:int,
    alpha:np.ndarray,
    beta:np.ndarray,
    A:float,
    B:float
) -> np.ndarray:
    """ McMurphi-Davidson recursion for the compuation of the Hermite Gaussian expansion coefficients.
        See Equations (73), (74) and (75) in Helgaker and Taylor.

        Args:
            i (int): orbital angular momentum number of gaussian a
            j (int): orbital angular momentum number of gaussian b
            t (int): number of nodes in the Hermite polynomial
            alpha (np.ndarray): exponents of the gaussian a. Shape must be broadcastable with beta.
            beta (np.ndarray): exponents of the gaussian b. Shape must be broadcastable with alpha.
            A (float): origin of gaussian a
            B (float): origin of gaussian b

        Returns:
            E (np.ndarray): 
                the expansion coefficients for the given gaussians. The shape matches the broadcast result of alpha and beta.
    """

    # TODO: dynamic programming

    Q = A - B
    p = alpha + beta
    q = alpha * beta / p
    
    if (t < 0) or (t > (i+j)):
        # trivial case, out of bounds
        return 0.0
    elif i == j == t == 0:
        # trivial case, return pre-exponential factor K_AB
        return np.exp(-q * Q*Q)
    elif j == 0:
        # decrement index i
        return 1.0 / (2.0 * p) * MMD_E(i-1, j, t-1, alpha, beta, A, B) \
            - q * Q / alpha * MMD_E(i-1, j, t, alpha, beta, A, B) \
            + (t + 1) * MMD_E(i-1, j, t+1, alpha, beta, A, B)
    else:
        # decrement index j
        return 1.0 / (2.0 * p) * MMD_E(i, j-1, t-1, alpha, beta, A, B) \
            + q * Q / beta * MMD_E(i, j-1, t, alpha, beta, A, B) \
            + (t + 1) * MMD_E(i, j-1, t+1, alpha, beta, A, B)

def MMD_R(
    t:int,
    u:int,
    v:int,
    n:int,
    alpha:np.ndarray,
    P:np.ndarray,
    C:np.ndarray
) -> np.ndarray:
    """ McMurphi-Davidson recursion for the computation of the coulomb auxiliary hermite integrals.
        See Equations (189), (190), (191) and (192) in Helgaker and Taylor.

        Args:
            t (int): order of the derivative in x-coordinate
            u (int): order of the derivative in y-coordinate
            v (int): order of the derivative in z-coordinate
            n (int): order of the boys function
            alpha (np.ndarray): 
                exponent of composite gaussian ab. Each value defines a composite gaussian.
                Must be of shape broadcastable with C and P without last dimension ndim.
            P (np.ndarray): center of gaussian a. Must be of shape (..., ndim) broadcastable with C.
            C (np.ndarray): center of gaussian b. Must be of shape (..., ndim) broadcastable with P.

        Returns:
            R (np.ndarray): hermite integral values of same shape as alpha.
    """

    # compute distance
    PC = P - C

    if (t < 0) or (u < 0) or (v < 0):
        # trivial case: out of bounds
        return 0.0

    elif t == u == v == 0:
        # trivial case: evaluate boys function
        T = alpha * (PC*PC).sum(axis=-1)
        return (-2 * alpha) ** n * hyp1f1(n+0.5, n+1.5, -T) / (2.0 * n + 1.0)

    elif t == u == 0:
        # decrement index v
        return (v-1) * MMD_R(t, u, v-2, n+1, alpha, P, C) + PC[..., 2] * MMD_R(t, u, v-1, n+1, alpha, P, C)

    elif t == 0:
        # decrement index u
        return (u-1) * MMD_R(t, u-2, v, n+1, alpha, P, C) + PC[..., 1] * MMD_R(t, u-1, v, n+1, alpha, P, C)

    else:
        # decrement index t
        return (t-1) * MMD_R(t-2, u, v, n+1, alpha, P, C) + PC[..., 0] * MMD_R(t-1, u, v, n+1, alpha, P, C)


