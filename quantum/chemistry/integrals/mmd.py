import numpy as np
from scipy.special import hyp1f1
from functools import lru_cache

class ExpansionCoefficients(object):
    """ McMurphi-Davidson recursion for the compuation of the 
        Hermite Gaussian expansion coefficients and their derivatives.

        Args:
            alpha (np.ndarray):
                exponents of the gaussian a. Shape must be broadcastable
                with beta.
            beta (np.ndarray): 
                exponents of the gaussian b. Shape must be broadcastable
                with alpha.
            Q (float):
                vector pointing from origin of gaussian a to origin of gaussian b.
    """

    def __init__(
        self,
        alpha:np.ndarray,
        beta:np.ndarray,
        Q:float
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.Q = Q
        self.p = (alpha + beta)
        self.q = (alpha * beta) / self.p

    @lru_cache(maxsize=2048)
    def compute(self, i:int, j:int, t:int) -> np.ndarray:
        """ Compute the Hermite Gaussian expansion coefficients by
            Equations (73), (74) and (75) in Helgaker and Taylor.

            Args:
                i (int): orbital angular momentum number of gaussian a
                j (int): orbital angular momentum number of gaussian b
                t (int): number of nodes in the Hermite polynomial
        
            Returns:
                E (np.ndarray): 
                    the expansion coefficients for the given gaussians.
                    The shape matches the broadcast result of alpha and
                    beta.
        """

        if (t < 0) or (t > (i+j)):
            # trivial case, out of bounds
            return 0.0
        elif i == j == t == 0:
            # trivial case, return pre-exponential factor K_AB
            return np.exp(-self.q * self.Q*self.Q)
        elif j == 0:
            # decrement index i
            return 1.0 / (2.0 * self.p) * self.compute(i-1, j, t-1) \
                - self.q * self.Q / self.alpha * self.compute(i-1, j, t) \
                + (t + 1) * self.compute(i-1, j, t+1)
        else:
            # decrement index j
            return 1.0 / (2.0 * self.p) * self.compute(i, j-1, t-1) \
                + self.q * self.Q / self.beta * self.compute(i, j-1, t) \
                + (t + 1) * self.compute(i, j-1, t+1)

    @lru_cache(maxsize=1024)
    def deriv(self, i:int, j:int, t:int, n:int) -> np.ndarray:
        """ Compute the derivative of the Hermite Gaussian expansion
            coefficients by Equation (20) in 'On the evaluation of
            derivatives of Gaussian integrals' (1992) by Helgaker
            and Taylor.

            Args:
                i (int): orbital angular momentum number of gaussian a
                j (int): orbital angular momentum number of gaussian b
                t (int): number of nodes in the Hermite polynomial
                n (int): n-th derivative to take
        
            Returns:
                E_deriv (np.ndarray): 
                    derivative of the expansion coefficients for the
                    given gaussians. The shape matches the broadcast
                    result of alpha and beta.
        """

        if (t < 0) or (t > (i+j)) or (n < 0):
            # trivial case: out of bounds
            return 0.0

        if n == 0:
            # trivial case: 0-th derivative
            return self.compute(i, j, t)
        
        if i == j == t == 0:
            # decrement index n
            return -2 * self.q * (
                self.Q * self.deriv(0, 0, 0, n-1) + \
                n      * self.deriv(0, 0, 0, n-2)
            )

        if j == 0:
            # decrement index i
            return 1.0 / (2.0 * self.p) * self.deriv(i-1, j, t-1, n) - \
                self.q / self.alpha * (
                    self.Q * self.deriv(i-1, j, t, n) + \
                    n      * self.deriv(i-1, j, t, n-1)
                ) + \
                (t + 1) * self.deriv(i-1, j, t+1, n)

        else:
            # decrement index j
            return 1.0 / (2.0 * self.p) * self.deriv(i, j-1, t-1, n) + \
                self.q / self.beta * (
                    self.Q * self.deriv(i, j-1, t, n) + \
                    n      * self.deriv(i, j-1, t, n-1)
                ) + \
                (t + 1) * self.deriv(i, j-1, t+1, n)


class HermiteIntegrals(object):
    """ McMurphi-Davidson recursion for the computation of the coulomb
        auxiliary hermite integrals.

        Args:
            alpha (np.ndarray): 
                exponent of composite gaussian ab. Each value defines a
                composite gaussian. Must be of shape broadcastable with
                C and P without last dimension ndim.
            P (np.ndarray): 
                center of gaussian a. Must be of shape (..., ndim)
                broadcastable with C.
            C (np.ndarray):
                center of gaussian b. Must be of shape (..., ndim)
                broadcastable with P.
    """

    def __init__(
        self,
        alpha:np.ndarray,
        P:np.ndarray,
        C:np.ndarray
    ) -> None:
        self.alpha = alpha
        self.PC = P - C

    @lru_cache(maxsize=1024)
    def compute(self, t:int, u:int, v:int, n:int) -> np.ndarray:
        """ Compute the coulomd auxiliary hermite intergrals by
            Equations (189), (190), (191) and (192) in Helgaker
            and Taylor.
        
            Args:
                t (int): order of the derivative in x-coordinate
                u (int): order of the derivative in y-coordinate
                v (int): order of the derivative in z-coordinate
                n (int): order of the boys function
        
            Returns:
                R (np.ndarray):
                    hermite integral values of same shape as alpha.
        """

        if (t < 0) or (u < 0) or (v < 0):
            # trivial case: out of bounds
            return 0.0

        elif t == u == v == 0:
            # trivial case: evaluate boys function
            T = self.alpha * (self.PC*self.PC).sum(axis=-1)
            return (-2 * self.alpha) ** n * hyp1f1(n+0.5, n+1.5, -T) / (2.0 * n + 1.0)

        elif t == u == 0:
            # decrement index v
            return (v-1) * self.compute(t, u, v-2, n+1) + self.PC[..., 2] * self.compute(t, u, v-1, n+1)

        elif t == 0:
            # decrement index u
            return (u-1) * self.compute(t, u-2, v, n+1) + self.PC[..., 1] * self.compute(t, u-1, v, n+1)

        else:
            # decrement index t
            return (t-1) * self.compute(t-2, u, v, n+1) + self.PC[..., 0] * self.compute(t-1, u, v, n+1)


