import numpy as np
from scipy.special import factorial

class LegendreFunction(object):
    """ Naive Implementation of the Legendre Function for integer degrees

        Args:
            m (int): Order
            v (int): Degree
    """

    def __init__(self, m:int, v:int) -> None:
        # save order and degree
        self.v = v if v >= 0 else (-v-1)
        self.m = abs(m)
        # create legendre polynomials
        self.lgdr = np.polynomial.legendre.Legendre([0] * self.v + [1]).deriv(self.m)
        self.lgdr_dx = np.polynomial.legendre.Legendre([0] * self.v + [1]).deriv(self.m+1)
        # scalar used to handle negative orders
        self.s = ((-1) ** self.m) if m >= 0 else (factorial(self.v - self.m) / factorial(self.v + self.m))

    def __call__(self, x:np.ndarray) -> np.ndarray:
        """ Evaluate the legendre function at specific position

            Args:
                x (np.ndarray): position at which to evaluate the function

            Returns:
                y (np.ndarray): the function values at the given positions
        """
        if self.m > self.v:
            # reduces to zero for order higher than degree
            return np.zeros_like(x)

        # evaluate legendre polynomial for positive order
        return self.s * self.lgdr(x) * (1 - x*x)**(self.m/2.0)

    def deriv(self, x:np.ndarray) -> np.ndarray:
        """ Compute the gradient of the legendre function at given position 

            Args:
                x (np.ndarray): position at which to evaluate the gradient

            Returns:
                dydx (np.ndarray): the gradient evaluated at the specific position
        """
        if self.m > self.v:
            # reduces to zero for order higher than degree
            return np.zeros_like(x)

        y = 1.0 - x*x
        return self.s * (
            self.lgdr_dx(x) * y**(self.m/2.0) - \
            self.lgdr(x) * self.m * y**(self.m/2.0-1) * x
        )
