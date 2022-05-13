import numpy as np

class LegendreFunction(object):
    """ Naive Implementation of the Legendre Function for integer degrees

        Args:
            m (int): Order
            v (int): Degree
    """

    def __init__(self, m:int, v:int) -> None:
        self.m = m
        self.lgdr = np.polynomial.legendre.Legendre([0] * v + [1]).deriv(m)
        self.lgdr_dx = np.polynomial.legendre.Legendre([0] * v + [1]).deriv(m+1)

    def __call__(self, x:np.ndarray) -> np.ndarray:
        """ Evaluate the legendre function at specific position

            Args:
                x (np.ndarray): position at which to evaluate the function

            Returns:
                y (np.ndarray): the function values at the given positions
        """
        return (-1)**self.m * self.lgdr(x) * (1 - x*x)**(self.m/2.0)

    def deriv(self, x:np.ndarray) -> np.ndarray:
        """ Compute the gradient of the legendre function at given position 

            Args:
                x (np.ndarray): position at which to evaluate the gradient

            Returns:
                dydx (np.ndarray): the gradient evaluated at the specific position
        """
        y = 1.0 - x*x
        return (-1)**self.m * (
            self.lgdr_dx(x) * y**(self.m/2.0) - \
            self.lgdr(x) * self.m * y**(self.m/2.0-1) * x
        )
