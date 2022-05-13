import numpy as np
from quantum.core.wave import WaveFunction

class InfiniteSquareWell1D(WaveFunction):
    """ Analytic Solution to the 1-dimensional infinte square well potential. 

        Args:
            n (int): 
                index of the energy level defining the quantum state 
                of the particle
            L (float):
                size of the square well
    """

    def __init__(self, n:int, L:float) -> None:
        self.n = n
        self.L = L
        # compute constant factors
        self.a = np.sqrt(2.0 / L)
        self.b = n * np.pi / L
        # compute analytic energy
        super(InfiniteSquareWell1D, self).__init__(
            E = n**2 * np.pi**2 / (2 * self.L**2)
        )

    def ti_psi(self, x:np.ndarray) -> np.ndarray:
        """ Compute the time-indipendent component of the wave function

            Args:
                x (np.ndarray): 
                    the position of shape (..., ndim) at which to evaluate the 
                    wave function. Note that all but the first dimension will be
                    ignored.

            Returns:
                y (np.ndarray): the amplitudes of the wave function at the given positions
        """
        x = x[..., 0]
        # check boundaties and shift x
        out_mask = np.abs(x) > 0.5 * self.L
        x = x - 0.5 * self.L
        # compute wave function and set all values
        # out of the well to zero
        y = self.a * np.sin(self.b * x)
        y[out_mask] = 0
        # multiply by time component
        return y
    
    def ti_gradient(self, x:np.ndarray) -> np.ndarray:
        """ Evaluate the (time-indipendent) spatial gradient at a given position

            Args:
                x (np.ndarray):
                    the position at which to evaluate the gradient. Again all 
                    but the first dimension are ignored.

            Returns:
                dydx (np.ndarray): the gradient at the given position

        """
        x = x[..., :1]
        # check boundaties and shift x
        out_mask = np.abs(x) > 0.5 * self.L
        x = x - 0.5 * self.L
        # compute wave function and set all values
        # out of the well to zero
        y_dx = self.a * self.b * np.cos(self.b * x)
        y_dx[out_mask] = 0
        # multiply by time component
        return y_dx

