import numpy as np
from quantum.core.wave import WaveFunction
from typing import Union

class DoubleSlit1D(WaveFunction):
    """ Analytic Solution to the Double Slit Setup in two-dimension. The travel dimension 
        of the particles is modeled by the time dimension. 

        Args:
            slit_dist (float): ten the two slits. Default is 1.0.
            slit_width (float): the width of the slit. Default is 0.2.
            vel_x (float): the velocity of the particle in travel direction. Default is 0.1.
    """

    def __init__(self,
        slit_dist:float =1.0,
        slit_width:float =0.2,
        vel_x:float =0.1
    ) -> None:
        # save values
        self.sd = slit_dist
        self.sw = slit_width
        self.vx = vel_x
        # initialize wave function
        super(DoubleSlit1D, self).__init__(
            E=0.5 * vel_x**2
        )

    def ti_psi(self, x:np.ndarray) -> np.ndarray:
        """ Cannot seperate time from spatial dimensions! """
        raise NotImplementedError()

    def psi(self, x:np.ndarray, t:Union[float, np.ndarray]) -> np.ndarray:
        """ Evaluate the wave function at a given position and time.

            Args:
                x (np.ndarray):
                    position of shape (..., ndim) at which to evaluate the wave function.
                    Note that the position is to be interpreted as displacement orthogonal to
                    the traveling dimension. Also only the first positional dimension will be
                    considered. All further dimensions are ignored.
                t (Union[float, np.ndarray]):
                    time at which to evaluate the wave function. Note that there is a linear
                    relation between the time t and the traveled distance of the particle along
                    the traveling dimension, i.e. x_0 = v_x * t

            Returns:
                y (np.ndarray): the wave function values at the given position and time
        """
        # take first dimension only
        x = x[..., 0]
        # compute standard deviation and normalization factor
        st = self.sw * (1 + (1j*t) / (2 * self.sw**2))
        N = (2 * np.pi * st**2)**(-0.25)
        # compute components of complex values
        A1 = -(x - self.sd)**2 / (4 * self.sw * st)  # y-component
        A2 = -(-x - self.sd)**2 / (4 * self.sw * st) # -y-component
        B = (-(self.vx * x * t) / 2.0)
        # combine
        return N * (np.exp(A1 + 1j * B) + np.exp(A2 + 1j * B))
        
    def gradient(self, x:np.ndarray, t:Union[float, np.ndarray]) -> np.ndarray:
        """ Compute the spatial gradient of the wave function at given position and time 

            Args:
                x (np.ndarray):
                    position of shape (..., ndim) at which to evaluate the gradient.
                    Note that only the first dimension is taken into account.
                t (Union[float, np.ndarray]):
                    time at which to evaluate the gradient.

            Returns:
                dydx (np.ndarray):
                    gradient of the wave function at given position and time.
        """
        # take first dimension only
        x = x[..., :1]
        # compute standard deviation and normalization factor
        st = self.sw * (1 + (1j*t) / (2 * self.sw**2))
        N = (2 * np.pi * st**2)**(-0.25)
        # compute components of complex values
        A1 = -(x - self.sd)**2 / (4 * self.sw * st)  # y-component
        A2 = -(-x - self.sd)**2 / (4 * self.sw * st) # -y-component
        B = (-(self.vx * x * t) / 2.0)
        # compute gradient
        return N * (
            (-(x - self.sd) / (2.0 * self.sw * st) + 1j * self.vx * t) * np.exp(A1 + 1j * B) + \
            ((-x - self.sd) / (2.0 * self.sw * st) + 1j * self.vx * t) * np.exp(A2 + 1j * B)
        )
