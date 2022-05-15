import numpy as np
from quantum.core.wave import WaveFunction
from typing import Union

class GaussianWavePacket(WaveFunction):
    """ Gaussian Wave Packet moving through space. Represents a free particle in 
        n-dimensional space moving with a specified velocity.

    Args:
        v (np.ndarray): 
            the velocity of the wave packet. The dimension indicates the
            dimension of the space through which the wave packet moves
        x0 (np.ndarray): 
            the initial position of the wave packet. Must match the
            shape of the velocity
        s0 (float): 
            the square width of the wave packet
    """

    def __init__(
        self,
        v:np.ndarray,
        x0:np.ndarray,
        s0:float
    ) -> None:
        # save velocity and initial values
        self.v = np.asarray(v)
        self.x0 = np.asarray(x0)
        self.s0 = s0
        # get the dimensionality
        self.ndim = 1 if len(self.v.shape) == 0 else self.v.shape[0]
        # initialize the wave function
        super(GaussianWavePacket, self).__init__(
            E=-0.5 * (self.v**2).sum()
        )

    def ti_psi(self, x:np.ndarray) -> np.ndarray:
        """ Cannot Separate spatial dimensions from time dimension! 
            Gaussian Wave Packet has no time-indipendent solution
        """
        raise NotImplementedError()

    def psi(self, x:np.ndarray, t:Union[float, np.ndarray]) -> np.ndarray:
        """ Compute the time-dependent wave function at a given position and time

            Args:
                x (np.ndarray):
                    positions of shape (..., ndim) where ndim must match with the
                    dimensionality of the specified velocity v and initial position x0
                t (Union[float, np.ndarray]):
                    time at which to evaluate the wave function

            Returns:
                y (np.ndarray): wave function values at given position and time
        """
        # check dimensionality
        assert x.shape[-1] == self.ndim, "Dimensionality mismatch"
        # compute time-dependent scale and position
        st = self.s0 * (1.0 + (1j * t) / (2.0 * self.s0**2))
        xt = self.x0 + t * self.v
        # normalization factor
        N = (2.0 * np.pi) ** (1.0/self.ndim) * st**0.5
        # compute real and imaginary parts
        real = -((x - xt)**2).sum(axis=-1) / (4.0 * self.s0 * st)
        imag = (self.v * (x - xt)).sum(axis=-1) - 0.5 * (self.v**2).sum() * t
        # combine
        return N * np.exp(real + 1j * imag)

    def gradient(self, x:np.ndarray, t:Union[float, np.ndarray]) -> np.ndarray:
        """ Compute the spatial gradient of the wave function at given position and time

            Args:
                x (np.ndarray):
                    position of shape (..., ndim) at which to evaluate the
                    gradient of the wave function
                t (Union[float, np.ndarray]):
                    time at which to evaluate the gradient

            Returns:
                dydx (np.ndarray): 
                    gradient of the wave function at given position and time
        """
        # check dimensionality
        assert x.shape[-1] == self.ndim, "Dimensionality mismatch"
        # compute time-dependent scale and position
        st = self.s0 * (1.0 + (1j * t) / (2.0 * self.s0**2))
        xt = self.x0 + t * self.v
        # normalization factor
        N = (2.0 * np.pi) ** (1.0/self.ndim) * st**0.5
        # compute real and imaginary parts
        real = -((x - xt)**2).sum(axis=-1, keepdims=True) / (4.0 * self.s0 * st)
        imag = self.v * (x - xt).sum(axis=-1, keepdims=True) - 0.5 * (self.v**2).sum() * t
        # compute gradient of both parts
        real_dx = -(x - xt) / (2.0 * self.s0 * st)
        imag_dx = self.v
        # combine to compute overall gradient
        return N * (real_dx + 1j * imag_dx) * np.exp(real + 1j * imag)
