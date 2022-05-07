import numpy as np
from typing import Union
from quantum.core.wave import WaveFunction

class SquareWellPotential(object):
    """ (Infinite) Square Well Potential Energy Landscape 
        centered around the origin
        
        Args:
            L (Union[float, np.ndarray]): size of the well in each dimension
            E_inf (float): Energy value used outside the well 
    """

    def __init__(
        self,
        L:Union[float, np.ndarray],
        E_inf:float =1e5
    ) -> None:
        # save values
        self.a = 0.5 * L
        self.E_inf = E_inf

    def __call__(
        self,
        x:np.ndarray
    ) -> np.ndarray:
        mask = np.abs(x) >= self.a
        mask = np.logical_or.reduce(mask, axis=-1)
        return np.where(mask, self.E_inf, 0)

class HarmonicOscillatorPotential(object):
    """ Harmonic Oscillator Potential Energy Landscape
        centered at origin

        Args:
            w (Union[float, np.ndarray]): the angulat frequency of oscillation omega (per dimension)
            m (float): mass of the particle (default is 1.0)
    """
    
    def __init__(
        self,
        w:Union[float, np.ndarray],
        m:float =1.0
    ) -> None:
        # save values
        self.w = w
        self.m = m

    def __call__(
        self,
        x:np.ndarray
    ) -> np.ndarray:
        return 0.5 * self.m * ((self.w * x)**2).sum(axis=-1)
