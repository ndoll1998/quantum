import numpy as np
from quantum.core.wave import WaveFunction

class InfiniteSquareWell1D(WaveFunction):

    def __init__(self, n:int, L:float) -> None:
        self.n = n
        self.L = L
        # compute constant factors
        self.a = np.sqrt(2.0 / L)
        self.b = n * np.pi / L
        # compute analytic energy
        self.E = n**2 * np.pi**2 / (2 * self.L**2)

    def __call__(self, x, t):
        # check dimension
        assert x.shape[-1] == 1, x.shape
        x = x[..., 0]
        # check boundaties and shift x
        out_mask = np.abs(x) > 0.5 * self.L
        x = x - 0.5 * self.L
        # compute wave function and set all values
        # out of the well to zero
        y = self.a * np.sin(self.b * x)
        y[out_mask] = 0
        # multiply by time component
        return y * np.exp(-1j * self.E * t)
    
    def gradient(self, x, t):
        # check boundaties and shift x
        out_mask = np.abs(x) > 0.5 * self.L
        x = x - 0.5 * self.L
        # compute wave function and set all values
        # out of the well to zero
        y_dx = self.a * self.b * np.cos(self.b * x)
        y_dx[out_mask] = 0
        # multiply by time component
        return y_dx * np.exp(-1j * self.E * t)

