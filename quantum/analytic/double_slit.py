import numpy as np
from quantum.core.wave import WaveFunction

class DoubleSlit1D(WaveFunction):

    def __init__(self,
        slit_dist:float =1.0,
        slit_width:float =0.2,
        vel_x:float =0.1
    ) -> None:
        # save values
        self.sd = slit_dist
        self.sw = slit_width
        self.vx = vel_x
        
    def __call__(self, x, t) -> np.ndarray:
        # make sure input is 1d
        assert x.shape[-1] == 1
        x = x[..., 0]
        # compute standard deviation and normalization
        st = self.sw * (1 + (1j*t) / (2 * self.sw**2))
        N = (2 * np.pi * st**2)**(-0.25)
        # compute components of complex
        A1 = -(x - self.sd)**2 / (4 * self.sw * st)  # y-component
        A2 = -(-x - self.sd)**2 / (4 * self.sw * st) # -y-component
        B = (-(self.vx * x * t) / 2.0)
        # combine
        return N * (np.exp(A1 + 1j * B) + np.exp(A2 + 1j * B))
        
    def gradient(self, x, t) -> np.ndarray:
        # compute standard deviation and normalization
        st = self.sw * (1 + (1j*t) / (2 * self.sw**2))
        N = (2 * np.pi * st**2)**(-0.25)
        # compute components of complex
        A1 = -(x - self.sd)**2 / (4 * self.sw * st)  # y-component
        A2 = -(-x - self.sd)**2 / (4 * self.sw * st) # -y-component
        B = (-(self.vx * x * t) / 2.0)
        # compute gradient
        return N * (
            (-(x - self.sd) / (2.0 * self.sw * st) + 1j * self.vx * t) * np.exp(A1 + 1j * B) + \
            ((-x - self.sd) / (2.0 * self.sw * st) + 1j * self.vx * t) * np.exp(A2 + 1j * B)
        )
