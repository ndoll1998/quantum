import numpy as np
from quantum.core.wave import WaveFunction
from quantum.utils.rk4 import rk4

class BohmianMechanics(object):
    """ Compute Particle Trajectories following the de Broglie-Bohm theory (for spin=0) """

    def __init__(
        self,
        wave:WaveFunction
    ) -> None:
        # save wave function
        self.wave = wave

    def dxdt(self, x, t) -> np.ndarray:
        """ Computes the bohmian velocity at given query positions and time """
        # evaluate wave function and gradient
        y = self.wave(x, t)
        g = self.wave.gradient(x, t)
        # avoid devision by zero or overflows
        mask = np.abs(y) <= 1e-7
        y[mask], g[mask, :] = 1.0, 0.0
        # compute bohmian velocity
        return (g / y.reshape(-1, 1)).imag

    def trajectory(
        self,
        q:np.ndarray,
        t:np.ndarray
    ) -> np.ndarray:
        """ Compute Particle Trajectories """
        qs = [q]
        for t0, t1 in zip(t[:-1], t[1:]):
            _, q = rk4(t=t0, dt=t1-t0, x=qs[-1], f=self.dxdt)
            qs.append(q)
        # stack trajectory points to build full trajectories
        return np.stack(qs, axis=0)
