import numpy as np
from quantum.core.wave import WaveFunction
from quantum.utils.rk4 import rk4

class BohmianMechanics(object):
    """ Compute Particle Trajectories following the de Broglie-Bohm theory (for spin=0) 

        Args:
            wave (WaveFunction):
                the wave function instance from which to derive the mechanical movement
                of a particle
    """

    def __init__(
        self,
        wave:WaveFunction
    ) -> None:
        # save wave function
        self.wave = wave

    def dxdt(self, x:np.ndarray, t:Union[float, np.ndarray]) -> np.ndarray:
        """ Computes the time-derivative (velocity) at given query positions and time
            following the idea of bohmian mechanics

            Args:
                x (np.ndarray): position at which to evaluate the time-derivative
                t (Union[float, np.ndarray): the time at which to evaluate the derivative

            Returns:
                dydt (np.ndarray): the time-derivative at the given position and time
        """
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
        """ Compute Particle Trajectories 

            Args:
                q (np.ndarray):
                    query positions of shape (..., ndim), i.e. initial particle positions from which the 
                    trajectories evolve
                t (np.ndarray):
                    the times at which to evaluate the trajectories.
            
            Returns:
                Q (np.ndarray):
                    the trajectories in shape of (n, ..., ndim) where n is the number of time-steps at
                    which to evaluate the trajectories
        """
        qs = [q]
        for t0, t1 in zip(t[:-1], t[1:]):
            _, q = rk4(t=t0, dt=t1-t0, x=qs[-1], f=self.dxdt)
            qs.append(q)
        # stack trajectory points to build full trajectories
        return np.stack(qs, axis=0)

    def simulate(
        self,
        q:np.ndarray,
        t0:float,
        dt:float
    ) -> iter:
        """ Iteratively compute the particle trajectories

            Args:
                q (np.ndarray):
                    query positions of shape (..., ndim), i.e. initial particle positions from which the 
                    trajectories evolve
                t0 (float):
                    the initial point in time at which to start the trajectories
                dt (float):
                    the progressive step in time per iteration

            Returns:
                Q_iter (iter): 
                    iterator yielding tuples of the form (t, q) where t is the current time and
                    q are the corresponding trajectories in shape of (..., ndim)
        """
        t = t0
        while True:
            t, q = rk4(t=t, dt=dt, x=q, f=self.dxdt)
            yield t, q
