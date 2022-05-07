import numpy as np
from scipy import sparse
from scipy.integrate import simps
from scipy.interpolate import RegularGridInterpolator
from functools import reduce
from typing import Sequence, Union

class WaveFunction(object):
    """ Wave Function represeting a specefic solution to the Schrödinger Equation 
        Args:
            x (List[np.ndarray]): Axes of the regular grid on which the equation is evaluated
            y (np.ndarray): Values on the regular grid
            E (float): total energy corresponding to the wave function
    """

    def __init__(
        self, 
        x:Sequence[np.ndarray],
        y:np.ndarray,
        E:float
    ) -> None:
        # save total energy
        self.E = E
        
        # normalize wave function
        N2 = (y * y.conjugate()).real
        for xi in reversed(x):
            N2 = simps(N2, x=xi, axis=-1)
        y /= np.sqrt(N2)
        
        # create regular grid interpolator
        # boundary-conditions assume that psi vanished 
        # outside of grid thus set fill_value to zero
        self.psi = RegularGridInterpolator(
            points=x,
            values=y,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

        # compute the positional gradient of the wave function
        gradient = []
        for i, n in enumerate(y.shape):
            # get the grid distance along current dimension
            dx = x[i][1] - x[i][0]
            # build derivative operator
            d = np.ones(n) / (2*dx)
            D = sparse.spdiags([-d, d], [-1, 1], n, n)
            # apply operator along i-th dimension
            g = (D @ y.swapaxes(i, 0)).swapaxes(i, 0)
            gradient.append(g)
        # stack derivatives to form gradient
        gradient = np.stack(gradient, axis=-1)

        # create a linear interpolator for the gradient
        self.psi_dx = RegularGridInterpolator(
            points=x,
            values=gradient,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

    def __call__(
        self,
        x:np.ndarray,
        t:float
    ) -> np.ndarray:
        """ Evaluate Wave Function at a given time 
            Args:
                x (np.ndarray): 
                    inputs of shape (n, ndim) at which to approximate
                    the wave function using linear interpolation
                t (Union[float, np.ndarray]): 
                    Time at which to evaluate the wave function.
                    Expects either a single float of an array of shape (n,) 
                    or broadcastable with shape (n,)
            Returns:
                psi (np.ndarray):
                    Wave function values evaluated at given inputs
        """
        return self.psi(x) * np.exp(-1j * self.E * t)

    def pdf(
        self,
        x:np.ndarray,
        t:float
    ) -> np.ndarray:
        """ Evaluate the probability of at a given position and time 
            Args:
                x (np.ndarray): position of shape (n, ndim)
                t (Union[float, np.ndarray]): 
                    Time at which to evaluate the wave function.
                    Expects either a single float of an array of shape (n,) 
                    or broadcastable with shape (n,)
            Returns:
                pdf (np.ndarray):
                    Density values evaluated at given inputs
        """
        y = self(x, t)
        return (y * y.conjugate()).real


    def gradient(
        self,
        x:np.ndarray,
        t:float
    ) -> np.ndarray:
        """ Approximates the gradient using finite differences 
            Args:
                x (np.ndarray): 
                    inputs of shape (n, ndim) at which to approximate
                    the gradient using linear interpolation
                t (Union[float, np.ndarray]): 
                    Time at which to evaluate the wave function.
                    Expects either a single float of an array of shape (n,)
            Returns:
                ddx_psi (np.ndarray):
                    Values of the gradient of the wave function at given inputs
        """
        return self.psi_dx(x) * np.exp(-1j * self.E * t)

    def __mul__(self, other:Union[float, "WaveFunction"]) -> Union["SuperPositionAll", "ScaledWaveFunction"]:
        if isinstance(other, WaveFunction):
            return SuperPositionAll(self, other)
        else:
            # return scaled wave function with scale alpha
            return ScaledWaveFunction(self, other)

    def __rmul__(self, other:Union[float, "WaveFunction"]) -> Union["SuperPositionAll", "ScaledWaveFunction"]:
        # return scaled wave function with scale alpha
        return self.__mul__(other)
   
    def __add__(self, other:"WaveFunction") -> "SuperPositionAny":
        # return superposition of both wave functions
        return SuperPositionAny(self, other)

class ScaledWaveFunction(WaveFunction):

    def __init__(
        self,
        wave:WaveFunction,
        alpha:float
    ) -> None:
        # save wave and scale value
        self.wave = wave
        self.alpha = alpha

    def __call__(self, x, t):
        return self.alpha * self.wave(x, t)

    def gradient(self, x, t):
        return self.alpha * self.wave.gradient(x, t)
 
class SuperPositionAll(WaveFunction):
    """ Describes a single particle in all states given by the wave functions.
        The resulting wave function is only normalized if all input wave functions are normalized.
    """    

    def __init__(self,
        *waves:WaveFunction
    ) -> None:
        # save waves
        self.waves = waves

    def __call__(self, x, t):
        return reduce(np.mul, (wave(x, t) for wave in self.waves))

    def gradient(self, x, t):
        # compute all values and gradients
        values = [wave(x, t) for wave in self.waves]
        grads = [wave.gradient(x, t) for wave in self.waves]
        # compute overall gradient by product rule
        return sum([
            g * reduce(np.mul, values[:i] + values[i+1:])
            for i, g in enumerate(grads)
        ])

class SuperPositionAny(WaveFunction):
    """ Describes a single particle in any of the states given by their wave function.
        The resulting wave function is only normalized if the input wave functions are scaled accordingly.
    """

    def __init__(self,
        *waves:WaveFunction
    ) -> None:
        # normalize superposition wave function
        self.waves = waves

    def __call__(self, x, t):
        return sum(wave(x, t) for wave in self.waves)

    def gradient(self, x, t):
        return sum(wave.gradient(x, t) for wave in self.waves)
