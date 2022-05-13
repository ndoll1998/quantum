import numpy as np
from abc import ABC, abstractmethod
from functools import reduce
from typing import Union

class WaveFunction(ABC):
    """ Abstract Wave Function describing the state of an quantum system.
        Inheriting classes must overwrite the abstract method `ti_psi`.

        Args:
            E (float): total energy of the corresponding quantum state
    """

    def __init__(self, E:float) -> None:
        # save total energy
        self.E = E

    @abstractmethod
    def ti_psi(self, x:np.ndarray) -> np.ndarray:
        """ Evaluate the Time-Indipendent component of the wave function
            at the given positions.

            Args:
                x (np.ndarray): 
                    the positions in the shape of (..., ndim) at which to 
                    evaluate the function. `ndim` refers to the number
                    of spacial dimensions of the wave function.
            
            Returns:
                y (np.ndarray):
                    the amplitude of the wave function at the given inputs.
        """
        raise NotImplementedError()
    
    def ti_gradient(self, x:np.ndarray, dx:Union[float, np.ndarray]=1e-5) -> np.ndarray:
        """ Approximate the spatial gradient of the time-indipendent component of 
            the wave function at some given position using finite differences.

            Args:
                x (np.ndarray): 
                    the positions in the shape of (..., ndim) at which to 
                    approximate the gradient.
                dx (Union[float, np.ndarray]):
                    distance used by finite difference gradient approximation.
                    Either given as float of as array of shape (ndim,) where each
                    entry defines the distance value for the corresponding dimension.

            Returns:
                dydx (np.ndarray):
                    the spatial gradient at the given inputs.
        """
        # get the number of dimensions
        ndim = x.shape[-1]
        # apply difference to each dimension in the input
        # separately and in both directions
        x = np.stack((
            np.expand_dims(x, -2) + np.eye(ndim) * dx,
            np.expand_dims(x, -2) - np.eye(ndim) * dx
        ), axis=0)
        # evaluate wave function and approximate gradient
        y = self.ti_psi(x)
        dydx = (y[0, ...] - y[1, ...]) / (2.0 * dx)
        # return gradient approximation
        return dydx

    def psi(self, x:np.ndarray, t:Union[float, np.ndarray]) -> np.ndarray:
        """ Evaluate the time-dependent wave function at given position and time 

            Args:
                x (np.ndarray): 
                    the positions in the shape of (..., ndim) at which to 
                    evaluate the function. `ndim` refers to the number
                    of spacial dimensions of the wave function.
                t (Union[float, np.ndarray]):
                    the time at which to evaluate the wave function.
                    Either given as single float which applies to all
                    positions or given as an array matching the shape
                    of x but lacking the very last dimension as time 
                    is one-dimensional.

            Returns:
                y (np.ndarray):
                    the time-dependent wave function values at the given
                    position and time.
        """
        return self.ti_psi(x) * np.exp(-1j * self.E * t)

    def gradient(self, x:np.ndarray, t:Union[float, np.ndarray]) -> np.ndarray:
        """ Evaluate the time-dependent spatial gradient at given position and time 

            Args:
                x (np.ndarray): 
                    the positions in the shape of (..., ndim) at which to 
                    evaluate the gradient.
                t (Union[float, np.ndarray]):
                    the time at which to evaluate the wave function.
                    Either given as single float which applies to all
                    positions or given as an array.
        """
        return self.ti_gradient(x) * np.exp(-1j * self.E * t)
    
    def pdf(self, x:np.ndarray, t:float) -> np.ndarray:
        """ Evaluate the probability density values of particle being
            at a specific position at a specific time. Note that the density
            values are actual probabilities only if the wave function is
            normalized.
            
            Args:
                x (np.ndarray): 
                    the positions in the shape of (..., ndim) at which to 
                    compuate the density values.
                t (Union[float, np.ndarray]):
                    the time at which to evaluate the density values.
            Returns:
                pdf (np.ndarray):
                    Density values evaluated at given inputs
        """
        # evaluate time-dependent wave function
        y = self.psi(x, t)
        # compute squared absolute value of complex output
        return (y * y.conjugate()).real

    def __call__(self, x:np.ndarray, t:Union[float, np.ndarray]) -> np.ndarray:
        """ Shorthand to evaluate the time-dependent wave function. 
            For more information see `WaveFunction.psi`
        """
        return self.psi(x, t)

    def __mul__(self, other:Union[float, "WaveFunction"]) -> Union["SuperPositionAll", "ScaledWaveFunction"]:
        """ Multiply wave function by either a float to scale it or another wave 
            function creating an superposition of both states simultaneously
        """
        return SuperPositionAll(self, other) if isinstance(other, WaveFunction) else ScaledWaveFunction(self, other)

    def __rmul__(self, other:Union[float, "WaveFunction"]) -> Union["SuperPositionAll", "ScaledWaveFunction"]:
        """ See `WaveFunction.__mul__` """
        return self.__mul__(other)
   
    def __add__(self, other:"WaveFunction") -> "SuperPositionAny":
        """ Add a wave function creating a superposition representing the either one of both states """
        return SuperPositionAny(self, other)


class ScaledWaveFunction(WaveFunction):
    """ Scaled Wave Function scaling the amplitude of a given wave function 
        by a constant factor. Note that the resulting wave function is no
        longer normalized and usually only represent intermediate states
        which are combined to superpositions.

        Args:
            wave (WaveFunction): the parent wave function to scale
            alpha (float): the scaling factor
    """

    def __init__(
        self,
        wave:WaveFunction,
        alpha:float
    ) -> None:
        # save wave and scale value
        self.wave = wave
        self.alpha = alpha

    def ti_psi(self, x:np.ndarray) -> np.ndarray:
        return self.alpha * self.wave.ti_psi(x)

    def ti_gradient(self, x:np.ndarray) -> np.ndarray:
        return self.alpha * self.wave.ti_gradient(x)
    
    def psi(self, x:np.ndarray, t:Union[float, np.ndarray]) -> np.ndarray:
        return self.alpha * self.wave.psi(x, t)

    def gradient(self, x:np.ndarray, t:Union[float, np.ndarray]) -> np.ndarray:
        return self.alpha * self.wave.gradient(x, t)
 

class SuperPositionAll(WaveFunction):
    """ Superposition of the states represented by the given wave functions. The superposition
        represents the state in which the underlying quantum system is in all given states simulateously.
        Note that the resulting wave function is only normalized if all input waves are normalized.

        Args:
            *waves (WaveFunction): the wave functions over which to generate the superposition
    """

    def __init__(self,
        *waves:WaveFunction
    ) -> None:
        # make sure all waves are of the same type
        assert all(isinstance(w, type(waves[0])) for w in waves[1:]), "All waves must be of the same type"
        self.waves = waves

    def ti_psi(self, x:np.ndarray) -> np.ndarray:
        return reduce(np.mul, (wave.ti_psi(x) for wave in self.waves))

    def ti_gradient(self, x:np.ndarray) -> np.ndarray:
        # compute all values and gradients
        values = [wave.ti_psi(x) for wave in self.waves]
        grads = [wave.ti_gradient(x) for wave in self.waves]
        # compute overall gradient by product rule
        return sum([
            g * reduce(np.mul, values[:i] + values[i+1:])
            for i, g in enumerate(grads)
        ])
    
    def psi(self, x:np.ndarray, t:Union[float, np.ndarray]) -> np.ndarray:
        return reduce(np.mul, (wave.psi(x, t) for wave in self.waves))

    def gradient(self, x:np.ndarray, t:Union[float, np.ndarray]) -> np.ndarray:
        # compute all values and gradients
        values = [wave.psi(x, t) for wave in self.waves]
        grads = [wave.gradient(x, t) for wave in self.waves]
        # compute overall gradient by product rule
        return sum([
            g * reduce(np.mul, values[:i] + values[i+1:])
            for i, g in enumerate(grads)
        ])


class SuperPositionAny(WaveFunction):
    """ Superposition of the states represented by the given wave functions. The superposition
        represents the state in which the underlying quantum system is in either one of the given states.
        Note that the resulting wave function is only normalized if the input waves are scaled accordingly.

        Args:
            *waves (WaveFunction): the wave functions over which to generate the superposition
    """

    def __init__(self,
        *waves:WaveFunction
    ) -> None:
        assert all(isinstance(w, type(waves[0])) for w in waves[1:]), "All waves must be of the same type"
        self.waves = waves

    def ti_psi(self, x:np.ndarray) -> np.ndarray:
        return sum(wave.ti_psi(x) for wave in self.waves)

    def ti_gradient(self, x:np.ndarray) -> np.ndarray:
        return sum(wave.ti_gradient(x) for wave in self.waves)
    
    def psi(self, x:np.ndarray, t:Union[float, np.ndarray]) -> np.ndarray:
        return sum(wave.psi(x, t) for wave in self.waves)

    def gradient(self, x:np.ndarray, t:Union[float, np.ndarray]) -> np.ndarray:
        return sum(wave.gradient(x, t) for wave in self.waves)
