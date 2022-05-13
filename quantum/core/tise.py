import numpy as np
import scipy.sparse.linalg
from scipy import sparse
from scipy.integrate import simps
from scipy.interpolate import RegularGridInterpolator
from functools import reduce
from typing import Callable, List
from quantum.core.wave import WaveFunction

class TISEWaveFunction(WaveFunction):
    """ Wave Function representing a specific solution to the time-indipendent 
        schroedinger equation. Interpolates values on a regular grid.

        Args:
            axes (List[np.ndarray]): 
                the axes with shapes (m1,), ..., (mn) spanning the regular grid. 
                The number of axes indicates the spatial dimension of the wave 
                function.
            values (np.ndarray):
                the values on the points of the regular grid regular grid.
                Shape of values must match axes, i.e. (m1, ..., mn)
            E (float):
                the total energy of the quantum state
    """

    def __init__(
        self,
        axes:List[np.ndarray],
        values:np.ndarray,
        E:float
    ) -> None:
        # initialize wave function
        super(TISEWaveFunction, self).__init__(E=E)

        # normalize values
        N2 = (values * values.conjugate()).real
        for xi in reversed(axes):
            N2 = simps(N2, x=xi, axis=-1)
        values /= np.sqrt(N2)

        # create regular grid interpolator
        # boundary-conditions assume that psi vanished 
        # outside of grid thus set fill_value to zero
        self.ti_psi_interp = RegularGridInterpolator(
            points=axes,
            values=values,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        # compute the positional gradient of the wave function
        gradient = []
        for i, n in enumerate(values.shape):
            # get the grid distance along current dimension
            dx = axes[i][1] - axes[i][0]
            # build derivative operator
            d = np.ones(n) / (2*dx)
            D = sparse.spdiags([-d, d], [-1, 1], n, n)
            # apply operator along i-th dimension
            g = (D @ values.swapaxes(i, 0)).swapaxes(i, 0)
            gradient.append(g)
        # stack derivatives to form gradient
        gradient = np.stack(gradient, axis=-1)

        # create a linear interpolator for the gradient
        self.ti_gradient_interp = RegularGridInterpolator(
            points=axes,
            values=gradient,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

    def ti_psi(self, x:np.ndarray) -> np.ndarray:
        """ Approximate the wave amplitude at given positions using 
            linear interpolation over regular grid.

            Args:
                x (np.ndarray): 
                    the positions of shape (..., n) at which to approximate the
                    wave function values. Note that the dimensions n must match
                    the number of axes spanning the regular grid
            
            Returns:
                y (np.ndarray): the approximative wave amplitudes at the given inputs
        """
        return self.ti_psi_interp(x)

    def ti_gradient(self, x:np.ndarray) -> np.ndarray:
        """ Approximate the gradient of the wave function at given positions using 
            linear interpolation over regular grid.

            Args:
                x (np.ndarray): 
                    the positions of shape (..., n) at which to approximate the
                    gradient.

            Returns:
                dydx (np.ndarray): the approximative gradients at the given inputs
        """
        return self.ti_gradient_interp(x)


class TISE(object):
    """ (multi-dimensional) Time-Indipendent Schroedinger Equation for a given potential energy function. 
        The dimensionality is limited by the dimensionality of the potential energy function.

        Args:
            V (Callable[[np.ndarray], np.ndarray]): the potential energy function
    """

    def __init__(
        self, 
        V: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        # save potential energy function
        self.V = V

    def hamiltonian(
        self,
        grid:np.ndarray,
        dx:float
    ) -> np.ndarray:
        """ Compute the approximative Hamiltonian operator using finite differences
        
            Args:
                grid (np.ndarray): n-dimensional regular grid
                dx (float): distance between points on regular grid
        """

        shape = grid.shape[:-1] 
        # compute energy potential on grid
        V = self.V(grid)
        # build hamiltonian operator
        d = np.ones(max(grid.shape[:-1]))
        D = [
            sparse.spdiags([d, -2*d, d], [-1, 0, 1], n, n) 
            for n in grid.shape[:-1]
        ]
        D = -0.5 * reduce(sparse.kronsum, D)
        return (1.0 / dx**2) * D + sparse.diags(V.flatten(), 0)

    def solve(
        self, 
        bounds:np.ndarray,
        dx:float,
        **kwargs
    ) -> List[TISEWaveFunction]:
        """ Solve the Equation as an Eigenvalue Problem and returns a list of wave functions
        
            Args:
                bounds (np.ndarray): 
                    boundaries of the regular grid on which the equation will be evaluated 
                    given as array of shape (ndim, 2) where ndim is the number of positional 
                    dimensions of the schrödinger equation. Note that ndim is only limited by
                    the potential energy function `V`.
                dx (float): distance between points on regular grid
                **kwargs (Any): 
                    extra keyword arguments passed to the eigenpair solver. 
                    See scipy.sparse.linalg.eigsh for more information.
        
            Returns:
                waves (List[WaveFunction]): 
                    A List of wave functions corresponding to the different (stable) quantum states.
                    Specific wave functions can be solved for by setting **kwargs accordingly.
        """

        # set defaults in kwargs
        kwargs['which'] = kwargs.get('which', "SM")  # smallest eigenvalues
        kwargs['return_eigenvectors'] = True         # must return eigenvectors

        bounds = np.asarray(bounds)
        # build regular grid
        axes = [np.arange(b, e, dx) for (b, e) in bounds]
        grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
        shape = grid.shape[:-1]

        # compute hamiltonian operator and solve equation
        H = self.hamiltonian(grid, dx)
        E, Y = sparse.linalg.eigsh(H, **kwargs)

        return [
            TISEWaveFunction(
                axes=axes,
                values=Y[:, i].reshape(shape),
                E=E[i]
            ) for i in range(E.shape[0])
        ]
