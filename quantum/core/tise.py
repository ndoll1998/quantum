import numpy as np
import scipy.sparse.linalg
from scipy import sparse
from functools import reduce
from typing import Callable, List
from quantum.core.wave import WaveFunction

class TISE(object):
    """ Time-Indipendent Schrödinger Equation """

    def __init__(
        self, 
        V: Callable[[np.ndarray], np.ndarray]
    ) -> None:
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
    ) -> List[WaveFunction]:
        """ Solve the Equation as an Eigenvalue Problem and returns a list of Wave Functions
            Args:
                bounds (np.ndarray): 
                    boundaries of the regular grid on which the equation will be evaluated 
                    given as array of shape (dim, 2) where dim is the number of positional 
                    dimensions of the schrödinger equation
                dx (float): distance between points on regular grid
                **kwargs (Any): 
                    extra keyword arguments passed to the solver. 
                    See scipy.sparse.linalg.eigsh for more information.
        
            Returns:
                psi (List[WaveFunction]): A List of k Wave Functions corresponding to the smallest k energies
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
            WaveFunction(
                x=axes,
                y=Y[:, i].reshape(shape),
                E=E[i]
            ) for i in range(E.shape[0])
        ]
