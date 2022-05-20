import numpy as np
from scipy.special import factorial, factorial2
from quantum.core.wave import WaveFunction
from typing import List

class GaussianOrbital(WaveFunction):
    """ Cartesian Gaussian Type Orbital (GTO) """    

    def __init__(
        self,
        alpha:np.ndarray,
        coeff:np.ndarray,
        origin:np.ndarray,
        angular:np.ndarray
    ) -> None:
        # initialize wave function
        # TODO: compute energy of gaussian orbital
        super(GaussianOrbital, self).__init__(E=None)
        # save values
        self.alpha = np.asarray(alpha)
        self.coeff = np.asarray(coeff)
        self.origin = np.asarray(origin)
        self.angular = np.asarray(angular)

        L = self.angular.sum()
        # compute normalization factor of each primitive gaussian
        self.N = (2.0 * self.alpha / np.pi) ** 0.75 * np.sqrt(
            (8.0 * self.alpha)**L * np.prod(factorial(self.angular)) / \
            np.prod(factorial(2 * self.angular))
        )
        # compute and apply normalization factor of contracted gaussian
        prefactor = np.pi**1.5 * np.prod(factorial2(2 * self.angular - 1)) / 2.0**L
        self.coeff /= np.sqrt(
            prefactor * np.sum(
                self.N.reshape(-1, 1) * self.N.reshape(1, -1) * \
                self.coeff.reshape(-1, 1) * self.coeff.reshape(1, -1) / \
                (self.alpha.reshape(-1, 1) + self.alpha.reshape(1, -1))**(L + 1.5)
            )
        )

    def ti_psi(self, x:np.ndarray) -> np.ndarray:
        # compute distance to origin
        x = x - self.origin
        R2 = np.sum(x**2, axis=-1, keepdims=True)
        # evaluate primitive gaussians and polynomial term
        g = np.exp(-self.alpha * R2)
        p = np.prod(x**self.angular, axis=-1)
        # compute weighted sum
        return p * np.sum(self.N * self.coeff * g, axis=-1)

class MolecularOrbital(WaveFunction):

    def __init__(
        self,
        coeff:np.ndarray,
        basis:List[GaussianOrbital],
        E:float
    ) -> None:
        # save coefficients and basis set
        self.coeff = coeff
        self.basis = basis
        # initialize wave function  
        super(MolecularOrbital, self).__init__(E=E)
    
    def ti_psi(self, x:np.ndarray) -> np.ndarray:
        # evaluate all atomic orbitals and compute weighted sum
        y = np.stack([ao.ti_psi(x) for ao in self.basis], axis=-1)
        return np.sum(self.coeff * y, axis=-1)
