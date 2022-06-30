import numpy as np
from scipy.special import factorial, factorial2
from quantum.core.wave import WaveFunction
from typing import List

class GaussianOrbital(WaveFunction):
    """ Cartesian Gaussian Type Orbital (GTO)

        Args:
            alpha (np.ndarray): 
                the exponents of the primitive gaussians. Each value defines to 
                exactly one primitive gaussian. Must be of shape (n,).
            ceoff (np.ndarray):
                the scaling coefficients of the primitive gassians. Each value
                defines the weight of the corresponding gaussian. Must be of shape (n,).
            origin (np.ndarray):
                the origin of the orbital in units of Bohr. Must be of shape (ndim,) where
                ndim is the dimensionality of the orbital (usually ndim=3).
            angular (np.ndarray):
                the degree of the polynomial in each spatial dimension. Loosely referred
                to as the angular quantum numbers. Must be of shape (ndim,).
    """

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
        """ Evaluate the time-indipendent component of the wave GTO at a given position 

            Args:
                x (np.ndarray): 
                    the position of at which to evaluate the GTO. Must be of shape (..., ndim).

            Returns:
                y (np.ndarray):
                    the amplitudes of the wave function at the given positions 
        """
        # compute distance to origin
        x = x - self.origin
        R2 = np.sum(x*x, axis=-1, keepdims=True)
        # evaluate primitive gaussians and polynomial term
        g = np.exp(-self.alpha * R2)
        p = np.prod(x**self.angular, axis=-1)
        # compute weighted sum
        return p * np.sum(self.N * self.coeff * g, axis=-1)

class MolecularOrbital(WaveFunction):
    """ Molecular Orbital defined as a linear combination of Gaussian Type Orbitals 

        Args:
            coeff (np.ndarray): the weights of the GTOs. Must be of shape (n,) where n is the number of GTOs in the basis.
            basis (List[GaussianOrbital]): the basis build of n GTOs. Each GTO must be of the exact same spatial dimension ndim.
            E (float): Energy of the molecular orbital (usually in units of Hartree)
    """

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
        """ Evaluate the time-indipendent component of the MO 

            Args:
                x (np.ndarray):
                    the position at which to evaluate the wave function. Must be of shape (..., ndim).
            
            Returns:
                y (np.ndarray):
                    the amplitudes of the wave function at the given positions 
        """
        # evaluate all atomic orbitals and compute weighted sum
        y = np.stack([ao.ti_psi(x) for ao in self.basis], axis=-1)
        return np.sum(self.coeff * y, axis=-1)
