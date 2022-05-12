import numpy as np
from quantum.core.wave import WaveFunction
from quantum.utils.legendre import LegendreFunction
from scipy.special import factorial, genlaguerre

class HydrogenLike(WaveFunction):
    """ Wave Function describing the electron of an Hydrogen-Like atom.
        Note that the wave function expects inputs in spherical coordinate system

        Args:
            n (int): principal quantum number
            l (int): azimuthal (or angular momentum) quantum number
            m (int): magnetic quantum number
    """

    def __init__(self, n:int, l:int, m:int) -> None:
        # check conditions on quantum numbers
        assert 0 <= l <= n - 1
        assert -l <= m <= l
        # save charge and quantum numbers
        self.n = n
        self.l = l
        self.m = m
        # TODO: compute total energy
        self.E = -1.0

        # compute some constants
        self.r_factor = np.sqrt((2.0/n)**3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
        self.sh_factor = (-1)**m * np.sqrt((2*l + 1) * factorial(l - m) / (4 * np.pi * factorial(l + m)))
        # polynomials
        self.laguerre = genlaguerre(n-l-1, 2*l+1)
        self.laguerre_dx = np.polyder(self.laguerre)
        self.legendre = LegendreFunction(m, l)
        self.legendre_dx = self.legendre.deriv

    def radial(self, r:np.ndarray) -> np.ndarray:
        """ Compute the normalized radial part of the wave function 
            
            Args:
                r (np.ndarray): radii to evaluate the radial function at
            
            Returns:
                R_nl (np.ndarray): radial function values at r in units such that the Bohr radius :math: $a_0 = 1$
        """
        # get quantum numbers as shorthands
        n, l = self.n, self.l
        # compute the final expression
        x = (2.0 / n) * r
        return self.r_factor * np.exp(-r / n) * (x ** l) * self.laguerre(x)

    def spherical_harmonics(self,
        theta:np.ndarray,
        phi:np.ndarray
    ) -> np.ndarray:
        """ Compute the normalized angular function part of the wave function 

            Args:
                theta (np.ndarray): the polar angle
                phi (np.ndarray): the azimuthal angle
            
            Returns:
                Y_lm (np.ndarray): spherical harmonics values for the given angles
        """
        # get quantum numbers as shorthands
        l, m = self.l, self.m
        # compute the polar and azimuthal component
        f = self.sh_factor * self.legendre(np.cos(theta))
        g = np.exp(1j * m * phi)
        # return spherical harmonics
        return f * g

    def psi(self, 
        x:np.ndarray
    ) -> np.ndarray:
        """ Evaluate the Time-Independent Wave Function at the given coordinates in spherical coordinate system
            
            Args:
                x (np.ndarray): input in spherical coordinates

            Returns:
                y (np.ndarray): wave-function values at given input, i.e. :math: $y = \psi(x, y, z)$
        """
        # check input shape
        assert x.shape[-1] == 3
        r, theta, phi = x[..., 0], x[..., 1], x[..., 2]
        # compute wave function
        return self.radial(r) * self.spherical_harmonics(theta, phi)

    def gradient(
        self,
        x:np.ndarray,
        t:float
    ) -> np.ndarray:
        # check input shape
        assert x.shape[-1] == 3
        r, theta, phi = x[..., 0], x[..., 1], x[..., 2]
        
        # get quantum numbers as shorthands
        n, l, m = self.n, self.l, self.m
        
        x = (2.0 / n) * r
        # evaluate function values of all components
        radial = self.r_factor * np.exp(-r / n) * (x ** l) * self.laguerre(x)
        angular_polar = self.sh_factor * self.legendre(np.cos(theta))
        angular_azimutal = np.exp(1j * m * phi)

        # avoid division by zero
        ct = np.clip(np.cos(theta), -0.999, 0.999)        
        # compute derivative of radial component
        radial_dr = self.r_factor * (1.0/n) * x**(l-1) * np.exp(-r / n) * (
            (2*l - x) * self.laguerre(x) + 2 * x * self.laguerre_dx(x)
        )
        # compute partial derviatives of angular components
        angular_dtheta = self.sh_factor * self.legendre_dx(np.cos(theta)) * -np.sin(theta)
        angular_dphi = 1j * m * angular_azimutal

        return np.stack((
            radial_dr * angular_polar  * angular_azimutal,
            radial    * angular_dtheta * angular_azimutal / (np.sin(phi) * r + 1e-5),
            radial    * angular_polar  * angular_dphi     / (r + 1e-5)
        ), axis=-1) * np.exp(-1j * self.E * t)