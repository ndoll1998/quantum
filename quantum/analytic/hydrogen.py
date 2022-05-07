import numpy as np
from quantum.core.wave import WaveFunction
from scipy.special import factorial, genlaguerre, lpmv

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
        self.E = 1.0

    def radial(self, r:np.ndarray) -> np.ndarray:
        """ Compute the normalized radial part of the wave function 
            
            Args:
                r (np.ndarray): radii to evaluate the radial function at
            
            Returns:
                R_nl (np.ndarray): radial function values at r in units such that the Bohr radius :math: $a_0 = 1$
        """
        # get quantum numbers as shorthands
        n, l = self.n, self.l
        # some frequently occuring terms
        x = (2.0 / n)
        xr = x * r
        # compute the final expression
        return np.sqrt(x**3 * factorial(n - l - 1) / (2 * n * factorial(n + l))) * \
            np.exp(-r / n) * (xr ** l) * genlaguerre(n-l-1, 2*l+1)(xr)

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
        f = (-1)**m * np.sqrt((2*l + 1) * factorial(l - m) / (4 * np.pi * factorial(l + m))) * lpmv(m, l, np.cos(theta))
        g = np.exp(1j * m * phi)
        # return spherical harmonics
        return f * g

    def psi(self, 
        x:np.ndarray
    ) -> np.ndarray:
        """ Evaluate the Time-Independent Wave Function at the given coordinates in spherical coordinate system
            
            Args:
                r (np.ndarray): radius
                theta (np.ndarray): the polar angle
                phi (np.ndarray): the azimuthal angle

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
        
        # some frequently occuring terms
        x = (2.0 / n)
        xr = x * r
        # compute radial component and its derivative
        radial = self.radial(r)
        dfdr = np.sqrt(x**3 * factorial(n - l - 1) / (2 * n * factorial(n + l))) * \
            1.0/n * xr**(l-1) * np.exp(-r / n) * (
                (2*l - xr) * genlaguerre(n-l-1, 2*l+1)(xr) + \
                2 * xr * np.polyder(genlaguerre(n-l-1, 2*l+1))(xr)
            )

        # frequently occuring terms
        ct = np.cos(theta)
        lgdr = lpmv(m, l, ct)
        a = (-1)**m * np.sqrt((2*l + 1) * factorial(l - m) / (4 * np.pi * factorial(l + m)))
        # compute the polar and azimuthal component
        f = a * lgdr
        g = np.exp(1j * m * phi)
        
        # avoid division by zero
        ct = np.clip(ct, -0.999, 0.999)
        # compute partial derviatives of angular components
        dfdt = a * (-((1 - ct**2)**-0.5) * lpmv(m+1, l, ct) - (m * ct) / (1 - ct**2) * lgdr) * -np.sin(theta)
        dgdp = 1j * m * g
        
        # combine to compute final partial derivatives
        return np.stack((
            dfdr * f * g,
            radial * dfdt * g,
            radial * f * dgdp
        ), axis=-1) * np.exp(1j * self.E * t)
