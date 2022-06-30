import numpy as np
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
from itertools import product
from typing import List, Tuple
import warnings
from quantum.core.tise import TISE
from .orbital import GaussianOrbital, MolecularOrbital
from .utils import MMD_E, MMD_R

class ElectronicTISE(TISE):
    """ Electronic Schroedinger Equation 

        Args:
            basis (List[GaussianOrbitals]): basis in which the equation is defined
            C (np.ndarray): the nuclei positions (i.e. atom positions) in units of Bohr and shape of (n, 3) where n refers to the number of atoms.
            Z (np.ndarray): the principle quantum numbers (i.e. nuclei charges) of the each atim given in the shape (n,)
    """

    def __init__(self, 
        basis:List[GaussianOrbital],
        C:np.ndarray,
        Z:np.ndarray
    ) -> None:
        # save basis
        self.basis = basis
        # compute molecular integrals
        self.S = self.overlap(basis)
        self.T = self.kinetic(basis)
        self.V_en = self.electron_nuclear_attraction(basis, C, Z)
        self.V_ee = self.electron_electron_repulsion(basis)
        # compute nuclear-nuclear repulsion energy
        self.E_nn = self.nuclear_nuclear_repulsion(C, Z)

    def overlap(self, basis:List[GaussianOrbital]) -> np.ndarray:
        """ Compute the overlap matrix of the given basis.
            See Equations (100) and (101) in Helgaker and Taylor.

            Args:
                basis (List[GaussianOrbital]): orbitals over which to compute the pairwise overlaps

            Returns:
                S (np.ndarray): 
                    the overlap matrix of shape (m, m) where m is the number of orbitals in the basis.
                    S_ij refers to the overlap value of the orbitals with index i and j in the basis.
        """
        # create matrix to store values in
        n = len(basis)
        S = np.empty((n, n))
        # process each pair of basis elements
        for ii, jj in product(range(n), repeat=2):
            a, b = basis[ii], basis[jj]
            # gather all values from GTOs
            (i, k, m), (j, l, n) = a.angular, b.angular
            (Ax, Ay, Az), (Bx, By, Bz) = a.origin, b.origin
            # reshape to compute pairwise values when combined
            alpha, beta = a.alpha.reshape(-1, 1), b.alpha.reshape(1, -1)
            c1c2 = a.coeff.reshape(-1, 1) * b.coeff.reshape(1, -1)
            n1n2 = a.N.reshape(-1, 1) * b.N.reshape(1, -1)
            # compute overlap between GTOs (contracted gaussians) i and j
            # reshape to compute pairwise overlap values of primitive gaussians and
            # finally sum over all pairs
            S[ii, jj] = (
                c1c2 * n1n2 * \
                MMD_E(i, j, 0, alpha, beta, Ax, Bx) * \
                MMD_E(k, l, 0, alpha, beta, Ay, By) * \
                MMD_E(m, n, 0, alpha, beta, Az, Bz) * \
                (np.pi / (alpha + beta)) ** 1.5
            ).sum()

        return S

    def kinetic(self, basis:List[GaussianOrbital]) -> np.ndarray:
        """ Compute the kinetic energy matrix for the given basis.
            See Equations (100) and (116) in Helgaker and Taylor.

            Args:
                basis (List[GaussianOrbital]): orbitals

            Returns:
                T (np.ndarray): 
                    the kintec energy matrix of shape (m, m) where m is the number of orbitals in the basis.
        """
        # create matrix to store values in
        n = len(basis)
        T = np.empty((n, n))
        # process each pair of basis elements
        for ii, jj in product(range(n), repeat=2):
            a, b = basis[ii], basis[jj]
            # gather all values from GTOs
            (i, k, m), (j, l, n) = a.angular, b.angular
            (Ax, Ay, Az), (Bx, By, Bz) = a.origin, b.origin
            alpha, beta = a.alpha, b.alpha
            # reshape to compute pairwise values when combined
            alpha, beta = alpha.reshape(-1, 1), beta.reshape(1, -1)
            c1c2 = a.coeff.reshape(-1, 1) * b.coeff.reshape(1, -1)
            n1n2 = a.N.reshape(-1, 1) * b.N.reshape(1, -1)
            # compute directional overlaps - actually only the expansion coefficients
            # factor sqrt(pi/p) is factored out of the equations and added to the final expression
            Sx = MMD_E(i, j, 0, alpha, beta, Ax, Bx)
            Sy = MMD_E(k, l, 0, alpha, beta, Ay, By)
            Sz = MMD_E(m, n, 0, alpha, beta, Az, Bz)
            # compute kinetic terms in each direction
            # similarly to the overlaps only using the expansion coefficients here
            Tx = j * (j - 1) * MMD_E(i, j-2, 0, alpha, beta, Ax, Bx) - \
                2.0 * beta * (2.0 * j + 1.0) * Sx + \
                4.0 * beta**2 * MMD_E(i, j+2, 0, alpha, beta, Ax, Bx)
            Ty = l * (l - 1) * MMD_E(k, l-2, 0, alpha, beta, Ay, By) - \
                2.0 * beta * (2.0 * l + 1.0) * Sy + \
                4.0 * beta**2 * MMD_E(k, l+2, 0, alpha, beta, Ay, By)
            Tz = n * (n - 1) * MMD_E(m, n-2, 0, alpha, beta, Az, Bz) - \
                2.0 * beta * (2.0 * n + 1.0) * Sz + \
                4.0 * beta**2 * MMD_E(m, n+2, 0, alpha, beta, Az, Bz)
            # compute final value
            T[ii, jj] = -0.5 * np.sum(
                c1c2 * n1n2 * \
                (Tx * Sy * Sz + Sx * Ty * Sz + Sx * Sy * Tz) * \
                (np.pi / (alpha + beta))**1.5
            )

        return T

    def electron_nuclear_attraction(
        self, 
        basis:List[GaussianOrbital], 
        C:np.ndarray,
        Z:np.ndarray
    ) -> np.ndarray:
        """ Compute the electron-nuclear attraction energy matrix for the given basis.
            See Equations (199) and (204) in Helgaker and Taylor.

            Args:
                basis (List[GaussianOrbital]): orbitals
                C (np.ndarray): the nuclei positions in shape (n, 3) where n refers to the number of nuclei
                Z (np.ndarray): the nuclei charges given in shape (n,)

            Returns:
                V_en (np.ndarray): 
                    the attaction energy matrix of shape (m, m) where m is the number of orbitals in the basis.
        """
        # create matrix to store values in
        n = len(basis)
        V = np.empty((n, n))
        # process each pair of basis elements
        for ii, jj in product(range(n), repeat=2):
            a, b = basis[ii], basis[jj]
            # gather all values from GTOs
            (i, k, m), (j, l, n) = a.angular, b.angular
            (Ax, Ay, Az), (Bx, By, Bz) = a.origin, b.origin
            alpha, beta = a.alpha, b.alpha
            # reshape to compute pairwise values when combined
            # last dimension is for broadcasting to number of nuclei in C
            alpha, beta = alpha.reshape(-1, 1, 1), beta.reshape(1, -1, 1)
            c1c2 = a.coeff.reshape(-1, 1, 1) * b.coeff.reshape(1, -1, 1)
            n1n2 = a.N.reshape(-1, 1, 1) * b.N.reshape(1, -1, 1)
        
            # compute all gaussian composite centers
            # here last dimension is re-used for coordinates
            P = (alpha * a.origin + beta * b.origin) / (alpha + beta)
            # add dimension for nuclei, note that last dimension of coordinates
            # is reduced in computations of R and thus the remaining last dimension
            # refers to the number of nuclei again
            P = np.expand_dims(P, -2)

            # compute
            V[ii, jj] = np.sum(
                2.0 * np.pi / (alpha + beta) * c1c2 * n1n2 * \
                -Z * sum((
                    MMD_E(i, j, t, alpha, beta, Ax, Bx) * \
                    MMD_E(k, l, u, alpha, beta, Ay, By) * \
                    MMD_E(m, n, v, alpha, beta, Az, Bz) * \
                    MMD_R(t, u, v, 0, alpha + beta, P, C)
                    for t, u, v in product(range(i+j+1), range(k+l+1), range(m+n+1))
                ))
            )

        # return potential
        return V

    def electron_electron_repulsion(self, basis:List[GaussianOrbital]) -> np.ndarray:
        """ Compute the electron-electron repulsion energy matrix for the given basis.
            See Equations (199) and (205) in Helgaker and Taylor.

            Args:
                basis (List[GaussianOrbital]): orbitals

            Returns:
                V_ee (np.ndarray): 
                    the repulsion energy matrix of shape (m, m) where m is the number of orbitals in the basis.
        """
        # create matrix to store values in
        n = len(basis)
        V = np.empty((n, n, n, n))
        # process each pair of basis elements
        for ii, jj, kk, ll in product(range(n), repeat=4):
            a, b, c, d = basis[ii], basis[jj], basis[kk], basis[ll]
            
            # gather all values from GTOs a and b
            (i1, k1, m1), (j1, l1, n1) = a.angular, b.angular
            (Ax, Ay, Az), (Bx, By, Bz) = a.origin, b.origin
            alpha, beta = a.alpha, b.alpha
            # reshape to compute pairwise values when combined
            # last dimension is for broadcasting to number of nuclei in C
            alpha, beta = alpha.reshape(-1, 1), beta.reshape(1, -1)
            c1c2 = a.coeff.reshape(-1, 1) * b.coeff.reshape(1, -1)
            n1n2 = a.N.reshape(-1, 1) * b.N.reshape(1, -1)
        
            # gather all values from GTOs c and d
            (i2, k2, m2), (j2, l2, n2) = c.angular, d.angular
            (Cx, Cy, Cz), (Dx, Dy, Dz) = c.origin, d.origin
            gamma, delta = c.alpha, d.alpha
            # reshape to compute pairwise values when combined
            # last dimension is for broadcasting to number of nuclei in C
            gamma, delta = gamma.reshape(-1, 1), delta.reshape(1, -1)
            c3c4 = c.coeff.reshape(-1, 1) * d.coeff.reshape(1, -1)
            n3n4 = c.N.reshape(-1, 1) * d.N.reshape(1, -1)
        
            # compute all gaussian composite centers
            P_ab = (alpha.reshape(-1, 1, 1) * a.origin + beta.reshape(1, -1, 1) * b.origin) 
            P_ab /= (alpha.reshape(-1, 1, 1) + beta.reshape(1, -1, 1))
            P_cd = (gamma.reshape(-1, 1, 1) * c.origin + delta.reshape(1, -1, 1) * d.origin) 
            P_cd /= (gamma.reshape(-1, 1, 1) + delta.reshape(1, -1, 1))
            # compute composite exponents
            p1 = (alpha + beta)[:, :, None, None]
            p2 = (gamma + delta)[None, None, :, :]
            q12 = p1 * p2 / (p1 + p2)

            # compute repulsion
            V[ii, jj, kk, ll] = np.sum(
                # scaling factor
                2.0 * np.pi**2.5 / (p1 * p2 * np.sqrt(p1 + p2)) * \
                # outer summation corresping to basis elements a and b
                sum((
                    (
                        c1c2 * n1n2 * \
                        MMD_E(i1, j1, t1, alpha, beta, Ax, Bx) * \
                        MMD_E(k1, l1, u1, alpha, beta, Ay, By) * \
                        MMD_E(m1, n1, v1, alpha, beta, Az, Bz)
                    )[:, :, None, None] * \
                    # inner summation corresponding to c and d
                    sum((
                        c3c4 * n3n4 * (-1)**(t2+u2+v2) * \
                        MMD_E(i2, j2, t2, gamma, delta, Cx, Dx) * \
                        MMD_E(k2, l2, u2, gamma, delta, Cy, Dy) * \
                        MMD_E(m2, n2, v2, gamma, delta, Cz, Dz) * \
                        MMD_R(t1+t2, u1+u2, v1+v2, 0, q12, 
                            P_ab[:, :, None, None, :], 
                            P_cd[None, None, :, :, :]
                        )
                        for t2, u2, v2 in product(range(i2+j2+1), range(k2+l2+1), range(m2+n2+1))
                    ))
                    for t1, u1, v1 in product(range(i1+j1+1), range(k1+l1+1), range(m1+n1+1))
                ))
            )
        
        # return potentials
        return V 

    def nuclear_nuclear_repulsion(self, C:np.ndarray, Z:np.ndarray) -> float:
        """ compute the nuclear-nuclear repulsion energy

            Args:
                C (np.ndarray): the nuclei positions in shape (3, n) where n refers to the number of nuclei
                Z (np.ndarray): the nuclei charges given in shape (n,)
                
            Returns:
                E_nn (float): the repulsion energy value
        """
        # compute pairwise distances between nuclei
        # and set diagonal from one to avoid divison by zero
        R = pdist(C, metric='euclidean')
        R = squareform(R)
        np.fill_diagonal(R, 1.0)
        # compute pairwise product of the nuclei charge
        # and set diagonal to zero to avoid repulsion from self
        ZZ = Z.reshape(-1, 1) * Z.reshape(1, -1)
        np.fill_diagonal(ZZ, 0)
        # compute repulsion
        return 0.5 * np.sum(ZZ / R)
 
    def restricted_hartree_fock(
        self,
        num_occ_orbitals:int =1,
        max_cycles:int =20,
        tol:float =1e-5
    ) -> Tuple[float, List[MolecularOrbital]]:
        """ Implements the restricted hartree-fock method also known as the self-consistent field (SCF) method 

            Args:
                num_occ_orbitals (int): 
                    the number of occupied orbitals, i.e. half the number of electrons in the system.
                    Defaults to 1.
                max_cycles (int): the maximum number of SCF cycles to do.
                tol (float): tolerance value used to detect convergence.

            Returns:
                E (float): the total estimated energy in units of Hartree, i.e. the estimated electronic energy plus the nuclear-nuclear repulsion energy
                MOs (List[MolecularOrbital]): the molecular orbitals found by the SCF method
        """

        # build core hamiltonian matrix
        H_core = self.T + self.V_en
        # define initial density matrix
        n = len(self.basis)
        P = np.zeros((n, n))
        # store electronic energy of previous 
        # iteration for convergence check
        E_elec_prev = float('inf')

        # scf cylcle
        for _ in range(max_cycles):

            # compute the two-electron term
            J, K = self.V_ee, self.V_ee.swapaxes(1, 3)
            G = np.sum(P.reshape(1, 1, n, n) * (J - 0.5 * K), axis=(2, 3))

            # form fock operator and diagonalize to obtain
            # molecular orbital energies and coefficients
            F = H_core + G
            E_mol, C = linalg.eigh(F, self.S)

            # form next density matrix
            P = C[:, :num_occ_orbitals]
            P = 2.0 * P @ P.T.conjugate()
            
            # estimate expected electronic energy
            E_elec = np.sum(P * (H_core + 0.5 * G))    
            # check for convergence
            if abs(E_elec - E_elec_prev) < tol:
                break

            # update old electronic energy value
            E_elec_prev = E_elec
        else:
            # convergence not met
            warnings.warn("Convergence not met in SCF cycle!", UserWarning)            

        # build all molecular orbitals
        MOs = [
            MolecularOrbital(
                coeff=C[:, i],
                basis=self.basis,
                E=E_mol[i]
            )
            for i in range(n)
        ]
        # return electronic energy and molecular orbitals
        return E_elec + self.E_nn, MOs

    def solve(
        self,
        num_occ_orbitals:int =1,
        max_cycles:int =20,
        tol:float =1e-5
    ) -> List[MolecularOrbital]:
        """ Solve the electronic schroedinger equation using hartree-fock method.

            Args:
                num_occ_orbitals (int): 
                    the number of occupied orbitals, i.e. half the number of electrons in the system.
                    Defaults to 1.
                max_cycles (int): the maximum number of SCF cycles to do.
                tol (float): tolerance value used to detect convergence.

            Returns:
                MOs (List[MolecularOrbital]): the molecular orbitals found by the SCF method
        """
        # use restricted hartree fock
        _, MOs = self.restricted_hartree_fock(
            num_occ_orbitals=num_occ_orbitals,
            max_cycles=max_cycles,
            tol=tol
        )
        # return all molecular orbitals
        return MOs
