import warnings
import numpy as np
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
from functools import cached_property
from typing import List, Tuple

from quantum.core.tise import TISE
from .structs import Molecule
from .orbital import GaussianOrbital, MolecularOrbital
# import integrals
from .integrals.overlap import Overlap
from .integrals.kinetic import Kinetic
from .integrals.attraction import ElectronNuclearAttraction
from .integrals.repulsion import ElectronElectronRepulsion
from .integrals.helpers import (
    create_expansion_coefficients,
    create_R_PC,
    create_R_PP
)

class ElectronicTISE(TISE):
    """ Electronic Schroedinger Equation 

        Args:
            basis (List[GaussianOrbitals]): basis in which the equation is defined
            C (np.ndarray): the nuclei positions (i.e. atom positions) in units of Bohr and shape of (n, 3) where n refers to the number of atoms.
            Z (np.ndarray): the principle quantum numbers (i.e. nuclei charges) of the each atom given in the shape (n,)
    """

    def __init__(self, 
        basis:List[GaussianOrbital],
        C:np.ndarray,
        Z:np.ndarray
    ) -> None:
        # save basis
        self.basis = basis
        # save nuclei info
        self.C = C
        self.Z = Z        

        n = len(self.basis)
        # create all expansion coefficients instance
        # one for each pair of GTOs
        self.expan_coeffs = np.asarray([
            create_expansion_coefficients(
                A_origin=A.origin,
                A_alpha=A.alpha,
                B_origin=B.origin,
                B_alpha=B.alpha
            )
            for A in self.basis
            for B in self.basis
        ]).reshape(n, n, 3)

    @classmethod
    def from_molecule(cls, molecule:Molecule) -> "ElectronicTISE":
        """ Initialize an electronic schroedinger equation directly from
            a molecule instance.
    
            Args:
                molecule (Molecule): the molecule instance
            
            Returns:
                tise (ElectronicTISE): schroedinger equation defined by the molecule
        """
        return cls(
            basis=molecule.basis,
            C=molecule.origins,
            Z=molecule.Zs
        )

    @cached_property
    def S(self) -> np.ndarray:
        """ Overlap Matrix
            
            Returns:
                S (np.ndarray): 
                    the overlap matrix of shape (m, m) where m is the number of orbitals in the basis.
                    S_ij refers to the overlap value of the orbitals with index i and j in the basis.
        """

        # create empty matrix to hold values
        S = np.empty((len(self.basis), len(self.basis)))

        # compute all values
        # make use of symmetric property of one-electron integrals
        for i, A in enumerate(self.basis):
            for j, B in enumerate(self.basis[:i+1]):
                S[i, j] = S[j, i] = Overlap.compute(
                    # GTO A
                    A_coeff=A.coeff,
                    A_alpha=A.alpha,
                    A_angular=A.angular,
                    A_norm=A.N,
                    # GTO B
                    B_coeff=B.coeff,
                    B_alpha=B.alpha,
                    B_angular=B.angular,
                    B_norm=B.N,
                    # GTO pair AB
                    Ex=self.expan_coeffs[i, j, 0],
                    Ey=self.expan_coeffs[i, j, 1],
                    Ez=self.expan_coeffs[i, j, 2]
                )

        # return overlap matrix
        return S

    @cached_property
    def T(self) -> np.ndarray:
        """ Kinetic Energy Matrix 

            Returns:
                T (np.ndarray): 
                    the kintec energy matrix of shape (m, m) where m is the number of orbitals in the basis.
        """
        # create empty matrix to hold values
        T = np.empty((len(self.basis), len(self.basis)))

        # compute all values
        # make use of symmetric property of one-electron integrals
        for i, A in enumerate(self.basis):
            for j, B in enumerate(self.basis[:i+1]):
                T[i, j] = T[j, i] = Kinetic.compute(
                    # GTO A
                    A_coeff=A.coeff,
                    A_alpha=A.alpha,
                    A_angular=A.angular,
                    A_norm=A.N,
                    # GTO B
                    B_coeff=B.coeff,
                    B_alpha=B.alpha,
                    B_angular=B.angular,
                    B_norm=B.N,
                    # GTO pair AB
                    Ex=self.expan_coeffs[i, j, 0],
                    Ey=self.expan_coeffs[i, j, 1],
                    Ez=self.expan_coeffs[i, j, 2]
                )

        # return overlap matrix
        return T

    @cached_property
    def V_en(self) -> np.ndarray:
        """ Electron-Nuclear Attraction Matrix """
        # create empty matrix to hold values
        V = np.empty((len(self.basis), len(self.basis)))

        # compute all values
        # make use of symmetric property of one-electron integrals
        for i, A in enumerate(self.basis):
            for j, B in enumerate(self.basis[:i+1]):
                V[i, j] = V[j, i] = ElectronNuclearAttraction.compute(
                    Z=self.Z,
                    # GTO A
                    A_coeff=A.coeff,
                    A_alpha=A.alpha,
                    A_angular=A.angular,
                    A_norm=A.N,
                    # GTO B
                    B_coeff=B.coeff,
                    B_alpha=B.alpha,
                    B_angular=B.angular,
                    B_norm=B.N,
                    # GTO pair AB
                    Ex=self.expan_coeffs[i, j, 0],
                    Ey=self.expan_coeffs[i, j, 1],
                    Ez=self.expan_coeffs[i, j, 2],
                    # global
                    R_PC=create_R_PC(
                        C=self.C,
                        A_origin=A.origin,
                        A_alpha=A.alpha,
                        B_origin=B.origin,
                        B_alpha=B.alpha
                    )
                )

        # return overlap matrix
        return V
    
    @cached_property
    def V_ee(self) -> np.ndarray:
        """ Electron-Electron Repulsion Tensor """
        return np.asarray([
            ElectronElectronRepulsion.compute(
                # GTO A
                A_coeff=A.coeff,
                A_alpha=A.alpha,
                A_angular=A.angular,
                A_norm=A.N,
                # GTO B
                B_coeff=B.coeff,
                B_alpha=B.alpha,
                B_angular=B.angular,
                B_norm=B.N,
                # GTO C
                C_coeff=C.coeff,
                C_alpha=C.alpha,
                C_angular=C.angular,
                C_norm=C.N,
                # GTO D
                D_coeff=D.coeff,
                D_alpha=D.alpha,
                D_angular=D.angular,
                D_norm=D.N,
                # GTO pair AB
                Ex_AB=self.expan_coeffs[i, j, 0],
                Ey_AB=self.expan_coeffs[i, j, 1],
                Ez_AB=self.expan_coeffs[i, j, 2],
                # GTO pair CD
                Ex_CD=self.expan_coeffs[k, l, 0],
                Ey_CD=self.expan_coeffs[k, l, 1],
                Ez_CD=self.expan_coeffs[k, l, 2],
                # global
                R_PP=create_R_PP(
                    A_origin=A.origin,
                    A_alpha=A.alpha,
                    B_origin=B.origin,
                    B_alpha=B.alpha,
                    C_origin=C.origin,
                    C_alpha=C.alpha,
                    D_origin=D.origin,
                    D_alpha=D.alpha,
                )
            )
            for i, A in enumerate(self.basis)
            for j, B in enumerate(self.basis)
            for k, C in enumerate(self.basis)
            for l, D in enumerate(self.basis)
        ]).reshape(len(self.basis), len(self.basis), len(self.basis), len(self.basis))

    @cached_property
    def E_nn(self) -> float:
        """ compute the nuclear-nuclear repulsion energy

            Returns:
                E_nn (float): the repulsion energy value
        """
        # compute pairwise distances between nuclei
        # and set diagonal to one to avoid divison by zero
        R = pdist(self.C, metric='euclidean')
        R = squareform(R)
        np.fill_diagonal(R, 1.0)
        # compute pairwise product of the nuclei charge
        # and set diagonal to zero to avoid repulsion from self
        ZZ = self.Z.reshape(-1, 1) * self.Z.reshape(1, -1)
        np.fill_diagonal(ZZ, 0)
        # compute repulsion
        return 0.5 * np.sum(ZZ / R)
 
    def restricted_hartree_fock(
        self,
        num_occ_orbitals:int =1,
        max_cycles:int =20,
        tol:float =1e-5
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """ Implements the restricted hartree-fock method also known as the self-consistent field (SCF) method 

            Args:
                num_occ_orbitals (int): 
                    the number of occupied orbitals, i.e. half the number of electrons in the system.
                    Defaults to 1.
                max_cycles (int): the maximum number of SCF cycles to do.
                tol (float): tolerance value used to detect convergence.

            Returns:
                E (float): 
                    the total estimated energy in units of Hartree, i.e. the estimated 
                    electronic energy plus the nuclear-nuclear repulsion energy
                C (np.ndarray): The eigenvectors of F, i.e. molecular orbital coefficient matrix
                F (np.ndarray): 
                    diagonal entries of the fock matrix in the molecular orbital 
                    basis C (i.e. F is diagonal). In other words these are the eigenvalues of the fock matrix, i.e. the energies of the corresponding molecular orbitals.
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

        # return electronic energy and molecular orbitals
        return E_elec + self.E_nn, C, E_mol

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
        _, C, F = self.restricted_hartree_fock(
            num_occ_orbitals=num_occ_orbitals,
            max_cycles=max_cycles,
            tol=tol
        )
        # build all molecular orbitals
        return [
            MolecularOrbital(
                coeff=C[:, i],
                basis=self.basis,
                E=F[i]
            )
            for i in range(F.shape[0])
        ]
