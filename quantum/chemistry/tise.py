import warnings
import numpy as np
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
from functools import cached_property
from typing import List, Tuple, Optional

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
            mol (Molecule): the molecule specifying the schroedinger equation
            num_occ_orbitals (int): 
                the number of occupied orbitals, i.e. half the number of electrons in the system.
                Defaults to sum(Z)//2 where Z is the list of atom charges.
    """

    def __init__(self, mol:Molecule, num_occ_orbitals:Optional[int] =None) -> None:
        # save the molecule and number of occupied orbitals
        self.mol = mol
        self.n_occ = num_occ_orbitals or (mol.Zs.sum() // 2)
        # shorthands to all kinds of values from the molecule
        self.Z = mol.Zs
        self.C = mol.origins
        self.basis = mol.basis
        self.basis_ids = mol.basis_atom_ids

    @cached_property
    def E(self) -> np.ndarray:
        """ Expansion coefficient instances for all combinations of basis elements """
        return np.asarray([
            create_expansion_coefficients(
                A_origin=A.origin,
                A_alpha=A.alpha,
                B_origin=B.origin,
                B_alpha=B.alpha
            )
            for A in self.basis
            for B in self.basis
        ]).reshape((len(self.basis), len(self.basis), 3))

    @cached_property
    def R_PC(self) -> np.ndarray:
        """ Hermite integral instances for electron-nuclear interaction """
        return np.asarray([
            create_R_PC(
                C=self.C,
                A_origin=A.origin,
                A_alpha=A.alpha,
                B_origin=B.origin,
                B_alpha=B.alpha
            )
            for A in self.basis
            for B in self.basis
        ]).reshape((len(self.basis), len(self.basis)))
            
    @cached_property
    def R_PP(self) -> np.ndarray:
        """ Hermite integral instances for electron-electron interaction """
        return np.asarray([
            create_R_PP(
                A_origin=A.origin,
                A_alpha=A.alpha,
                B_origin=B.origin,
                B_alpha=B.alpha,
                C_origin=C.origin,
                C_alpha=C.alpha,
                D_origin=D.origin,
                D_alpha=D.alpha,
            )
            for A in self.basis
            for B in self.basis
            for C in self.basis
            for D in self.basis
        ]).reshape(len(self.basis), len(self.basis), len(self.basis), len(self.basis))

    @cached_property
    def S(self) -> np.ndarray:
        """ Overlap Matrix """
        # create empty matrix to hold values, note that
        # overlap with self is always 1.0
        S = np.empty((len(self.basis), len(self.basis)))
        np.fill_diagonal(S, 1.0)
        # create empty matrix to hold values

        # compute all values
        # make use of symmetric property of one-electron integrals
        for i, A in enumerate(self.basis):
            for j, B in enumerate(self.basis[:i]):
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
                    Ex=self.E[i, j, 0],
                    Ey=self.E[i, j, 1],
                    Ez=self.E[i, j, 2]
                )

        # return overlap matrix
        return S

    @cached_property
    def S_grad(self) -> np.ndarray:
        """ Gradient of the Overlap Matrix """
        # create empty matrix to hold values, initialize
        # with zeros as the matrix is very sparse
        S_grad = np.zeros((len(self.mol), len(self.basis), len(self.basis), 3))

        # compute all values
        # make use of symmetric property of one-electron integrals
        for i, A in enumerate(self.basis):
            # overlap with self is constant thus has gradient 0
            # meaning value (i, j) doesn't need to be computed
            for j, B in enumerate(self.basis[:i]):
                dSdA, dSdB = Overlap.gradient(
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
                    Ex=self.E[i, j, 0],
                    Ey=self.E[i, j, 1],
                    Ez=self.E[i, j, 2]
                )
                # set all values
                S_grad[self.basis_ids[i], [i, j], [j, i], :] += dSdA
                S_grad[self.basis_ids[j], [i, j], [j, i], :] += dSdB

        # return overlap matrix
        return S_grad

    @cached_property
    def T(self) -> np.ndarray:
        """ Kinetic Energy Matrix """
        # create empty matrix to hold values, note
        # that kinetic energy matrix is hollow
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
                    Ex=self.E[i, j, 0],
                    Ey=self.E[i, j, 1],
                    Ez=self.E[i, j, 2]
                )

        # return overlap matrix
        return T
    
    @cached_property
    def T_grad(self) -> np.ndarray:
        """ Gradient of the Kinetic Energy Matrix """
        # create empty matrix to hold values, again
        # very sparse matrix so initialize with zeros
        T_grad = np.zeros((len(self.mol), len(self.basis), len(self.basis), 3))

        # compute all values
        # make use of symmetric property of one-electron integrals
        for i, A in enumerate(self.basis):
            # again gradient is hollow meaning gradient of
            # (i, j) is zero so skip computing diagonal
            for j, B in enumerate(self.basis[:i]):
                dTdA, dTdB = Kinetic.gradient(
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
                    Ex=self.E[i, j, 0],
                    Ey=self.E[i, j, 1],
                    Ez=self.E[i, j, 2]
                )
                # set all values
                T_grad[self.basis_ids[i], [i, j], [j, i], :] += dTdA
                T_grad[self.basis_ids[j], [i, j], [j, i], :] += dTdB

        # return overlap matrix
        return T_grad

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
                    Ex=self.E[i, j, 0],
                    Ey=self.E[i, j, 1],
                    Ez=self.E[i, j, 2],
                    # global
                    R_PC=self.R_PC[i, j]
                )

        # return overlap matrix
        return V
    
    @cached_property
    def V_en_grad(self) -> np.ndarray:
        """ Gradient of the Electron-Nuclear Attraction Matrix """
        # create empty matrix to hold values
        V_grad = np.zeros((len(self.mol), len(self.basis), len(self.basis), 3))

        # compute gradient w.r.t orbital origins
        # make use of symmetric property of one-electron integrals
        for i, A in enumerate(self.basis):
            for j, B in enumerate(self.basis[:i+1]):
                dVdA, dVdB = ElectronNuclearAttraction.gradient(
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
                    Ex=self.E[i, j, 0],
                    Ey=self.E[i, j, 1],
                    Ez=self.E[i, j, 2],
                    # global
                    R_PC=self.R_PC[i, j]
                )
                # handle diagonal by adding second term
                V_grad[self.basis_ids[i], [i, j], [j, i], :] += dVdA
                V_grad[self.basis_ids[j], [i, j], [j, i], :] += dVdB

        # add gradient w.r.t atom origins
        for i, A in enumerate(self.basis):
            for j, B in enumerate(self.basis[:i+1]):
                V_grad[:, [i, j], [j, i], :] += ElectronNuclearAttraction.gradient_wrt_C(
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
                    Ex=self.E[i, j, 0],
                    Ey=self.E[i, j, 1],
                    Ez=self.E[i, j, 2],
                    # global
                    R_PC=self.R_PC[i, j]
                )[:, None, :]

        # return overlap matrix
        return V_grad
    
    @cached_property
    def V_ee(self) -> np.ndarray:
        """ Electron-Electron Repulsion Tensor """
        # create empty matrix to hold values
        V = np.empty((len(self.basis), len(self.basis), len(self.basis), len(self.basis)))

        # compute all values
        for i, A in enumerate(self.basis):
            for j, B in enumerate(self.basis[:i+1]):
                for k, C in enumerate(self.basis):
                    for l, D in enumerate(self.basis[:k+1]):
                        # make use of full symmetric property
                        if (i*(i+1)//2 + j) >= ((k*(k+1))//2 + l):
                            V[
                                [i, i, j, j, k, k, l, l],
                                [j, j, i, i, l, l, k, k],
                                [k, l, k, l, i, j, i, j],
                                [l, k, l, k, j, i, j, i]
                            ] = ElectronElectronRepulsion.compute(
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
                                Ex_AB=self.E[i, j, 0],
                                Ey_AB=self.E[i, j, 1],
                                Ez_AB=self.E[i, j, 2],
                                # GTO pair CD
                                Ex_CD=self.E[k, l, 0],
                                Ey_CD=self.E[k, l, 1],
                                Ez_CD=self.E[k, l, 2],
                                # global
                                R_PP=self.R_PP[i, j, k, l]
                            )
        
        return V

    @cached_property
    def V_ee_grad(self) -> np.ndarray:
        """ Gradient of the Electron-Electron Repulsion Tensor """
        # create empty matrix to hold values
        V_grad = np.zeros((len(self.mol), len(self.basis), len(self.basis), len(self.basis), len(self.basis), 3))

        # compute all values
        for i, A in enumerate(self.basis):
            for j, B in enumerate(self.basis[:i+1]):
                for k, C in enumerate(self.basis):
                    for l, D in enumerate(self.basis[:k+1]):
                        # make use of full symmetric property
                        if (i*(i+1)//2 + j) >= (k*(k+1)//2 + l):
                            dVdA, dVdB, dVdC, dVdD = ElectronElectronRepulsion.gradient(
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
                                Ex_AB=self.E[i, j, 0],
                                Ey_AB=self.E[i, j, 1],
                                Ez_AB=self.E[i, j, 2],
                                # GTO pair CD
                                Ex_CD=self.E[k, l, 0],
                                Ey_CD=self.E[k, l, 1],
                                Ez_CD=self.E[k, l, 2],
                                # global
                                R_PP=self.R_PP[i, j, k, l]
                            )

                            # set all values
                            V_grad[
                                self.basis_ids[i],
                                [i, i, j, j, k, k, l, l],
                                [j, j, i, i, l, l, k, k],
                                [k, l, k, l, i, j, i, j],
                                [l, k, l, k, j, i, j, i],
                            :] += dVdA
                            V_grad[
                                self.basis_ids[j],
                                [i, i, j, j, k, k, l, l],
                                [j, j, i, i, l, l, k, k],
                                [k, l, k, l, i, j, i, j],
                                [l, k, l, k, j, i, j, i],
                            :] += dVdB
                            V_grad[
                                self.basis_ids[k],
                                [i, i, j, j, k, k, l, l],
                                [j, j, i, i, l, l, k, k],
                                [k, l, k, l, i, j, i, j],
                                [l, k, l, k, j, i, j, i],
                            :] += dVdC
                            V_grad[
                                self.basis_ids[l],
                                [i, i, j, j, k, k, l, l],
                                [j, j, i, i, l, l, k, k],
                                [k, l, k, l, i, j, i, j],
                                [l, k, l, k, j, i, j, i],
                            :] += dVdD
                            
        return V_grad
    
    @cached_property
    def E_nn(self) -> float:
        """ Nuclear-Nuclear Repulsion Energy """
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
    
    @cached_property
    def E_nn_grad(self) -> np.ndarray:
        """ Gradient of the Nuclear-Nuclear Repulsion Energy """
        # compute pairwise distances between nuclei
        # and set diagonal to one to avoid divison by zero
        R = pdist(self.C, metric='euclidean')
        R = squareform(R)
        np.fill_diagonal(R, 1.0)
        # compute gradient
        return self.Z[:, None] * np.sum(
            self.Z[None, :, None] * (self.C[None, :, :] - self.C[:, None, :]) / \
            (R[:, :, None] ** 3),
            axis=1
        )
    
    def restricted_hartree_fock(
        self,
        max_cycles:int =20,
        tol:float =1e-5
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """ Implements the restricted hartree-fock method also known as the self-consistent field (SCF) method 

            Args:
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
        for i in range(max_cycles):

            # compute the two-electron term
            J, K = self.V_ee, self.V_ee.swapaxes(1, 3)
            G = np.sum(P.reshape(1, 1, n, n) * (J - 0.5 * K), axis=(2, 3))

            # form fock operator and diagonalize to obtain
            # molecular orbital energies and coefficients
            F = H_core + G
            E_mol, C = linalg.eigh(F, self.S)

            # form next density matrix
            P = C[:, :self.n_occ]
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
            warnings.warn("Convergence not met in SCF cycle after %i iterations!" % i, UserWarning)            

        # return electronic energy and molecular orbitals
        return E_elec + self.E_nn, C, E_mol

    def solve(
        self,
        max_cycles:int =20,
        tol:float =1e-5
    ) -> List[MolecularOrbital]:
        """ Solve the electronic schroedinger equation using hartree-fock method.

            Args:
                max_cycles (int): the maximum number of SCF cycles to do.
                tol (float): tolerance value used to detect convergence.

            Returns:
                MOs (List[MolecularOrbital]): the molecular orbitals found by the SCF method
        """
        # use restricted hartree fock
        _, C, F = self.restricted_hartree_fock(
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
