import numpy as np
from .structs import Molecule
from .tise import ElectronicTISE
from copy import deepcopy
from types import SimpleNamespace
from abc import ABC, abstractmethod
from typing import Optional, List

class GeometryOptimizer(ABC):
    """ Abstract Geometry Optimizer

        Geometry Optimizers aim to find a low-energy state of a molecule
        by minimize the hartree-fock energy w.r.t atom positions within
        the molecule.

        Args:
            mol (Molecule): Molecule instance to optimize
            num_occ_orbitals (int): 
                the number of occupied orbitals, i.e. half the number of electrons in the system.
                Defaults to sum(Z)//2 where Z is the list of atom charges.
            rhf_max_cycles (Optional[int]):
                maximum number of SCF-cycles to run in restricted
                hartree-fock routine (see `ElectronicTISE.restricted_hartree_fock`)
            rhf_tol (Optional[float]):
                tolerance threshold used to detect convergence in SCF-cyle
                (see `ElectronicTISE.restricted_hartree_fock`)
    """

    def __init__(self, 
        mol:Molecule, 
        num_occ_orbitals:Optional[int] =None,
        rhf_max_cycles:Optional[int] =None,
        rhf_tol:Optional[float] =None
    ) -> None:
        self.n_occ = num_occ_orbitals or (mol.Zs.sum()//2)
        # build hartree-fock keyword arguments
        self.rhf_kwargs = {}
        if rhf_max_cycles is not None:
            self.rhf_kwargs['max_cycles'] = rhf_max_cycles
        if rhf_tol is not None:
            self.rhf_kwargs['tol'] = rhf_tol

        # internal state values changing with every update step
        self.state = SimpleNamespace(
            mol=None,
            tise=None,
            # rhf solution state
            E=None,
            C=None,
            F=None
        )
        # set initial state from copy of given molecule
        # just to make sure the original is kept untouched
        self.update_state(deepcopy(mol))

    @property
    def molecule(self) -> Molecule:
        """ Current molecule """
        return self.state.mol

    def update_state(self, mol:Molecule):
        """ Update internal state w.r.t a given molecule. 
            Note that this is quiet expensive as the state update
            requires the computation of the hartree-fock energy.
        """
        # update molecule and tise
        self.state.mol = mol
        self.state.tise = ElectronicTISE(mol, self.n_occ)
        # solve and save solution in state
        sol_state = self.state.tise.restricted_hartree_fock(**self.rhf_kwargs)
        self.state.E, self.state.C, self.state.F = sol_state

    @abstractmethod
    def step(self, mol:Molecule) -> Molecule:
        """ (abstract) Perform a single optimization step. 
            Overwrite to implement custom geometry optimization logic.
            
            Args:
                mol (Molecule): the original molecule to update
            
            Returns:
                mol (Molecule): the updated molecule
        """
        raise NotImplementedError()

    def optimize(self, max_iters:int, tol:float =1e-5) -> iter:
        """ Optimize the molecule geometry until some convergence criteria is met.
            
            Args:
                max_iters (int):
                    maximum number of update steps to perform
                tol (float): 
                    tolerance value used to detect convergence based on
                    hartree-fock energy of molecule state. Defaults to 1e-5.

            Yields:
                E (float): 
                    Hartree-Fock Energy of last optimization step
                mol (Molecule):
                    Molecule state of last optimization step
        """

        # iterate step function until convergence criteria is met
        for _ in range(max_iters):
            # get energy before update for convergence check
            prev_E = self.state.E
            # perform an update step
            mol = self.step(self.state.mol)
            self.update_state(mol=mol)
            # yield energy and molecule after optimization step
            yield self.state.E, self.state.mol
            # check for convergence
            if abs(self.state.E - prev_E) < tol:
                break

class GradientDescentGeometryOptimizer(GeometryOptimizer):
    """ First-order Gradient Descent Geometry Optimizer

        Args:
            mol (Molecule): Molecule instance to optimize
            num_occ_orbitals (int): 
                the number of occupied orbitals, i.e. half the number of electrons in the system.
                Defaults to sum(Z)//2 where Z is the list of atom charges.
            alpha (float): step size of gradient descent update. Defaults to 0.1.
            rhf_max_cycles (Optional[int]):
                maximum number of SCF-cycles to run in restricted
                hartree-fock routine (see `ElectronicTISE.restricted_hartree_fock`)
            rhf_tol (Optional[float]):
                tolerance threshold used to detect convergence in SCF-cyle
                (see `ElectronicTISE.restricted_hartree_fock`)
    """

    def __init__(
        self, 
        mol:Molecule,
        num_occ_orbitals:int =None,
        alpha:float =0.1,
        rhf_max_cycles:Optional[int] =None,
        rhf_tol:Optional[float] =None
    ) -> None:
        # initialize geometry optimizer
        super(GradientDescentGeometryOptimizer, self).__init__(
            mol=mol,
            num_occ_orbitals=num_occ_orbitals,
            rhf_max_cycles=rhf_max_cycles,
            rhf_tol=rhf_tol
        )
        # save step size
        self.alpha = alpha    

    def compute_origin_gradients(self) -> np.ndarray:
        """ Compute the gradient of the Hartree-Fock energy w.r.t. the
            atom origins of the current molecule. See Equation (C.12) in
            'Modern Quantum Chemistry' (1989) by Szabo and Ostlund.

            Returns:
                origin_grads (np.ndarray):
                    gradients of the atom origins in the shape (n, 3)
                    where n is the number of atoms and 3 corresponds to
                    the spacial dimensions in the cartesian coordinate system.
        """
        # compute density matrix and scaled density matrix
        # based on hf solution in state
        F_part = self.state.F[:self.n_occ]
        C_part = self.state.C[:, :self.n_occ]
        P = 2.0 * C_part @ C_part.T.conjugate()
        Q = 2.0 * (C_part * F_part) @ C_part.T.conjugate()
        
        # get all one-electron integral gradients
        dSdx = self.state.tise.S_grad
        dTdx = self.state.tise.T_grad
        dVdx = self.state.tise.V_en_grad
        # build gradient of G from gradient of two-electron integral
        dJdx = self.state.tise.V_ee_grad
        dKdx = self.state.tise.V_ee_grad.swapaxes(2, 4)
        dGdx = np.sum(P[None, None, None, :, :, None] * (dJdx - 0.5 * dKdx), axis=(3, 4))

        # compute gradient of energy w.r.t. basis origins
        # this doesn't include the nuclear-nuclear repulsion term
        # yet as it can't be computed on basis-level but only on
        # atom-level
        dEdx = (
            # core hamiltonian term
            (P.T[None, :, :, None] * (dTdx + dVdx)) + \
            # electron-electron repulsion term
            0.5 * (P.T[None, :, :, None] * dGdx) + \
            # overlap term
            -(Q.T[None, :, :, None] * dSdx)
        ).sum(axis=(1, 2))

        # add nuclear-nuclear repulsion gradient and return
        return dEdx + self.state.tise.E_nn_grad
    
    def step(self, mol:Molecule) -> Molecule:
        """ Implements a single gradient descent step.
            Atom origins are updated inplace.

            Args:
                mol (Molecule): the original molecule to update
            
            Returns:
                mol (Molecule): the updated molecule
        """
        # compute new origins
        grads = self.compute_origin_gradients()
        origins = mol.origins - self.alpha * grads
        # update origins inplace because why not
        for i, atom in enumerate(mol.atoms):
            atom.origin = origins[i, :]
        # return molecule
        return mol
