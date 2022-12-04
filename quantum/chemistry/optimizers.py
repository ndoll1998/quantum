import numpy as np
from .structs import Molecule
from .tise import ElectronicTISE
from copy import deepcopy
from types import SimpleNamespace
from abc import ABC, abstractmethod
from typing import Optional, List

class GeometryOptimizer(ABC):
    """
        Args:
            mol (Molecule): Molecule instance to optimize
            num_occ_orbitals (int): 
                the number of occupied orbitals, i.e. half the number
                of electrons in the system. Defaults to 1.
            rhf_num_cycles (Optional[int]):
                maximum number of SCF-cycles to run in restricted
                hartree-fock routine (see `ElectronicTISE.restricted_hartree_fock`)
            rhf_tol (Optional[float]):
                tolerance threshold used to detect convergence in SCF-cyle
                (see `ElectronicTISE.restricted_hartree_fock`)
    """

    def __init__(self, 
        mol:Molecule, 
        num_occ_orbitals:int =1,
        rhf_num_cycles:Optional[int] =None,
        rhf_tol:Optional[float] =None
    ) -> None:
        self.n_occ = num_occ_orbitals
        # build hartree-fock keyword arguments
        self.rhf_kwargs = {'num_occ_orbitals': self.n_occ}
        if rhf_num_cycles is not None:
            self.rhf_kwargs['num_cycles'] = rhf_num_cycles
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
        return self.state.mol

    def update_state(self, mol:Molecule):
        # update molecule and tise
        self.state.mol = mol
        self.state.tise = ElectronicTISE.from_molecule(mol)
        # solve and save solution in state
        sol_state = self.state.tise.restricted_hartree_fock(**self.rhf_kwargs)
        self.state.E, self.state.C, self.state.F = sol_state

    @abstractmethod
    def step(self, mol:Molecule) -> Molecule:
        """ Perform a single optimization step """
        raise NotImplementedError()

    def optimize(self, max_iters:int, tol:float =1e-5) -> List[float]:
        """ Optimize the molecule geometry until some convergence criteria is met.
            
            Args:
                tol (float): 
                    tolerance value used to detect convergence based on
                    hartree-fock energy of molecule state. Defaults to 1e-5.
        """

        history = [self.state.E]
        # iterate step function until convergence criteria is met
        for _ in range(max_iters):
            # get energy before update for convergence check
            prev_E = self.state.E
            # perform an update step
            mol = self.step(self.state.mol)
            self.update_state(mol=mol)
            # add new energy state to history
            history.append(self.state.E)
            # check for convergence
            if abs(self.state.E - prev_E) < tol:
                break

        return history

class GradientDescentGeometryOptimizer(GeometryOptimizer):
    """
        Args:
            mol (Molecule): Molecule instance to optimize
            num_occ_orbitals (int): 
                the number of occupied orbitals, i.e. half the number
                of electrons in the system. Defaults to 1.
            alpha (float): step size of gradient descent update. Defaults to 0.1.
            rhf_num_cycles (Optional[int]):
                maximum number of SCF-cycles to run in restricted
                hartree-fock routine (see `ElectronicTISE.restricted_hartree_fock`)
            rhf_tol (Optional[float]):
                tolerance threshold used to detect convergence in SCF-cyle
                (see `ElectronicTISE.restricted_hartree_fock`)
    """

    def __init__(
        self, 
        mol:Molecule,
        num_occ_orbitals:int =1,
        alpha:float =0.1,
        rhf_num_cycles:Optional[int] =None,
        rhf_tol:Optional[float] =None
    ) -> None:
        # initialize geometry optimizer
        super(GradientDescentGeometryOptimizer, self).__init__(
            mol=mol,
            num_occ_orbitals=num_occ_orbitals,
            rhf_num_cycles=rhf_num_cycles,
            rhf_tol=rhf_tol
        )
        # save step size
        self.alpha = alpha    


    def compute_origin_gradients(self) -> np.ndarray:
        """ Compute the gradient of the Hartree-Fock energy w.r.t. the
            atom origins of the current molecule. See Equation (C.12) in
            'Modern Quantum Chemistry' (1989) by Szabo and Ostlund.
        """
        # compute density matrix and scaled density matrix
        # based on hf solution in state
        F_part = self.state.F[:self.n_occ]
        C_part = self.state.C[:, :self.n_occ]
        P = 2.0 * C_part @ C_part.T.conjugate()
        Q = 2.0 * (C_part * F_part) @ C_part.T.conjugate()
        
        # compute gradient of G
        J_grad, K_grad = self.state.tise.V_ee_grad, self.state.tise.V_ee_grad.swapaxes(1, 3)
        G_grad = np.sum(P[None, None, :, :, None] * (J_grad - 0.5 * K_grad), axis=(2, 3))        
        # compute gradient on basis-level
        dEdx = (
            P[:, :, None] * (self.state.tise.T_grad + self.state.tise.V_en_grad) + \
            P[:, :, None] * G_grad - \
            Q[:, :, None] * self.state.tise.S_grad
        )

        # build atom id vector, indicates which atom a specific
        # basis element corresponds to
        atom_ids = np.asarray([i for i, atom in enumerate(self.state.mol.atoms) for _ in range(len(atom))])
        # transform gradient from basis-level to atom-level by summation
        # and add nuclear-nuclear repulsion energy gradient
        return self.state.tise.E_nn_grad(self.state.mol) + np.asarray([
            dEdx[atom_ids == i, :, :].sum(axis=(0, 1))
            for i in range(len(self.state.mol.atoms))
        ]) * 2.0
    
    def step(self, mol:Molecule) -> Molecule:
        # compute new origins
        grads = self.compute_origin_gradients()
        origins = mol.origins - self.alpha * grads
        # update origins inplace because why not
        for i, atom in enumerate(mol.atoms):
            atom.origin = origins[i, :]
        # return molecule
        return mol
