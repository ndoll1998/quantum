import numpy as np
from .structs import Molecule
from .tise import ElectronicTISE
from .utils import MMD_E, MMD_R, MMD_dEdx
from scipy.spatial.distance import pdist, squareform
from itertools import product
from copy import deepcopy

class GeometryOptimization(object):
    """ Molecule geometry optimization by minimizaing the Hartree-Fock 
        energy using gradient descent. Gradients of molecular integrals
        are computed as proposed in 'On the evaluation of derivatives of
        Gaussian integrals' (1992) by Helgaker and Taylor.

        Args:
            step (float): step-size applied to the gradient
            tol (float): tolerance value used to detect convergence
    """

    def __init__(
        self,
        step:float =0.5,
        tol:float =1e-5
    ) -> None:
        self.step = step
        self.tol = tol

    def overlap_deriv(self, molecule:Molecule) -> np.ndarray:
        """ Compute derivative of overlap matrix for a given molecule.
        
            Args:
                molecule (Molecule): molecule for which to compute the gradient
        
            Returns:
                dSdA (np.ndarray): 
                    the overlap derivative matrix of shape (m, m, n, 3) where
                    m is the number of basis elements and n the number of atoms.
        """
        # read values from molecule
        Z = molecule.Zs
        C = molecule.origins
        basis = molecule.basis
        # number of atoms and length of basis
        n_atoms = len(molecule)
        n_basis = len(basis)
        # overlap derivative matrix
        dS_dA = np.zeros((n_basis, n_basis, n_atoms, 3))
        
        # compute gradient of overlap of all combinations of atoms
        for ai, aj in product(range(n_atoms), repeat=2):
            for bi, bj in product(
                range(len(molecule.atoms[ai])), 
                range(len(molecule.atoms[aj]))
            ):
                # get basis elements
                a = molecule.atoms[ai].basis[bi]
                b = molecule.atoms[aj].basis[bj]
                # compute global index of basis elements
                ii = sum(map(len, molecule.atoms[:ai])) + bi
                jj = sum(map(len, molecule.atoms[:aj])) + bj
                # check global index
                assert basis[ii] == a
                assert basis[jj] == b            
                
                # gather all values from GTOs
                (i, k, m), (j, l, n) = a.angular, b.angular
                (Ax, Ay, Az), (Bx, By, Bz) = a.origin, b.origin
                # reshape to compute pairwise values when combined
                alpha, beta = a.alpha.reshape(-1, 1), b.alpha.reshape(1, -1)
                c1c2 = a.coeff.reshape(-1, 1) * b.coeff.reshape(1, -1)
                n1n2 = a.N.reshape(-1, 1) * b.N.reshape(1, -1)
                
                # compute all expansion coefficients
                E = np.stack((
                    MMD_E(i, j, 0, alpha, beta, Ax, Bx),
                    MMD_E(k, l, 0, alpha, beta, Ay, By),
                    MMD_E(m, n, 0, alpha, beta, Az, Bz)
                ), axis=0)
                # compute all derivatives of the expansion coefficients
                E_dx = np.stack((
                    MMD_dEdx(i, j, 0, 1, alpha, beta, Ax, Bx),
                    MMD_dEdx(k, l, 0, 1, alpha, beta, Ay, By),
                    MMD_dEdx(m, n, 0, 1, alpha, beta, Az, Bz)
                ), axis=0)
                # apply product rule to compute final derivative
                dS_dAx = np.stack([
                    (E_dx[0] * E[1] * E[2]),  # x-derivative
                    (E[0] * E_dx[1] * E[2]),  # y-derivative
                    (E[0] * E[1] * E_dx[2]),  # z-derivative
                ], axis=0)

                # scale and multiply with integral of hermitian
                dS_dAx *= c1c2 * n1n2 * (np.pi / (alpha + beta)) ** 1.5
                # sum over combinations of alpha and beta
                dS_dAx = np.sum(dS_dAx, axis=(1, 2))

                # add to gradient matrix
                dS_dA[ii, jj, ai, :] += dS_dAx
                dS_dA[ii, jj, aj, :] -= dS_dAx

        return dS_dA

    def kinetic_deriv(self, molecule:Molecule) -> np.ndarray:
        """ Compute derivative of kinetic energy matrix for a given molecule.
        
            Args:
                molecule (Molecule): molecule for which to compute the gradient
        
            Returns:
                dTdA (np.ndarray): 
                    the kinetic energy derivative matrix of shape (m, m, n, 3) where
                    m is the number of basis elements and n the number of atoms.
        """
        # read values from molecule
        Z = molecule.Zs
        C = molecule.origins
        basis = molecule.basis
        # number of atoms and length of basis
        n_atoms = len(molecule)
        n_basis = len(basis)
        # overlap derivative matrix
        dT_dA = np.zeros((n_basis, n_basis, n_atoms, 3))
        
        # compute gradient of overlap of all combinations of atoms
        for ai, aj in product(range(n_atoms), repeat=2):
            for bi, bj in product(
                range(len(molecule.atoms[ai])), 
                range(len(molecule.atoms[aj]))
            ):
                # get basis elements
                a = molecule.atoms[ai].basis[bi]
                b = molecule.atoms[aj].basis[bj]
                # compute global index of basis elements
                ii = sum(map(len, molecule.atoms[:ai])) + bi
                jj = sum(map(len, molecule.atoms[:aj])) + bj
                # check global index
                assert basis[ii] == a
                assert basis[jj] == b            
            
                # gather all values from GTOs
                (i, k, m), (j, l, n) = a.angular, b.angular
                (Ax, Ay, Az), (Bx, By, Bz) = a.origin, b.origin
                # reshape to compute pairwise values when combined
                alpha, beta = a.alpha.reshape(-1, 1), b.alpha.reshape(1, -1)
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
        
                # compute all directional overlap derivatives
                Sx_dx = MMD_dEdx(i, j, 0, 1, alpha, beta, Ax, Bx)
                Sy_dy = MMD_dEdx(k, l, 0, 1, alpha, beta, Ay, By)
                Sz_dz = MMD_dEdx(m, n, 0, 1, alpha, beta, Az, Bz)
                # compute partial derivatives of kinetic terms
                Tx_dx = j * (j - 1) * MMD_dEdx(i, j-2, 0, 1, alpha, beta, Ax, Bx) - \
                    2.0 * beta * (2.0 * j + 1.0) * MMD_dEdx(i, j, 0, 1, alpha, beta, Ax, Bx) + \
                    4.0 * beta**2 * MMD_dEdx(i, j+2, 0, 1, alpha, beta, Ax, Bx)
                Ty_dy = l * (l - 1) * MMD_dEdx(k, l-2, 0, 1, alpha, beta, Ay, By) - \
                    2.0 * beta * (2.0 * l + 1.0) * MMD_dEdx(k, l, 0, 1, alpha, beta, Ay, By) + \
                    4.0 * beta**2 * MMD_dEdx(k, l+2, 0, 1, alpha, beta, Ay, By)
                Tz_dz = n * (n - 1) * MMD_dEdx(m, n-2, 0, 1, alpha, beta, Az, Bz) - \
                    2.0 * beta * (2.0 * n + 1.0) * MMD_dEdx(m, n, 0, 1, alpha, beta, Az, Bz) + \
                    4.0 * beta**2 * MMD_dEdx(m, n+2, 0, 1, alpha, beta, Az, Bz)

                # build gradient
                dT_dAx = np.stack((
                    (Tx_dx * Sy * Sz + Sx_dx * Ty * Sz + Sx_dx * Sy * Tz),
                    (Tx * Sy_dy * Sz + Sx * Ty_dy * Sz + Sx * Sy_dy * Tz),
                    (Tx * Sy * Sz_dz + Sx * Ty * Sz_dz + Sx * Sy * Tz_dz)
                ), axis=0)
                # scale and multiply with integral of hermitian 
                dT_dAx = c1c2 * n1n2 * dT_dAx * (np.pi / (alpha + beta)) ** 1.5
                dT_dAx = -0.5 * np.sum(dT_dAx, axis=(1, 2))
                # add to total gradient
                dT_dA[ii, jj, ai, :] += dT_dAx
                dT_dA[ii, jj, aj, :] -= dT_dAx

        return dT_dA

    def electron_nuclear_attraction_deriv(self, molecule:Molecule) -> np.ndarray:
        """ Compute the derivative of electron-nuclear attraction energy for the given molecule.
            
            Args:
                molecule (Molecule): molecule for which to compute the gradient
        
            Returns:
                dVdA (np.ndarray): 
                    the attraction energy derivative matrix of shape (m, m, n, 3) where
                    m is the number of basis elements and n the number of atoms.
        """

        # read values from molecule
        Z = molecule.Zs
        C = molecule.origins
        basis = molecule.basis
        # number of atoms and length of basis
        n_atoms = len(molecule)
        n_basis = len(basis)
        # overlap derivative matrix
        dV_dA = np.zeros((n_basis, n_basis, n_atoms, 3))
        
        # compute gradient of overlap of all combinations of atoms
        for ai, aj in product(range(n_atoms), repeat=2):
            for bi, bj in product(
                range(len(molecule.atoms[ai])), 
                range(len(molecule.atoms[aj]))
            ):
                # get basis elements
                a = molecule.atoms[ai].basis[bi]
                b = molecule.atoms[aj].basis[bj]
                # compute global index of basis elements
                ii = sum(map(len, molecule.atoms[:ai])) + bi
                jj = sum(map(len, molecule.atoms[:aj])) + bj
                # check global index
                assert basis[ii] == a
                assert basis[jj] == b            

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

                # compute all expansion coefficients
                Ex = np.stack([MMD_E(i, j, t, alpha, beta, Ax, Bx) for t in range(i+j+1)], axis=0)
                Ey = np.stack([MMD_E(k, l, u, alpha, beta, Ay, By) for u in range(k+l+1)], axis=0)
                Ez = np.stack([MMD_E(m, n, v, alpha, beta, Az, Bz) for v in range(m+n+1)], axis=0)

                # compute all derivatives of expansion coefficients
                Ex_dx = np.stack([MMD_dEdx(i, j, t, 1, alpha, beta, Ax, Bx) for t in range(i+j+1)], axis=0)
                Ey_dy = np.stack([MMD_dEdx(k, l, u, 1, alpha, beta, Ay, By) for u in range(k+l+1)], axis=0)
                Ez_dz = np.stack([MMD_dEdx(m, n, v, 1, alpha, beta, Az, Bz) for v in range(m+n+1)], axis=0)

                # reshape to compute outer products when combined
                Ex, Ex_dx = Ex[:, None, None, ...], Ex_dx[:, None, None, ...]
                Ey, Ey_dy = Ey[None, :, None, ...], Ey_dy[None, :, None, ...]
                Ez, Ez_dz = Ez[None, None, :, ...], Ez_dz[None, None, :, ...]

                # compute all auxiliary hermite integras
                R = np.stack([
                        MMD_R(t, u, v, 0, alpha + beta, P, C)
                        for t, u, v in product(range(i+j+2), range(k+l+2), range(m+n+2))
                ], axis=0)
                R = R.reshape(i+j+2, k+l+2, m+n+2, alpha.shape[0], beta.shape[1], C.shape[0])

                # get the derivatives from the hermite integrals
                R_dx = R[1:, :-1, :-1, ...]
                R_dy = R[:-1, 1:, :-1, ...]
                R_dz = R[:-1, :-1, 1:, ...]
                # get the hermite integrals
                R = R[:-1, :-1, :-1, ...]

                # compute the gradient w.r.t (A - B)
                dV_dR = np.stack([
                    Ex_dx * Ey * Ez * R,
                    Ex * Ey_dy * Ez * R,
                    Ex * Ey * Ez_dz * R,
                ], axis=0)
                # compute the gradient w.r.t. P
                dV_dP = np.stack([
                    Ex * Ey * Ez * R_dx,
                    Ex * Ey * Ez * R_dy,
                    Ex * Ey * Ez * R_dz
                ], axis=0)

                # prefactor
                f = 2.0 * np.pi / (alpha + beta) * c1c2 * n1n2 * -Z
                # combine to obtain gradient w.r.t A and B
                dV_dA[ii, jj, ai, :] += np.sum(f * (alpha / (alpha + beta) * dV_dP + dV_dR), axis=tuple(range(1, 7)))
                dV_dA[ii, jj, aj, :] += np.sum(f * (beta  / (alpha + beta) * dV_dP - dV_dR), axis=tuple(range(1, 7)))

        return dV_dA

    def electron_electron_repulsion_deriv(
        self,
        molecule:Molecule
    ) -> np.ndarray:
        """ Compute the derivative of the electron-electron repulsion energy matrix for the given molecule.

            Args:
                molecule (Molecule): molecule for which to compute the gradient
        
            Returns:
                dVdA (np.ndarray): 
                    the repulsion energy derivative matrix of shape (m, m, m, m, n, 3) where
                    m is the number of basis elements and n the number of atoms.
        """
        # TODO: not implemented!
        m, n = len(molecule.basis), len(molecule)
        return np.zeros((m, m, m, m, n, 3))

    def nuclear_nuclear_repulsion_deriv(
        self, 
        molecule:Molecule
    ) -> np.ndarray:
        """ Compute the derivative of nuclear-nuclear repulsion energy for the given molecule.
            
            Args:
                molecule (Molecule): molecule for which to compute the gradient
        
            Returns:
                dVdA (np.ndarray): 
                    the repulsion energy derivative matrix of shape (n, 3) where
                    n the number of atoms.
        """
        

        Z = molecule.Zs
        C = molecule.origins
        # compute pairwise distances between nuclei
        # and set diagonal to one to avoid divison by zero
        R = pdist(C, metric='euclidean')
        R = squareform(R)
        np.fill_diagonal(R, 1.0)
        # compute gradient
        return Z[:, None] * np.sum(
            Z[None, :, None] * (C[None, :, :] - C[:, None, :]) / \
            (R[:, :, None] ** 3),
            axis=1
        )

    def hartree_fock_energy_deriv(
        self, 
        molecule:Molecule,
        **kwargs
    ):
        """ Compute the derivative of the hartree fock energy w.r.t the
            atom origins. See equation (C.12) in 'Modern Quantum Chemistry'
            (1989) by Szabo and Ostlund.

            Args:
                molecule (Molecule): molecule state at which to compute the gradient
                **kwargs (Any): 
                    additional keyword arguments forwarded to the restricted
                    hartree fock routine (see `ElectronicTISE.restricted_hartree_fock`)

            Return:
                E (float): hartree fock energy of the given molecule
                dEdx (np.ndarray): gradient of the hartree fock energy
        """
        # read kwargs
        num_occ_orbitals = kwargs.get('num_occ_orbitals', 1)
        # solve using restricted hartree fock
        tise = ElectronicTISE.from_molecule(molecule)
        E, C, F = tise.restricted_hartree_fock(**kwargs)
        
        # compute density matrix and scaled density matrix
        F_part = F[:num_occ_orbitals]
        C_part = C[:, :num_occ_orbitals]
        P = 2.0 * C_part @ C_part.T.conjugate()
        Q = 2.0 * (C_part * F_part) @ C_part.T.conjugate()

        # compute gradients of molecular integrals
        S_dx = self.overlap_deriv(molecule)
        T_dx = self.kinetic_deriv(molecule)
        V_en_dx = self.electron_nuclear_attraction_deriv(molecule)
        V_ee_dx = self.electron_electron_repulsion_deriv(molecule)
        V_nn_dx = self.nuclear_nuclear_repulsion_deriv(molecule)
        # compute gradient of G
        J_dx, K_dx = V_ee_dx, V_ee_dx.swapaxes(1, 3)
        G_dx = np.sum(P[None, None, :, :, None, None] * (J_dx - 0.5 * K_dx), axis=(2, 3))
        # combine to full gradient
        dEdx = (
            P[:, :, None, None] * (T_dx + V_en_dx) + \
            P[:, :, None, None] * G_dx - \
            Q[:, :, None, None] * S_dx
        ).sum(axis=(0, 1)) + V_nn_dx

        # return energy and gradient
        return E, dEdx

    def optimize(
        self,
        molecule:Molecule,
        iterations:int,
        return_energy_history:bool =False,
        **kwargs
    ) -> Molecule:
        """ Optimize geometry of given molecule 

            Args:
                molecule (Molecule): molecule to optimize
                iterations (int): maximum number of update iterations
                return_energy_history (bool): 
                    whether to return the energy history of the optimization.
                    Defaults to False.
                **kwargs (Any):
                    additional keyword arguments forwarded to the restricted
                    hartree fock routine (see `ElectronicTISE.restricted_hartree_fock`)

            Return:
                molecule (Molecule): molecule with optimized structure
                Es (List[float]): energy history
        """
        # copy molecule to avoid overwriting the given instance
        molecule = deepcopy(molecule)

        # track energies for each iteration
        Es = [float('inf')]
        for _ in range(iterations):
            
            # compute gradient and compute new geometry
            E, dEdx = self.hartree_fock_energy_deriv(molecule, **kwargs)
            C_new = molecule.origins - self.step * dEdx
            # update atom origins
            for j, atom in enumerate(molecule.atoms):
                atom.origin = C_new[j, :]

            # add energy value to list
            Es.append(E)

            # check for convergence
            if abs(Es[-1] - Es[-2]) < self.tol:
                break

        # return molecule and energy history
        return (molecule, Es[1:]) if return_energy_history else molecule
