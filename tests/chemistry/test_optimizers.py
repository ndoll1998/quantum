import pytest
import numpy as np
from quantum.chemistry.structs import Atom
from quantum.chemistry.orbital import GaussianOrbital
from quantum.chemistry.tise import ElectronicTISE
from quantum.chemistry.optimizers import GradientDescentGeometryOptimizer
from copy import deepcopy

# STO-3G basis for hydrogen atom
H_basis = [
    GaussianOrbital(
        alpha=[0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00],
        coeff=[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00],
        origin=[0, 0, 0.0],
        angular=[0, 0, 0]
    )
]
# STO-3G basis for oxygen atom
O_basis = [
    GaussianOrbital(
        alpha=[0.1307093214E+03, 0.2380886605E+02, 0.6443608313E+01],
        coeff=[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00],
        origin=[0, 0, 0.0],
        angular=[0, 0, 0]
    ),
    GaussianOrbital(
        alpha=[0.5033151319E+01, 0.1169596125E+01, 0.3803889600E+00],
        coeff=[-0.9996722919E-01, 0.3995128261E+00,0.7001154689E+00],
        origin=[0, 0, 0.0],
        angular=[0, 0, 0]
    ),
    GaussianOrbital(
        alpha=[0.5033151319E+01, 0.1169596125E+01, 0.3803889600E+00],
        coeff=[0.1559162750E+00, 0.6076837186E+00, 0.3919573931E+00],
        origin=[0, 0, 0.0],
        angular=[1, 0, 0]
    ),
    GaussianOrbital(
        alpha=[0.5033151319E+01, 0.1169596125E+01, 0.3803889600E+00],
        coeff=[0.1559162750E+00, 0.6076837186E+00, 0.3919573931E+00],
        origin=[0, 0, 0.0],
        angular=[0, 1, 0]
    ),
    GaussianOrbital(
        alpha=[0.5033151319E+01, 0.1169596125E+01, 0.3803889600E+00],
        coeff=[0.1559162750E+00, 0.6076837186E+00, 0.3919573931E+00],
        origin=[0, 0, 0.0],
        angular=[0, 0, 1]
    )
]

class TestGradientDescentGeometryOptimizer(object):

    def random_water(self):
        # create atoms
        H1 = Atom(H_basis, Z=1)
        H2 = Atom(H_basis, Z=1)
        O = Atom(O_basis, Z=8)
        # randomize origins
        H1.origin = np.random.uniform(-1, 1, size=3)
        H2.origin = np.random.uniform(-1, 1, size=3)
        O.origin = np.random.uniform(-1, 1, size=3)
        # return molecule
        return H1+H2+O

    @pytest.mark.parametrize('num_runs', range(3))
    def test_gradient_water(self, num_runs):

        mol = self.random_water()
        # get molecule parameters
        basis = mol.basis
        # these are constant, note that
        # atom origins are assumed constant
        C = mol.origins
        Z = mol.Zs

        # compute gradient
        optim = GradientDescentGeometryOptimizer(mol)
        grads = optim.compute_origin_gradients()
        # subtract nuclear-nuclear repulsion term as this is not
        # accounted for in approximation (only the GTO origins are
        # permuted but the atom positions stay constant, thus the
        # nuclear-nuclear repulsion gradient is zero)
        # note that correctness of nuclear-nuclear repulsion
        # is tested seperately (see `TestIntegralGradients` in
        # `tests/chemistry/test_tise.py`)
        grads -= optim.state.tise.E_nn_grad(mol)

        # finite difference approximation for each dimension
        for d in range(3):
            eps = 1e-5
            dx = np.eye(3)[d] * eps
            # approximate gradient of each atom
            for ai, A in enumerate(mol.atoms):

                dEdA = 0.0
                # sum up gradient approximations of each
                # basis element of the current atom
                for bi, b in enumerate(A.basis):
                    # global index in basis
                    i = sum(map(len, mol.atoms[:ai])) + bi
                    # permute basis
                    basis_pos_pert = deepcopy(basis)
                    basis_pos_pert[i].origin = b.origin + dx
                    basis_neg_pert = deepcopy(basis)
                    basis_neg_pert[i].origin = b.origin - dx
                    # solve for permuted energies
                    E_p, _, _ = ElectronicTISE(basis_pos_pert, C, Z).restricted_hartree_fock()
                    E_n, _, _ = ElectronicTISE(basis_neg_pert, C, Z).restricted_hartree_fock()
                    # finite difference approximation
                    dEdA += (E_p - E_n) / (2.0 * eps)

                # compute
                np.testing.assert_allclose(grads[ai, d], dEdA, atol=1e-3)
