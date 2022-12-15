import pytest
import numpy as np
from quantum.chemistry.structs import Atom
from quantum.chemistry.orbital import GaussianOrbital
from quantum.chemistry.tise import ElectronicTISE
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
# STO-3G basis for carbon atom
C_basis = [
    GaussianOrbital(
        alpha=[0.7161683735E+02, 0.1304509632E+02, 0.3530512160E+01],
        coeff=[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+0],
        origin=[0, 0, 0.0],
        angular=[0, 0, 0]
    ),
    GaussianOrbital(
        alpha=[0.2941249355E+01, 0.6834830964E+00, 0.2222899159E+00],
        coeff=[-0.9996722919E-01, 0.3995128261E+00, 0.7001154689E+00],
        origin=[0, 0, 0.0],
        angular=[0, 0, 0]
    ),
    GaussianOrbital(
        alpha=[0.2941249355E+01, 0.6834830964E+00, 0.2222899159E+00],
        coeff=[0.1559162750E+00, 0.6076837186E+00, 0.3919573931E+00],
        origin=[0, 0, 0.0],
        angular=[1, 0, 0]
    ),
    GaussianOrbital(
        alpha=[0.2941249355E+01, 0.6834830964E+00, 0.2222899159E+00],
        coeff=[0.1559162750E+00, 0.6076837186E+00, 0.3919573931E+00],
        origin=[0, 0, 0.0],
        angular=[0, 1, 0]
    ),
    GaussianOrbital(
        alpha=[0.2941249355E+01, 0.6834830964E+00, 0.2222899159E+00],
        coeff=[0.1559162750E+00, 0.6076837186E+00, 0.3919573931E+00],
        origin=[0, 0, 0.0],
        angular=[0, 0, 1]
    )
]

class TestRhfEnergy(object):
    """ Compare RHF energies of different molecules to values reported 
        in tests of https://github.com/jjgoings/McMurchie-Davidson
    """

    def test_rhf_energy_hydrogen(self):
        # create hydrogen atoms
        H1 = Atom(basis=H_basis, Z=1)
        H2 = Atom(basis=H_basis, Z=1)    
        # convert from angstrom to bohr distance
        H2.origin = [0, 0, 0.74 / 0.529177]

        # solve
        tise = ElectronicTISE(H1+H2)
        E, _, _ = tise.restricted_hartree_fock()
        # check with expectation
        np.testing.assert_allclose(E, -1.1167592920796137, atol=1e-10)
    
    def test_rhf_energy_water(self):
        # build hydrogen and oxygen atoms
        H1 = Atom(basis=H_basis, Z=1)
        H2 = Atom(basis=H_basis, Z=1)
        O = Atom(basis=O_basis, Z=8)
        # position as dipole
        H1.origin = np.asarray([-0.866811829, 0.601435779, 0]) / 0.529177
        H2.origin = np.asarray([0.866811829, 0.601435779, 0]) / 0.529177
        O.origin =  np.asarray([0, -0.075791844, 0]) / 0.529177

        # solve
        tise = ElectronicTISE(O+H1+H2)
        E, _, _ = tise.restricted_hartree_fock()
        # check with expectation
        np.testing.assert_allclose(E, -74.94207976229343, atol=1e-10)
    
    def test_rhf_energy_methane(self):
        # build hydrogen and carbon atoms
        H1 = Atom(basis=H_basis, Z=1)
        H2 = Atom(basis=H_basis, Z=1)
        H3 = Atom(basis=H_basis, Z=1)
        H4 = Atom(basis=H_basis, Z=1)
        C = Atom(basis=C_basis, Z=6)

        # position hydrogen around origin, i.e. around carbon atom
        H1.origin = np.asarray([1, 1, 1]) * 0.626425042 / 0.529177
        H2.origin = np.asarray([1, -1, -1]) * 0.626425042 / 0.529177
        H3.origin = np.asarray([-1, 1, -1]) * 0.626425042 / 0.529177
        H4.origin = np.asarray([-1, -1, 1]) * 0.626425042 / 0.529177

        # solve
        tise = ElectronicTISE(C+H1+H2+H3+H4)
        E, _, _ = tise.restricted_hartree_fock()
        # check with expectation
        np.testing.assert_allclose(E, -39.726851323525985, atol=1e-10)


class TestIntegralsWater(object):
    """ Compare integrals with values reported in:
        https://chemistry.montana.edu/callis/courses/chmy564/460water.pdf
    """

    def water(self):
        # build hydrogen and oxygen atoms
        H1 = Atom(basis=H_basis, Z=1)
        H2 = Atom(basis=H_basis, Z=1)
        O = Atom(basis=O_basis, Z=8)
        # set origins
        H1.origin = np.asarray([0.0, 0.751155, -0.465285]) / 0.529177
        H2.origin = np.asarray([0.0, -0.751155, -0.465285]) / 0.529177
        O.origin =  np.asarray([0.0, 0.0, 0.116321]) / 0.529177
        return O+H1+H2

    def test_overlap(self):
        # compute overlap matrix and compare
        np.testing.assert_allclose(
            ElectronicTISE(self.water()).S,
            # see Figure 1
            np.asarray([
                [1.000, 0.237, 0.000,  0.000,  0.000,  0.055,  0.055],
                [0.237, 1.000, 0.000,  0.000,  0.000,  0.479,  0.479],
                [0.000, 0.000, 1.000,  0.000,  0.000,  0.000,  0.000],
                [0.000, 0.000, 0.000,  1.000,  0.000,  0.313, -0.313],
                [0.000, 0.000, 0.000,  0.000,  1.000, -0.242, -0.242],
                [0.055, 0.479, 0.000,  0.313, -0.242,  1.000,  0.256],
                [0.055, 0.479, 0.000, -0.313, -0.242,  0.256,  1.000]
            ]),
            atol=1e-3
        )

    def test_kinetic(self):
        np.testing.assert_allclose(
            ElectronicTISE(self.water()).T,
            # see Figure 2
            np.asarray([
                [29.003, -0.168, 0.000,  0.000,  0.000, -0.002, -0.002],
                [-0.168,  0.808, 0.000,  0.000,  0.000,  0.132,  0.132],
                [ 0.000,  0.000, 2.529,  0.000,  0.000,  0.000,  0.000],
                [ 0.000,  0.000, 0.000,  2.529,  0.000,  0.229, -0.229],
                [ 0.000,  0.000, 0.000,  0.000,  2.529, -0.177, -0.177],
                [-0.002,  0.132, 0.000,  0.229, -0.177,  0.760,  0.009],
                [-0.002,  0.132, 0.000, -0.229, -0.177,  0.009,  0.760]
            ]),
            atol=1e-3
        )
        
    def test_attraction(self):
        np.testing.assert_allclose(
            ElectronicTISE(self.water()).V_en,
            # see Figure 3
            np.asarray([
                [-61.733,  -7.447,  0.000,   0.000,   0.019, -1.778, -1.778],
                [ -7.447, -10.151,  0.000,   0.000,   0.226, -3.920, -3.920],
                [  0.000,   0.000, -9.993,   0.000,   0.000,  0.000,  0.000],
                [  0.000,   0.000,  0.000, -10.152,   0.000, -2.277,  2.277],
                [  0.019,   0.226,  0.000,   0.000, -10.088,  1.837,  1.837],
                [ -1.778,  -3.920,  0.000,  -2.277,   1.837, -5.867, -1.652],
                [ -1.778,  -3.920,  0.000,   2.277,   1.837, -1.652, -5.867]
            ]),
            atol=1e-3
        )
    
    def test_rhf_energy(self):
        # solve
        tise = ElectronicTISE(self.water())
        E, _, _ = tise.restricted_hartree_fock()
        # check with expectation
        np.testing.assert_allclose(E, -74.96175778258913, atol=1e-10)


class TestIntegralGradients(object):
    
    def random_water(self):
        # create atoms
        H1 = Atom(H_basis, Z=1)
        H2 = Atom(H_basis, Z=1)
        H3 = Atom(H_basis, Z=1)
        O = Atom(O_basis, Z=8)
        # randomize origins
        H1.origin = np.random.uniform(-1, 1, size=3)
        H2.origin = np.random.uniform(-1, 1, size=3)
        H3.origin = np.random.uniform(-1, 1, size=3)
        O.origin = np.random.uniform(-1, 1, size=3)
        # return molecule
        return H1+H2+O

    @pytest.mark.parametrize('num_runs', range(3))
    def test_overlap_gradient_water(self, num_runs):
        # create molecule and compute analytical gradient
        mol = self.random_water()
        S_grad = ElectronicTISE(mol).S_grad

        eps = 1e-5
        for d in range(3):
            dx = np.eye(3)[d] * eps

            for i, A in enumerate(mol.atoms):
                # perturb origin of i-th atom
                mol_pos_pert = deepcopy(mol)
                mol_neg_pert = deepcopy(mol)
                mol_pos_pert.atoms[i].origin = A.origin + dx
                mol_neg_pert.atoms[i].origin = A.origin - dx
                # approximate partial derivative
                dSdx = (
                    ElectronicTISE(mol_pos_pert).S - \
                    ElectronicTISE(mol_neg_pert).S
                ) / (2.0 * eps)
                # check
                m = (mol.basis_atom_ids != i) # should all be zero
                np.testing.assert_allclose(S_grad[i, m, m, d], 0, atol=eps)
                np.testing.assert_allclose(S_grad[i, :, :, d], dSdx, atol=eps)
    
    @pytest.mark.parametrize('num_runs', range(3))
    def test_kinetic_gradient_water(self, num_runs):
        # create molecule and compute analytical gradient
        mol = self.random_water()
        T_grad = ElectronicTISE(mol).T_grad

        eps = 1e-5
        for d in range(3):
            dx = np.eye(3)[d] * eps

            for i, A in enumerate(mol.atoms):
                # perturb origin of i-th atom
                mol_pos_pert = deepcopy(mol)
                mol_neg_pert = deepcopy(mol)
                mol_pos_pert.atoms[i].origin = A.origin + dx
                mol_neg_pert.atoms[i].origin = A.origin - dx
                # approximate partial derivative
                dTdx = (
                    ElectronicTISE(mol_pos_pert).T - \
                    ElectronicTISE(mol_neg_pert).T
                ) / (2.0 * eps)
                # check
                m = (mol.basis_atom_ids != i) # should all be zero
                np.testing.assert_allclose(T_grad[i, m, m, d], 0, atol=eps)
                np.testing.assert_allclose(T_grad[i, :, :, d], dTdx, atol=eps)
    
    @pytest.mark.parametrize('num_runs', range(3))
    def test_attraction_gradient_water(self, num_runs):
        # create molecule and compute analytical gradient
        mol = self.random_water()
        V_grad = ElectronicTISE(mol).V_en_grad

        eps = 1e-5
        for d in range(3):
            dx = np.eye(3)[d] * eps

            for i, A in enumerate(mol.atoms):
                # perturb origin of i-th atom
                mol_pos_pert = deepcopy(mol)
                mol_neg_pert = deepcopy(mol)
                mol_pos_pert.atoms[i].origin = A.origin + dx
                mol_neg_pert.atoms[i].origin = A.origin - dx
                # approximate partial derivative
                dVdx = (
                    ElectronicTISE(mol_pos_pert).V_en - \
                    ElectronicTISE(mol_neg_pert).V_en
                ) / (2.0 * eps)
                # check
                np.testing.assert_allclose(V_grad[i, :, :, d], dVdx, atol=eps)
    
    @pytest.mark.parametrize('num_runs', range(3))
    def test_electron_electron_repulsion_gradient_water(self, num_runs):
        # create molecule and compute analytical gradient
        mol = self.random_water()
        V_grad = ElectronicTISE(mol).V_ee_grad

        eps = 1e-5
        for d in range(3):
            dx = np.eye(3)[d] * eps

            for i, A in enumerate(mol.atoms):
                # perturb origin of i-th atom
                mol_pos_pert = deepcopy(mol)
                mol_neg_pert = deepcopy(mol)
                mol_pos_pert.atoms[i].origin = A.origin + dx
                mol_neg_pert.atoms[i].origin = A.origin - dx
                # approximate partial derivative
                dVdx = (
                    ElectronicTISE(mol_pos_pert).V_ee - \
                    ElectronicTISE(mol_neg_pert).V_ee
                ) / (2.0 * eps)
                # check
                np.testing.assert_allclose(V_grad[i, ..., d], dVdx, atol=eps)
    
    @pytest.mark.parametrize('num_runs', range(3))
    def test_nuclear_nuclear_repulsion_gradient_water(self, num_runs):
        # create molecule and compute analytical gradient
        mol = self.random_water()
        E_grad = ElectronicTISE(mol).E_nn_grad

        eps = 1e-5
        for d in range(3):
            dx = np.eye(3)[d] * eps
            
            for i, A in enumerate(mol.atoms):
                # perturb origin of i-th atom
                mol_pos_pert = deepcopy(mol)
                mol_neg_pert = deepcopy(mol)
                mol_pos_pert.atoms[i].origin = A.origin + dx
                mol_neg_pert.atoms[i].origin = A.origin - dx
                # approximate partial derivative
                dEdx = (
                    ElectronicTISE(mol_pos_pert).E_nn - \
                    ElectronicTISE(mol_neg_pert).E_nn
                ) / (2.0 * eps)
                # check
                np.testing.assert_allclose(E_grad[i, d], dEdx, atol=eps)
