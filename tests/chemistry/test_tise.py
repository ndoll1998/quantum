import numpy as np
from quantum.chemistry.structs import Atom
from quantum.chemistry.orbital import GaussianOrbital
from quantum.chemistry.tise import ElectronicTISE

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

class TestElectronicTISE(object):

    def test_rhf_energy_hydrogen(self):
        # create hydrogen atoms
        H1 = Atom(basis=H_basis, Z=1)
        H2 = Atom(basis=H_basis, Z=1)    
        # convert from angstrom to bohr distance
        H2.origin = [0, 0, 0.74 / 0.529177]

        # solve
        tise = ElectronicTISE.from_molecule(H1+H2)
        E, _, _ = tise.restricted_hartree_fock()
        # check with expectation
        assert np.allclose(E, -1.1167592920796137, atol=1e-10)
    
    def test_rhf_energy_oxygen(self):
        # build hydrogen and oxygen atoms
        H1 = Atom(basis=H_basis, Z=1)
        H2 = Atom(basis=H_basis, Z=1)
        O = Atom(basis=O_basis, Z=8)

        # position as dipole
        H1.origin = np.asarray([-0.866811829, 0.601435779, 0]) / 0.529177
        H2.origin = np.asarray([0.866811829, 0.601435779, 0]) / 0.529177
        O.origin =  np.asarray([0, -0.075791844, 0]) / 0.529177

        # solve
        tise = ElectronicTISE.from_molecule(O+H1+H2)
        E, _, _ = tise.restricted_hartree_fock()
        # check with expectation
        assert np.allclose(E, -74.94207976229343, atol=1e-10)
    
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
        tise = ElectronicTISE.from_molecule(C+H1+H2+H3+H4)
        E, _, _ = tise.restricted_hartree_fock()
        # check with expectation
        np.testing.assert_allclose(E, -39.726851323525985, atol=1e-10)
