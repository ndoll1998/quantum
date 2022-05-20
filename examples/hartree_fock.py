import numpy as np
from quantum.chemistry import GaussianOrbital, ElectronicTISE

if __name__ == '__main__':

    # hydrogen 1s orbital in STO-3G basis
    # taken from basis-set-exchange
    H1_1s = GaussianOrbital(
        alpha=[0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00],
        coeff=[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00],
        origin=[0.0, 0, 0.0],
        angular=[0, 0, 0]
    )
    H2_1s = GaussianOrbital(
        alpha=[0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00],
        coeff=[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00],
        origin=[0.0, 0.0, 1.4],
        angular=[0, 0, 0]
    )

    # H-H molecule
    molecule = [H1_1s, H2_1s]
    
    # nuclei positions
    C = np.asarray([
        [0, 0, 0],
        [0, 0, 1.4]
    ])
    # principal quantum number (charge) per atom
    Z = np.asarray([1, 1])

    # solve electroinic schroedinger equation using hartree-fock
    tise = ElectronicTISE(molecule, C, Z)
    wfs = tise.solve()
