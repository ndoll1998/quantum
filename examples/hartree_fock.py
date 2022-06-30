import numpy as np
from quantum.chemistry import GaussianOrbital, ElectronicTISE
import matplotlib.pyplot as plt

def dissociation_curve_H2():
    # energies at different nuclei-distances
    ds = np.linspace(0.5, 8, 128) # distances in bohr
    Es = []

    for x in ds:
        # H-H molecule
        molecule = [
            # hydrogen 1s orbital in STO-3G basis taken
            # from basis-set-exchange
            GaussianOrbital(
                alpha=[0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00],
                coeff=[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00],
                origin=[0.0, 0, 0.0],
                angular=[0, 0, 0]
            ),
            # hydrogen 1s orbital in STO-3G basis with origin shifted by x
            GaussianOrbital(
                alpha=[0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00],
                coeff=[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00],
                origin=[0.0, 0.0, x],
                angular=[0, 0, 0]
            )
        ]
        # nuclei positions
        C = np.asarray([
            [0, 0, 0],
            [0, 0, x]
        ])
        # principal quantum number (charge) per atom
        Z = np.asarray([1, 1])

        # solve electroinic schroedinger equation using hartree-fock
        E, _ = ElectronicTISE(molecule, C, Z).restricted_hartree_fock()
        Es.append(E)

    # convert distances to angstrom for plotting
    ds *= 0.529177

    # find bonding energy
    min_idx = np.argmin(Es)
    min_x, min_y = ds[min_idx], Es[min_idx]
    
    # create figure and setup axis
    fig, ax = plt.subplots()
    ax.set(
        title="H-H Dissociation Curve",
        xlabel="Bond Distance $r$ in Angstrom ($\AA$)",
        ylabel="Total Energy $E$ in Hartree ($Ha$)"
    )
    ax.grid()
    # plot energy curve and mark minimum
    ax.plot(ds, Es, label="$E(r)$")
    ax.axvline(x=min_x, linestyle='--', label="$r=%.02f$" % min_x)
    # add legend
    ax.legend()

    return fig

if __name__ == '__main__':

    fig = dissociation_curve_H2()
    fig.savefig("./docs/H2_dissociation.png")
    fig.savefig("/mnt/c/users/Nicla/OneDrive/Bilder/img.png")
