import quantum
import numpy as np
import matplotlib.pyplot as plt

def dissociation_curve_H2():
    # energies at different nuclei-distances
    ds = np.linspace(0.5, 8, 128) # distances in bohr
    Es = []

    # hydrogen 1s orbital in STO-3G basis
    basis = [
        quantum.chemistry.GaussianOrbital(
            alpha=[0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00],
            coeff=[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00],
            origin=[0.0, 0, 0.0],
            angular=[0, 0, 0]
        )
    ]
    # create atoms and molecule
    H1 = quantum.chemistry.Atom(basis=basis, Z=1)
    H2 = quantum.chemistry.Atom(basis=basis, Z=1)    
    molecule = H1 + H2

    for x in ds:
        # update distance between nuclei by updating position of H2
        H2.origin = [0, 0, x]
        # solve electroinic schroedinger equation using hartree-fock
        E, _, _ = quantum.chemistry.ElectronicTISE(molecule).restricted_hartree_fock()
        Es.append(E)

    # convert distances to angstrom for plotting
    ds *= 0.529177

    # find bonding energy
    min_idx = np.argmin(Es)
    min_x, min_y = ds[min_idx], Es[min_idx]
    
    # create figure and setup axis
    fig, ax = plt.subplots(figsize=(8, 5))
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
    fig.savefig("/mnt/c/users/Nicla/OneDrive/Bilder/img.png")
