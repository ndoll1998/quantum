import quantum
import matplotlib.pyplot as plt
from itertools import combinations

def geometry_optimization():
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
    # some random initial position
    H2.origin = [3, -1, 4]

    optim = quantum.chemistry.GradientDescentGeometryOptimizer(
        mol=H1+H2,
        alpha=0.4
    )
    # optimize molecular geometry
    Es = optim.optimize(max_iters=50, tol=-1)

    # create figure for plots
    fig = plt.figure(figsize=(8, 4), tight_layout=True)
    ax = fig.add_subplot(1, 2, 1)
    # plot optimization course
    ax.plot(Es)
    ax.grid()
    ax.set(
        title="Gradient Descent Optimization",
        xlabel="iteration",
        ylabel="Energy in Hartree ($Ha$)"
    )
    
    # plot atom origins and bonds
    C = optim.molecule.origins
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(C[:, 0], C[:, 1], C[:, 2], color='blue')
    # bonds between all atoms
    for i, j in combinations(range(len(optim.molecule)), 2):
        ax.plot(C[(i, j), 0], C[(i, j), 1], C[(i, j), 2], color='blue')
    ax.set(title="Molecular Geometry")

    return fig

if __name__ == '__main__':

    fig = geometry_optimization()
    fig.savefig("/mnt/c/users/Nicla/OneDrive/Bilder/img.png")
