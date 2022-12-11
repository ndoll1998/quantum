import quantum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap
from itertools import combinations

# color scheme
ATOM_COLORS = ListedColormap([
    "tab:orange",
    "tab:blue",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:olive",
    "tab:cyan"
])
BOND_COLOR = "tab:gray"

# atomix symbols used for plot legend
ATOM_SYMBOLS = [
    None, 
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Uut", "Fl", "Uup", "Lv", "Uus", "Uuo"
]
# valence radii of each atom in bohr units
# used to identify bonds for visualization
ATOM_VALENCE_RADII = np.array([
    0,
    230, 930, 680, 350, 830, 680, 680, 680, 640,
    1120, 970, 1100, 1350, 1200, 750, 1020, 990,
    1570, 1330, 990, 1440, 1470, 1330, 1350, 1350,
    1340, 1330, 1500, 1520, 1450, 1220, 1170, 1210,
    1220, 1210, 1910, 1470, 1120, 1780, 1560, 1480,
    1470, 1350, 1400, 1450, 1500, 1590, 1690, 1630,
    1460, 1460, 1470, 1400, 1980, 1670, 1340, 1870,
    1830, 1820, 1810, 1800, 1800, 1990, 1790, 1760,
    1750, 1740, 1730, 1720, 1940, 1720, 1570, 1430,
    1370, 1350, 1370, 1320, 1500, 1500, 1700, 1550,
    1540, 1540, 1680, 1700, 2400, 2000, 1900, 1880,
    1790, 1610, 1580, 1550, 1530, 1510, 1500, 1500,
    1500, 1500, 1500, 1500, 1500, 1500, 1600, 1600,
    1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600,
    1600, 1600, 1600, 1600, 1600, 1600
], dtype=np.float32) / 1000.0 * (1.0/0.529177)

def geometry_optimization():
    
    # create atoms at random positions
    H1 = quantum.chemistry.Atom.from_BSE("STO-3G", "H", origin=np.random.uniform(-1, 1, size=3))
    H2 = quantum.chemistry.Atom.from_BSE("STO-3G", "H", origin=np.random.uniform(-1, 1, size=3))
    H3 = quantum.chemistry.Atom.from_BSE("STO-3G", "H", origin=np.random.uniform(-1, 1, size=3))
    H4 = quantum.chemistry.Atom.from_BSE("STO-3G", "H", origin=np.random.uniform(-1, 1, size=3))
    C = quantum.chemistry.Atom.from_BSE("STO-3G", "C", origin=np.random.uniform(-1, 1, size=3))

    # optimize molecular geometry
    optim = quantum.chemistry.GradientDescentGeometryOptimizer(
        mol=H1+H2+H3+H4+C,
        alpha=0.03,
        rhf_max_cycles=1000
    )
    
    Es, Cs = [], []
    for i, (E, mol) in enumerate(optim.optimize(max_iters=50, tol=1e-5), 1):
        print("Step %i: RHF Energy %.05f" % (i, E))
        Es.append(E)
        Cs.append(mol.origins * 0.529177)

    print("Final RHF Energy: %.06f" % Es[-1])
   
 
    # create figure for plots
    fig = plt.figure(figsize=(8, 4), tight_layout=True)
    ax1 = fig.add_subplot(1, 2, 1)
    # plot optimization course
    E_curve, = ax1.plot(Es)
    ax1.grid()
    ax1.set(
        title="Gradient Descent Optimization",
        xlabel="iteration",
        ylabel="Energy in Hartree ($Ha$)"
    )
    
    # get atom charges and final positions
    # and convert to angstrom for visualization
    Z = optim.molecule.Zs
    C = Cs[-1]

    # set up axis
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set(title="Molecular Geometry ($\AA$)")
    # plot atoms
    atoms = ax2.scatter(
        C[:, 0], C[:, 1], C[:, 2],
        c=Z,
        cmap=ATOM_COLORS,
        s=100
    )
    # plot bonds between all atoms
    bonds = {
        (i, j): ax2.plot(
            C[(i, j), 0], C[(i, j), 1], C[(i, j), 2],
            color=BOND_COLOR
        )[0]
        for i, j in combinations(range(len(optim.molecule.atoms)), 2)
    }


    # add legend
    handles, labels = atoms.legend_elements()
    ax2.legend(
        handles=handles, 
        labels=[ATOM_SYMBOLS[int(l[14:-2])] for l in labels]
    )

    def update(k):
        # update energy curve
        E_curve.set_data(list(range(k+1)), Es[:k+1])
        # update atom positions
        atoms._offsets3d = (Cs[k][:, 0], Cs[k][:, 1], Cs[k][:, 2])
        # update bonds
        for (i, j), line in bonds.items():
            # get atoms
            Ai = optim.molecule.atoms[i]
            Aj = optim.molecule.atoms[j]
            # update bond points
            C = Cs[k][(i, j), :]
            line.set_data(C[:, 0], C[:, 1])
            line.set_3d_properties(C[:, 2])
            # check if bond is effective
            d = np.linalg.norm(C[0, :] - C[1, :])
            v = d < ATOM_VALENCE_RADII[Ai.Z] + ATOM_VALENCE_RADII[Aj.Z]
            # update visibility
            line.set_visible(v)

        return E_curve, atoms, *bonds.values()

    # animate
    anim = animation.FuncAnimation(
        fig, update, 
        frames=len(Es),
        interval=20,
        blit=True
    )

    return fig, anim

if __name__ == '__main__':
    
    fig, anim = geometry_optimization()
    anim.save("/mnt/c/users/Nicla/OneDrive/Bilder/anim.gif", fps=4)
    fig.savefig("/mnt/c/users/Nicla/OneDrive/Bilder/img.png")
