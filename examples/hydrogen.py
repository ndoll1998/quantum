import quantum
import numpy as np
from scipy.integrate import simps
from skimage.measure import marching_cubes
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def hydrogen(
    t:float
) -> plt.Figure:

    # create analytic wave function
    wf = quantum.analytic.HydrogenLike(5, 3, 1)
    # radius containing the wave function
    R = 16

    # create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5), subplot_kw={"projection": "3d"}, tight_layout=True)
    fig.suptitle("Hydrogen n=%i,l=%i,m=%i" % (wf.n, wf.l, wf.m))
    
    #
    # Spherical Harmonics
    #

    # create spherical grid
    theta, phi = np.meshgrid(
        np.linspace(-np.pi, np.pi, 300),
        np.linspace(0, np.pi, 150),
        sparse=True
    )
    
    # compute spherical harmonic values
    Y = wf.spherical_harmonics(theta, phi)
    rho = (Y * Y.conjugate()).real
    
    # convert spherical to cartesian for plotting
    x = quantum.utils.spherical.spherical_to_cartesian(rho, theta, phi)

    # create face colors depending on amplitude
    facecolors = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=rho.min(), vmax=rho.max()),
        cmap=plt.cm.jet
    ).to_rgba(rho)
    
    # plot spherical harmonic
    ax1.plot_surface(x[0], x[1], x[2], rstride=1, cstride=1, color='b', facecolors=facecolors)
    ax1.set(title="Spherical Harmonic")
    #
    # Probability Distribution
    #
    
    # create spherical grid
    x = np.stack(
        np.meshgrid(
            np.linspace(0, 16, 100),
            np.linspace(0, 2*np.pi, 100),
            np.linspace(0, np.pi, 50),
            indexing='ij'
        ), axis=-1
    )
    
    # evaluate wave function on grid
    Y = wf.pdf(x, t=0)

    # use marching cubes algorithm to build mesh
    # from probability distribution
    v, f, _, _ = marching_cubes(
        volume=Y,
        level=1e-5,
        allow_degenerate=False, 
        spacing=[
            R/(x.shape[0]-1), 
            2*np.pi/(x.shape[1]-1), 
            np.pi/(x.shape[2]-1)
        ]
    )

    # update radius to maximum radius of interest
    R = v[:, 0].max()
    # compute phase for each face
    P = wf.psi(v, t=0).imag
    P = P[f].mean(axis=1)

    # convert vertices from spherical to cartesian coordinate system
    v = quantum.utils.spherical.spherical_to_cartesian(*v.T)
    v = np.stack(v, axis=-1)

    # create cyclic colormap
    facecolors = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=P.min(), vmax=P.max()),
        cmap=plt.cm.jet
    ).to_rgba(P)

    # plot
    ax2.add_collection(
        Poly3DCollection(
            v[f, :], 
            facecolors=facecolors, 
            alpha=0.5
        )
    )
    ax2.set(
        title="Probability Mesh",
        xlim=[-R, R],
        ylim=[-R, R],
        zlim=[-R, R],
    )

    #
    # Electron Trajectories
    # 
   
    # sample k points from the probability distribution
    # as initial query points for the trajectories 
    q = quantum.utils.sampling.inverse_transfer_sampling(
        pdf=partial(wf.pdf, t=0), 
        num_samples=200,
        bounds=[[0, R], [0, 2*np.pi], [0, np.pi]],
        delta=0.05
    )
    # specify sampling points in time
    t = np.arange(0, 50, 0.05)
     
    # compute bohmian trajectories
    Q = quantum.extra.BohmianMechanics(wave=wf).trajectory(q=q, t=t)
    # convert trajectories to cartesian coordinates
    Q = quantum.utils.spherical.spherical_to_cartesian(Q[..., 0], Q[..., 1], Q[..., 2])
    Q = np.stack(Q, axis=-1)

    # plot nucleus
    ax3.scatter(0, 0, 0, alpha=1, color="black")
    # plot all trajectories
    for i in range(Q.shape[1]):
        ax3.plot(Q[:, i, 0], Q[:, i, 1], Q[:, i, 2], alpha=0.3)
    ax3.set(
        title="Electron Trajectories",
        # use the same limits as for probability
        xlim=ax2.get_xlim(),
        ylim=ax2.get_ylim(),
        zlim=ax2.get_zlim(),
    )

    return fig

if __name__ == '__main__':

    fig = hydrogen(t=0)
    fig.savefig("/mnt/c/users/Nicla/OneDrive/Bilder/img.png")
