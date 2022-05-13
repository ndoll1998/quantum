import quantum
import numpy as np
from scipy.integrate import simps
from functools import partial
import matplotlib.pyplot as plt

def hydrogen(
    t:float
) -> plt.Figure:

    # create analytic wave function
    wf = quantum.analytic.HydrogenLike(5, 3, 1)
    # create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5), subplot_kw={"projection": "3d"}, tight_layout=True)
    fig.suptitle("Hydrogen n=%i,l=%i,m=%i" % (wf.n, wf.l, wf.m))
    
    #
    # Spherical Harmonics
    #

    # create spherical grid
    theta, phi = np.meshgrid(
        np.linspace(0, 2*np.pi, 300),
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

    s = quantum.utils.sampling.inverse_transfer_sampling(
        pdf=partial(wf.pdf, t=0), 
        num_samples=50000,
        bounds=[[0, 8], [0, 2*np.pi], [0, np.pi]],
        delta=0.05
    )

    Y = wf.pdf(s, t=0)

    # convert to cartesian coordinates
    x = quantum.utils.spherical.spherical_to_cartesian(s[:, 0], s[:, 1], s[:, 2])

    # plot
    ax2.scatter(*x, alpha=Y / Y.max() * 0.03, marker='.')
    ax2.set(title="Distribution")    

    #
    # Electron Trajectories
    # 

    # take the first k points sampled from the distribution
    # as initial query points for the trajectories
    q = s[:200]
    t = np.arange(0, 20, 0.01)
     
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
    ax3.set(title="Electron Trajectories")

    return fig

if __name__ == '__main__':

    fig = hydrogen(t=0)
    plt.show()
    fig.savefig("/mnt/c/users/Nicla/OneDrive/Bilder/img.png")
