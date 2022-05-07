import quantum
import numpy as np
import matplotlib.pyplot as plt

def infinite_square_well_trajectories(
    num_particles:int
):
    # initial positions and time steps of the trajectories
    q = np.linspace(-0.7, 0.7, num_particles).reshape(-1, 1)
    t = np.arange(0, 2, 0.001)

    # the wave function used is a superposition of
    # the first two eigenstates to the infinite 
    # square well potential

    # analytical solutions
    wf_analytic = quantum.analytic.InfiniteSquareWell1D(n=1, L=1.0) + \
        quantum.analytic.InfiniteSquareWell1D(n=2, L=1.0)
    # compute trajectories from analytical solution
    Q_analytic = quantum.extra.BohmianMechanics(wf_analytic).trajectory(q=q, t=t)

    # numeric solution
    V = quantum.extra.potentials.SquareWellPotential(L=1.0)
    wf_numeric = np.sqrt(0.5) * quantum.SuperPositionAny(
        *quantum.TISE(V).solve([[-1, 1]], dx=0.01, k=2)
    )
    # compute trajectories from numerical solution
    Q_numeric = quantum.extra.BohmianMechanics(wf_numeric).trajectory(q=q, t=t)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    fig.suptitle("Bohmian Trajectories for 1D infinite square well")
    # plot analytic trajectories
    ax1.plot(t, Q_analytic[..., 0], color='black', alpha=0.5)
    ax1.set(
        title="Analytical Bohmian Trajectories",
        ylabel="x"
    )
    ax1.grid()
    # plot numeric trajectories
    ax2.plot(t, Q_numeric[..., 0], color='black', alpha=0.5)
    ax2.set(
        title="Numerical Bohmian Trajectories",
        ylabel="x", xlabel="t",
        xlim=(t.min(), t.max())
    )
    ax2.grid()
    return fig

if __name__ == '__main__':

    fig = infinite_square_well_trajectories(num_particles=20)
    fig.savefig("/mnt/c/users/Nicla/OneDrive/Bilder/img.png")
