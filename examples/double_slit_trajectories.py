import quantum
import numpy as np
import matplotlib.pyplot as plt

def double_slit_trajectories(
    num_particles:int
) -> plt.Figure:
    
    # using analytic solution
    wf = quantum.analytic.DoubleSlit1D()
    # create initial particle positions within slits
    # and time steps at which to evaluate for their trajectories
    q = np.random.uniform(-1, 1, size=(num_particles, 1))
    q = q + np.sign(q) * (0.5 * wf.sd)
    t = np.arange(0, 2, 0.01)

    # compute bohmian trajectories
    Q = quantum.extra.BohmianMechanics(wave=wf).trajectory(q=q, t=t)
    # plot trajectories
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(t, Q[..., 0], color='black', alpha=0.05)
    ax.set(
        title="Bohmian Trajectories for Double Slit Experiment",
        xlabel="t", ylabel="x",
        xlim=(t.min(), t.max())
    )
    ax.grid()
    return fig

if __name__ == '__main__':

    fig = double_slit_trajectories(num_particles=1000)
    fig.savefig("/mnt/c/users/Nicla/OneDrive/Bilder/img.png")
