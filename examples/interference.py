import quantum
import numpy as np
import matplotlib.pyplot as plt

def interference_pattern():
    
    # superposition, particle can start out in any of the two slits
    wf = quantum.analytic.GaussianWavePacket(v=[0.1, 0.0], x0=[0.0, 1.0], s0=0.2) + \
        quantum.analytic.GaussianWavePacket(v=[0.1, 0.0], x0=[0.0, -1.0], s0=0.2)

    # create times at which to evaluate and
    # positions (which depend on time)
    t = np.arange(0, 2, 0.01)
    x = np.stack(np.meshgrid(
        t * 0.1, # x = vt
        np.arange(-15, 15, 0.05),
        indexing='ij'
    ), axis=-1)

    # TODO: vectorize this
    y = [wf.pdf(x[i, :], t) for i, t in enumerate(t)]
    y = np.asarray(y)
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.imshow(y, cmap='plasma')
    ax.set(
        title="Interference Pattern of superimposing two freely moving particles",
        ylabel="$t = x_0/t$", xlabel="$x_1$",
        yticks=range(0, len(t), 40), yticklabels=t[::40],
        xticks=range(0, 601, 60), xticklabels=np.arange(-15, 15.1, 0.05*60)
    )
    return fig

def interference_trajectories(
    num_particles:int
) -> plt.Figure:

    # superposition, particle can start out in any of the two slits
    wf = quantum.analytic.GaussianWavePacket(v=[0.1, 0.0], x0=[0.0, 1.0], s0=0.2) + \
        quantum.analytic.GaussianWavePacket(v=[0.1, 0.0], x0=[0.0, -1.0], s0=0.2)

    # create initial particle positions within slits
    q = np.random.uniform(-1, 1, size=(num_particles, 1))
    q = np.concatenate((
        np.zeros_like(q), 
        1.0 * q + 0.5 * np.sign(q)
    ), axis=-1)
    # time steps at which to evaluate for their trajectories
    t = np.arange(0, 2, 0.01)

    # compute bohmian trajectories
    Q = quantum.extra.BohmianMechanics(wave=wf).trajectory(q=q, t=t)
    # plot trajectories
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(t, Q[..., 1], color='black', alpha=0.01)
    ax.set(
        title="Bohmian Trajectories for Double Slit Experiment",
        xlabel="t", ylabel="x",
        xlim=(t.min(), t.max())
    )
    ax.grid()
    return fig

if __name__ == '__main__':

    # fig = interference_pattern()
    fig = interference_trajectories(num_particles=5000)
    plt.show()
    # fig.savefig("/mnt/c/users/Nicla/OneDrive/Bilder/img.png")
