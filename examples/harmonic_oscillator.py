import quantum
import numpy as np
import matplotlib.pyplot as plt

def harmonic_oscillator(
    n:int,
    t:float,
) -> plt.Figure:
    
    # solve time-indipendent schrödinger equation 
    # for 2d harmonic oscillator potential
    V = quantum.extra.potentials.HarmonicOscillatorPotential(w=1.0)
    wf = quantum.TISE(V).solve([[-10, 10], [-10, 10]], dx=0.1, k=n+1)[-1]
    
    # evaluate wave function on grid
    x = np.stack(np.meshgrid(
        np.arange(-5, 5, 0.1),
        np.arange(-5, 5, 0.1)
    ), axis=-1)
    y = wf(x, t=t)

    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle("2D Harmonic Oscillator for E=$%.02f$" % wf.E)
    # plot real components
    ax1.imshow(y.real, cmap='plasma', extent=(-5, 5, -5, 5))
    ax1.set(title="Real Component")
    # plot imaginary component
    ax2.imshow(y.imag, cmap='plasma', extent=(-5, 5, -5, 5))
    ax2.set(title="Imaginary Component")
    # return figure
    return fig

if __name__ == '__main__':

    fig = harmonic_oscillator(n=13, t=0.1)
    fig.savefig("/mnt/c/users/Nicla/OneDrive/Bilder/img.png")
