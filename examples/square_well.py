import quantum
import numpy as np
import matplotlib.pyplot as plt

def infinite_square_well_1d(
    n:int,
    t:float,
) -> plt.Figure:
    
    # analytical solution
    wf_analytic = quantum.analytic.InfiniteSquareWell1D(n=n+1, L=1.0)
    # numeric solution
    V = quantum.extra.potentials.SquareWellPotential(L=1.0)
    wf_numeric = quantum.TISE(V).solve([[-1, 1]], dx=0.01, k=n+1)[-1]
    
    # compare energies
    print("Analytic Energy:", wf_analytic.E)
    print("Numeric Energy: ", wf_numeric.E)

    # inputs at which to evaluate
    x = np.arange(-1, 1, 0.01).reshape(-1, 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.suptitle("1D Infinite Square Well")
    # plot analytic solution
    y_analytic = wf_analytic(x, t=t)
    ax1.plot(x, y_analytic.real, label="real")
    ax1.plot(x, y_analytic.imag, label="imag")
    ax1.set(
        title="Analytic Wave Function for E=%.02f" % wf_analytic.E,
        xlabel="x", ylabel="$\psi_%i$" % n
    )
    ax1.legend()
    ax1.grid()
    # plot numeric solution
    y_numeric = wf_numeric(x, t=t)
    ax2.plot(x, y_numeric.real, label="real")
    ax2.plot(x, y_numeric.imag, label="imag")
    ax2.set(
        title="Numeric Wave Function for E=%.02f" % wf_numeric.E,
        xlabel="x", ylabel="$\psi_%i$" % n
    )
    ax2.legend()
    ax2.grid()
    # return figure
    return fig

if __name__ == '__main__':

    fig = infinite_square_well_1d(n=2, t=0.3)
    fig.savefig("/mnt/c/users/Nicla/OneDrive/Bilder/img.png")
