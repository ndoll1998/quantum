import numpy as np
from typing import Callable, Tuple

def rk4(
    t:float,
    dt:float,
    x:np.ndarray,
    f:Callable[[np.ndarray, float], np.ndarray]
) -> Tuple[float, np.ndarray]:
    """ Implementation of a single RK4 step, a 4-th Order Runge-Kutta 
        method to iteratively solve initial-value-problems (IVP)
        
        Args:
            t (float): the time at which to approximate a solution
            dt (float): the time step
            x (np.ndarray): the input at which to approximate a solution
            f (Callable[[np.ndarray, float], np.ndarray]): 
                the right hand side to the initial value problem

        Returns:
            t' (float): the next point in time
            y (np.ndarray): the value of the IVP at the time t' and position x
    """
    k1 = dt * f(x, t)
    k2 = dt * f(x + 0.5*k1, t + 0.5*dt)
    k3 = dt * f(x + 0.5*k2, t + 0.5*dt)
    k4 = dt * f(x + k3, t + dt)
    return t + dt, x + (k1 + 2*(k2+k3) + k4)/6.0
