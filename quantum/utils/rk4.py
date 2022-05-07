import numpy as np
from typing import Callable, Tuple

def rk4(
    t:float,
    dt:float,
    x:np.ndarray,
    f:Callable[[np.ndarray, float], np.ndarray]
) -> Tuple[float, np.ndarray]:
    k1 = dt * f(x, t)
    k2 = dt * f(x + 0.5*k1, t + 0.5*dt)
    k3 = dt * f(x + 0.5*k2, t + 0.5*dt)
    k4 = dt * f(x + k3, t + dt)
    return t + dt, x + (k1 + 2*(k2+k3) + k4)/6.0
