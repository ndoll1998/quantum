import numpy as np

def spherical_to_cartesian(
    r:np.ndarray, 
    theta:np.ndarray, 
    phi:np.ndarray
) -> np.ndarray:
    """ Convert spherical coordinates to cartesian

        Args:
            r (np.ndarray): radius
            theta (np.ndarray): polar angle
            phi (np.ndarray): azimuth angle
    
        Returns:
            x (np.ndarray): x-coordinate
            y (np.ndarray): y-coordinate
            z (np.ndarray): z-coordinate
    """
    x = r * (np.cos(phi) * np.sin(theta))
    y = r * (np.sin(phi) * np.sin(theta))
    z = r * (np.ones_like(phi) * np.cos(theta))
    return x, y, z

def cartesian_to_spherical(
    x:np.ndarray,
    y:np.ndarray,
    z:np.ndarray
) -> np.ndarray:
    """ Convert cartesian coordinates to spherical

        Args:
            x (np.ndarray): x-coordinate
            y (np.ndarray): y-coordinate
            z (np.ndarray): z-coordinate
    
        Returns:
            r (np.ndarray): radius
            theta (np.ndarray): polar angle
            phi (np.ndarray): azimuth angle
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x)
    return r, theta, phi

def sample_unit_sphere(
    n:int,
    d:int
) -> np.ndarray:
    """ Uniformly sample points in the unit sphere
        
        Args:
            n (int): number of points to sample
            d (int): dimension of the unit sphere

        Returns:
            x (np.ndarray): sampled points of shape (n, d)
    """
    # sample radius and direction
    r = np.random.uniform(0, 1, size=(n, 1))
    v = np.random.normal(size=(n, d))
    # normalize to stay in unit sphere
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    # compute final samples
    return r ** (1.0/d) * v
