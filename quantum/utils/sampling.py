import numpy as np
from scipy.interpolate import interpn, interp1d
from typing import Union, Callable, Tuple

def metropolis(
    pdf:Callable[[np.ndarray], np.ndarray],
    sampler:Callable[[], Tuple[np.ndarray, float]],
    num_samples:int,
    burn:int =100,
    keep_rejected:bool =True
) -> np.ndarray:
    """ Implementation of the Metropolis-Hastings Algorithm for sampling from an arbitrary 
        (multivariate) probability density function and proposal sampler.

        Args:
            pdf (Callable[[np.ndarray], np.ndarray]): the target probability density function from which to sample
            sampler (Callable[[], Tuple[np.ndarray, float]]):
                the sampler used to generate proposal states. Must return the sampled vector and it's (unnormalized) probability.
                The sampled vector indicates the dimensionality and data type.
            num_samples (int): the number of points to sample from the target distribution
            burn (int): 
                the number of burn iterations, i.e. number of iterations to omit before recording. 
                Default is 100.
            keep_rejected (bool): whether to keep rejected samples (doubles). Default is True.

        Returns:
            samples (np.ndarray): the points sampled following the given density function in an array of shape (num_samples, num_dims).
    """
    # sample initial vector and evaluate target density
    x, x_prob = sampler()
    x_dens = pdf(x)
    # create empty array to store samples in
    samples = np.empty((num_samples, *x.shape), dtype=x.dtype)

    for i in range(burn + num_samples):

        # find next sample
        while True:
            # sample from proposal distribution
            y, y_prob = sampler(i)
            y_dens = pdf(y)

            # check acceptance
            accepted = x_prob * y_dens >= np.random.uniform(0, 1) * y_prob * x_dens
            if accepted or keep_rejected or (i < burn):
                break
        
        # overwrite if new sample is accepted
        if accepted:
            x, x_prob, x_dens = y, y_prob, y_dens
        
        # add to samples
        if i >= burn:
            samples[i-burn, ...] = x

    return samples

def metropolis_normal(
    pdf:Callable[[np.ndarray], np.ndarray],
    num_samples:int,
    burn:int =100,
    keep_rejected:bool =True,
    dim:int =1,
    loc:Union[float, np.ndarray] =0.0,
    scale:Union[float, np.ndarray] =1.0
) -> np.ndarray:
    """ Implementation of the Metropolis-Hastings Algorithm for sampling from an arbitrary 
        (multivariate) probability density function using the normal distribution as proposal
        function. 

        Args:
            pdf (Callable[[np.ndarray], np.ndarray]): the target probability density function from which to sample
            num_samples (int): the number of points to sample from the target distribution
            burn (int): 
                the number of burn iterations, i.e. number of iterations to omit before recording. 
                Default is 100.
            keep_rejected (bool): whether to keep rejected samples (doubles). Default is False.
            dim (int): Dimensionalty of the samples to be generated. Default is 1.
            loc (Union[float, np.ndarray]):
                Mean of the distribution. Defaults to 0.0. For more information see `numpy.random.normal`.
            scale (Union[float, np.ndarray]): 
                Standard deviation of the normal distribution used by the proposal sampler.
                Defaults to 1.0. For more information see `numpy.random.normal`.

        Returns:
            samples (np.ndarray): the points sampled following the given density function in an array of shape (num_samples, num_dims).
    """

    def normal_sampler(i=0):
        x = np.random.normal(loc=loc, scale=scale * (i/num_samples), size=dim)
        p = np.exp(-0.5 * ((x - loc)**2 / scale).sum())
        return x, p

    return metropolis(
        pdf=pdf,
        sampler=normal_sampler,
        num_samples=num_samples,
        burn=burn,
        keep_rejected=keep_rejected
    )


def inverse_transfer_sampling(
    pdf:Callable[[np.ndarray], np.ndarray],
    num_samples:int,
    bounds:np.ndarray,
    delta:float =0.1
):
    """ (n-dimensional) Inverse Transform Sampling for arbitrary probability density functions
    
        Args:
            pdf (Callable[[np.ndarray], np.ndarray]): The target density function from which to sample
            num_samples (int): the number of samples to generate
            bounds (np.ndarray): 
                the bounds in shape of (dim, 2) on which to evaluate the pdf. 
                This indicates the dimensionality of the sampled points. Also it 
                limits the minimum and maximum values a sample can take in the 
                corresponding dimension. Ideally choosen such that the pdf vanishes
                in both limits.
            delta (float): the distance of points on the regular grid
        
        Returns:
            samples (np.ndarray): the points sampled following the given density function in an array of shape (num_samples, num_dims).
    """

    dim = bounds.shape[0] if isinstance(bounds, np.ndarray) else len(bounds)
    # create empty array to store samples
    samples = np.empty((num_samples, dim))
    
    # create regular grid on which to evaluate the pdf
    axes = [np.arange(b, e, delta) for b, e in bounds]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    # evaluate pdf on regular grid    
    pdf = pdf(grid)

    # sample each dimension one at a time
    for d in range(dim):

        # compute the marginal density for the current dimension
        # given the previously already sampled dimensions
        marginal = np.sum(pdf, axis=tuple(range(d+1, dim)))
        if d > 0:
            marginal = interpn(axes[:d], marginal, samples[:, :d])
        
        # compute normalized cumulative density from marginal density 
        cdf = np.cumsum(marginal, axis=-1)
        cdf /= cdf[..., -1:]

        # inverse sampling
        u = np.random.uniform(0, 1, size=num_samples)
        if d > 0:
            for i in range(num_samples):
                inv_cdf = interp1d(cdf[i, :], axes[d], bounds_error=False, fill_value=axes[d][0])
                samples[i, d] = inv_cdf(u[i])
        else:
            inv_cdf = interp1d(cdf, axes[d], bounds_error=False, fill_value=axes[d][0])
            samples[:, d] = inv_cdf(u)

    return samples
