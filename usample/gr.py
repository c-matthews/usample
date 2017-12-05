"""
Code for computing the GR of the umbrella sampling windows.
"""
import numpy as np

def get_gr(x):
    """
    Return the maximum GR across all dimensions, for multiple walkers.

    Parameters
    ----------
    x : 3d array
        NxWxD array, a trajectory of N steps with W walkers in D dimensions.

    Returns
    -------
    gr : float
        The maximum GR across all dimensions.
    """
    
    xs = np.shape(x)
     
    ndim = xs[2]
    
    gr = np.zeros( (ndim, 1) )
    
    for ii in range(ndim):
         
        xi=np.squeeze( x[:,:,ii] )
        
        gr[ii] = compute_gr( xi )
        
    return np.max( gr )


def compute_gr(x):
    """
    Compute the Gelman-Rubin (GR) estimate for a trajectory.

    Parameters
    ----------
    x : 2d array
        NxW array, with a trajectory of N steps of W walkers. 

    Returns
    -------
    gr : float
        The GR statistic, with smaller numbers usually indicating better sampling.
    """
    # For information see
    # http://astrostatistics.psu.edu/RLectures/diagnosticsMCMC.pdf
    # page 23 
    xs = np.shape(x) 
    
    N = xs[0]
    Nw = xs[1]
    
    W =  np.sum(  (x - np.mean(x,0) )**2 )  / ( Nw * (N-1) )
    
    
    B_N = np.sum(  (np.mean(x,0) - np.mean(x) )**2 )  / ( Nw  - 1 )
    
    sig2 = (N-1) * W / N +  B_N
    
    R = (Nw +1 ) * sig2 / (W* Nw )  - (N -1 ) / (Nw * N )
    
    R = np.sqrt( sig2 / W )
    
    
    return R -1.0

    
    