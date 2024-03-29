# -*- coding: utf-8 -*-
"""
Container for the primary EMUS routines.
"""
import numpy as np
try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp
try:
    import usample.linalg as lm
    import usample.autocorrelation as autocorrelation
    from .usutils import unpackNbrs
except ImportError:
    import linalg as lm
    import autocorrelation as autocorrelation
    from usutils import unpackNbrs

def calculate_obs(psis,z,f1data,f2data=None):
    """Estimates the value of an observable or ratio of observables.

    Parameters
    ----------
    psis : 3D data structure
        Data structure containing psi values.  See documentation for a detailed explanation.
    z : 1D array
        Array containing the normalization constants
    f1data : 2D data structure
        Trajectory of observable in the numerator.  First dimension corresponds to the umbrella index and the second to the point in the trajectory.
    f2data : 2D data structure, optional
        Trajectory of observable in the denominator.

    Returns
    -------
    avg : float
        The estimate of <f_1>/<f_2>.

    """
    f1avg = 0
    f2avg = 0
    for i,psi_i in enumerate(psis):
        psi_xi = np.array(psi_i)
        psi_i_sum = np.sum(psi_xi,axis=1)
        f1_i = np.array(f1data[i])/psi_i_sum
        if f2data is None:
            f2_i = 1./psi_i_sum
        else:
            f2_i = np.array(f2data[i])/psi_i_sum
        f1avg_i = np.average(f1_i)
        f2avg_i = np.average(f2_i)
        f1avg += z[i]*f1avg_i
        f2avg += z[i]*f2avg_i
    return f1avg / f2avg

def calculate_pmf(cv_trajs, psis, domain, z, nbins = 100,kT=1.):
    """Calculates the free energy surface for an umbrella sampling run.

    Parameters
    ----------
    cv_trajs : 2D data structure
        Data structure containing trajectories in the collective variable space.  See documentation for more detail.
    psis : 3D data structure
        Data structure containing psi values.  See documentation for a detailed explanation.
    domain : tuple
        Tuple containing the dimensions of the space over which to construct the pmf, e.g. (-180,180) or ((0,1),(-3.14,3.14)) z (1D array or list): Normalization constants for each state
    nbins : int or tuple, optional
        Number of bins to use.  If int, uses that many bins in each dimension.  If tuple, e.g. (100,20), uses 100 bins in the first dimension and 20 in the second.
    kT : float, optional
        Value of kT to scale the PMF by.  If not provided, set to 1.0

    Returns
    -------
    pmf : nd array
        Returns the potential of mean force as a d dimensional array, where d is the number of collective variables.

    """
    if domain is None:
        raise NotImplementedError

    domain = np.asarray(domain)
    if len(np.shape(domain)) == 1:
        domain = np.reshape(domain,(1,len(domain)))
    ndims = np.shape(domain)[0]
    if type(nbins) is int: # Make nbins to an iterable in the 1d case.
        nbins = [nbins]*ndims
    domainwdth = domain[:,1] - domain[:,0]

    # Calculate the PMF
    hist = np.zeros(nbins)
    for i,xtraj_i in enumerate(cv_trajs):
#        xtraj_i = (xtraj_i - domain[:,0])%domainwdth + domain[:,0]
        hist_i = np.zeros(nbins) # Histogram of umbrella i
        for n,coord in enumerate(xtraj_i):
            psi_i_n = psis[i][n]
            # We find the coordinate of the bin we land in.
            coordbins = (coord - domain[:,0])/domainwdth*nbins
            coordbins = tuple(coordbins.astype(int))
            weight = 1./np.sum(psi_i_n)
            hist_i[coordbins] += weight
        hist+=hist_i/len(xtraj_i)*z[i]
    pmf =-kT* np.log(hist)
    pmf -= min(pmf.flatten())


    # Calculate the centers of each histogram bin.
    return pmf

def calculate_zs(psis,neighbors=None,nMBAR=0,tol=1.E-15,use_iats=False,iat_method='ipce'):
    """Calculates the normalization constants for the states.

    Parameters
    ----------
    nMBAR : int, optional (default 0)
         Maximum number of MBAR iterations to perform.
    tol : float, optional (default 1.0E-15)
        If the relative residual falls beneath the tolerance, the MBAR iteration is truncated.
    use_iats : bool, optional
        If true, estimate integrated autocorrelation time in each MBAR iteration.  Likely unnecessary unless dynamics are expected to be drastically different in each state. If iats is provided, the iteration will use those rather than estimating them in each step.
    iat_method : string, optional
        Routine to use for calculating integrated autocorrelation times.  Currently accepts 'ipce', 'acor', and 'icce'.

    Returns
    -------
    z : 1D array
        Values for the Normalization constant in each state.
    F : 2D array
        Matrix to take the eigenvalue of for MBAR.
    iats : 1D array
        Estimated values of the autocorrelation time.  Only returned if use_iats is true.

    """
    L = len(psis) # Number of States
    Npnts = np.array([len(psis_i) for psis_i in psis])
    Npnts = (1.0 * Npnts ) / np.max(Npnts)


    if use_iats:
        z,F,iats = emus_iter(psis,neighbors=neighbors,return_iats=use_iats,iat_method=iat_method)
    else:
        z,F = emus_iter(psis,neighbors=neighbors,return_iats=use_iats,iat_method=iat_method)
        iats = np.ones(z.shape)


    # we perform the self-consistent polishing iteration
    for n in range(nMBAR):
        z_old = np.copy(z)
        z_old[ z_old<1e-100 ] = 1e-100
        Apart = Npnts/z_old
        Amat = np.outer(np.ones(L),Apart)
        Amat /= np.outer(np.ones(L),iats)
        if use_iats:
            z, F, iats = emus_iter(psis,Amat,neighbors=neighbors,return_iats=True,iat_method=iat_method)
        else:
            z, F = emus_iter(psis,Amat,neighbors=neighbors)
        # Check if we have converged.

        if np.max(np.abs(z-z_old)/z_old) < tol:
            break

    if use_iats:
        return z, F, iats
    else:
        return z, F


def emus_iter(psis, Avals=None, neighbors=None, return_iats = False,iat_method='ipce'):
    """Performs one step of the the EMUS iteration.

    Parameters
    ----------
    psis : 3D data structure
        Data structure containing psi values.  See documentation for a detailed explanation.
    Avals : 2D matrix, optional
        Weights in front of :math:`\psi` in the overlap matrix.
    neighbors : 2D array, optional
        List showing which states neighbor which.  See neighbors_harmonic in usutils.
    return_iats : bool, optional
        Whether or not to calculate integrated autocorrelation times of :math:`\psi_ii^*` for each window.
    iat_method : string, optional
        Routine to use for calculating said iats.  Accepts 'ipce', 'acor', and 'icce'.

    Returns
    -------
    z : 1D array
        Normalization constants for each state
    F : 2D array
        The overlap matrix constructed for the eigenproblem.
    iats : 1D array
        If return_iats chosen, returns the iats that have been estimated.
    """

    # Initialize variables
    L = len(psis) # Number of Windows
    F = np.zeros((L,L)) # Initialize F Matrix
    # Take care of defaults..
    if return_iats:
        iats = np.ones(L)
        iatroutine = ac._get_iat_method(iat_method)
    if Avals is None:
        Avals = np.ones((L,L))
    if neighbors is None:
        neighbors = np.outer(np.ones(L),range(L)).astype(int)

    for i in range(L):
        nbrs_i = neighbors[i]
        A_nbs = Avals[i][nbrs_i]
        nbr_index = list(nbrs_i).index(i)
        Fi_out = calculate_Fi(psis[i],nbr_index,A_nbs,return_iats)
        if return_iats:
            Fi, trajs = Fi_out
            iats[i] = iatroutine(trajs[nbr_index])[0]
#            print iats[i]
        else:
            Fi = Fi_out
        # Unpack the Neighbor list
        F[i] = unpackNbrs(Fi,nbrs_i,L)

    z = lm.stationary_distrib(F)
#    print F
#    print z
    if return_iats:
        return z, F, iats
    else:
        return z, F

def calculate_Fi(psi_i, i, Avals_i=None, return_trajs=False):
    """
    Calculates the values of a single row in the F matrix.
    If neighborlists are being used, psi_i, and Avals_i should be the
    neighborlisted data structure, and the row will be need to be unpacked
    using the neighborlist .

    Parameters
    ----------
        psi_i : 2D array-like
            Values of :math:`\psi` collected in window i.  The j'th column
            corresponds to the j'th neighboring window.
        i : int
            Index of the window where the data was collected.
        Avals_i : 1D array-like
            Weights in front of :math:`\psi_{ij} in the overlap matrix.
        return_trajs : bool, optional
            Whether or not to return the trajectories that are averaged to
            calculate the values of F.  These can be useful for estimating
            autocorrelation times and performing error analysis.

    Returns
    -------
        Fi : 1D numpy array
            The (neighborlisted) row in the F matrix
        trajs : 2D numpy array
            If return_trajs is True, returns the trajectories used in
            calculating the values of F

    """
    # Setup
    L = np.shape(psi_i)[1] # Number of neighboring windows
    Fi = np.zeros(L)
    # Take care of defaults
    if Avals_i is None:
        Avals_i = np.ones(L)

    psi_i = np.array(psi_i)
    denom = np.dot(psi_i,Avals_i)
    lse = logsumexp( psi_i,axis=1,b=Avals_i)

    if return_trajs:
        trajs = np.zeros(psi_i.shape)
    for j in range(L):
        Ftraj = psi_i[:,j] - lse
        Fi[j] = np.exp( logsumexp(Ftraj) - np.log(len(Ftraj)))
        #Ftraj = psi_i[:,j]/denom # traj \psi_j/{\sum_k \psi_k A_k}
        #Fi[j] = np.average(Ftraj)
        Fi[j] *= Avals_i[i]
        if return_trajs:
            trajs[:,j] = Ftraj
    if return_trajs:
        return Fi, trajs
    else:
        return Fi
