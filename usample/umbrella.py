# -*- coding: utf-8 -*-
"""
Class object for each individual umbrella.
"""
import numpy as np 
try:
    from .gr import get_gr 
    from .makecv import get_cv
except ImportError:
    from gr import get_gr
    from makecv import get_cv
import copy

def g_lnprob(p , lpf , biasinfo, lpfargs, lpfkwargs): 
    """
    The global function returning the log probability with the bias.

    Parameters
    ----------
    p : 1D array
        The point for which to return the log probability.
    lpf : function pointer
        The function handle for the log probability.
    biasinfo : list
        A list containing the information about the particular biasing function.

    Returns
    -------
    p1 : float
        The biased log probability
    p2 : float
        The bias (so the log probability = p1-p2 ).
    """
        
    L = lpf(p , *lpfargs, **lpfkwargs ) 
    
    if (np.isfinite(L) ):
        bias = g_get_bias( p , L , biasinfo )
    else:
        bias = 0
      
    return (L + bias, bias)

def g_get_bias( p , L , biasinfo ):
    """
    A global function returning the bias imposed by the umbrella.

    Parameters
    ----------
    p : 1D array
        The point for which to return the log probability.
    L : float
        The current unbiased log probability.
    biasinfo : list
        A list containing the information about the particular biasing function.

    Returns
    -------
    bias : float
        The bias introduced by the umbrella
    """
     
    center,cvfn,sigma,tempfac = biasinfo
    
    if (cvfn is None):
        return L * tempfac
    
    cvstyle = cvfn[0]
    
    if (cvstyle=="line"):
    
        cv = get_cv( p , cvfn ) 
        
        xp = 1.0 - np.abs(cv - center) / (1.0*sigma)
        xp = np.array(xp)
        L = np.array(L).squeeze()
        
        kk = xp <= 0 
        kc = np.invert(kk)
        
        xp[kk] = -np.inf
        xp[kc] =  tempfac*L[kc] + np.log(xp[kc] )
        
        return xp 
    
    if (cvstyle=="grid"):
    
        cv = get_cv( p , cvfn ) 
        
        xp  = 1.0 - np.abs(np.array(cv).T - center) / (1.0*sigma)
        xp  = np.array(xp)
        L = np.array(L).squeeze()
        
        kk = xp <= 0 
        kc = np.invert(kk)
        
        xp[kk] = -np.inf
        xp[kc] =  np.log(xp[kc] )
        
        if (xp.ndim==1):
            xp = np.sum(xp)
        else:
            xp = np.sum(xp,axis=1).squeeze()
            
        
        return xp+tempfac*L

    return 0


def initiate_pool(i):
    """
    Helper function used in the parallelization of the code.
    Should return/do nothing, only executed upon completion of the loop.
    """
    pass

class Umbrella:
    """
    The umbrella object itself, containing trajectory and biasing information for this distribution.
    """
    
    def __init__(self,lpf, ic, nows, sampler=None, comm=None, ranks=None, lpfargs=[], lpfkwargs={}, samplerargs={}, temp=1.0, center=0.0, cvfn=None,sigma=1   ):
        """
        Initializer for the umbrella class.

        Parameters
        ----------
        lpf : function
            The function handle for the log probability to be sampled.
        ic : 1d array
            Initial condition for the sampling to start.
        nows : integer
            The total number of windows (umbrellas) to use.
        sampler : ParallelSampler object
            The sampler to use to sample the umbrellas.
        comm : MPI4PY communicator (optional)
            The communicator to use to sample this window in parallel.
        ranks : list (optional)
            A list of which mpi processes should run this window.
        lpfargs : list (optional)
            Arguments for the LPF function.
        lpfkwargs : dictionary (optional)
            KwArgs for the LPF function.
        samplerargs : dictionary (optional)
            Arguments for the sampler we shall run.
        temp : float (optional, default=1.0)
            The temperature to use within this umbrella.
        center : float (optional, default=0.0)
            The center of the umbrella's biasing distribution to use within this umbrella.
        cvfn : list (optional)
            The details of the collective variable to use in order to bias the sampling.
        sigma : float (optional, default=1)
            The scaling parameter for the umbrella sampling bias function.

        Returns
        -------
        umbrella : Umbrella
            The initialized umbrella object.
        """
        self.lpf = lpf
        self.lpfargs = lpfargs
        self.lpfkwargs = lpfkwargs
        
        self.temp = temp
            
        self.center = center
             
        self.sigma = sigma
        self.cvfn = cvfn
        
        self.tempfac = 1.0 / self.temp - 1.0 
        self.nows = nows
        
        self.biasinfo = [self.center,self.cvfn,self.sigma,self.tempfac]
        
        # Initialize the positions around the ic  
        self.ic = np.array(ic).squeeze()
        
        if (self.ic.ndim==1):
            self.p = (1e-5) * np.random.normal(size=(nows, len(ic) ) ) + self.ic 
        else:
            self.p = self.ic
        
        self.lnprob0 = None  
        self.blobs0 = None  
        
        self.traj_pos = []
        self.traj_prob = []
        self.traj_blob = []
        
        self.tsteps = 0
        self.acorval = 0
        
        self.pool = None
            
        if not comm==None:
            from mpi4py import MPI
            try:
                import usample.mpi_pool as mpi_pool
            except ImportError:
                import mpi_pool
            if MPI.COMM_WORLD.Get_rank() in ranks:
                self.pool = mpi_pool.MPIPool( comm=comm )
                self.pool.map( initiate_pool , range(  self.pool.size ) )
                 

        # Setup the sampler. At the moment, just use emcee with g_lnprob as the log likelihood eval
        self.sampler = sampler(self.nows,  np.shape(self.p)[1] , g_lnprob, pool=self.pool , args=[self.lpf , self.biasinfo, lpfargs, lpfkwargs ], **samplerargs )
         
    def get_bias(self, p , L ):
        """
        Return the log of the biasing function.

        Parameters
        ----------
        p : 1d array
            Point to evaluate the bias for upon.
        L : float
            The log probability at this point. 

        Returns
        -------
        bias : float
            The bias introduced by the umbrella object.
        """ 
        return g_get_bias( p , L , self.biasinfo )
    
    def get_state(self):
        """
        Return the current position, log probability, and bias data of the umbrella.

        Returns
        -------
        p : Array
            An array containing data from all points sampling this umbrella.
        lpf : Array
            The biased log probabilities from all points.
        blobs : Array
            The log of the bias functions for all points.
        """ 
        return [self.p , self.lnprob0, self.blobs0]
    
    def set_state(self,z):
        """
        Sets the state of the umbrella to be a new value.

        Parameters
        ----------
        z : list
            A list containing three arrays: [positions, lpfs, biases].  
        """ 
        p,lnprob0,blobs0 = z
        
        self.p = np.copy( np.array(p) )
        self.lnprob0 = np.copy( np.array( lnprob0 ) )
        self.blobs0 = np.copy( np.array( blobs0 ) )
        
        return
    
    def get_traj(self):
        """
        Return the total trajectory information from this umbrella. 

        Returns
        -------
        traj_pos : Array
            The trajectory of all walkers sampling this umbrella.
        traj_prob : Array
            The biased log probability of each walker at each step.
        traj_blob : Array
            The introduced bias at each step, for each walker.
        """
        return [self.traj_pos , self.traj_prob, self.traj_blob]
    
    def get_acor(self):
        """
        Return the maximum autocorrelation time as reported by the sampler.

        Returns
        -------
        acor : float
            The maximum autocorrelation time computed so far.
        """
        return self.acorval
    
    def set_traj(self,z):
        """
        Overwrites the trajectory of the umbrella.

        Parameters
        ----------
        z : list
            A list containing three arrays: [positions, lpfs, biases].  
        """  
        traj_pos,traj_prob,traj_blob = z
        
        self.traj_pos = copy.deepcopy( traj_pos )
        self.traj_prob = copy.deepcopy( traj_prob ) 
        self.traj_blob = copy.deepcopy( traj_blob )
        
        return
    
 
    def sample(self, nsteps, thin=1):
        """
        Sample the distribution defined by this umbrella, using the given sampler.

        Parameters
        ----------
        nsteps : integer
            The number of steps to run.
        thin : integer (optional, default=1)
            How frequently to save steps, for thinning the trajectory.  
        """ 
        counter = 0 
        for (pos , prob , rstate , blobs ) in  self.sampler.sample( self.p , lnprob0=self.lnprob0 , blobs0=self.blobs0, iterations=nsteps ):
            
            if ((counter % thin )==0):
                self.traj_pos.append( pos.copy() )
                self.traj_prob.append( prob.reshape(np.shape(blobs)) - blobs  )
                self.traj_blob.append( np.array(blobs).copy() )
                
            counter += 1
         
        self.p = pos
        self.lnprob0 = prob
        self.blobs0 = blobs
        self.tsteps += nsteps 
        
        self.gr =  get_gr( np.array(self.traj_pos ) )
        
        try:
            self.acorval = np.max( self.sampler.acor )
        except:
            self.acorval = self.tsteps
        
        
        
        
        
        
